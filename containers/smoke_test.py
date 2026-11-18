"""Is this runtime able to run the pipeline? One JSON report, on one GPU, in a few minutes.

    python containers/smoke_test.py                          # torch, attention at 32k, LoRA, Triton kernels, imports
    python containers/smoke_test.py --model Qwen/Qwen3.5-4B  # also: vLLM generates with that (cached) model

Run it through the site so it gets a GPU and the container: ``bash slurm/submit.sh --site lumi gen
slurm/jobs/smoke.sbatch``. It exits 1 if a required check fails. What each check guards against:

| check | required | why |
|---|---|---|
| torch_gpu | yes | the GPU is visible to PyTorch (ROCm reports through the torch.cuda API) |
| matmul_bf16 | yes | bf16 works and is fast enough (a rough TFLOP/s figure) |
| sdpa_32k | yes | causal attention over 32,768 tokens without the O(n^2) math kernel; training and generation need it |
| lora_backward | yes | PEFT LoRA forward/backward with gradient checkpointing on a tiny random Llama |
| scisum_import | yes | the package and its settings load (data root, profiles) |
| vllm_import | yes | the generation/grading engine is installed |
| fla_gated_delta | no (yes for Qwen3.5 students) | the Triton kernel of Qwen3.5's linear attention; without it transformers falls back to a slow, memory-hungry path |
| vllm_generate | no (with --model) | vLLM loads a real model and generates on this GPU |
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPORT: dict = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""), "checks": {}}
REQUIRED = ("torch_gpu", "matmul_bf16", "sdpa_32k", "lora_backward", "scisum_import", "vllm_import")


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                info = fn(*a, **kw) or {}
                REPORT["checks"][name] = {"ok": True, "seconds": round(time.time() - t0, 2), **info}
            except Exception as e:  # noqa: BLE001 - every failure is a finding
                REPORT["checks"][name] = {"ok": False, "seconds": round(time.time() - t0, 2),
                                          "error": f"{type(e).__name__}: {e}"[:600],
                                          "trace": traceback.format_exc()[-1500:]}
        return run
    return deco


@check("torch_gpu")
def torch_gpu():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
            "devices": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


@check("matmul_bf16")
def matmul_bf16():
    import torch
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0, n = time.time(), 20
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"tflops": round(2 * 8192 ** 3 * n / dt / 1e12, 1)}


@check("sdpa_32k")
def sdpa_32k():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out = {}
    q = torch.randn(1, 16, 32768, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, 32768, 128, device="cuda", dtype=torch.bfloat16).repeat_interleave(4, 1)
    v = torch.randn_like(k)
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with sdpa_kernel([backend]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y.sum().backward()
            torch.cuda.synchronize()
            out[name] = {"ok": True, "seconds": round(time.time() - t0, 2),
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        q.grad = None
    assert any(x["ok"] for x in out.values()), f"no memory-efficient attention kernel at 32k tokens: {out}"
    return out


@check("lora_backward")
def lora_backward():
    import peft
    import torch
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1000, max_position_embeddings=4096)
    m = LlamaForCausalLM(cfg).to("cuda", dtype=torch.bfloat16)
    m.gradient_checkpointing_enable()
    m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "down_proj"]))
    x = torch.randint(0, 1000, (1, 2048), device="cuda")
    loss = m(input_ids=x, labels=x).loss
    loss.backward()
    assert torch.isfinite(loss).item()
    return {"loss": round(float(loss), 3), "transformers": transformers.__version__, "peft": peft.__version__}


@check("fla_gated_delta")
def fla_gated_delta():
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    b, t, h, d = 1, 4096, 4, 64
    q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(b, t, h, d, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(b, t, h, device="cuda", dtype=torch.float32))
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    assert torch.isfinite(o.float()).all().item()
    import fla
    return {"fla": getattr(fla, "__version__", "?")}


@check("scisum_import")
def scisum_import():
    from scisum.config import get_settings
    s = get_settings()
    return {"data_root": str(s.data_root), "data_root_exists": Path(s.data_root).exists(), "base_model": s.base_model}


@check("vllm_import")
def vllm_import():
    import vllm
    return {"vllm": vllm.__version__}


@check("vllm_generate")
def vllm_generate(model: str, max_len: int):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=0.85, trust_remote_code=True)
    t0 = time.time()
    out = llm.generate(["Summarise: the cat sat on the mat."], SamplingParams(max_tokens=64, temperature=0.0), use_tqdm=False)
    return {"model": model, "seconds_generate": round(time.time() - t0, 2), "text": out[0].outputs[0].text[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="", help="a cached HF model id or path for the vLLM generation check")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", default="", help="also write the report here (default runs/logs/smoke-<host>-<job>.json)")
    a = ap.parse_args()
    torch_gpu()
    if REPORT["checks"]["torch_gpu"]["ok"]:
        matmul_bf16()
        sdpa_32k()
        lora_backward()
        fla_gated_delta()
    scisum_import()
    vllm_import()
    if a.model:
        vllm_generate(a.model, a.max_model_len)
    failed = [c for c in REQUIRED if not REPORT["checks"].get(c, {}).get("ok")]
    REPORT["required_failed"] = failed
    REPORT["ok"] = not failed
    text = json.dumps(REPORT, indent=1)
    print(text)
    out = Path(a.out or f"runs/logs/smoke-{REPORT['host']}-{REPORT['job'] or 'local'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
