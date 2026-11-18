"""Controlled inference benchmark of one student (base model + optional LoRA adapter) on fixed test papers.

    python scripts/analysis/bench_inference.py --student-env configs/students/<tag>.env [--adapter PATH] \\
        --out runs/<run>/bench/<ckpt>.json [--n-papers 200] [--latency-papers 20] [--warmup 3] [--repeats 2]

Run it on a GPU node, alone on its device(s), never next to another job on the same GPU. It measures, on
the same papers and with the same decoding as the evaluation (greedy, the evaluation's token cap):

1. **Single-request latency** (one paper at a time, after ``--warmup`` papers that are not counted):
   time to first token (TTFT), time per output token (TPOT) and end-to-end time per paper; mean, p50, p95.
   TTFT is measured as the time of a 1-token generation of the same prompt (prefill + first token);
   prefix caching is off, so that call does not warm the full one. When vLLM exposes per-request timing
   metrics they are recorded too (``ttft_source`` says which was used).
2. **Offline throughput at full batching**: all ``--n-papers`` prompts in one ``generate`` call (vLLM
   schedules them with continuous batching), repeated ``--repeats`` times; papers/hour, output and prompt
   tokens/s, and GPU-seconds per paper (= GPUs x wall seconds / papers), the cost axis of the Pareto plots.
3. **Memory**: weights on disk (base and adapter), the KV-cache bytes per token computed from the model
   config (full-attention layers only: linear-attention / recurrent layers keep a fixed-size state), the
   KV cache one paper needs at the median and p95 context, and the device memory the engine took (which
   reflects ``--gpu-frac``, not the model's need).
4. **Model load time** (engine construction: weight loading, memory profiling, graph capture). The engine's
   import time is recorded separately (``import_s``): it depends on the filesystem, not on the model.
5. A hardware and software fingerprint (GPU model and count, CUDA or ROCm, driver, torch, vLLM,
   transformers) so that only like-for-like numbers are compared.

``--adapter`` serves the LoRA adapter through vLLM's LoRA path, as the evaluation does. To measure a merged
model instead, merge it first (``scripts/analysis/merge_adapter.py``) and pass the merged directory with
``--model-dir``; compare the two JSON files. ``--engine hf`` is a fallback when vLLM is not available: latency
only, with Hugging Face ``generate`` (much slower; not comparable with vLLM numbers).

Prompts come from the student's built prompt shards (``data/<model_tag>/prompts/test/part-*.jsonl.gz``,
written by ``scisum data build``), so they are byte-identical to the evaluation's. Papers are the first N of
``--ids`` (default: the packaged test-paper list) in that file's order. Output: one JSON file, schema
``scisum.bench_inference/1`` (see ``docs/ANALYSIS.md``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import describe, parse_env_file, read_jsonl  # noqa: E402

SCHEMA = "scisum.bench_inference/1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_IDS = ("data/eval/test_ids.json", "data/eval/eval_ids.json", "data/eval/test_papers.json")


# ---- inputs -------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[bench {datetime.now(UTC).strftime('%H:%M:%S')}Z] {msg}", file=sys.stderr, flush=True)



def apply_env(paths: list[str]) -> dict[str, str]:
    """Load ``KEY=VALUE`` profiles into ``os.environ`` (later files win) before ``Settings`` is built."""
    merged: dict[str, str] = {}
    for p in paths:
        if p:
            merged.update(parse_env_file(p))
    os.environ.update(merged)
    return merged


def load_ids(path: str | None) -> list[str] | None:
    cands = [path] if path else [str(REPO / p) for p in DEFAULT_IDS]
    for c in cands:
        if c and Path(c).exists():
            data = json.loads(Path(c).read_text())
            if isinstance(data, dict):       # {"ids": [...]} or {"test": [...]}
                data = data.get("ids") or data.get("test") or []
            return [str(x) for x in data]
    if path:
        raise SystemExit(f"--ids {path}: no such file")
    return None


def load_prompts(prompt_dir: Path, ids: list[str] | None, n: int) -> list[dict]:
    """The first ``n`` prompt records, in the order of ``ids`` (or of the shards when there is no list)."""
    parts = sorted(prompt_dir.glob("part-*.jsonl.gz"))
    if not parts:
        raise SystemExit(f"no prompt shards in {prompt_dir}: build them first "
                         "(`scisum data build --split test` with this student's profile)")
    want = set(ids[: n * 3]) if ids else None    # enough ids even if a few have no prompt; stops reading early
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for p in parts:
        if want is not None and len(by_id) >= len(want):
            break
        for r in read_jsonl(p):
            if want is None or r["id"] in want:
                by_id[r["id"]] = {"id": r["id"], "prompt": r["prompt"],
                                  "prompt_tokens": int((r.get("meta") or {}).get("prompt_tokens") or 0)}
                order.append(r["id"])
                if want is None and len(order) >= n:
                    break
        if want is None and len(order) >= n:
            break
    seq = [i for i in ids if i in by_id] if ids else order
    if len(seq) < n:
        print(f"warning: only {len(seq)} of the {n} requested papers have prompts", file=sys.stderr)
    return [by_id[i] for i in seq[:n]]


# ---- model facts --------------------------------------------------------------------------------


def kv_bytes_per_token(config: dict, dtype_bytes: int = 2) -> dict:
    """KV-cache bytes one token occupies, from a Hugging Face config dict (``text_config`` if nested).

    Only layers with a growing KV cache count: ``layer_types`` entries other than full/sliding attention
    (linear attention, Mamba, recurrence) keep a fixed-size state and are reported separately.
    Sliding-window layers are counted as full here (an upper bound at long context)."""
    c = config.get("text_config") or config
    n_layers = int(c.get("num_hidden_layers") or 0)
    heads = int(c.get("num_attention_heads") or 0)
    kv_heads = int(c.get("num_key_value_heads") or heads or 0)
    head_dim = int(c.get("head_dim") or (int(c.get("hidden_size") or 0) // heads if heads else 0))
    types = c.get("layer_types") or c.get("layers_block_type")
    if types:
        kv_layers = sum(1 for t in types if "attention" in str(t) and "linear" not in str(t))
    else:
        kv_layers = n_layers
    per_tok = 2 * kv_layers * kv_heads * head_dim * dtype_bytes
    return {"bytes_per_token": per_tok, "kv_layers": kv_layers, "layers": n_layers,
            "fixed_state_layers": n_layers - kv_layers, "kv_heads": kv_heads, "head_dim": head_dim,
            "dtype_bytes": dtype_bytes, "source": "computed from config"}


def dir_size_gb(path: Path, patterns: tuple[str, ...] = ("*.safetensors", "*.bin")) -> float | None:
    if not path or not path.exists():
        return None
    total = sum(f.stat().st_size for pat in patterns for f in path.rglob(pat))
    return round(total / 2**30, 3) if total else None


def local_model_dir(model: str, revision: str | None) -> Path | None:
    """The local directory of a model: a path as given, else the Hugging Face cache snapshot (no network)."""
    if Path(model).exists():
        return Path(model)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, revision=revision or None, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached, or no huggingface_hub: report as unknown
        return None


def read_config(model_dir: Path | None) -> dict:
    if model_dir and (model_dir / "config.json").exists():
        return json.loads((model_dir / "config.json").read_text())
    return {}


# ---- fingerprint --------------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def fingerprint() -> dict:
    """GPU model/count/memory, CUDA or ROCm, driver, library versions, CPU, Slurm job."""
    hw: dict = {"gpus": [], "gpu_count": 0, "backend": None, "driver": None}
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpu_count"] = torch.cuda.device_count()
            for i in range(hw["gpu_count"]):
                p = torch.cuda.get_device_properties(i)
                hw["gpus"].append({"name": p.name, "total_gb": round(p.total_memory / 2**30, 1)})
        hw["backend"] = ("rocm " + torch.version.hip) if getattr(torch.version, "hip", None) else (
            ("cuda " + torch.version.cuda) if torch.version.cuda else "cpu")
    except ImportError:
        pass
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        hw["driver"] = "nvidia " + drv.splitlines()[0]
    else:
        out = _run(["rocm-smi", "--showdriverversion"]) or _run(["amd-smi", "version"])
        m = re.search(r"(?:Driver version|driver)[^\d]*([\d.]+)", out, re.I)
        hw["driver"] = ("amdgpu " + m.group(1)) if m else (out.splitlines()[0] if out else None)
    cpu = ""
    with contextlib.suppress(OSError):
        cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name")), "")
    hw["cpu"] = cpu or platform.processor()
    sw = {k: _version(k) for k in ("torch", "vllm", "transformers", "peft", "flash_attn")}
    sw["python"] = platform.python_version()
    prov = {"host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
            or os.environ.get("HIP_VISIBLE_DEVICES")}
    git = _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    if git:
        prov["git_commit"] = git + ("-dirty" if _run(["git", "-C", str(REPO), "status", "--porcelain"]) else "")
    return {"hardware": hw, "software": sw, "provenance": prov}


def device_free_gb() -> list[float]:
    try:
        import torch

        return [round(torch.cuda.mem_get_info(i)[0] / 2**30, 2) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


# ---- engines ------------------------------------------------------------------------------------


class VllmEngine:
    """vLLM offline engine; the LoRA adapter (if any) goes through vLLM's LoRA path, as in evaluation."""

    name = "vllm"

    def __init__(self, s, served: dict, adapter: str | None, max_model_len: int, tp: int, gpu_frac: float):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self._sp = SamplingParams
        self.llm = LLM(dtype="bfloat16", max_model_len=max_model_len, tensor_parallel_size=tp,
                       gpu_memory_utilization=gpu_frac, enable_lora=adapter is not None,
                       max_lora_rank=int(getattr(s, "lora_r", 64)), max_loras=1, seed=0,
                       enable_prefix_caching=False, trust_remote_code=bool(getattr(s, "trust_remote_code", False)),
                       **served)
        self.lora = LoRARequest("bench", 1, adapter) if adapter else None
        self.gpus = tp

    def generate(self, prompts: list[str], max_tokens: list[int]) -> list[dict]:
        params = [self._sp(temperature=0.0, top_p=1.0, max_tokens=m) for m in max_tokens]
        outs = self.llm.generate(prompts, params, use_tqdm=False,
                                 **({"lora_request": self.lora} if self.lora else {}))
        res = []
        for o in outs:
            c = o.outputs[0]
            m = getattr(o, "metrics", None)
            ttft = None
            if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                ttft = m.first_token_time - m.arrival_time
            res.append({"text": c.text, "out_tokens": len(c.token_ids), "prompt_tokens": len(o.prompt_token_ids),
                        "finish_reason": c.finish_reason, "metrics_ttft_s": ttft})
        return res


class HfEngine:
    """Hugging Face ``generate`` fallback (latency only)."""

    name = "hf"

    def __init__(self, s, model: str, adapter: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = s.hf_kwargs() if hasattr(s, "hf_kwargs") else {}
        self.tok = AutoTokenizer.from_pretrained(model, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cuda", **kw)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.gpus = 1

    def generate(self, prompts: list[str], max_tokens: list[int]) -> list[dict]:
        import torch

        res = []
        for p, m in zip(prompts, max_tokens, strict=True):
            ids = self.tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=m, do_sample=False)
            new = out[0, ids.shape[1]:]
            res.append({"text": self.tok.decode(new, skip_special_tokens=True), "out_tokens": int(new.shape[0]),
                        "prompt_tokens": int(ids.shape[1]), "finish_reason": None, "metrics_ttft_s": None})
        return res


# ---- measurements -------------------------------------------------------------------------------


def max_new_tokens(prompt_tokens: int, cap: int, max_model_len: int) -> int:
    """The evaluation's rule: the configured cap, but never past the engine window."""
    return max(1, min(cap, max_model_len - prompt_tokens - 8))


def measure_latency(engine, recs: list[dict], cap: int, max_model_len: int, warmup: int, clock=time.perf_counter) -> dict:
    """One paper at a time: a 1-token call (TTFT) then the full generation (end-to-end)."""
    for r in recs[:warmup]:
        engine.generate([r["prompt"]], [max_new_tokens(r["prompt_tokens"], cap, max_model_len)])
    rows = []
    for r in recs[warmup:]:
        t0 = clock()
        engine.generate([r["prompt"]], [1])
        ttft = clock() - t0
        t1 = clock()
        out = engine.generate([r["prompt"]], [max_new_tokens(r["prompt_tokens"], cap, max_model_len)])[0]
        e2e = clock() - t1
        n = out["out_tokens"]
        rows.append({"id": r["id"], "prompt_tokens": out["prompt_tokens"], "out_tokens": n, "ttft_s": ttft,
                     "metrics_ttft_s": out.get("metrics_ttft_s"), "e2e_s": e2e,
                     "tpot_s": (e2e - ttft) / (n - 1) if n > 1 else None})
    have_metrics = rows and all(x["metrics_ttft_s"] is not None for x in rows)
    ttft_key = "metrics_ttft_s" if have_metrics else "ttft_s"
    return {"n": len(rows), "warmup": warmup,
            "ttft_source": "engine request metrics" if have_metrics else "1-token generation",
            "ttft_s": describe(x[ttft_key] for x in rows),
            "ttft_1token_s": describe(x["ttft_s"] for x in rows),
            "tpot_s": describe(x["tpot_s"] for x in rows if x["tpot_s"] is not None),
            "e2e_s": describe(x["e2e_s"] for x in rows),
            "out_tokens": describe(x["out_tokens"] for x in rows),
            "prompt_tokens": describe(x["prompt_tokens"] for x in rows),
            "per_paper": rows}


def _json_ok(text: str) -> bool:
    t = text.split("</think>")[-1]
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return False
    try:
        json.loads(t[a:b + 1])
        return True
    except json.JSONDecodeError:
        return False


def measure_throughput(engine, recs: list[dict], cap: int, max_model_len: int, repeats: int,
                       clock=time.perf_counter) -> dict:
    """All prompts in one call, ``repeats`` times; the median repeat is the reported one."""
    prompts = [r["prompt"] for r in recs]
    caps = [max_new_tokens(r["prompt_tokens"], cap, max_model_len) for r in recs]
    reps = []
    outs: list[dict] = []
    for _ in range(max(1, repeats)):
        t0 = clock()
        outs = engine.generate(prompts, caps)
        wall = clock() - t0
        n_out = sum(o["out_tokens"] for o in outs)
        n_in = sum(o["prompt_tokens"] for o in outs)
        reps.append({"wall_s": wall, "output_tokens": n_out, "prompt_tokens": n_in})
    reps_sorted = sorted(reps, key=lambda x: x["wall_s"])
    med = reps_sorted[len(reps_sorted) // 2]
    n = len(recs)
    wall = med["wall_s"]
    return {"n": n, "repeats": len(reps), "wall_s": round(wall, 3), "wall_s_all": [round(x["wall_s"], 3) for x in reps],
            "papers_per_s": n / wall if wall else None, "papers_per_hour": 3600 * n / wall if wall else None,
            "output_tokens_per_s": med["output_tokens"] / wall if wall else None,
            "prompt_tokens_per_s": med["prompt_tokens"] / wall if wall else None,
            "gpus": engine.gpus, "gpu_seconds_per_paper": engine.gpus * wall / n if n else None,
            "out_tokens_mean": med["output_tokens"] / n if n else None,
            "json_parse_share": sum(_json_ok(o["text"]) for o in outs) / n if n else None,
            "hit_token_limit_share": (sum(o["finish_reason"] == "length" for o in outs) / n
                                      if n and outs and outs[0]["finish_reason"] is not None else None)}


# ---- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-env", required=True, help="configs/students/<tag>.env")
    ap.add_argument("--recipe-env", default="", help="optional extra profile (sourced after the student's)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (served as an adapter)")
    ap.add_argument("--model-dir", default=None, help="serve this directory instead (e.g. a merged model)")
    ap.add_argument("--out", required=True, help="runs/<run>/bench/<ckpt>.json")
    ap.add_argument("--n-papers", type=int, default=200, help="papers in the throughput phase")
    ap.add_argument("--latency-papers", type=int, default=20, help="papers in the single-request phase")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2, help="throughput repetitions (median reported)")
    ap.add_argument("--ids", default=None, help="JSON list of paper ids (default: the packaged test papers)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompts-dir", default=None, help="override data/<model_tag>/prompts/<split>")
    ap.add_argument("--max-model-len", type=int, default=0, help="default: SCISUM_MAX_SEQ_LEN")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--label", default="", help="free text stored in the output (e.g. 'merged')")
    a = ap.parse_args(argv)

    env = apply_env([a.student_env, a.recipe_env])
    sys.path.insert(0, str(REPO))
    from scisum.config import Settings

    s = Settings()
    max_model_len = a.max_model_len or int(getattr(s, "max_seq_len", 32768))
    cap = int(getattr(s, "gen_max_new_tokens", 3000))
    prompt_dir = Path(a.prompts_dir) if a.prompts_dir else Path(s.model_data_dir) / "prompts" / a.split
    ids = load_ids(a.ids)
    need = max(a.n_papers, a.latency_papers + a.warmup)
    t_read = time.perf_counter()
    recs = load_prompts(prompt_dir, ids, need)
    log(f"{len(recs)} prompts from {prompt_dir} in {time.perf_counter() - t_read:.0f}s")
    ids_used = [r["id"] for r in recs]

    if a.model_dir:
        served = {"model": str(Path(a.model_dir).resolve()), "tokenizer": str(Path(a.model_dir).resolve())}
        model_dir = Path(a.model_dir)
    else:
        try:
            from scisum.evaluate import serve_model_args

            served = serve_model_args(s)
        except Exception as e:  # noqa: BLE001 - keep the benchmark usable if the helper moved
            print(f"warning: serve_model_args unavailable ({e}); serving {s.base_model} as declared", file=sys.stderr)
            served = {"model": s.base_model, **({"revision": s.base_model_revision} if s.base_model_revision else {})}
        model_dir = local_model_dir(served["model"], served.get("revision"))
    cfg = read_config(model_dir)
    kv = kv_bytes_per_token(cfg) if cfg else None

    fp = fingerprint()
    t_imp = time.perf_counter()
    if a.engine == "vllm":
        import vllm  # noqa: F401 - imported here so that its (filesystem-bound) import time is not load time
    import_s = time.perf_counter() - t_imp
    log(f"{a.engine} imported in {import_s:.0f}s")
    free0 = device_free_gb()
    t0 = time.perf_counter()
    if a.engine == "vllm":
        engine = VllmEngine(s, served, a.adapter, max_model_len, a.tp, a.gpu_frac)
    else:
        engine = HfEngine(s, served["model"], a.adapter)
    load_s = time.perf_counter() - t0
    free1 = device_free_gb()
    log(f"engine {a.engine} loaded in {load_s:.0f}s; latency phase: {a.warmup} warm-up + {a.latency_papers} papers")

    lat_recs = recs[:a.latency_papers + a.warmup]
    latency = measure_latency(engine, lat_recs, cap, max_model_len, a.warmup)
    log(f"latency done: e2e p50 {latency['e2e_s']['p50']:.1f}s; throughput phase: {a.n_papers} papers x {a.repeats}")
    throughput = None
    if a.engine == "vllm":
        throughput = measure_throughput(engine, recs[:a.n_papers], cap, max_model_len, a.repeats)

    ctx = [r["prompt_tokens"] + (throughput or {}).get("out_tokens_mean", 0) for r in recs[:a.n_papers]]
    memory = {
        "weights_gb": dir_size_gb(model_dir) if model_dir else None,
        "adapter_gb": dir_size_gb(Path(a.adapter)) if a.adapter else None,
        "kv": kv,
        "kv_gb_per_paper_p50": round(kv["bytes_per_token"] * describe(ctx)["p50"] / 2**30, 3) if kv and ctx else None,
        "kv_gb_per_paper_p95": round(kv["bytes_per_token"] * describe(ctx)["p95"] / 2**30, 3) if kv and ctx else None,
        "device_free_gb_before": free0, "device_free_gb_after_load": free1,
        "engine_reserved_gb": (round(sum(free0) - sum(free1), 2) if free0 and free1 else None),
        "note": "engine_reserved_gb follows --gpu-frac (vLLM pre-allocates); weights + KV per paper is the need",
    }
    out = {
        "schema": SCHEMA, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": a.label, "student": env.get("STUDENT_TAG") or Path(a.student_env).stem,
        "base_model": getattr(s, "base_model", None), "revision": getattr(s, "base_model_revision", None),
        "served_model": served.get("model"), "adapter": a.adapter,
        "adapter_mode": "merged" if a.model_dir else ("lora" if a.adapter else "base"),
        "engine": a.engine,
        "settings": {"max_model_len": max_model_len, "tp": a.tp, "gpu_frac": a.gpu_frac, "dtype": "bfloat16",
                     "temperature": 0.0, "max_new_tokens_cap": cap, "prefix_caching": False, "seed": 0},
        "papers": {"split": a.split, "ids_file": a.ids or "default", "n_latency": latency["n"],
                   "n_throughput": (throughput or {}).get("n", 0),
                   "ids_sha256": hashlib.sha256("\n".join(ids_used).encode()).hexdigest()[:16],
                   "prompt_tokens": describe(r["prompt_tokens"] for r in recs[:a.n_papers])},
        "import_s": round(import_s, 2), "load_s": round(load_s, 2), "latency": latency, "throughput": throughput, "memory": memory, **fp,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(a.out)
    lt = latency
    print(f"{out['student']} {out['adapter_mode']}: load {load_s:.0f}s; TTFT p50 {lt['ttft_s']['p50']:.2f}s, "
          f"e2e p50 {lt['e2e_s']['p50']:.1f}s p95 {lt['e2e_s']['p95']:.1f}s"
          + (f"; {throughput['papers_per_hour']:.0f} papers/h, {throughput['output_tokens_per_s']:.0f} tok/s, "
             f"{throughput['gpu_seconds_per_paper']:.2f} GPU-s/paper" if throughput else "")
          + f" -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
