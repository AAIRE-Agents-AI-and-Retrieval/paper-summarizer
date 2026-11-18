"""Does vLLM apply a run's LoRA adapters? A few prompts, base vs. two checkpoints, nothing written.

Run on a GPU (the MIG slice is enough with --max-model-len 8192):

    SUM_MODEL_ENV=slurm/models/qwen3.5-9b.env sbatch slurm/probe_vllm_adapter.sbatch

Prints, per checkpoint, whether its output differs from the base model and from the other
checkpoint, and whether it parses as the schema. Written after Student C's evaluations turned out
to be the base model three times over (2026-10-18): identical text from three different adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from scisum.config import get_settings
from scisum.evaluate import adapter_path, check_adapter_applied, load_records, serve_model_args
from scisum.metrics import parse_prediction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    s = get_settings()
    served = serve_model_args(s)               # exactly what an evaluation would build
    print(f"serving {served['model']} as {served.get('hf_overrides') or 'declared'}", flush=True)

    budget = a.max_model_len - a.max_tokens - 8
    recs = sorted((r for r in load_records(s, "val", "curve", None)
                   if r["meta"]["prompt_tokens"] <= budget), key=lambda r: r["meta"]["prompt_tokens"])[:a.n]
    prompts = [r["prompt"] for r in recs]
    print(f"{len(prompts)} prompts of {[r['meta']['prompt_tokens'] for r in recs]} tokens", flush=True)

    llm = LLM(dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.85,
              enable_lora=True, max_lora_rank=s.lora_r, max_loras=1, seed=0, disable_log_stats=True,
              trust_remote_code=s.trust_remote_code, **served)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    def run(lora=None) -> list[str]:
        kw = {"lora_request": lora} if lora else {}
        return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False, **kw)]

    out = {"base": run()}
    for i, ck in enumerate(a.ckpts, start=1):
        lora = LoRARequest(ck, i, str(adapter_path(s, a.run, ck)))
        check_adapter_applied(llm, lora, prompts)          # raises if the adapter changes nothing
        out[ck] = run(lora)

    for name, texts in out.items():
        digest = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
        parsed = [parse_prediction(t)[0] for t in texts]
        ok = sum(isinstance(p, dict) for p in parsed)
        print(f"{name:8s} md5={digest} json={ok}/{len(texts)} first={json.dumps(texts[0][:110])}")
    names = list(out)
    distinct = len({hashlib.md5("".join(out[n]).encode()).hexdigest() for n in names})
    print(f"distinct outputs: {distinct} of {len(names)}")


if __name__ == "__main__":
    main()
