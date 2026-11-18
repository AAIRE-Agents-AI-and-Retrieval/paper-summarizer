"""Merge a LoRA adapter into its base model and save the result, for benchmarking a merged deployment.

    python scripts/analysis/merge_adapter.py --student-env configs/students/<tag>.env --adapter PATH --out DIR

Runs on a GPU node (or a CPU node with enough RAM: about 2 bytes x parameters plus overhead). The base model
is loaded with the class training used (``AutoModelForCausalLM``), so a vision-language checkpoint such as
Qwen3.5 is saved as its text-only causal-LM class; check with ``bench_inference.py --model-dir DIR`` that vLLM
serves it and that its outputs match the adapter-served ones on a few papers before trusting the numbers.
Untested on the five new students: treat the first merge of each as a check, not a given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import parse_env_file  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student-env", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ.update(parse_env_file(a.student_env))
    sys.path.insert(0, str(REPO))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scisum.config import Settings

    s = Settings()
    kw = s.hf_kwargs()
    base = AutoModelForCausalLM.from_pretrained(s.base_model, torch_dtype=torch.bfloat16, **kw)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(s.base_model, **kw).save_pretrained(out)
    print(f"merged {a.adapter} into {s.base_model} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
