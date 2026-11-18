"""What a new student looks like from the pipeline's point of view, before any GPU time is spent.

Instantiates the architecture on the meta device (no weights, no memory), so it answers in seconds:
which linear modules exist to put LoRA on, whether the head-slice loss can reach `.model` and
`.lm_head`, and what the chat template wraps an assistant answer in.

    srun -p short -c 4 --mem=32G -t 15 .venv/bin/python scripts/inspect_model.py
"""

from __future__ import annotations

import sys
from collections import Counter

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scisum.config import get_settings
from scisum.prompt import assistant_wrapper


def main() -> int:
    s = get_settings()
    print(f"model: {s.base_model}  revision={s.base_model_revision or 'main'}")
    cfg = AutoConfig.from_pretrained(s.base_model, **s.hf_kwargs())
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, **({"trust_remote_code": True} if s.trust_remote_code else {}))

    linears: Counter[str] = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            linears[name.split(".")[-1]] += 1
    print(f"\nlinear modules by name ({sum(linears.values())} in total):")
    for name, n in linears.most_common():
        print(f"  {name:28} {n:4d}")

    print("\nthe head-slice loss needs a transformer trunk and a head:")
    base = model
    print(f"  .model          {type(getattr(base, 'model', None)).__name__}")
    print(f"  .lm_head        {type(getattr(base, 'lm_head', None)).__name__}")
    inner = getattr(base, "model", None)
    print(f"  .model.layers   {len(getattr(inner, 'layers', []) or [])} layers"
          if hasattr(inner, "layers") else "  .model.layers   MISSING - the trunk is nested deeper")
    for attr in ("language_model", "text_model", "vision_tower", "visual"):
        if hasattr(inner, attr) or hasattr(base, attr):
            print(f"  nested: .{attr} is present - check which one carries the decoder layers")

    tok = AutoTokenizer.from_pretrained(s.base_model, **s.hf_kwargs())
    print(f"\ntokenizer: {len(tok):,} tokens, eos={tok.eos_token!r} ({tok.eos_token_id})")
    print(f"assistant turn wrapper: {assistant_wrapper(tok)}")
    sug = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
           if n in linears]
    missing = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
               if n not in linears]
    other = [n for n in linears if n not in sug and "experts" not in n]
    print(f"\nstandard targets present: {sug}")
    print(f"standard targets absent:  {missing}")
    print(f"other linear names:       {other}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
