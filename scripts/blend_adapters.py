"""Blend two LoRA adapters of the same base into one exact adapter of twice the rank (CPU, ~2 min).

    python scripts/blend_adapters.py --a runs/q35/ckpt/ex20000/adapter \
        --b runs/q35prefcc/ckpt/ex1233/adapter --weight 0.5 --out runs/q35ccmix50/ckpt/mix50/adapter

Written 2026-11-09 (step 0 of ``paper_docs/12_reactive_training.md``): the "strong fixed baseline" that
the reactive-training lesson asks for. In OPAL's run most of the agent's gain was weight averaging,
which a fixed recipe also gets. Here the question is whether moving part of the way from Student C
(``a``) towards the preference-tuned ``prefcc`` (``b``) keeps the fact-recall gain and recovers the NLI
and judge faithfulness that ``prefcc`` lost against C.

The blended weight change is ``(1 - w) * dW_a + w * dW_b`` with ``dW = (alpha / r) * B @ A`` per module.
Since ``(1-w) B_a A_a + w B_b A_b = [(1-w) B_a, w B_b] @ [A_a; A_b]``, it is exactly a LoRA of rank
``2r`` with the same scale ``alpha / r`` (so ``lora_alpha`` doubles too). Averaging A and B separately
would not be exact (the product is bilinear). ``w = 0`` is ``a`` and ``w = 1`` is ``b``.

Both adapters must have the same base model, rank, alpha, target modules and tensor names, and plain
LoRA (no DoRA, no rsLoRA). The script checks the identity on a few modules in float64 before writing,
writes to a temporary directory and renames it, and records the inputs in ``BLEND.json`` next to the
adapter. Evaluating it needs ``max_lora_rank >= 2r`` in vLLM: use ``slurm/models/qwen3.5-9b-r128.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CHECKED_KEYS = ("r", "lora_alpha", "target_modules", "base_model_name_or_path", "use_dora", "use_rslora",
                "bias", "fan_in_fan_out")


def _config(path: Path) -> dict:
    return json.loads((path / "adapter_config.json").read_text())


def blend_tensors(ta: dict[str, torch.Tensor], tb: dict[str, torch.Tensor], w: float) -> dict[str, torch.Tensor]:
    """A' = [A_a; A_b] (rows), B' = [(1-w) B_a, w B_b] (columns), for every lora_A/lora_B pair."""
    if set(ta) != set(tb):
        raise SystemExit(f"tensor names differ: {sorted(set(ta) ^ set(tb))[:4]}")
    out = {}
    for k, a in ta.items():
        b = tb[k]
        if a.shape != b.shape:
            raise SystemExit(f"{k}: shapes differ {tuple(a.shape)} vs {tuple(b.shape)}")
        if ".lora_A." in k:
            out[k] = torch.cat([a, b], dim=0)
        elif ".lora_B." in k:
            out[k] = torch.cat([(1.0 - w) * a, w * b], dim=1).to(a.dtype)
        else:
            raise SystemExit(f"{k}: not a lora_A/lora_B tensor; this script handles plain LoRA only")
    return out


def check_identity(ta: dict, tb: dict, tm: dict, w: float, n: int = 4) -> float:
    """Largest relative error of B'A' against (1-w) B_a A_a + w B_b A_b over ``n`` modules (float64)."""
    worst = 0.0
    for ka in sorted(k for k in ta if ".lora_A." in k)[:: max(1, len(ta) // (2 * n))][:n]:
        kb = ka.replace(".lora_A.", ".lora_B.")
        want = (1 - w) * ta[kb].double() @ ta[ka].double() + w * tb[kb].double() @ tb[ka].double()
        got = tm[kb].double() @ tm[ka].double()
        worst = max(worst, float((got - want).norm() / want.norm().clamp_min(1e-30)))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="adapter at weight 1 - w (Student C)")
    ap.add_argument("--b", required=True, type=Path, help="adapter at weight w (the preference-tuned one)")
    ap.add_argument("--weight", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    x = ap.parse_args()
    if not 0.0 <= x.weight <= 1.0:
        raise SystemExit("--weight must be in [0, 1]")
    ca, cb = _config(x.a), _config(x.b)
    for k in CHECKED_KEYS:
        va, vb = ca.get(k), cb.get(k)
        if (sorted(va) if isinstance(va, list) else va) != (sorted(vb) if isinstance(vb, list) else vb):
            raise SystemExit(f"adapter_config {k} differs: {va} vs {vb}")
    if ca.get("use_dora") or ca.get("use_rslora"):
        raise SystemExit("DoRA / rsLoRA adapters are not handled")
    if x.out.exists():
        raise SystemExit(f"{x.out} exists; remove it first")
    fa, fb = x.a / "adapter_model.safetensors", x.b / "adapter_model.safetensors"
    ta, tb = load_file(str(fa)), load_file(str(fb))
    tm = blend_tensors(ta, tb, x.weight)
    err = check_identity(ta, tb, tm, x.weight)
    print(f"identity check: largest relative error {err:.2e}")
    if err > 1e-5:
        raise SystemExit("blend does not reproduce the weighted sum of the two weight changes")
    cfg = dict(ca, r=2 * ca["r"], lora_alpha=2 * ca["lora_alpha"])
    tmp = x.out.parent / f".writing-{x.out.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(tm, str(tmp / "adapter_model.safetensors"))
    (tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    (x.out.parent / "BLEND.json").write_text(json.dumps({
        "a": str(x.a), "b": str(x.b), "weight_b": x.weight, "sha256_16_a": sha(fa), "sha256_16_b": sha(fb),
        "rank": cfg["r"], "lora_alpha": cfg["lora_alpha"], "identity_rel_error": err,
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    tmp.replace(x.out)
    print(f"wrote {x.out} (rank {cfg['r']}, alpha {cfg['lora_alpha']}, w={x.weight})")


if __name__ == "__main__":
    main()
