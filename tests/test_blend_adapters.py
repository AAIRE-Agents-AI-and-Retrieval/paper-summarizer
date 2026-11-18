import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("blend_adapters", Path(__file__).parent.parent / "scripts" / "blend_adapters.py")
blend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blend)


def _adapter(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in ("layers.0.q_proj", "layers.1.down_proj"):
        out[f"base_model.model.{m}.lora_A.weight"] = torch.randn(4, 6, generator=g)
        out[f"base_model.model.{m}.lora_B.weight"] = torch.randn(5, 4, generator=g)
    return out


def test_blend_is_exact_weighted_sum_of_weight_changes():
    ta, tb = _adapter(0), _adapter(1)
    for w in (0.0, 0.25, 1.0):
        tm = blend.blend_tensors(ta, tb, w)
        assert tm["base_model.model.layers.0.q_proj.lora_A.weight"].shape == (8, 6)
        assert tm["base_model.model.layers.0.q_proj.lora_B.weight"].shape == (5, 8)
        assert blend.check_identity(ta, tb, tm, w) < 1e-6
