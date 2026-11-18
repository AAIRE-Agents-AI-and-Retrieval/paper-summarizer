"""scripts/promote_final.py: a finished run's final rolling adapter becomes its milestone ex<N>."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("promote_final", Path(__file__).resolve().parent.parent / "scripts" / "promote_final.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def make_run(root: Path, finished: bool = True, step_examples: int = 1163, done: bool = True) -> SimpleNamespace:
    run = root / "q35x"
    step = run / "ckpt" / "step-00000146"
    (step / "adapter").mkdir(parents=True)
    (step / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    if done:
        (step / "DONE").write_text("ok")
    (step / "state.json").write_text(json.dumps({"examples": step_examples, "step": 146}))
    (run / "ckpt" / "LATEST").write_text("step-00000146\n")
    (run / "state.json").write_text(json.dumps({"examples": 1163, "step": 146, "tokens": 5, "finished": finished}))
    return SimpleNamespace(run_dir=lambda name: root / name, base_model="Qwen/Qwen3.5-9B")


def test_promotes_the_final_adapter_with_a_ready_marker(tmp_path):
    s = make_run(tmp_path)
    dest = pf.promote(s, "q35x", val=False)
    assert dest.name == "ex1163"
    assert (dest / "adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
    ready = json.loads((dest / "READY").read_text())
    assert ready["milestone"] == 1163 and ready["step"] == 146 and ready["promoted_from"] == "ckpt/step-00000146"
    assert ready["val_loss"] is None
    assert pf.promote(s, "q35x", val=False) == dest          # idempotent


@pytest.mark.parametrize("kw", [{"finished": False}, {"step_examples": 1160}, {"done": False}])
def test_refuses_an_unfinished_or_stale_checkpoint(tmp_path, kw):
    s = make_run(tmp_path, **kw)
    with pytest.raises(SystemExit):
        pf.promote(s, "q35x", val=False)
    assert not (tmp_path / "q35x" / "ckpt" / "ex1163").exists()
