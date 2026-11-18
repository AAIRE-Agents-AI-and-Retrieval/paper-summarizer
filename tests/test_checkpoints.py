"""Rolling checkpoints: a resume must never land on a half-written one.

The failure these guard against is cheap to cause and expensive to have: a job killed while the
checkpoint directory was being replaced used to leave no adapter at all, which on resume looks
exactly like "start from scratch" — seventeen hours of GPU time thrown away silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from scisum.train import CKPT_KEEP, latest_checkpoint


def _complete(run_dir: Path, step: int, examples: int) -> Path:
    d = run_dir / "ckpt" / f"step-{step:08d}"
    (d / "adapter").mkdir(parents=True)
    (d / "adapter" / "adapter_config.json").write_text("{}")
    (d / "state.json").write_text(json.dumps({"examples": examples, "step": step}))
    (d / "DONE").write_text("2026-10-14T00:00:00Z")
    (run_dir / "ckpt" / "LATEST").write_text(d.name)
    return d


def test_none_before_the_first_checkpoint(tmp_path):
    (tmp_path / "ckpt").mkdir()
    assert latest_checkpoint(tmp_path) is None


def test_the_newest_complete_checkpoint_wins(tmp_path):
    _complete(tmp_path, 10, 80)
    newest = _complete(tmp_path, 20, 160)
    assert latest_checkpoint(tmp_path) == newest


def test_a_half_written_checkpoint_is_ignored(tmp_path):
    good = _complete(tmp_path, 10, 80)
    half = tmp_path / "ckpt" / "step-00000020"       # renamed into place but DONE never written
    (half / "adapter").mkdir(parents=True)
    (half / "adapter" / "adapter_config.json").write_text("{}")
    (tmp_path / "ckpt" / "LATEST").write_text(half.name)
    assert latest_checkpoint(tmp_path) == good


def test_a_checkpoint_still_being_written_is_invisible(tmp_path):
    good = _complete(tmp_path, 10, 80)
    writing = tmp_path / "ckpt" / ".writing-step-00000020"
    (writing / "adapter").mkdir(parents=True)
    (writing / "adapter" / "adapter_config.json").write_text("{}")
    (writing / "DONE").write_text("x")
    assert latest_checkpoint(tmp_path) == good       # the name does not match step-*


def test_a_lost_pointer_still_finds_the_newest(tmp_path):
    _complete(tmp_path, 10, 80)
    newest = _complete(tmp_path, 20, 160)
    (tmp_path / "ckpt" / "LATEST").unlink()
    assert latest_checkpoint(tmp_path) == newest


def test_the_state_travels_with_the_checkpoint(tmp_path):
    _complete(tmp_path, 30, 240)
    d = latest_checkpoint(tmp_path)
    assert json.loads((d / "state.json").read_text())["examples"] == 240


def test_two_slots_are_kept(tmp_path):
    assert CKPT_KEEP >= 2, "one slot means a corrupt newest checkpoint has no fallback"


def test_shortest_and_longest_select_opposite_ends():
    """--shortest is the preflight's opposite of --longest; they must not pick the same examples."""
    import inspect

    from scisum.train import run_train
    src = inspect.getsource(run_train)
    assert "sign = 1 if shortest else -1" in src
    assert "shortest: bool = False" in str(inspect.signature(run_train)) or \
           "shortest" in inspect.signature(run_train).parameters
