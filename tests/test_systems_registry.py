"""Systems under test are found from run names (scisum.factcheck.system_pred_path), not a hand-edited table."""

import gzip
import json

import pytest

from scisum.config import Settings
from scisum.factcheck import read_system, system_pred_path


def _pred(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "text": json.dumps({"tldr": f"t{i}"})}) + "\n")


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.delenv("SCISUM_SYSTEM_PATHS", raising=False)
    monkeypatch.setenv("SCISUM_DATA_ROOT", str(tmp_path / "data"))
    return Settings(repo_root=tmp_path)


def test_run_name_picks_the_final_checkpoint_not_base(s):
    ev = s.run_dir("k2h-3.7b-sft50k") / "eval"
    for name in ("test-base-n1246", "test-ex20000-n1246", "test-ex50000-n1246", "test-ex5000-n1246"):
        _pred(ev / f"{name}.pred.jsonl.gz", ["a"])
    assert system_pred_path(s, "k2h-3.7b-sft50k").name == "test-ex50000-n1246.pred.jsonl.gz"
    assert system_pred_path(s, "k2h-3.7b-sft50k@ex20000").name == "test-ex20000-n1246.pred.jsonl.gz"
    assert system_pred_path(s, "k2h-3.7b-sft50k@base").name == "test-base-n1246.pred.jsonl.gz"
    assert system_pred_path(s, "nobody") is None
    with pytest.raises(ValueError, match="SCISUM_SYSTEM_PATHS"):
        read_system(s, "nobody", {"a"})


def test_anchors_are_found_in_the_release_data(s, tmp_path):
    _pred(tmp_path / "data" / "anchors" / "q35prefcc" / "eval" / "test-ex1233-n1246.pred.jsonl.gz", ["a", "b"])
    _pred(tmp_path / "data" / "anchors" / "q35" / "eval" / "test-ex20000-n15000.pred.jsonl.gz", ["a"])
    got = read_system(s, "C_prefcc", {"a"})
    assert got == {"a": {"tldr": "ta"}}
    assert system_pred_path(s, "C").name == "test-ex20000-n15000.pred.jsonl.gz"


def test_explicit_path_override(s, tmp_path, monkeypatch):
    p = tmp_path / "elsewhere" / "x.pred.jsonl.gz"
    _pred(p, ["z"])
    monkeypatch.setenv("SCISUM_SYSTEM_PATHS", json.dumps({"mine": str(p)}))
    assert read_system(s, "mine", {"z"}) == {"z": {"tldr": "tz"}}


def test_run_names_with_dots_round_trip_through_grade_files(s):
    from scisum.factcheck import fdir

    (fdir(s) / "grades-k2h-3.7b-prefcc.jsonl.gz").write_bytes(gzip.compress(b""))
    labs = sorted(p.name[len("grades-"):-len(".jsonl.gz")] for p in fdir(s).glob("grades-*.jsonl.gz"))
    assert labs == ["k2h-3.7b-prefcc"]
