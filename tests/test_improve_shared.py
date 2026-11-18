"""The student-independent pool files are copied into each student's improve directory, once, never over its own."""

import gzip
import json

from scisum.config import Settings
from scisum.improve import SHARED_POOL_FILES, idir


def test_shared_files_seed_a_new_student_dir_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.delenv("SCISUM_IMP_SHARED_DIR", raising=False)
    monkeypatch.setenv("SCISUM_DATA_ROOT", str(tmp_path / "data"))
    shared = tmp_path / "data" / "improve_pool"
    shared.mkdir(parents=True)
    for name in (*SHARED_POOL_FILES, "pool.jsonl.gz"):
        (shared / name).write_bytes(gzip.compress(json.dumps({"id": "x", "src": name}).encode() + b"\n"))
    s = Settings(repo_root=tmp_path, imp_dir="improve-k2h-3.7b")
    d = idir(s)
    assert sorted(p.name for p in d.iterdir()) == sorted(SHARED_POOL_FILES)     # the pool is the student's own
    (d / "refs.jsonl.gz").write_bytes(gzip.compress(b'{"id": "mine"}\n'))
    idir(s)
    assert b"mine" in gzip.decompress((d / "refs.jsonl.gz").read_bytes())


def test_no_shared_dir_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.delenv("SCISUM_IMP_SHARED_DIR", raising=False)
    monkeypatch.setenv("SCISUM_DATA_ROOT", str(tmp_path / "nothing"))
    d = idir(Settings(repo_root=tmp_path, imp_dir="improve-x"))
    assert list(d.iterdir()) == []
