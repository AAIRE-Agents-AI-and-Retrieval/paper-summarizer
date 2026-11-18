"""The packaged corpus: the eligible papers, their teacher summaries, the splits and the parsed bodies.

Layout under ``Settings.data_root`` (``SCISUM_DATA_ROOT``, default ``<repo>/data``); see ``docs/DATA.md``::

    corpus/index.jsonl.gz     one row per eligible paper: arXiv metadata (``doc``), the teacher summary
                              (``summary``), the teacher model, cohort, citation rank, body statistics
    corpus/splits.json        train / val / test ids in their fixed order (seed 20261014), curve and loss subsets
    corpus/index.json         counts and the reasons papers were dropped (details in index_drops.jsonl)
    corpus/bodies.sqlite      table ``bodies(id TEXT PRIMARY KEY, parsed BLOB)``: zlib-compressed JSON
                              ``[[title, text, kind], ...]``, the GROBID body as the teacher run parsed it

The rows of ``index.jsonl.gz`` and ``splits.json`` are the files the original run used, unchanged; the bodies
replace the GROBID TEI files and their parser, which are not distributed.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
import zlib
from pathlib import Path

import orjson

from scisum.vendor.tei_sections import ParsedBody

INDEX = "index.jsonl.gz"
SPLITS = "splits.json"
BODIES = "bodies.sqlite"


def corpus_dir(s) -> Path:
    return Path(s.data_root) / "corpus"


def read_index_file(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]


def encode_body(rows: list[list[str]]) -> bytes:
    return zlib.compress(json.dumps(rows, ensure_ascii=False).encode("utf-8"), 6)


def decode_body(blob: bytes) -> list[list[str]]:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


class BodyStore:
    """Read-only access to ``bodies.sqlite``; one connection per thread and process (safe after ``fork``)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path}: the packaged bodies are missing (see docs/DATA.md)")
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        import os

        c = getattr(self._local, "conn", None)
        if c is None or getattr(self._local, "pid", None) != os.getpid():
            c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            self._local.conn, self._local.pid = c, os.getpid()
        return c

    def rows(self, paper_id: str) -> list[list[str]]:
        r = self._conn().execute("SELECT parsed FROM bodies WHERE id = ?", (paper_id,)).fetchone()
        if r is None:
            raise KeyError(paper_id)
        return decode_body(r[0])

    def parsed(self, paper_id: str) -> ParsedBody:
        return ParsedBody.from_rows(self.rows(paper_id))

    def __contains__(self, paper_id: str) -> bool:
        return self._conn().execute("SELECT 1 FROM bodies WHERE id = ?", (paper_id,)).fetchone() is not None

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM bodies").fetchone()[0]


_STORES: dict[str, BodyStore] = {}


def body_store(s) -> BodyStore:
    p = str(corpus_dir(s) / BODIES)
    if p not in _STORES:
        _STORES[p] = BodyStore(Path(p))
    return _STORES[p]
