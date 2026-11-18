"""The packaged corpus readers: index rows, splits and the body store, on a tiny corpus built in a temp dir."""

from __future__ import annotations

import gzip
import json
import sqlite3

import orjson

from scisum.config import Settings
from scisum.corpus import BodyStore, decode_body, encode_body, read_index_file
from scisum.dataset import load_splits, read_index
from scisum.prompt import fit_body


def _corpus(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    rows = [{"id": "1111.0001", "doc": {"title": "T"}, "summary": {}}, {"id": "hep-th/9711200", "doc": {}, "summary": {}}]
    with gzip.open(c / "index.jsonl.gz", "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")
    (c / "splits.json").write_text(json.dumps({"train": ["1111.0001"], "val": [], "test": ["hep-th/9711200"]}))
    con = sqlite3.connect(c / "bodies.sqlite")
    con.execute("CREATE TABLE bodies (id TEXT PRIMARY KEY, parsed BLOB NOT NULL)")
    body = [["Introduction", "intro text", "body"], ["References", "[1] x", "references"],
            ["Figure and table captions", "Fig 1", "captions"]]
    con.execute("INSERT INTO bodies VALUES (?, ?)", ("1111.0001", encode_body(body)))
    con.commit()
    con.close()
    return c, rows, body


def test_round_trip_of_a_body():
    rows = [["Ü title", "text with ünïcode", "body"]]
    assert decode_body(encode_body(rows)) == rows


def test_readers_use_the_packaged_corpus(tmp_path):
    c, rows, body = _corpus(tmp_path)
    s = Settings(data_root=tmp_path)
    assert read_index(s) == rows == read_index_file(c / "index.jsonl.gz")
    assert load_splits(s)["test"] == ["hep-th/9711200"]
    store = BodyStore(c / "bodies.sqlite")
    assert "1111.0001" in store and "hep-th/9711200" not in store and store.count() == 1
    parsed = store.parsed("1111.0001")
    assert [x.kind for x in parsed.sections] == ["body", "references", "captions"]
    text, trunc, n, dropped = fit_body("1111.0001", parsed, 10_000, ("body", "captions"))
    assert text == "Introduction\nintro text\n\nFigure and table captions\nFig 1" and not trunc
    assert n == 2 and dropped == len("[1] x")
