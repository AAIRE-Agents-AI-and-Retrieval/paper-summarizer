"""The student's prompt must be byte-identical to the one the teacher summaries were written from.

The prompt pieces are copied into :mod:`scisum.vendor` from the teacher pipeline. Their output was recorded with
the *original* code for 208 papers from all three
splits, including the 20 longest of the first 3,000 of each split, so section dropping is exercised: the
SHA-256 of the system prompt, of ``fit_body``'s body at four character budgets (with its truncation flag, section
count and dropped characters), of the user message and the full message list at 60,000 characters, of the
abstract-only message and of the canonical target. This test recomputes them from the packaged corpus with the
vendored code. It is skipped when ``data/`` is not present (a code-only checkout).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import orjson
import pytest

from scisum import prompt as P
from scisum.config import Settings
from scisum.corpus import BODIES, INDEX, BodyStore

GOLDEN = Path(__file__).parent / "fixtures" / "prompt_golden.jsonl"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _golden() -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]
    return lines[0]["_header"], lines[1:]


def test_system_prompt_and_schema_are_the_teachers():
    header, _ = _golden()
    assert _h(P.SYSTEM_PROMPT) == header["system_prompt_sha256"]
    assert P.SUMMARY_KEYS == ("tldr", "problem", "contributions", "method", "results", "limitations",
                              "keywords", "domain")


@pytest.fixture(scope="module")
def corpus():
    c = Settings().corpus_dir
    if not (c / INDEX).exists() or not (c / BODIES).exists():
        pytest.skip(f"packaged corpus not present under {c}")
    header, golden = _golden()
    wanted = {g["id"] for g in golden}
    needles = {i: f'"id":"{i}"'.encode() for i in wanted}
    rows: dict[str, dict] = {}
    with gzip.open(c / INDEX, "rb") as f:
        for line in f:
            head = line[:80]
            for i, n in needles.items():
                if n in head:
                    rows[i] = orjson.loads(line)
                    break
    return header, golden, rows, BodyStore(c / BODIES)


def test_prompts_rebuilt_from_the_package_match_the_original_code(corpus):
    header, golden, rows, store = corpus
    kinds = tuple(header["kinds"])
    assert len(rows) == len(golden), "every golden paper must be in the packaged index"
    bad = []
    for g in golden:
        row, parsed = rows[g["id"]], store.parsed(g["id"])
        for b, want in g["budgets"].items():
            text, trunc, n_sec, dropped = P.fit_body(g["id"], parsed, int(b), kinds)
            got = {"body_sha256": _h(text), "chars": len(text), "truncated": trunc, "sections": n_sec,
                   "dropped_chars": dropped}
            if got != want:
                bad.append((g["id"], b, "body"))
            if b == "60000":
                if _h(json.dumps(P.messages(row["doc"], text), ensure_ascii=False)) != g["messages_sha256"]:
                    bad.append((g["id"], b, "messages"))
                if _h(P.build_user_message(row["doc"], text, False)) != g["user_sha256"]:
                    bad.append((g["id"], b, "user"))
        if _h(P.build_user_message(row["doc"], "", True)) != g["abstract_only_sha256"]:
            bad.append((g["id"], "-", "abstract_only"))
        if _h(P.canonical_target(row["summary"])) != g["target_sha256"]:
            bad.append((g["id"], "-", "target"))
    assert not bad, f"{len(bad)} mismatches, e.g. {bad[:5]}"


def test_golden_set_exercises_section_dropping():
    _, golden = _golden()
    assert len(golden) >= 200
    assert {g["split"] for g in golden} == {"train", "val", "test"}
    truncated = sum(g["budgets"]["20000"]["truncated"] for g in golden)
    assert truncated >= 50, "the long papers must force section dropping"
