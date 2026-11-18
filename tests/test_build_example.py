"""Fitting a paper into the prompt budget: the result must always fit, and never be cut blind.

A character-based fake tokenizer stands in for the real one, so the budget logic is tested without
a model: the dangerous case is a paper whose body is one enormous section, where dropping sections
cannot help and the text itself has to be cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from scisum.config import Settings
from scisum.dataset import build_example
from tests.test_metrics import GOOD


class FakeEncoding:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTok:
    """``chars_per_token`` characters make one token; the chat template adds a fixed header."""

    eos_token_id = 99

    def __init__(self, chars_per_token: float = 4.0):
        self.cpt = chars_per_token

    def __call__(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
        return FakeEncoding(list(range(int(len(text) / self.cpt) + 1)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        """Shaped like a real chat template: one block per message, header only when asked."""
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


@dataclass
class FakeSection:
    title: str
    text: str
    kind: str = "body"

    def as_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class FakeTEI:
    sections: list = field(default_factory=list)


@pytest.fixture
def row():
    return {"id": "1234.5678", "teacher": "t", "cohort": "top", "rank": 1, "pages": 9,
            "body_chars": 100, "teacher_truncated": False, "teacher_prompt_tokens": 100,
            "doc": {"title": "A Paper", "authors": ["A B"], "categories": ["cs.CL"],
                    "abstract": "The abstract."},
            "summary": GOOD}


class FakeStore:
    """Stands in for ``data/corpus/bodies.sqlite``: every paper has the given sections."""

    def __init__(self, sections):
        self.sections = sections

    def parsed(self, paper_id):
        return FakeTEI(self.sections)


def _build(monkeypatch, row, sections, cpt=4.0, **over):
    s = Settings(**over)
    monkeypatch.setattr("scisum.corpus.body_store", lambda settings: FakeStore(sections))
    return s, build_example(s, FakeTok(cpt), row)


def test_a_paper_without_a_body_is_dropped(monkeypatch, row):
    class Empty:
        def parsed(self, paper_id):
            raise KeyError(paper_id)

    monkeypatch.setattr("scisum.corpus.body_store", lambda settings: Empty())
    assert build_example(Settings(), FakeTok(), row) == {"drop": "tei_error:KeyError"}


def test_short_paper_keeps_its_whole_body(monkeypatch, row):
    s, ex = _build(monkeypatch, row, [FakeSection("Intro", "x" * 4_000)])
    assert "drop" not in ex
    assert ex["meta"]["trunc_kind"] == "none" and ex["meta"]["fits"]
    assert len(ex["prompt_ids"]) <= s.max_prompt_tokens


def test_long_paper_drops_sections_first(monkeypatch, row):
    sections = [FakeSection("Introduction", "i" * 20_000), FakeSection("Related work", "r" * 20_000),
                FakeSection("Method", "m" * 20_000), FakeSection("Results", "e" * 20_000)]
    s, ex = _build(monkeypatch, row, sections)
    assert "drop" not in ex
    assert len(ex["prompt_ids"]) <= s.max_prompt_tokens
    assert "Related work" not in ex["prompt_text"]      # the least useful section goes first
    assert "Results" in ex["prompt_text"]


def test_one_enormous_section_still_fits_and_keeps_the_instructions(monkeypatch, row):
    # 400k characters in a single section: no choice of sections can fit, and at 1.5 characters per
    # token the first guess at a character budget is far too generous.
    s, ex = _build(monkeypatch, row, [FakeSection("All", "z" * 400_000)], cpt=1.5)
    assert "drop" not in ex
    assert ex["meta"]["trunc_kind"] in ("sections", "hard")
    assert len(ex["prompt_ids"]) <= s.max_prompt_tokens
    assert ex["prompt_text"].endswith("assistant\n")    # the instructions and header survive
    assert len(ex["prompt_ids"]) == len(FakeTok(1.5)(ex["prompt_text"], add_special_tokens=False).input_ids)


def test_a_body_that_overshoots_by_a_little_still_converges(monkeypatch, row):
    """The regression that produced 21 000-token prompts.

    The body (30k characters in six sections) is well under the *character* budget but over the
    *token* budget, because this paper tokenises at 2 characters per token. Shrinking the character
    budget from its previous value then changes nothing — six attempts later the example fell into
    the hard-cut path, which used to slice the token ids and destroy the end of the prompt. The
    budget now shrinks from the body that was actually produced.
    """
    sections = [FakeSection(f"S{i}", "s" * 5_000) for i in range(6)]
    s, ex = _build(monkeypatch, row, sections, cpt=2.0)
    assert "drop" not in ex
    assert ex["meta"]["trunc_kind"] == "sections"
    assert len(ex["prompt_ids"]) <= s.max_prompt_tokens
    assert len(ex["prompt_ids"]) == len(FakeTok(2.0)(ex["prompt_text"], add_special_tokens=False).input_ids)


def test_empty_body_is_dropped(monkeypatch, row):
    _, ex = _build(monkeypatch, row, [FakeSection("", "")])
    assert ex["drop"] == "empty_body"
