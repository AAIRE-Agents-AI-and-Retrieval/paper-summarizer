"""The prompt the student is trained on, and the canonical form of a summary.

The student must see exactly what the teacher saw, or the 100k teacher summaries are labels for a
different task. The prompt text, the JSON schema and the section-selection logic are the teacher
pipeline's own, copied verbatim into :mod:`scisum.vendor`;
the two wrappers the teacher run used — :func:`fit_body` and :func:`build_user_message` — are repeated
here. ``tests/test_prompt_parity.py`` checks both against hashes recorded with the teacher's own code
(``tests/fixtures/prompt_golden.jsonl``).

The one deliberate difference is the body budget: the teacher had 160k characters of context, the
student has ``max_seq_len`` tokens, so :func:`fit_prompt` shrinks the body until prompt + target fit.
Every example records how it was cut, so the effect can be measured instead of guessed.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass

from scisum.vendor.teacher_prompts import (
    _KEYS_BLOCK,
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    FullText,
    select_fulltext,
)
from scisum.vendor.teacher_prompts import Section as SciSection
from scisum.vendor.tei_sections import doc_id

SUMMARY_KEYS: tuple[str, ...] = tuple(SUMMARY_SCHEMA["required"])
_STR_KEYS = ("tldr", "problem", "method", "results", "limitations", "domain")
_LIST_KEYS = ("contributions", "keywords")
_WORD = re.compile(r"\w+")


# ---- prompt (identical to the teacher's) --------------------------------------------------------
def build_user_message(doc: dict, body: str, abstract_only: bool) -> str:
    kind = ("abstract" if abstract_only else
            "abstract and full text (extracted from the PDF; references and appendices removed)")
    text = "" if abstract_only else f"\nFull text:\n{body}\n"
    return f"""Summarise the following paper from its title, {kind}.

Title: {doc.get("title", "")}
Authors: {", ".join((doc.get("authors") or [])[:8])}
Categories: {", ".join(doc.get("categories") or [])}
Abstract:
{doc.get("abstract", "")}
{text}
{_KEYS_BLOCK}"""


def messages(doc: dict, body: str, abstract_only: bool = False) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(doc, body, abstract_only)}]


_MARK = "\u2063CONTENT\u2063"   # an invisible separator that no tokenizer template will rewrite


def assistant_wrapper(tok) -> tuple[str, str]:
    """What the model's own chat template puts around an assistant answer: ``(prefix, suffix)``.

    Qwen3 gives ``("", "<|im_end|>")``. K2-Horizon gives ``("</ifm|think>", "<|ifm|im_end|>")``,
    because its template *always* opens a reasoning block in the generation prompt and raises if an
    assistant message carries no thinking field — so the student has to learn to open the block,
    close it immediately and answer. Reading this from the template instead of hardcoding it means
    a third model needs no code, and the training target can never drift from what the model is
    served with.
    """
    cached = getattr(tok, "_scisum_assistant_wrapper", None)
    if cached is not None:   # rendering a template twice per example is not free
        return cached
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": _MARK, "think": ""}],
                                   tokenize=False)
    # Most templates render the generation prompt as an exact prefix of the full conversation. Some do not
    # (granite-4.2-3b's): then the shared prefix is used, provided the generation prompt has nothing beyond it
    # but whitespace and the marker lies after it.
    k = len(gen) if full.startswith(gen) else len(os.path.commonprefix([gen, full]))
    if gen[k:].strip() or _MARK not in full[k:]:
        raise RuntimeError(f"cannot read the assistant turn from this chat template: {full[-200:]!r}")
    prefix, _, suffix = full[k:].partition(_MARK)
    out = (prefix, suffix.rstrip("\n"))
    with contextlib.suppress(AttributeError, TypeError):
        tok._scisum_assistant_wrapper = out
    return out


def fit_body(arxiv_id: str, parsed, max_chars: int, kinds: tuple[str, ...]) -> tuple[str, bool, int, int]:
    """The sections of ``kinds``, cut to ``max_chars`` by dropping the least useful ones first.

    ``parsed`` is anything with ``.sections`` of ``title``/``text``/``kind`` items: a
    :class:`scisum.vendor.tei_sections.ParsedBody` from the packaged corpus.
    """
    keep = [s for s in parsed.sections if s.text and s.kind in kinds]
    dropped = sum(len(s.text) for s in parsed.sections if s.text and s.kind not in kinds)
    ft = FullText(doc_id=doc_id(arxiv_id), source_url="",
                  sections=[SciSection(title=s.title, text=s.text) for s in keep])
    text, truncated = select_fulltext(ft, max_chars)
    return text, truncated, len(ft.sections), dropped


# ---- target -------------------------------------------------------------------------------------
def canonical_target(summary: dict) -> str:
    """The 8 content fields, in schema order, as the JSON string the student learns to emit.

    ``json.dumps`` defaults (``", "`` / ``": "`` separators, unescaped non-ASCII) rather than a
    compact or indented form: it is what a JSON parser and a reader both expect, and the ~1.5% of
    output tokens the separators cost buys robustness in the generated text.
    """
    return json.dumps({k: summary[k] for k in SUMMARY_KEYS}, ensure_ascii=False)


MIN_DISTINCT_10 = 0.80
_MIN_CHARS = {"tldr": 10, "problem": 10, "method": 10, "results": 10, "domain": 3}


def distinct_n(text: str, n: int = 10) -> float:
    """Share of distinct word n-grams: 1.0 for ordinary prose, near 0 for a generation loop.

    Measured on all 100 827 teacher summaries: the median is 1.000, the 1st percentile 0.965 and
    the worst 0.583, so the threshold at 0.80 rejects the 8 pathological ones and nothing else.
    An earlier rule ("any 8-gram repeated four times") rejected 851 perfectly good summaries,
    because a formula or a long technical term ("irreducible subfactor planar algebra with a
    distributive biprojection lattice") repeats legitimately in a dense scientific summary.
    """
    words = _WORD.findall(text.lower())
    if len(words) < n + 5:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


@dataclass(slots=True)
class Check:
    ok: bool
    reason: str = ""


def validate_summary(obj: object) -> Check:
    """Is this teacher summary usable as a training target?"""
    if not isinstance(obj, dict):
        return Check(False, "not_object")
    missing = [k for k in SUMMARY_KEYS if k not in obj]
    if missing:
        return Check(False, f"missing:{','.join(missing)}")
    for k in _STR_KEYS:
        v = obj[k]
        if not isinstance(v, str):
            return Check(False, f"type:{k}")
        if len(v.strip()) < _MIN_CHARS.get(k, 0):  # "limitations" may be empty; "domain" may be "optics"
            return Check(False, f"empty:{k}")
    for k in _LIST_KEYS:
        v = obj[k]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            return Check(False, f"type:{k}")
    if not 2 <= len(obj["contributions"]) <= 10:
        return Check(False, "contributions_count")
    if not 3 <= len(obj["keywords"]) <= 20:
        return Check(False, "keywords_count")
    text = " ".join([obj[k] for k in _STR_KEYS] + obj["contributions"] + obj["keywords"])
    if len(text) > 25_000:
        return Check(False, "too_long")
    if distinct_n(text) < MIN_DISTINCT_10:
        return Check(False, "repetitive")
    return Check(True)
