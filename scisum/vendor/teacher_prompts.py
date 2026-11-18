"""The teacher's prompt pieces and section selection, copied from the teacher pipeline.

Contents: ``SYSTEM_PROMPT``, ``SUMMARY_SCHEMA``, ``_KEYS_BLOCK``, ``_PRIORITY_HINTS``, ``_section_priority``,
``select_fulltext``, and ``Section`` and ``FullText`` (only the fields and methods ``select_fulltext`` uses, as
dataclasses instead of pydantic models).
The strings are byte-identical to the originals; do not reformat them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sections worth keeping first when a paper does not fit the budget: what the paper did and found.
_PRIORITY_HINTS = (
    ("introduction", 0), ("conclusion", 1), ("discussion", 2), ("result", 2), ("experiment", 3),
    ("method", 3), ("approach", 3), ("model", 4), ("analysis", 4), ("related work", 8),
    ("background", 7), ("appendix", 9), ("acknowledg", 10), ("ethic", 10), ("broader impact", 10),
)

SYSTEM_PROMPT = """You are an expert scientific editor writing structured summaries of research papers for a
scientific search engine. Summaries must be faithful to the provided text, specific, and dense with the
concrete terms a researcher would search for (methods, models, datasets, metrics, domains). Never invent
results that are not stated. Write in English regardless of the input language. Output only JSON."""

SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string", "description": "1-2 sentences: what the paper does and finds"},
        "problem": {"type": "string", "description": "The problem or research question"},
        "contributions": {"type": "array", "items": {"type": "string"}, "description": "2-5 key contributions"},
        "method": {"type": "string", "description": "Approach, models, data, experimental setup"},
        "results": {"type": "string", "description": "Main findings, with numbers if given"},
        "limitations": {"type": "string", "description": "Stated or evident limitations; empty if none"},
        "keywords": {"type": "array", "items": {"type": "string"}, "description": "5-10 searchable keywords"},
        "domain": {"type": "string", "description": "Broad field, e.g. 'information retrieval'"},
    },
    "required": ["tldr", "problem", "contributions", "method", "results", "limitations", "keywords", "domain"],
    "additionalProperties": False,
}


_KEYS_BLOCK = """Return a JSON object with exactly these keys:
- "tldr": 1-2 sentences saying what the paper does and what it finds.
- "problem": the problem or research question addressed.
- "contributions": array of 2-5 short strings, each a concrete contribution.
- "method": the approach — models, algorithms, data, experimental setup.
- "results": the main quantitative or qualitative findings (include numbers when stated).
- "limitations": limitations stated or evident from the text (empty string if none).
- "keywords": array of 5-10 specific, searchable keywords or phrases.
- "domain": the broad research field in a few words."""


@dataclass
class Section:
    """One section of a paper's full text."""

    title: str = ""
    text: str = ""

    def as_text(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title else self.text


@dataclass
class FullText:
    """The body of a paper, reduced to what section selection uses."""

    doc_id: str
    source_url: str = ""
    sections: list[Section] = field(default_factory=list)

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    def text(self) -> str:
        return "\n\n".join(s.as_text() for s in self.sections).strip()


def _section_priority(title: str) -> int:
    t = title.lower()
    for hint, rank in _PRIORITY_HINTS:
        if hint in t:
            return rank
    return 5


def select_fulltext(fulltext: FullText, budget_chars: int) -> tuple[str, bool]:
    """Fit a paper's body into ``budget_chars``, dropping the least useful sections first.

    Returns ``(text, truncated)``. Sections are kept whole and in reading order; when the paper is
    too long, related work, appendices and acknowledgements go before results and methods do.
    """
    sections: list[Section] = [s for s in fulltext.sections if s.text]
    total = sum(len(s.as_text()) + 2 for s in sections)
    if total <= budget_chars:
        return fulltext.text(), False

    order = sorted(range(len(sections)), key=lambda i: (_section_priority(sections[i].title), i))
    keep: set[int] = set()
    used = 0
    for i in order:
        cost = len(sections[i].as_text()) + 2
        if used + cost <= budget_chars:
            keep.add(i)
            used += cost
    if not keep:  # one huge section: take the head of the first one
        return sections[0].as_text()[:budget_chars], True
    return "\n\n".join(sections[i].as_text() for i in sorted(keep)), True
