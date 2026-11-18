"""The parsed GROBID body as the prompt builder consumes it.

The corpus pipeline's TEI parser turns a GROBID TEI file into sections, each with a title, its text and a kind:
``body``, ``appendix``, ``captions`` (one section holding every figure and table caption), ``references`` or
``backmatter`` (acknowledgements, funding, declarations). The packaged corpus stores exactly those sections per
paper (``data/corpus/bodies.sqlite``), so the TEI files and the parser are not needed here. Only the fields
:func:`scisum.prompt.fit_body` reads are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("body", "appendix", "captions", "references", "backmatter")


@dataclass
class TeiSection:
    title: str
    text: str
    kind: str = "body"  # body | appendix | captions | references | backmatter


@dataclass
class ParsedBody:
    sections: list[TeiSection] = field(default_factory=list)

    @classmethod
    def from_rows(cls, rows: list) -> ParsedBody:
        """From the stored ``[[title, text, kind], ...]`` list."""
        return cls([TeiSection(title=t, text=x, kind=k) for t, x, k in rows])

    def to_rows(self) -> list[list[str]]:
        return [[s.title, s.text, s.kind] for s in self.sections]


def doc_id(arxiv_id: str) -> str:
    """The document id the teacher pipeline used for an arXiv paper."""
    return f"arxiv:{arxiv_id}"
