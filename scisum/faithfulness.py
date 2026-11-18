"""Is the summary supported by the paper? NLI entailment over retrieved evidence.

A summariser that invents results is worse than one that paraphrases badly, and no overlap metric
sees the difference. The standard way to measure it without a reference is sentence-level natural
language inference between the source document and each claim in the summary (SummaC, Laban et al.
2022; the same construction underlies AlignScore and MiniCheck).

A paper is far too long to pair every claim with every source sentence (a 13k-token body is ~500
sentences; 40 claims would be 20 000 NLI calls per paper), so the source is split into overlapping
chunks of ``chunk_sents`` sentences and only the ``k`` chunks a claim is lexically closest to
(BM25) are paired with it. The claim's score is the best entailment probability over those chunks,
and the paper's score is the mean over its claims — ``max`` over evidence, ``mean`` over claims, as
in SummaC-ZS.

This is a relative measure: it is read as "milestone 20k is more grounded than milestone 1k", never
as an absolute hallucination rate. The NLI model (``microsoft/deberta-large-mnli``, 400M) is fixed
and off-the-shelf so the numbers stay comparable across runs.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

log = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
_WORD = re.compile(r"[a-z0-9]+")
_CLAIM_FIELDS = ("tldr", "problem", "method", "results", "limitations")


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    """Sentences of at least ``min_chars``: a fragment that short carries nothing to check."""
    out = [s.strip() for s in _SENT.split(text or "") if s and len(s.strip()) >= min_chars]
    return out


def chunk_sentences(sents: list[str], size: int = 3, stride: int = 2) -> list[str]:
    if not sents:
        return []
    return [" ".join(sents[i:i + size]) for i in range(0, max(len(sents) - size + stride, 1), stride)]


def claims_of(obj: dict | None, max_claims: int = 40) -> list[str]:
    """The checkable statements in a summary: prose sentences and the contribution bullets."""
    if not isinstance(obj, dict):
        return []
    out: list[str] = []
    for k in _CLAIM_FIELDS:
        v = obj.get(k)
        if isinstance(v, str):
            out.extend(split_sentences(v))
    v = obj.get("contributions")
    if isinstance(v, list):
        out.extend(str(x).strip() for x in v if isinstance(x, str) and len(x.strip()) >= 25)
    return out[:max_claims]


class Bm25:
    """BM25 over one paper's chunks (small, built per paper, thrown away)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [_WORD.findall(d.lower()) for d in docs]
        self.k1, self.b = k1, b
        self.len = [len(t) for t in self.toks]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        self.tf = [Counter(t) for t in self.toks]
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = max(len(docs), 1)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def top_k(self, query: str, k: int) -> list[int]:
        q = _WORD.findall(query.lower())
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.len[i] or 1
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


class NliScorer:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels.get("entailment", max(labels.values()))
        log.info("NLI model %s on %s, entailment index %d", model_name, device, self.entail)

    def entailment(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(premises), self.batch_size):
                enc = self.tok(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                               truncation=True, max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                probs = self.model(**enc).logits.softmax(-1)[:, self.entail]
                out.extend(float(x) for x in probs.cpu())
        return out


def score_papers(items: list[tuple[str, dict | None, str]], scorer: NliScorer, *, k: int = 4,
                 chunk_sents: int = 3, max_claims: int = 40) -> dict[str, dict]:
    """``items`` is ``(paper_id, predicted summary object, source text)``.

    Returns per-paper faithfulness: mean best-entailment over claims, and the share of claims whose
    best evidence does not entail them.
    """
    pairs_p: list[str] = []
    pairs_h: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    for rid, obj, source in items:
        claims = claims_of(obj, max_claims)
        chunks = chunk_sentences(split_sentences(source), chunk_sents) or [source[:4000]]
        bm = Bm25(chunks)
        spans[rid] = []
        for c in claims:  # a short paper can have fewer than k chunks, so each claim records its own span
            start = len(pairs_p)
            for j in bm.top_k(c, k):
                pairs_p.append(chunks[j])
                pairs_h.append(c)
            spans[rid].append((start, len(pairs_p)))
    log.info("NLI: %d pairs over %d papers", len(pairs_p), len(items))
    probs = scorer.entailment(pairs_p, pairs_h) if pairs_p else []
    out: dict[str, dict] = {}
    for rid, claim_spans in spans.items():
        if not claim_spans:
            out[rid] = {"faithfulness": 0.0, "unsupported_rate": 1.0, "claims": 0}
            continue
        per_claim = [max(probs[a:b], default=0.0) for a, b in claim_spans]
        n = len(per_claim)
        out[rid] = {"faithfulness": round(sum(per_claim) / n, 5),
                    "unsupported_rate": round(sum(1 for p in per_claim if p < 0.5) / n, 5),
                    "claims": n}
    return out
