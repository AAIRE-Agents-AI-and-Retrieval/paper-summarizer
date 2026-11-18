"""Scoring a generated summary against the reference, without touching a GPU.

What is measured, and why these:

* **Schema compliance** — the summary is consumed by a search engine, so a reply that is not valid
  JSON with the eight required fields is worthless however well written: parse rate, repair rate,
  required fields, extra fields, the schema's own count ranges, and whether generation stopped on
  the EOS token instead of the token limit. Small models are known to drift here, which is exactly
  what fine-tuning should fix, so it is measured at every milestone.
* **ROUGE-1/2/L** (Lin, 2004; ``rouge_score``, Porter stemming) — the standard summarisation
  overlap metric, reported for the whole summary and per field. It rewards n-gram overlap with one
  reference and nothing else, which is why it is never read on its own here.
* **Keyword set F1** and **domain accuracy** — the two fields that are sets, not prose, where exact
  retrieval behaviour depends on getting them right.
* **Length ratio and repetition** (distinct-4, longest repeated 10-gram) — 7-9B models degrade into
  repetition at long outputs, and a summariser that pads is worse than one that stops.

:mod:`scisum.evaluate` adds the two metrics that need a model: BERTScore (semantic similarity) and
NLI faithfulness against the paper text (:mod:`scisum.faithfulness`).
"""

from __future__ import annotations

import json
import re
import statistics as st
from functools import lru_cache

from scisum.prompt import SUMMARY_KEYS

_STR_KEYS = ("tldr", "problem", "method", "results", "limitations", "domain")
_LIST_KEYS = ("contributions", "keywords")
_WORD = re.compile(r"[a-z0-9]+")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


# ---- parsing ------------------------------------------------------------------------------------
def parse_prediction(text: str) -> tuple[dict | None, dict]:
    """The JSON object a reply contains, plus how much repair it needed.

    ``repaired`` counts replies that were not clean JSON but could be recovered (a markdown fence,
    a preamble, trailing text, or an unterminated object). The model is expected to emit bare JSON;
    the recovery exists so that one stray token does not hide a perfectly good summary.
    """
    diag = {"json_ok": False, "repaired": False, "parse_error": ""}
    raw = text.strip()
    try:
        obj = json.loads(raw)
        diag["json_ok"] = True
        return (obj if isinstance(obj, dict) else None), diag
    except json.JSONDecodeError as e:
        diag["parse_error"] = str(e)[:120]
    candidate = _FENCE.sub("", raw).strip()
    start = candidate.find("{")
    if start < 0:
        return None, diag
    depth = 0
    in_str = esc = False
    for i, ch in enumerate(candidate[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(candidate[start:i + 1])
                except json.JSONDecodeError:
                    break
                diag["repaired"] = True
                return (obj if isinstance(obj, dict) else None), diag
    return None, diag


def schema_report(obj: dict | None) -> dict:
    """Field-level conformance to the summary schema."""
    if not isinstance(obj, dict):
        return {"schema_ok": False, "fields_present": 0, "extra_fields": 0, "type_errors": 1,
                "keywords_in_range": False, "contributions_in_range": False}
    present = [k for k in SUMMARY_KEYS if k in obj]
    extra = [k for k in obj if k not in SUMMARY_KEYS]
    type_errors = 0
    for k in _STR_KEYS:
        if k in obj and not isinstance(obj[k], str):
            type_errors += 1
    for k in _LIST_KEYS:
        if k in obj and not (isinstance(obj[k], list) and all(isinstance(x, str) for x in obj[k])):
            type_errors += 1
    kw = obj.get("keywords") if isinstance(obj.get("keywords"), list) else []
    con = obj.get("contributions") if isinstance(obj.get("contributions"), list) else []
    return {"schema_ok": len(present) == len(SUMMARY_KEYS) and not type_errors,
            "fields_present": len(present), "extra_fields": len(extra), "type_errors": type_errors,
            "keywords_in_range": 5 <= len(kw) <= 10, "contributions_in_range": 2 <= len(con) <= 5,
            "n_keywords": len(kw), "n_contributions": len(con)}


def render_summary(obj: dict | None) -> str:
    """The summary as one text, in a fixed order: what ROUGE and BERTScore compare."""
    if not isinstance(obj, dict):
        return ""
    parts = []
    for k in SUMMARY_KEYS:
        v = obj.get(k)
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v)
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts)


def field_text(obj: dict | None, key: str) -> str:
    if not isinstance(obj, dict):
        return ""
    v = obj.get(key)
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return v.strip() if isinstance(v, str) else ""


# ---- overlap metrics ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(pred: str, ref: str) -> dict:
    if not pred or not ref:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    sc = _scorer().score(ref, pred)
    return {k: round(v.fmeasure, 5) for k, v in sc.items()}


def norm_phrase(x: str) -> str:
    return " ".join(_WORD.findall(x.lower()))


def set_f1(pred: list[str], ref: list[str]) -> dict:
    p = {norm_phrase(x) for x in pred if norm_phrase(x)}
    r = {norm_phrase(x) for x in ref if norm_phrase(x)}
    if not p and not r:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"p": round(prec, 5), "r": round(rec, 5), "f1": round(f1, 5)}


def token_f1(pred: str, ref: str) -> float:
    p, r = _WORD.findall(pred.lower()), _WORD.findall(ref.lower())
    if not p or not r:
        return float(p == r)
    common = 0
    pool = list(r)
    for t in p:
        if t in pool:
            pool.remove(t)
            common += 1
    prec, rec = common / len(p), common / len(r)
    return round(2 * prec * rec / (prec + rec), 5) if prec + rec else 0.0


def repetition(text: str, n: int = 4, long_n: int = 10) -> dict:
    """distinct-n (1.0 = no repeated n-gram) and the worst repeat count of a long n-gram."""
    w = _WORD.findall(text.lower())
    if len(w) < n:
        return {"distinct4": 1.0, "max_repeat10": 0}
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    counts: dict[tuple, int] = {}
    worst = 0
    if len(w) >= long_n:
        for i in range(len(w) - long_n + 1):
            g = tuple(w[i:i + long_n])
            counts[g] = counts.get(g, 0) + 1
            worst = max(worst, counts[g])
    return {"distinct4": round(len(set(grams)) / len(grams), 5), "max_repeat10": worst}


# ---- one paper ----------------------------------------------------------------------------------
def score_one(pred_text: str, reference: dict, *, finish_reason: str = "stop", out_tokens: int = 0,
              target_prefix: str = "") -> dict:
    """Every GPU-free metric for one generated summary.

    ``target_prefix`` is what the model's own chat template puts before the answer and the model was
    trained to emit — nothing for Qwen3, ``</ifm|think>`` for K2-Horizon, whose template always opens
    a reasoning block. It is protocol, not malformedness, so it is removed before parsing: scored
    without it, a perfectly behaved K2 reads as 0.0% clean JSON.
    """
    # Both sides are stripped: the template's prefix can itself start with whitespace (Qwen3.5's is
    # "\n</think>\n\n"), and comparing it against lstrip()ped text never matched, so every
    # well-formed Student C reply was scored as "not clean JSON".
    text = pred_text.lstrip()
    prefix = target_prefix.lstrip()
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    obj, diag = parse_prediction(text)
    out = {**diag, **schema_report(obj), "out_tokens": out_tokens,
           "stopped_on_eos": finish_reason == "stop", "hit_token_limit": finish_reason == "length"}
    pred_all, ref_all = render_summary(obj), render_summary(reference)
    out["pred_chars"], out["ref_chars"] = len(pred_all), len(ref_all)
    out["length_ratio"] = round(len(pred_all) / max(len(ref_all), 1), 4)
    out.update({f"all_{k}": v for k, v in rouge(pred_all, ref_all).items()})
    for key in ("tldr", "problem", "method", "results", "limitations", "contributions"):
        out[f"{key}_rougeL"] = rouge(field_text(obj, key), field_text(reference, key))["rougeL"]
    kw = set_f1(obj.get("keywords") or [] if isinstance(obj, dict) else [], reference["keywords"])
    out.update({"keywords_p": kw["p"], "keywords_r": kw["r"], "keywords_f1": kw["f1"]})
    out["domain_exact"] = float(norm_phrase(field_text(obj, "domain")) == norm_phrase(reference["domain"]))
    out["domain_f1"] = token_f1(field_text(obj, "domain"), reference["domain"])
    out.update(repetition(pred_all))
    return out


# ---- aggregation --------------------------------------------------------------------------------
def aggregate(rows: list[dict], keys: list[str] | None = None) -> dict:
    """Mean (and median for the heavy-tailed ones) over per-paper scores."""
    if not rows:
        return {"papers": 0}
    keys = keys or sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool))})
    out: dict[str, float] = {"papers": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float, bool))]
        if not vals:
            continue
        out[k] = round(st.fmean(vals), 5)
        if k in ("out_tokens", "pred_chars", "ref_chars", "length_ratio"):
            out[f"{k}_median"] = round(st.median(vals), 3)
    return out


def group_by(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(field)), []).append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}
