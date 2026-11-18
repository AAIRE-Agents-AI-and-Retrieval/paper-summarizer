"""Where each grader model is served: one OpenAI-compatible base URL per served model name.

The fact test, the claim check, the judge and the preference stage's reward all talk to LLM graders
(``gemma4-31b-it``, ``qwen38-27b``; see ``docs/GRADERS.md``). The earlier work used one shared endpoint
for both. A self-hosted setup (for example two vLLM servers on LUMI) has one URL per model, so every
client asks this module for the URL of the model it is about to call. Resolution order, first hit wins:

1. ``SCISUM_GRADER_ENDPOINTS``: inline JSON, or the path of a JSON file, mapping a served model name to
   a base URL (``{"gemma4-31b-it": "http://nid005123:8000/v1", ...}``). A value may also be an object
   ``{"base_url": ..., "api_key_env": "NAME_OF_ENV_VAR"}``.
2. ``state/graders/*.json``, written by a grader server job when its server answers
   (``{"model", "base_url", "job", "started"}``); the newest file per model wins.
3. ``SCISUM_JUDGE_BASE`` (with ``JUDGE_API_KEY``): one endpoint for every model, as before.

The API key is optional: a self-hosted vLLM server without ``--api-key`` accepts any token, so
``"EMPTY"`` is sent when none is set. Keys and URLs are never written to logs or run.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from scisum.config import Settings


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str
    api_key: str
    source: str          # "env", "state/graders/<file>" or "judge_base": where the URL came from (never the URL)


def _from_env(model: str) -> tuple[str, str] | None:
    raw = os.environ.get("SCISUM_GRADER_ENDPOINTS", "").strip()
    if not raw:
        return None
    table = json.loads(raw if raw.startswith("{") else Path(raw).read_text())
    v = table.get(model)
    if v is None:
        return None
    if isinstance(v, str):
        return v, ""
    key = os.environ.get(v.get("api_key_env", ""), "") if v.get("api_key_env") else ""
    return v["base_url"], key


def _from_state(s: Settings, model: str) -> tuple[str, str] | None:
    d = s.state_dir / "graders"
    if not d.is_dir():
        return None
    best: tuple[float, str, str] | None = None
    for f in d.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("model") != model or not rec.get("base_url"):
            continue
        t = f.stat().st_mtime
        if best is None or t > best[0]:
            best = (t, rec["base_url"], f.name)
    return (best[1], best[2]) if best else None


def resolve(s: Settings, model: str) -> Endpoint:
    """The endpoint serving ``model``; raises with what to set when there is none."""
    api_key = s.judge_api_key or os.environ.get("JUDGE_API_KEY", "") or "EMPTY"
    hit = _from_env(model)
    if hit:
        return Endpoint(model, hit[0].rstrip("/"), hit[1] or api_key, "env")
    hit = _from_state(s, model)
    if hit:
        return Endpoint(model, hit[0].rstrip("/"), api_key, f"state/graders/{hit[1]}")
    if s.judge_base:
        return Endpoint(model, s.judge_base.rstrip("/"), api_key, "judge_base")
    raise RuntimeError(
        f"no endpoint for grader {model!r}: start its server (docs/GRADERS.md) so it writes state/graders/*.json, "
        "or set SCISUM_GRADER_ENDPOINTS, or SCISUM_JUDGE_BASE in .env")


def check(s: Settings, models: list[str], timeout: float = 20.0) -> list[dict]:
    """For each model: is an endpoint configured, does it answer ``GET /models``, and does it serve that name?"""
    import httpx

    out = []
    for m in models:
        row: dict = {"model": m, "configured": False, "reachable": False, "serves_model": False, "source": ""}
        try:
            ep = resolve(s, m)
        except RuntimeError as e:
            row["error"] = str(e)
            out.append(row)
            continue
        row.update(configured=True, source=ep.source)
        try:
            r = httpx.get(f"{ep.base_url}/models", headers={"Authorization": f"Bearer {ep.api_key}"}, timeout=timeout)
            r.raise_for_status()
            names = [d.get("id") for d in r.json().get("data", [])]
            row.update(reachable=True, serves_model=m in names, served=names)
        except (httpx.HTTPError, ValueError) as e:
            row["error"] = f"{type(e).__name__}"
        out.append(row)
    return out
