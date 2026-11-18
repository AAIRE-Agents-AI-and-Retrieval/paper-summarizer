"""Shared helpers of the analysis scripts: reading run artifacts, statistics, the student registry, colours.

Nothing here needs a GPU, a model or the network. The scripts in this directory import it as a sibling
module (``python scripts/analysis/<script>.py`` puts this directory on ``sys.path``).
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
import statistics as st
from collections.abc import Iterable, Sequence
from pathlib import Path

# ---- files --------------------------------------------------------------------------------------


def read_jsonl(path: Path | str) -> list[dict]:
    """Rows of a JSON-lines file, gzip or plain. Missing file: ``[]``. A truncated last line is skipped.

    Generation and grading append to gzip files member by member, so a file being written can end in a
    half line; skipping it keeps a live reading usable instead of crashing.
    """
    p = Path(path)
    if not p.exists():
        return []
    opener = gzip.open if p.suffix == ".gz" else open
    rows: list[dict] = []
    try:
        with opener(p, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except EOFError:      # a gzip member cut short by a killed writer: keep what was read
        pass
    return rows


def read_json(path: Path | str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically: ``ex9000`` before ``ex50000``."""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def last_match(directory: Path, pattern: str) -> Path | None:
    """The last file (in natural order) matching a glob pattern, or None. Eval names carry the checkpoint,
    so with ``test-ex*-n1246`` the last match is the latest checkpoint evaluated on those papers."""
    found = sorted(directory.glob(pattern), key=lambda p: natural_key(p.name))
    return found[-1] if found else None


_EXPORT = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$""")


def parse_env_file(path: Path | str) -> dict[str, str]:
    """``KEY=VALUE`` pairs of a shell profile (``configs/students/*.env``), without running it.

    Handles ``export``, single or double quotes and trailing comments outside quotes; ``$VAR`` and
    ``${VAR:-x}`` are expanded from keys set earlier in the same file only. ``source`` lines are ignored:
    the profiles of this repository write every setting out (one profile per student, one per recipe).
    """
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        m = _EXPORT.match(raw)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val[:1] in ("'", '"'):
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end > 0 else val[1:]
            if q == "'":
                out[key] = val
                continue
        else:
            val = val.split(" #", 1)[0].strip()

        def _sub(mm: re.Match) -> str:
            name, default = mm.group(1) or mm.group(3), mm.group(2)
            return out.get(name, default if default is not None else "")

        out[key] = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)", _sub, val)
    return out


# ---- students -----------------------------------------------------------------------------------

# A fixed order: a student's colour follows the student, never its rank.
STUDENT_ORDER = ("k2h-3.7b", "minicpm5-2b", "qwen35-4b", "k2h-7b", "granite42-3b")


def discover_students(students_dir: Path | str) -> list[dict]:
    """``configs/students/<tag>.env`` as dicts ``{tag, env_path, base_model, revision, params}``, in the
    fixed order, unknown tags after the known ones (alphabetical). Anchor profiles (``STUDENT_ROLE=anchor``)
    are returned too, flagged."""
    d = Path(students_dir)
    rows = []
    for p in sorted(d.glob("*.env")):
        env = parse_env_file(p)
        tag = env.get("STUDENT_TAG") or p.stem
        params = env.get("STUDENT_PARAMS")
        try:
            params_f = float(params) if params else None
        except ValueError:
            params_f = None
        rows.append({"tag": tag, "env_path": str(p), "base_model": env.get("SCISUM_BASE_MODEL", ""),
                     "revision": env.get("SCISUM_BASE_MODEL_REVISION", ""), "params": params_f,
                     "anchor": env.get("STUDENT_ROLE", "") == "anchor"})
    rank = {t: i for i, t in enumerate(STUDENT_ORDER)}
    return sorted(rows, key=lambda r: (rank.get(r["tag"], len(rank)), r["tag"]))


def run_names(tag: str) -> dict[str, str]:
    """The runs of one student: <tag>-sft50k, <tag>-prefcc and <tag>-prefcc-s2."""
    return {"sft": f"{tag}-sft50k", "prefcc": f"{tag}-prefcc", "prefcc_s2": f"{tag}-prefcc-s2"}


# ---- statistics ---------------------------------------------------------------------------------


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (numpy's default); NaN for an empty list."""
    v = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return float("nan")
    if len(v) == 1:
        return float(v[0])
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def describe(xs: Iterable[float]) -> dict:
    v = [float(x) for x in xs if x is not None]
    if not v:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {"n": len(v), "mean": st.fmean(v), "p50": quantile(v, 0.5), "p95": quantile(v, 0.95),
            "min": min(v), "max": max(v)}


def boot_ci(d: Sequence[float], n: int = 2000, seed: int = 0, level: float = 0.95) -> list[float]:
    """[mean, low, high]: percentile bootstrap of the mean (paired when ``d`` are per-paper differences)."""
    if len(d) < 2:
        return [st.fmean(d) if d else float("nan"), float("nan"), float("nan")]
    rnd = random.Random(seed)
    k = len(d)
    ms = sorted(st.fmean(rnd.choices(d, k=k)) for _ in range(n))
    tail = (1 - level) / 2
    return [st.fmean(d), ms[int(tail * n)], ms[min(n - 1, int((1 - tail) * n))]]


def perm_p(d: Sequence[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test of mean(d) = 0 (as in the C -> prefcc verdicts)."""
    if not d:
        return float("nan")
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dependency of the repository
        rnd = random.Random(seed)
        obs = abs(st.fmean(d))
        hits = sum(abs(st.fmean([x if rnd.random() < 0.5 else -x for x in d])) >= obs for _ in range(n))
        return (1 + hits) / (n + 1)
    x = np.asarray(d, dtype=float)
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(x)))
    null = np.abs((flips * x).mean(axis=1))
    return float((1 + (null >= abs(x.mean())).sum()) / (n + 1))


def fmt(x: float | None, digits: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def fmt_ci(ci: Sequence[float] | None, digits: int = 4, signed: bool = True) -> str:
    if not ci or ci[0] is None or (isinstance(ci[0], float) and math.isnan(ci[0])):
        return "–"
    if len(ci) < 3 or any(isinstance(c, float) and math.isnan(c) for c in ci[1:3]):
        return fmt(ci[0], digits, signed)
    return f"{fmt(ci[0], digits, signed)} [{fmt(ci[1], digits, signed)}, {fmt(ci[2], digits, signed)}]"


# ---- Pareto -------------------------------------------------------------------------------------


def pareto_front(points: Sequence[dict], quality: str, cost: str, higher_is_better: bool = True) -> list[int]:
    """Indices of the non-dominated points: no other point is at least as good on quality and at least as
    cheap, and strictly better on one. Points missing either value are never on the front."""
    sign = 1.0 if higher_is_better else -1.0
    ok = [i for i, p in enumerate(points) if _num(p.get(quality)) is not None and _num(p.get(cost)) is not None]
    front = []
    for i in ok:
        qi, ci = sign * points[i][quality], points[i][cost]
        dominated = False
        for j in ok:
            if j == i:
                continue
            qj, cj = sign * points[j][quality], points[j][cost]
            if qj >= qi and cj <= ci and (qj > qi or cj < ci):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: points[i][cost])


def _num(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ---- plotting style -----------------------------------------------------------------------------

# The validated categorical palette (light mode), slots in fixed order; anchors are neutral ink.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def student_style(tag: str, known: Sequence[str] = STUDENT_ORDER) -> dict:
    """Colour and marker of a student: slot by its place in the fixed order (unknown tags after it).

    Scatter plots carry more than three categories, so colour is never the only encoding: every student
    also has its own marker and a direct label."""
    order = list(known)
    if tag not in order:
        order.append(tag)
    i = order.index(tag) % len(PALETTE)
    return {"color": PALETTE[i], "marker": MARKERS[i % len(MARKERS)]}


def setup_matplotlib():
    """Agg backend, the repository's plain style; returns ``pyplot``."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    })
    return plt


def save_figure(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=160 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    return paths


def md_table(header: Sequence[str], rows: Iterable[Sequence], align: Sequence[str] | None = None) -> str:
    """A GitHub Markdown table; ``align`` per column: 'l' or 'r' (default: first left, rest right)."""
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---" if a == "l" else "--:" for a in align]
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(lines)
