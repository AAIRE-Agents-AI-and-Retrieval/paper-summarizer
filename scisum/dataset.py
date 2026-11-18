"""Dataset: from the teacher run's ledgers to tokenised examples, in three auditable steps.

The first two steps were run once by the original project and are packaged (``data/corpus/``, see
``docs/DATA.md``); here they only verify the package.

``scisum data index``
    The index: one record per usable paper with the arXiv metadata, the teacher summary that becomes
    the target, and the statistics needed to stratify (packaged as ``corpus/index.jsonl.gz``; every
    paper left out is in ``corpus/index_drops.jsonl`` with the reason). Checks it against the bodies.

``scisum data split``
    Seeded, stratified by (teacher model, cohort), into 70k train / 15k validation / 15k test, by
    paper. Also fixes the training **order** once and for all: evaluation milestones are positions
    in that order, so "the model after 5 000 examples" means exactly one checkpoint, reproducibly.
    Duplicate titles are collapsed in the index step, so no paper can appear in two splits.
    Recomputes the packaged ``corpus/splits.json`` and reports whether it is identical.

``scisum data build --split <s>``
    Rebuilds the teacher's prompt from the parsed GROBID body, fits it into ``max_seq_len`` tokens together
    with the target, tokenises it with the student's own chat template and writes token shards
    (:mod:`scisum.shards`) in training order, plus prompt text for the generation-based evaluations.
    Parts are independent and skipped when complete, so the build is resumable and parallel.
"""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

from scisum.config import Settings
from scisum.prompt import assistant_wrapper, canonical_target, fit_body, messages
from scisum.shards import ShardWriter

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
_TITLE_KEEP = str.maketrans("", "", "".join(c for c in map(chr, range(128)) if not c.isalnum()))


# ---- step 1: index ------------------------------------------------------------------------------
# The index was built once by the teacher project's pipeline from its three ledgers (summary, metadata,
# GROBID) and is packaged unchanged as data/corpus/index.jsonl.gz. Its rules, applied in this order:
# teacher prompt version "ft-grobid-v1"; not abstract-only; GROBID body of at least 2,000 characters;
# summary passes `validate_summary`; title and abstract present; TEI present; then one row per
# normalised title (the more-cited paper kept). Every dropped paper is listed with its reason in
# data/corpus/index_drops.jsonl. The ledgers and TEI files are not distributed (docs/DATA.md).
def normalise_title(title: str) -> str:
    return (title or "").lower().translate(_TITLE_KEEP)


def run_index(s: Settings) -> dict:
    """Check the packaged index against its bodies and return its statistics (``corpus/index.json``).

    Nothing is rebuilt: the ledgers it came from are not part of this repository.
    """
    from scisum.corpus import body_store

    rows = read_index(s)
    store = body_store(s)
    missing = [r["id"] for r in rows if r["id"] not in store]
    stats = json.loads((s.corpus_dir / "index.json").read_text())
    if len(rows) != stats["eligible"] or missing:
        raise RuntimeError(f"packaged corpus is inconsistent: {len(rows)} rows (index.json says "
                           f"{stats['eligible']}), {len(missing)} without a body, e.g. {missing[:3]}")
    log.info("packaged index: %d eligible papers, all with a body", len(rows))
    return stats


def read_index(s: Settings) -> list[dict]:
    """Every eligible paper, in citation-rank order (the packaged ``corpus/index.jsonl.gz``)."""
    from scisum.corpus import INDEX, read_index_file

    return read_index_file(s.corpus_dir / INDEX)


# ---- step 2: split -----------------------------------------------------------------------------
def _allocate(n: int, want: dict[str, int], total: int) -> dict[str, int]:
    """Split ``n`` items between splits in proportion to ``want`` (largest remainder)."""
    exact = {k: n * v / total for k, v in want.items()}
    base = {k: int(v) for k, v in exact.items()}
    left = n - sum(base.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - base[kv[0]], reverse=True)[:left]:
        base[k] += 1
    return base


EVAL_MARGIN = 100  # candidates above the asked-for 15 000, so tokenisation drops cannot shrink a split


def run_split(s: Settings) -> dict:
    """Recompute the splits from the index and check them against the packaged ``corpus/splits.json``.

    The packaged splits are the ones every run used; this only verifies them (written to
    ``state/splits.recomputed.json``). With the original settings (seed 20261014, 70k/15k/15k caps) they
    are identical. How they were made: assign every eligible paper to a split, proportionally within
    each (teacher, cohort) stratum.

    The splits are built with candidates, not final sizes: a handful of papers are still dropped
    when they are tokenised (a target over ``max_target_tokens``, a TEI that will not parse), so
    val and test get ``EVAL_MARGIN`` spare candidates and train gets every remaining paper. The
    consumers then take exactly the first ``n_train`` / ``n_val`` / ``n_test`` examples of the fixed
    order, which makes the asked-for sizes exact and the milestones well defined.
    """
    rows = read_index(s)
    needed = s.n_train + s.n_val + s.n_test
    if len(rows) < needed:
        log.warning("index has %d eligible papers, %d were asked for", len(rows), needed)
    want = {"val": min(s.n_val + EVAL_MARGIN, len(rows) // 6),
            "test": min(s.n_test + EVAL_MARGIN, len(rows) // 6)}
    want["train"] = len(rows) - want["val"] - want["test"]
    total = sum(want.values())
    rng = np.random.default_rng(s.seed)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        strata[(r["teacher"], r["cohort"])].append(r["id"])

    splits: dict[str, list[str]] = {k: [] for k in SPLITS}
    table = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        per = _allocate(len(ids), want, total)
        i = 0
        for sp in SPLITS:
            splits[sp].extend(ids[i:i + per[sp]])
            i += per[sp]
        table.append({"teacher": key[0], "cohort": key[1], "papers": len(ids), **per})

    for sp in SPLITS:  # fix every split's order once; milestones are positions in the train order
        arr = sorted(splits[sp])
        rng.shuffle(arr)
        splits[sp] = arr
    assert sum(len(v) for v in splits.values()) == len(rows), "every eligible paper belongs to a split"
    assert len({i for v in splits.values() for i in v}) == len(rows), "splits must be disjoint"

    by_id = {r["id"]: r for r in rows}
    curve = _stratified_subset(splits["val"][:s.n_val], by_id, s.curve_eval_n, rng)
    out = {"seed": s.seed, "candidates": {k: len(v) for k, v in splits.items()},
           "caps": {"train": s.n_train, "val": s.n_val, "test": s.n_test}, "strata": table,
           "curve_eval": curve, "loss_eval": curve[:s.loss_eval_n], "milestones": list(s.milestones),
           **{k: splits[k] for k in SPLITS}, **s.provenance()}
    s.state_dir.mkdir(parents=True, exist_ok=True)
    (s.state_dir / "splits.recomputed.json").write_text(json.dumps(out))
    summary = {k: v for k, v in out.items() if k not in SPLITS and k not in ("curve_eval", "loss_eval")}
    summary["curve_eval_n"] = len(curve)
    summary["loss_eval_n"] = len(out["loss_eval"])
    packaged = load_splits(s)
    summary["identical_to_packaged"] = {k: out[k] == packaged.get(k)
                                        for k in (*SPLITS, "curve_eval", "loss_eval")}
    if not all(summary["identical_to_packaged"].values()):
        log.warning("recomputed splits differ from the packaged ones: %s", summary["identical_to_packaged"])
    return summary


def _stratified_subset(ids: list[str], by_id: dict[str, dict], n: int, rng) -> list[str]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in ids:
        r = by_id[i]
        groups[(r["teacher"], r["cohort"])].append(i)
    sizes = {k: len(v) for k, v in groups.items()}
    per = _allocate(min(n, len(ids)), sizes, sum(sizes.values()))
    out: list[str] = []
    for k in sorted(groups):
        g = sorted(groups[k])
        rng.shuffle(g)
        out.extend(g[:per[k]])
    rng.shuffle(out)
    return out


def load_splits(s: Settings) -> dict:
    """The packaged splits: ``train``/``val``/``test`` ids in their fixed order, ``curve_eval``, ``loss_eval``.

    Consumers take the first ``n_train`` / ``n_val`` / ``n_test`` of each order, so 50,000 training papers
    are the first 50,000 of the 70,445 train candidates, and every run sees them in the same order.
    """
    return json.loads((s.corpus_dir / "splits.json").read_text())


# ---- step 3: build -----------------------------------------------------------------------------
_TOK = None


def tokenizer(s: Settings):
    """The student's tokenizer, loaded once per process."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(s.base_model, **s.hf_kwargs())
    return _TOK


def render_prompt(tok, doc: dict, body: str) -> str:
    """The exact string the model is trained and served on (chat template applied, no target)."""
    return tok.apply_chat_template(messages(doc, body), tokenize=False, add_generation_prompt=True)


def render_target(s: Settings, tok, summary: dict) -> str:
    """The assistant turn the student learns: whatever its template requires, then the JSON."""
    prefix, suffix = assistant_wrapper(tok)
    return prefix + canonical_target(summary) + suffix


def build_example(s: Settings, tok, row: dict) -> dict:
    """Prompt and target token ids for one paper, with the body cut to fit ``max_seq_len``.

    Returns ``{"drop": reason}`` instead when the paper cannot be used.
    """
    from scisum.corpus import body_store

    target = render_target(s, tok, row["summary"])
    target_ids = tok(target, add_special_tokens=False).input_ids
    if len(target_ids) > s.max_target_tokens:
        return {"drop": "target_too_long"}

    try:
        parsed = body_store(s).parsed(row["id"])
    except (KeyError, OSError, ValueError) as e:   # same drop family as the TEI errors of the original build
        return {"drop": f"tei_error:{type(e).__name__}"}

    budget = min(s.max_prompt_tokens, s.max_seq_len - len(target_ids))
    chars = min(s.max_body_chars, int(budget * 4.0))
    trunc_kind = "none"
    for attempt in range(6):
        body, section_trunc, n_sections, dropped_chars = fit_body(row["id"], parsed, chars, s.section_kinds)
        if not body.strip():
            return {"drop": "empty_body"}
        prompt_text = render_prompt(tok, row["doc"], body)
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        if len(prompt_ids) <= budget:
            if attempt:
                trunc_kind = "sections"
            break
        # Shrink from the body we actually got, not from the budget we asked for: when the body is
        # already smaller than the budget (the usual case for a paper that overshoots by a little),
        # lowering the budget changes nothing and the loop makes no progress at all.
        over = len(prompt_ids) - budget
        cpt = max(2.5, len(body) / max(1, len(prompt_ids)))
        chars = max(2_000, int(min(chars, len(body)) - over * cpt * 1.05))
    else:  # the section logic cannot get under the budget (one huge section): cut the text itself
        kept = body
        chars = min(chars, len(kept))
        for _ in range(8):
            head = int(chars * 0.6)
            body = kept[:head] + "\n[...]\n" + kept[-(chars - head):]
            prompt_text = render_prompt(tok, row["doc"], body)
            prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
            if len(prompt_ids) <= budget:
                break
            chars = int(chars * 0.8)
        trunc_kind = "hard"
    if len(prompt_ids) > budget:
        # Never truncate the token ids: the end of the prompt carries the instructions and the
        # assistant header, and a prompt cut there would teach the model from a broken example.
        return {"drop": "prompt_too_long"}
    return {
        "id": row["id"], "prompt_ids": prompt_ids, "target_ids": target_ids, "prompt_text": prompt_text,
        "target": target,
        "meta": {"id": row["id"], "teacher": row["teacher"], "cohort": row["cohort"], "rank": row["rank"],
                 "pages": row["pages"], "prompt_tokens": len(prompt_ids), "target_tokens": len(target_ids),
                 "body_chars": len(body), "body_sections": n_sections, "dropped_chars": dropped_chars,
                 "fits": trunc_kind == "none" and not section_trunc, "trunc_kind": trunc_kind,
                 "teacher_prompt_tokens": row["teacher_prompt_tokens"],
                 "teacher_body_chars": row["body_chars"], "teacher_truncated": row["teacher_truncated"]},
    }


def part_paths(s: Settings, split: str, part: int) -> dict[str, Path]:
    # The part inputs are shared between models (same papers, same order); only the tokenised
    # output and the rendered prompts are per model.
    return {"input": s.state_dir / "parts" / split / f"part-{part:04d}.jsonl",
            "stats": s.state_dir / "parts" / split / f"{s.model_tag}-part-{part:04d}.stats.json",
            "prompts": s.model_data_dir / "prompts" / split / f"part-{part:04d}.jsonl.gz"}


def _partition(s: Settings, split: str, ids: list[str], part_size: int) -> int:
    """Write one input file per part, in split order, so every worker reads only its own slice."""
    d = s.state_dir / "parts" / split
    d.mkdir(parents=True, exist_ok=True)
    n_parts = (len(ids) + part_size - 1) // part_size
    if all(part_paths(s, split, p)["input"].exists() for p in range(n_parts)):
        return n_parts  # already partitioned (reading the 600 MB index again would be pure cost)
    rows = {r["id"]: r for r in read_index(s)}
    n = 0
    for part, start in enumerate(range(0, len(ids), part_size)):
        p = part_paths(s, split, part)["input"]
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for rid in ids[start:start + part_size]:
                f.write(orjson.dumps(rows[rid]) + b"\n")
        tmp.replace(p)
        n = part + 1
    return n


def build_part(job: tuple) -> dict:
    """One part: tokenise its papers, write the token shard and/or the prompt file. Idempotent."""
    s, split, part, emit_tokens, emit_prompts, loss_ids = job
    paths = part_paths(s, split, part)
    tok = tokenizer(s)
    t0 = time.monotonic()
    writer = ShardWriter(s.tokens_dir(split), part) if emit_tokens else None
    loss_writer = ShardWriter(s.tokens_dir(f"{split}_loss"), part) if loss_ids else None
    gz = None
    if emit_prompts:
        paths["prompts"].parent.mkdir(parents=True, exist_ok=True)
        gz = gzip.open(paths["prompts"].with_suffix(".gz.tmp"), "wt")  # noqa: SIM115 - closed below
    drops: Counter[str] = Counter()
    kept = 0
    ptoks = ttoks = 0
    with paths["input"].open("rb") as f:
        for line in f:
            row = orjson.loads(line)
            ex = build_example(s, tok, row)
            if "drop" in ex:
                drops[ex["drop"]] += 1
                continue
            kept += 1
            ptoks += len(ex["prompt_ids"])
            ttoks += len(ex["target_ids"])
            if writer is not None:
                writer.add(ex["prompt_ids"], ex["target_ids"], ex["meta"])
            if loss_writer is not None and ex["id"] in loss_ids:
                loss_writer.add(ex["prompt_ids"], ex["target_ids"], ex["meta"])
            if gz is not None:
                gz.write(json.dumps({"id": ex["id"], "prompt": ex["prompt_text"], "target": ex["target"],
                                     "summary": row["summary"], "meta": ex["meta"]}, ensure_ascii=False) + "\n")
    if writer is not None:
        writer.close()
    if loss_writer is not None:
        loss_writer.close()
    if gz is not None:
        gz.close()
        paths["prompts"].with_suffix(".gz.tmp").replace(paths["prompts"])
    stats = {"split": split, "part": part, "kept": kept, "drops": dict(drops), "prompt_tokens": ptoks,
             "target_tokens": ttoks, "seconds": round(time.monotonic() - t0, 1), "host": os.uname().nodename}
    paths["stats"].write_text(json.dumps(stats))
    return stats


def _part_done(s: Settings, split: str, part: int) -> bool:
    return part_paths(s, split, part)["stats"].exists()


BUILD_MARGIN = 400   # candidates above the cap, so tokenisation drops cannot shrink a split


def run_build(s: Settings, split: str, nproc: int = 0, part_size: int = 2_000, force: bool = False) -> dict:
    """Tokenise a split. Only as many papers as the run will use are built.

    The split files hold every eligible paper so that any run can take a prefix of the same fixed
    order, but a run that trains on 20 000 papers has no reason to tokenise 70 445 of them — at a
    32k budget that is three quarters of an hour of CPU for nothing. The cap is the run's own
    `n_train`/`n_val`/`n_test` plus a margin.
    """
    sp = load_splits(s)
    cap = {"train": s.n_train, "val": s.n_val, "test": s.n_test}[split] + BUILD_MARGIN
    ids = list(sp[split])[:cap]
    emit_tokens = split == "train"
    emit_prompts = split in ("val", "test")
    loss_ids = frozenset(sp["loss_eval"]) if split == "val" else frozenset()
    n_parts = _partition(s, split, ids, part_size)
    todo = [p for p in range(n_parts) if force or not _part_done(s, split, p)]
    log.info("%s: %d papers, %d parts, %d to build", split, len(ids), n_parts, len(todo))
    jobs = [(s, split, p, emit_tokens, emit_prompts, loss_ids) for p in todo]
    nproc = nproc or min(len(todo), (os.cpu_count() or 8))
    t0 = time.monotonic()
    done = 0
    if jobs:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=max(1, min(nproc, len(jobs)))) as pool:
            for st in pool.imap_unordered(build_part, jobs, chunksize=1):
                done += 1
                log.info("part %03d: kept=%d drops=%s (%d/%d parts, %.0fs elapsed)", st["part"], st["kept"],
                         st["drops"] or "-", done, len(jobs), time.monotonic() - t0)
    return collect_stats(s, split, n_parts)


def collect_stats(s: Settings, split: str, n_parts: int | None = None) -> dict:
    d = s.state_dir / "parts" / split
    files = sorted(d.glob(f"{s.model_tag}-part-*.stats.json"))
    stats = [json.loads(p.read_text()) for p in files]
    drops: Counter[str] = Counter()
    for st in stats:
        drops.update(st["drops"])
    return {"split": split, "parts": len(stats), "parts_expected": n_parts, "papers": sum(x["kept"] for x in stats),
            "prompt_tokens": sum(x["prompt_tokens"] for x in stats),
            "target_tokens": sum(x["target_tokens"] for x in stats),
            "drops": dict(drops.most_common()), "build_seconds": round(sum(x["seconds"] for x in stats), 1)}


def read_prompts(s: Settings, split: str):
    """Prompt records of a split, in split order."""
    for p in sorted((s.model_data_dir / "prompts" / split).glob("part-*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def run_verify(s: Settings, sample: int = 300) -> dict:
    """Check the built dataset against its own invariants. Cheap enough to run after every build.

    The token shards are checked in full (the index arrays are 24 bytes per example): no prompt
    longer than the prompt budget, no example longer than ``max_seq_len``, every example ends in
    the EOS token. The stored prompt *text* of the evaluation splits is re-tokenised on a sample
    and compared with the recorded length — the check that would have caught the fitting loop
    failing to converge, which left 3.6% of examples with a prompt that had been cut mid-sentence.
    """
    from scisum.shards import ShardSet

    tok = tokenizer(s)
    out: dict = {"model": s.base_model, "max_prompt_tokens": s.max_prompt_tokens,
                 "max_seq_len": s.max_seq_len, "turn_end": tok.decode(
                     tok(assistant_wrapper(tok)[1], add_special_tokens=False).input_ids[-1:]),
                 "splits": {}}
    problems: list[str] = []
    for split in ("train", "val_loss"):
        d = s.tokens_dir(split)
        if not d.exists():
            continue
        ss = ShardSet(d)
        if not len(ss):
            continue
        over_prompt = int((ss.prompt_lengths > s.max_prompt_tokens).sum())
        over_total = int((ss.lengths > s.max_seq_len).sum())
        # The last token must be the one the model's own template ends an assistant turn with -
        # `<|im_end|>` for Qwen3, `<|ifm|im_end|>` for K2-Horizon, neither of which is necessarily
        # `tok.eos_token_id` (K2's is `<|ifm|endoftext|>`, which would never appear here).
        end_id = tok(assistant_wrapper(tok)[1], add_special_tokens=False).input_ids[-1]
        eos_bad = sum(1 for i in range(0, len(ss), max(1, len(ss) // sample))
                      if int(ss.get(i).tokens[-1]) != end_id)
        out["splits"][split] = {"examples": len(ss), "prompt_over_budget": over_prompt,
                                "total_over_max_seq": over_total, "sampled_without_eos": eos_bad,
                                "longest_prompt": int(ss.prompt_lengths.max()),
                                "longest_example": int(ss.lengths.max())}
        problems += [f"{split}: {over_prompt} prompts over budget"] if over_prompt else []
        problems += [f"{split}: {over_total} examples over max_seq_len"] if over_total else []
        problems += [f"{split}: {eos_bad} sampled examples do not end in EOS"] if eos_bad else []

    for split in ("val", "test"):
        # Streamed: a split is 15 000 prompts of ~50 kB, and holding one in memory to slice it is
        # how this check got itself OOM-killed. The split order is already a seeded shuffle, so the
        # first `sample` records are a random sample of it.
        checked = mismatch = over = papers = 0
        worst = 0
        for r in read_prompts(s, split):
            papers += 1
            if checked >= sample:
                continue
            checked += 1
            n = len(tok(r["prompt"], add_special_tokens=False).input_ids)
            worst = max(worst, n)
            mismatch += n != r["meta"]["prompt_tokens"]
            over += n > s.max_prompt_tokens
        if not papers:
            continue
        out["splits"][split] = {"papers": papers, "checked": checked, "token_count_mismatch": mismatch,
                                "prompt_over_budget": over, "longest_checked": worst}
        problems += [f"{split}: {mismatch}/{checked} prompts do not re-tokenise to the recorded length"] if mismatch else []
        problems += [f"{split}: {over}/{checked} prompts over the budget"] if over else []

    out["ok"] = not problems
    out["problems"] = problems
    for p in problems:
        log.error("verify: %s", p)
    log.info("verify: %s", "all invariants hold" if out["ok"] else f"{len(problems)} problems")
    return out
