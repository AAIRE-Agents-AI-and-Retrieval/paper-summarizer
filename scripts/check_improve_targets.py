"""Do the training targets built from samples look exactly like the ones built from the teacher?

Run on the smoke test's state (state/improve-smoke): a selection is forced to choose a student sample
in every scored paper, the arms are built, and the shards are read back.

    SCISUM_IMP_DIR=improve-smoke SUM_MODEL_ENV=slurm/models/q35-rft.env python scripts/check_improve_targets.py

Checks (the shards are compared as tokens): (1) rendering the teacher's summary object reproduces the stored target string exactly, so
a sample and the teacher go through one renderer; (2) a sample's target starts and ends with the
chat template's own prefix and suffix and parses back to the summary it came from; (3) the shards
hold the right prompt and target tokens; (4) pair shards alternate chosen and rejected, share the
prompt, and differ in target.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from scisum.config import get_settings
from scisum.dataset import render_target, tokenizer
from scisum.improve import (
    build_arm,
    candidate_objs,
    idir,
    parse_candidate,
    read_jsonl,
    select_pool,
    write_jsonl_atomic,
)
from scisum.prompt import assistant_wrapper
from scisum.shards import ShardSet


def main() -> None:
    s = get_settings()
    tok = tokenizer(s)
    prefix, suffix = assistant_wrapper(tok)
    pool = select_pool(s)
    samples = {r["id"]: r["samples"] for r in read_jsonl(idir(s) / "samples.jsonl.gz")}
    sel = {r["id"]: r for r in read_jsonl(idir(s) / "selection.jsonl.gz")}
    print(f"{len(pool)} pool papers, {sum(1 for r in sel.values() if 'candidates' in r)} scored")

    bad = [p["id"] for p in pool if render_target(s, tok, p["teacher"]) != p["target"]]
    assert not bad, f"the teacher's summary object does not re-render to its stored target for {bad}"
    print("1. render_target(teacher object) == stored target for every paper")

    forced = []
    for p in pool:
        r = dict(sel[p["id"]])
        objs = candidate_objs(s, p, samples[p["id"]], prefix)
        if "candidates" in r and objs[1][0] is not None:
            r["chosen"] = 1
            r["candidates"] = [dict(c) for c in r["candidates"]]
            r["candidates"][1]["reward"] = 9.0           # make the forced choice also the highest, so a pair can use it
        forced.append(r)
    backup = idir(s) / "selection.jsonl.gz.keep"
    shutil.copy(idir(s) / "selection.jsonl.gz", backup)
    write_jsonl_atomic(idir(s) / "selection.jsonl.gz", forced)
    try:
        for arm in ("rft", "pref"):
            print("build", arm, build_arm(s, arm))
    finally:
        shutil.copy(backup, idir(s) / "selection.jsonl.gz")
        backup.unlink()

    shards = ShardSet(s.tokens_dir(f"{s.imp_dir}-rft"))
    metas = shards.meta()
    swapped = [i for i, m in enumerate(metas) if m["source"] != "teacher"]
    assert swapped, "no example was built from a sample"
    for i in swapped:
        ex = shards.get(i)
        pid = metas[i]["id"]
        p = next(q for q in pool if q["id"] == pid)
        want = candidate_objs(s, p, samples[pid], prefix)[1][0]
        rendered = render_target(s, tok, want)
        assert rendered.startswith(prefix) and rendered.endswith(suffix), (rendered[:40], rendered[-30:])
        # the invariant is on tokens, not on decoded text (decoding may normalise whitespace)
        assert ex.tokens[:ex.prompt_len].tolist() == tok(p["prompt"], add_special_tokens=False).input_ids, pid
        assert ex.tokens[ex.prompt_len:].tolist() == tok(rendered, add_special_tokens=False).input_ids, pid
        back = parse_candidate(rendered.removesuffix(suffix), prefix)
        assert back is not None and all(back[k] == want[k] for k in want if k in back), pid
    print(f"2-3. {len(swapped)} sample-built targets: template prefix and suffix present, parse back to the same summary, prompts intact")

    pairs = ShardSet(s.tokens_dir(f"{s.imp_dir}-pref"))
    pm = pairs.meta()
    assert len(pairs) % 2 == 0 and len(pairs) > 0
    for k in range(len(pairs) // 2):
        c, r = pairs.get(2 * k), pairs.get(2 * k + 1)
        assert pm[2 * k]["role"] == "chosen" and pm[2 * k + 1]["role"] == "rejected" and pm[2 * k]["id"] == pm[2 * k + 1]["id"]
        assert c.prompt_len == r.prompt_len and np.array_equal(c.tokens[:c.prompt_len], r.tokens[:r.prompt_len])
        assert not np.array_equal(c.tokens[c.prompt_len:], r.tokens[r.prompt_len:])
        assert pm[2 * k]["reward"] - pm[2 * k + 1]["reward"] >= s.imp_pair_gap
    print(f"4. {len(pairs) // 2} pairs: chosen then rejected, same prompt, different targets, reward gap >= {s.imp_pair_gap}")
    print(json.dumps({"pairs": len(pairs) // 2, "sources": [m["source"] for m in pm]}))
    print("OK")


if __name__ == "__main__":
    main()
