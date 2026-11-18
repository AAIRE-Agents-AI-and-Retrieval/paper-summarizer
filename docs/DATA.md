# Data: what is packaged, where it came from, and how to verify it

> **Internal use only.** The paper bodies come from the publishers' PDFs of arXiv papers under many different
> licences, most of them not permissive. The bodies, and every per-paper summary, question and answer derived from
> them, are shipped for running this project's pipeline only. Do not re-publish them, do not put them in a public
> repository, a model card or a dataset release, and do not share them beyond the people working on this project.
> This follows the original project's data policy: bodies are never re-published, and summaries written from the body of a
> non-permissively licensed paper stay out of dataset releases. The code in this repository is Apache-2.0; the data
> is not covered by that licence.

The data sits in `data/` beside the code (git-ignored; shipped in the zip). Point `SCISUM_DATA_ROOT` elsewhere if
the repository lives on a small file system (on LUMI, keep it on `/scratch/<project>`). Token shards and prompt files
that the pipeline builds go under the same root (`data/<model>/`).

## Layout

Total 4.0 GB in 270 files (`MANIFEST.json`); the zip compresses it little, because the bodies and adapters are
already compressed.

| Directory | Size | Files |
|:--|--:|--:|
| `corpus/` | 2.04 GB (bodies 1.85 GB, index 189 MB) | 6 |
| `eval/` | 61 MB | 133 |
| `improve_pool/` | 89 MB | 4 |
| `anchors/` | 1.82 GB (three adapters of 591 MB each) | 127 |

| Directory | What it is | Used by |
|:--|:--|:--|
| `corpus/` | The 100,644 eligible papers: metadata, teacher summary, split, parsed body | `scisum data build` (SFT data and the evaluation prompts) |
| `eval/` | The test-side instruments: the 1,246 evaluation papers, two fact-test question sets, claim checks | fact test, claim check, judge, NLI |
| `improve_pool/` | The 3,000-paper preference pool and its training-side questions and reference-reader verdicts | the preference (prefcc) stage |
| `anchors/` | Student C and `prefcc` (adapters, training logs, test summaries, scores), C's preference candidates, and the logs and scores of runs A and B | re-grading and benchmarking the reference models; learning-curve baselines |
| `MANIFEST.json` | Every file with its size, SHA-256, row count and a one-line description | verification |

### `corpus/`

| File | Rows | Content |
|:--|--:|:--|
| `index.jsonl.gz` | 100,644 | One JSON row per eligible paper, in citation-rank order. Fields: `id` (arXiv id), `cohort` (`top`: the 100,000 most-cited arXiv papers by OpenAlex count; `random`: a 500-paper calibration cohort), `rank`, `teacher` (model that wrote the summary), `pages`, `body_chars`, `body_sections`, `teacher_input_chars`, `teacher_truncated`, `teacher_prompt_tokens`, `teacher_completion_tokens`, `doc` (arXiv `id`, `title`, `abstract`, `authors`, `categories`, `published`), `summary` (the teacher's eight-field summary: the training target). The rows are byte-identical to the original run's `state/index.jsonl`. |
| `index.json` | | Counts: 100,826 summary records, 100,644 eligible, 182 dropped (85 contribution count outside 2–10, 72 duplicate titles, 13 bodies under 2,000 characters, 10 abstract-only, 1 repetitive, 1 keyword count); teachers; cohorts. |
| `index_drops.jsonl` | 182 | Every dropped paper and its reason. |
| `splits.json` | | `train` (70,445 ids), `val` (15,101), `test` (15,098), each in its **fixed order**; `curve_eval` (1,000 validation papers for learning curves), `loss_eval` (its first 200, for held-out loss); strata; seed 20261014. |
| `splits_summary.json` | | Split sizes and strata. |
| `bodies.sqlite` | 100,644 | Table `bodies(id TEXT PRIMARY KEY, parsed BLOB)`. `parsed` is zlib-compressed UTF-8 JSON `[[title, text, kind], ...]`: the GROBID body exactly as the teacher run's parser produced it, every section with its kind (`body`, `appendix`, `captions` (one section with all figure and table captions), `references`, `backmatter`). The student prompt uses `body` and `captions`. Table `meta` records the format and parser. |

**Splits and the 50,000-paper training set.** The splits are stratified by (teacher model, cohort) and fixed
once: train, val and test hold candidates (more than the 70,000 / 15,000 / 15,000 caps, so papers dropped while
tokenising cannot shrink a split). A run takes the first `n_train` papers of the train order that tokenise, so
**50,000 papers = the first 50,000 usable papers of the 70,445 train candidates**, in the same order for every
student; a 20,000-paper run (Student C) used a prefix of the same sequence. `scisum data build` tokenises
`n_train + 400` candidates. `scisum data split` recomputes the splits from the index and reports whether they are
identical to the packaged ones (they are, with seed 20261014 and the original caps).

**Provenance.**
- Papers: the 100,000 most-cited arXiv papers by OpenAlex citation count (123–83,445 citations) plus a 500-paper
  random calibration cohort.
- Full text: GROBID TEI from the publishers' PDFs; body and figure/table captions only in the prompt.
- Targets: 100,826 structured summaries written from those full texts through OpenRouter (cost $132.25) by
  `deepseek/deepseek-v4-flash-0731` (88,404 eligible), `deepseek/deepseek-v4.1-flash` (8,142) and
  `meta/muse-spark-1.3-contributor` (4,098). They are machine-generated targets: this is distillation, not
  ground truth.
- The index rules, in order: teacher prompt version `ft-grobid-v1`; not abstract-only; GROBID body of at least
  2,000 characters; summary passes `scisum.prompt.validate_summary`; title and abstract present; TEI present; one
  row per normalised title (the more-cited paper kept).

**The prompt is the teacher's.** The student is trained on the teacher pipeline's production prompt, with the body fitted
into the token budget by dropping the least useful sections first. The prompt code is vendored in
`scisum/vendor/`; `tests/test_prompt_parity.py` checks it against hashes recorded with the original code for 208
papers, and a one-off check rebuilt 300 of Student C's stored test prompts byte for byte (see "Verification").
Changing the prompt invalidates the teacher targets.

### `eval/`

Copied unchanged from the original run's `state/`. To use them, the pipeline links or copies the directories to
`state/` under the same names (`state/factcheck`, `state/factcheck-b`, `state/claimcheck`, `state/improve/eval_ids.json`).

| File | Content |
|:--|:--|
| `eval_ids.json` | The 1,246 test papers every improvement arm was evaluated on: the 900 fact-test papers plus the first 500 of the test order, in test order (`scisum.improve.write_eval_ids`). Reference metrics on all of them; NLI on the first 300 (`faithfulness_n`); the judge on 500. |
| `factcheck/papers.json`, `factcheck-b/papers.json` | Fact-test papers, set A (300) and set B (600, disjoint), drawn from the first 3,000 test papers, stratified by archive: id, metadata, teacher summary and the paper text the question writer read (Student C's prompt body, cut at 90,000 characters). 899 papers have questions (one paper of set A yielded none). |
| `factcheck*/questions.jsonl.gz` | Ten questions per paper with gold answers, written by `gemma4-31b-it` from the paper text (1 problem, 1 motivation, 2 approach, 2 setup, 2 finding, 1 limitation, 1 scope). |
| `factcheck*/answers-<system>.jsonl.gz`, `grades-<system>.jsonl.gz` | The answers of `qwen38-27b` from one system's summary and their grades, for the teacher, Student C (`C`), `prefcc` (`C_prefcc`, `C_prefccs2`), every other arm of the original project, and the controls `closed_book`, `abstract`, `full_paper`. |
| `factcheck*/report.json` | Recall per system. |
| `claimcheck/<system>.<verifier>.jsonl.gz` | Claim check: every statement of a summary with the verifier's verdict (`supported`, `not_in_paper`, `contradicted`), for the same systems; verifiers `qwen38-27b` (evaluation side) and `gemma4-31b-it`. Controls: `teacher_perturbed` (numbers changed), `other_paper`. |
| `claimcheck/report.<verifier>.json` | Unsupported shares per system. |

### `improve_pool/`

The preference stage reuses these for every student: the questions and the reference readers do not depend on the
student.

| File | Rows | Content |
|:--|--:|:--|
| `pool.jsonl.gz` | 3,000 | The first 3,000 validation papers outside `curve_eval`: `paper` (the text the questions were written from: Student C's prompt body, ≤ 90,000 characters), `teacher` summary, `target`, cohort, teacher model, and Student C's own `prompt`/`prompt_tokens`. **For a new student, keep `id`, `paper`, `teacher` and rebuild `prompt`/`prompt_tokens`/`target` with that student's template**; the questions and verdicts below are keyed to `paper`. |
| `questions.jsonl.gz` | 2,551 | Fourteen questions per paper with gold answers, written by `qwen38-27b` (the training side; the evaluation's writer is `gemma4-31b-it`). |
| `questions-failed.jsonl.gz` | | Papers whose question writing failed. |
| `refs.jsonl.gz` | 2,551 | Per question, the verdicts of two reference readers (`gemma4-31b-it`): one with the whole paper, one closed-book. A question counts only if the whole-paper reader answers it correctly and the closed-book reader does not; a paper with fewer than 5 such questions contributes no pair. |

The original `prefcc` round finished 1,520 of these papers (sampled and scored) before its time limit; its pairs
came from that prefix.

### `anchors/`

| Path | Content |
|:--|:--|
| `q35/` | **Student C**: Qwen3.5-9B (revision `c202236`), LoRA r64/α128, SFT on 20,000 papers. `run.json` (settings, scrubbed), `metrics.jsonl` (the full training log), `state.json`, `ckpt/ex20000/adapter/` (PEFT adapter, 591 MB), `eval/` (test summaries and scores on 15,000 papers, the validation learning curve at 500–20,000 papers, untrained-base scores). |
| `q35prefcc/`, `q35prefccs2/` | **prefcc**, seeds 1 and 2 (pair order only): DPO β 0.1 + 1.0·NLL from C's adapter on 1,233 pairs. Adapter `ckpt/ex1233/adapter/`, training log, test summaries and scores on the 1,246 evaluation papers. |
| `k2/`, `main/` | Runs B (K2-Horizon-7B, pinned revision `4622f94`, 70,000 papers, 16k context) and A (Qwen3-4B-Instruct-2507, 70,000 papers): training logs and evaluation aggregates and per-paper scores (no adapters, no summaries). Baselines for learning curves and training cost. |
| `improve_c/` | Student C's candidate summaries on the pool (`samples.jsonl.gz`), their grades (`cand-grades.jsonl.gz`), the claim checks of the reward (`claims.gemma4-31b-it.jsonl.gz`), the selection (`selection.jsonl.gz`) and the statistics of `prefcc`'s pairs (`build.json`): what `prefcc` was trained from. |

What the anchors are for:
1. **Re-grading.** The original graders ran on a private endpoint whose served revision is not known. Graders you
   host yourself will score somewhat differently, so re-grade the teacher, C and `prefcc` summaries in `anchors/`
   with your graders before comparing a new student to them.
2. **Benchmarking.** C's and `prefcc`'s adapters let you measure the reference model's latency and memory on your
   own hardware.
3. **Learning curves.** `metrics.jsonl` of C, A and B give the expected shape of a run and its cost per paper.

Small JSON metadata files (`run.json`, aggregate `*.json`, `index.json`, `splits*.json`) were scrubbed on export:
the private grader endpoint field was removed, absolute paths replaced by roles (`<repo>`), and `host` values
blanked. Everything else is byte-identical to the original files.

## Verification

```bash
python - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("data"); m = json.loads((root / "MANIFEST.json").read_text())
bad = [f["path"] for f in m["files"]
       if hashlib.sha256((root / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
print(f"{len(m['files'])} files, {len(bad)} bad", bad[:5])
EOF
scisum data index          # the index and the bodies agree (every eligible paper has a body)
scisum data split          # the splits recompute identically from the index
python -m pytest tests/test_prompt_parity.py   # vendored prompt code == original code, on 208 papers
```

Results when the package was made (2026-11-17): all 270 files match their SHA-256; `scisum data index` found
100,644 papers, each with a body; `scisum data split` recomputed train, val, test, `curve_eval` and `loss_eval`
identically; the parity test passed on 208 papers; and the new code's `build_example` (vendored prompt code,
packaged bodies, Qwen3.5-9B tokenizer at revision `c202236`, 32,768 tokens with a 4,096-token reserve) rebuilt 300
of Student C's stored test prompts, targets and metadata byte for byte (287 whole bodies, 13 with sections
dropped).

## How the package was made

`data/` and `MANIFEST.json` were exported once from the original project's environment, where the TEI parser and
the TEI files exist. The golden prompt hashes in `tests/fixtures/prompt_golden.jsonl` were recorded with the
original prompt code at the same time. You never need to repeat the export. The teacher pipeline's ledgers, the TEI
files and the PDFs (170 GB) are not shipped: the parsed bodies replace them.
