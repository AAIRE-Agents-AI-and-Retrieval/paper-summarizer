# Instructions for coding agents

Read `README.md`, then `docs/START_HERE.md`.

## Ground rules

1. **Nothing long-running or computational on a login node.** Everything runs under Slurm through
   `scripts/pipeline.py` and `slurm/jobs/`. Reading logs, `squeue`, `pipeline.py status` and
   `pipeline.py next --dry-run` are the interactive things you may do.
2. **Never edit a running job script in place, and never resubmit just to pick up a fix.** Bash reads a
   script as it executes. So every script keeps its body in `main()`, called on the last line, and is replaced
   with `mv` (a new file), never patched in place. The `.sbatch` files are headers that `exec` a body from
   shared storage, so a queued job runs the current code.
3. **Every long step stays resumable and idempotent.**
   - Training resumes from the number of papers seen.
   - Generation appends and skips finished papers.
   - Scoring is keyed by `<split>-<ckpt>-n<papers>`.
   - Write to a temporary name, then rename.

   Never make a step depend on being run exactly once.
4. **The prompt and schema are frozen.** `scisum/vendor/` holds the original prompt, schema and section
   selection. `tests/test_prompt_parity.py` checks the output byte for byte against prompts recorded with the
   original code. Changing the prompt invalidates the 100,826 teacher summaries, so it would make a new
   dataset, not an edit.
5. **Numbers come from artifacts, never from a message:**
   - `runs/*/metrics.jsonl` and `runs/*/eval/*.json`;
   - the fact-test and claim-check files under `state/`;
   - the benchmark JSONs.

   `scripts/analysis/` regenerates every table and plot from them.
6. **Quality filters need evidence.** Measure a distribution before you add a filter, and record the
   measurement in the docstring.
7. **Secrets** (a hosted grader's key or URL) live in `.env` (chmod 600), never in git or in a log line.
   `JUDGE_API_KEY` and `SCISUM_JUDGE_BASE` are excluded from settings dumps.
8. **Data policy.** `data/` holds paper bodies, teacher summaries and evaluation questions for internal
   pipeline use only. Never commit them or publish them (`docs/DATA.md`).
9. **Tests must pass before a commit:** `ruff check scisum tests scripts && python -m pytest`. Tests never need
   a GPU or the network. Use this repository's own virtual environment or container.

## Where things are

| Path | Responsibility |
|---|---|
| `scisum/config.py` | `Settings` (env prefix `SCISUM_`) and every path, including `SCISUM_DATA_ROOT` |
| `scisum/corpus.py`, `scisum/vendor/` | The packaged corpus (index, sqlite body store) and the frozen prompt code |
| `scisum/prompt.py`, `scisum/dataset.py` | The prompt, the canonical target, and the steps from the corpus to token shards |
| `scisum/model.py`, `scisum/train.py`, `scisum/pref_train.py` | Model loading with LoRA; the SFT loop (WSD schedule, milestones, checkpoints, metrics); DPO + NLL |
| `scisum/evaluate.py`, `scisum/metrics.py`, `scisum/faithfulness.py`, `scisum/bertscore.py` | vLLM generation and every reference metric |
| `scisum/factcheck.py`, `scisum/claimcheck.py`, `scisum/judge.py`, `scisum/endpoints.py` | The fact test, the claim check, the judge, and grader endpoint resolution |
| `scisum/improve.py` | The prefcc stage: pool, samples, questions, reward, pairs |
| `configs/students/`, `configs/recipes/` | One profile per student; the `sft50k` and `prefcc` recipes |
| `slurm/site/`, `slurm/jobs/` | The site layer (LUMI, CUDA) and site-neutral jobs |
| `scripts/pipeline.py` | The stage graph: `status`, `next`, `run`, `driver`, `logs`, `cancel` |
| `scripts/analysis/` | Benchmark, training resources, results collection, learning curves, Pareto plots |
| `paper_docs/` | The original project's research record (read-only history) |
