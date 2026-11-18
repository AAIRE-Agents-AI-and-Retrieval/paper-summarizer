"""Command-line interface: ``scisum <group> <command>``.

Every stage is a separate command so that each can run in its own Slurm job, be re-run safely
(they are all idempotent) and be read in a log. ``scisum status`` is the one-screen overview.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import typer

from scisum.config import get_settings

app = typer.Typer(help="Distil and preference-tune small students as scientific-paper summarisers.",
                  no_args_is_help=True)
data_app = typer.Typer(help="Dataset: index, split, tokenise.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation: generate, score, watch for checkpoints.", no_args_is_help=True)
app.add_typer(data_app, name="data")
fact_app = typer.Typer(help="Fact-level evaluation against questions written from the paper, not the teacher.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
imp_app = typer.Typer(help="Reward-selected distillation: sample, score, select, build training sets.", no_args_is_help=True)
claim_app = typer.Typer(help="Claim-level faithfulness: every statement of a summary checked against the whole paper.",
                        no_args_is_help=True)
app.add_typer(claim_app, name="claimcheck")
app.add_typer(fact_app, name="factcheck")
app.add_typer(imp_app, name="improve")
graders_app = typer.Typer(help="LLM graders: which endpoint serves which model, and is it up.", no_args_is_help=True)
app.add_typer(graders_app, name="graders")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    import time

    logging.Formatter.converter = time.gmtime
    for noisy in ("httpx", "httpcore", "filelock", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


# ---- data ---------------------------------------------------------------------------------------
@data_app.command("index")
def data_index() -> None:
    """Check the packaged index (data/corpus/index.jsonl.gz) against the packaged bodies."""
    from scisum.dataset import run_index

    _echo(run_index(get_settings()))


@data_app.command("split")
def data_split() -> None:
    """Recompute the seeded, stratified split and check it equals the packaged data/corpus/splits.json."""
    from scisum.dataset import run_split

    _echo(run_split(get_settings()))


@data_app.command("build")
def data_build(split: str = typer.Option(..., help="train | val | test"),
               nproc: int = typer.Option(0, help="Worker processes (default: all CPUs)"),
               part_size: int = typer.Option(2000, help="Papers per part (one shard per part)"),
               force: bool = typer.Option(False, help="Rebuild parts that are already complete")) -> None:
    """Rebuild prompts from the TEI, tokenise and write shards. Resumable, parallel, idempotent."""
    from scisum.dataset import run_build

    _echo(run_build(get_settings(), split, nproc=nproc, part_size=part_size, force=force))


@data_app.command("verify")
def data_verify(sample: int = typer.Option(300, help="Prompts to re-tokenise per evaluation split")) -> None:
    """Check the built dataset against its invariants (budgets, EOS, prompt text vs token count)."""
    from scisum.dataset import run_verify

    r = run_verify(get_settings(), sample)
    _echo(r)
    if not r["ok"]:
        raise typer.Exit(1)


@data_app.command("stats")
def data_stats(out: Path = typer.Option(Path("state/build.json"))) -> None:
    """The dataset card as numbers: token distributions, truncation, teachers, drops."""
    from scisum.datastats import write_dataset_stats

    _echo(write_dataset_stats(get_settings(), out))


# ---- training -----------------------------------------------------------------------------------
@app.command()
def train(run: str = typer.Option("main", help="Run name (runs/<run>/)"),
          max_examples: int | None = typer.Option(None, help="Stop after this many examples (default n_train)"),
          max_seconds: float | None = typer.Option(None, help="Stop this many seconds into the chunk"),
          no_resume: bool = typer.Option(False, help="Start from scratch, ignoring any checkpoint"),
          smoke: int = typer.Option(0, help="Smoke test: train on this many examples only"),
          longest: bool = typer.Option(False, help="With --smoke: use the longest examples (memory check)"),
          shortest: bool = typer.Option(False, help="With --smoke: use the shortest examples (code-path check)")) -> None:
    """Train one chunk; exit 0 when the run is complete, 2 when it should be resubmitted."""
    import os

    # Set before torch touches the GPU. One sequence of 450-30,000 tokens per forward leaves the caching
    # allocator's blocks in every size; on a 44 GiB L40S a run at a 37 GB peak failed a 0.7 GiB request
    # after an hour of fragmentation warnings (2026-11-04). Expandable segments let blocks grow in place.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    s = get_settings()
    if s.objective == "preference":
        from scisum.pref_train import run_pref_train

        code = run_pref_train(s, run, max_examples=max_examples, max_seconds=max_seconds,
                              resume=not no_resume, smoke=smoke, shortest=shortest, longest=longest)
    else:
        from scisum.train import run_train

        code = run_train(s, run, max_examples=max_examples, max_seconds=max_seconds,
                         resume=not no_resume, smoke=smoke, longest=longest, shortest=shortest)
    raise typer.Exit(code)


# ---- evaluation ---------------------------------------------------------------------------------
@eval_app.command("generate")
def eval_generate(split: str = typer.Option("val"), run: str = typer.Option("base"),
                  ckpt: str = typer.Option("base", help="Checkpoint name under runs/<run>/ckpt, or 'base'"),
                  limit: int | None = typer.Option(None, help="Papers (default: the curve subset)"),
                  subset: str = typer.Option("curve", help="curve | all"),
                  max_model_len: int | None = typer.Option(None, help="Default: the model's max_seq_len"),
                  max_seconds: float | None = typer.Option(None, help="Stop between batches after this long")) -> None:
    """Generate summaries with vLLM (base weights + the checkpoint's LoRA adapter).

    Exits 3 when it stopped on ``--max-seconds`` with papers left, so a job script can tell an
    unfinished checkpoint from a finished one without parsing the result.
    """
    from scisum.evaluate import run_generate

    meta = run_generate(get_settings(), split=split, run=run, ckpt=ckpt, limit=limit, subset=subset,
                        max_model_len=max_model_len, max_seconds=max_seconds)
    _echo(meta)
    if meta.get("complete") is False:
        raise typer.Exit(3)


@eval_app.command("score")
def eval_score(split: str = typer.Option("val"), run: str = typer.Option("base"),
               ckpt: str = typer.Option("base"), limit: int | None = typer.Option(None),
               subset: str = typer.Option("curve"),
               faithfulness: bool = typer.Option(True, help="NLI groundedness against the paper text"),
               bertscore: bool = typer.Option(True)) -> None:
    """Score generated summaries: schema, ROUGE, BERTScore, keyword F1, repetition, faithfulness."""
    from scisum.evaluate import run_score

    _echo(run_score(get_settings(), split=split, run=run, ckpt=ckpt, limit=limit, subset=subset,
                    faithfulness=faithfulness, bertscore=bertscore))


@eval_app.command("judge")
def eval_judge(run: str = typer.Option("main"), ckpt: str = typer.Option("base"),
               split: str = typer.Option("val"), subset: str = typer.Option("curve"),
               limit: int | None = typer.Option(None), n: int | None = typer.Option(None),
               concurrency: int = typer.Option(8), force: bool = typer.Option(False),
               max_seconds: float | None = typer.Option(None, help="Stop and leave the rest for the next run")) -> None:
    """Score a checkpoint's summaries against the teacher's with an independent judge model."""
    import asyncio

    from scisum.judge import run_judge

    _echo(asyncio.run(run_judge(get_settings(), run=run, ckpt=ckpt, split=split, subset=subset,
                                limit=limit, n=n, force=force, concurrency=concurrency,
                                max_seconds=max_seconds)))


@eval_app.command("rescore-text")
def eval_rescore(run: str = typer.Option("main"), split: str = typer.Option("val")) -> None:
    """Recompute the text-only metrics of stored evaluations (no GPU, nothing regenerated)."""
    from scisum.evaluate import rescore_text

    _echo(rescore_text(get_settings(), run=run, split=split))


@eval_app.command("worker")
def eval_worker(run: str = typer.Option("main"), poll_s: float = typer.Option(60.0),
                max_seconds: float | None = typer.Option(None),
                split: str = typer.Option("val"), once: bool = typer.Option(False),
                limit: int | None = typer.Option(None, help="Papers per checkpoint (smoke tests)")) -> None:
    """Watch runs/<run>/ckpt/*/READY and evaluate each milestone as it appears."""
    from scisum.evaluate import run_worker

    _echo(run_worker(get_settings(), run=run, poll_s=poll_s, max_seconds=max_seconds, split=split,
                     once=once, limit=limit))


# ---- reporting ----------------------------------------------------------------------------------
@app.command()
def curves(run: str = typer.Option("main"), out_dir: Path | None = typer.Option(None)) -> None:
    """Draw the learning curves (loss, metrics per milestone, throughput) as png and svg."""
    from scisum.curves import draw_all

    for p in draw_all(get_settings(), run, out_dir):
        typer.echo(f"written: {p}")


@app.command()
def report(run: str = typer.Option("main")) -> None:
    """Collect every number into reports/REPORT.md and reports/results.json."""
    from scisum.report import write_report

    for p in write_report(get_settings(), run):
        typer.echo(f"written: {p}")


@app.command()
def status(run: str = typer.Option("main")) -> None:
    """One screen: dataset, training progress, evaluations, Slurm jobs."""
    from scisum.status import print_status

    print_status(get_settings(), run)


@app.command()
def env() -> None:
    """The effective settings (secrets masked)."""
    s = get_settings()
    d = json.loads(s.model_dump_json())
    d["judge_api_key"] = "set" if s.judge_api_key else "unset"        # never any part of the value
    d["max_prompt_tokens"] = s.max_prompt_tokens
    _echo(d)

@graders_app.command("check")
def graders_check(models: str = typer.Option("gemma4-31b-it,qwen38-27b", help="comma-separated served model names")) -> None:
    """For each grader: where its endpoint comes from, whether it answers, and whether it serves that name."""
    from scisum.endpoints import check

    rows = check(get_settings(), [m.strip() for m in models.split(",") if m.strip()])
    _echo(rows)
    raise typer.Exit(0 if all(r["serves_model"] for r in rows) else 1)


@fact_app.command("run")
def factcheck_run(systems: str = typer.Option("teacher", help="comma-separated labels: teacher, anchors (C, C_prefcc, "
                                                             "C_prefccs2, C_base) or run names, optionally run@ckpt"),
                  concurrency: int = typer.Option(8), max_seconds: float = typer.Option(3000),
                  n: int | None = typer.Option(None, help="papers (default: settings.fact_n_papers)")) -> None:
    """Questions (once), then answers and grades for each system. Resumable; stops before the walltime."""
    from scisum.factcheck import run_all

    s = get_settings()
    if n:
        s = s.model_copy(update={"fact_n_papers": n})
    _echo(asyncio.run(run_all(s, [x.strip() for x in systems.split(",") if x.strip()],
                              concurrency=concurrency, max_seconds=max_seconds)))


@fact_app.command("report")
def factcheck_report() -> None:
    """Recall, accuracy and the paired difference to the teacher, per system."""
    from scisum.factcheck import report

    _echo(report(get_settings()))


@fact_app.command("export")
def factcheck_export(n: int = typer.Option(40)) -> None:
    """Questions with gold answers and paper text, for a person to verify."""
    from scisum.factcheck import export_for_checking

    _echo({"written": str(export_for_checking(get_settings(), n=n))})


@fact_app.command("crossgrade")
def factcheck_crossgrade(label: str = typer.Option("C"), model: str = typer.Option("gemma4-31b-it")) -> None:
    """Re-grade one system's answers with a second model; report agreement and Cohen's kappa."""
    from scisum.factcheck import cross_grade

    _echo(asyncio.run(cross_grade(get_settings(), label, model)))


@claim_app.command("run")
def claimcheck_run(systems: str = typer.Option("teacher_perturbed,other_paper,teacher",
                                               help="comma-separated labels: controls, teacher, anchors or run names"),
                   n: int = typer.Option(200, help="papers, a seeded sample of the fact-test papers"),
                   n_controls: int = typer.Option(40), model: str | None = typer.Option(None, help="default: judge_model"),
                   concurrency: int = typer.Option(16), max_seconds: float = typer.Option(3000)) -> None:
    """Check every statement of each system's summaries against the paper. Resumable."""
    from scisum.claimcheck import run

    _echo(asyncio.run(run(get_settings(), [x.strip() for x in systems.split(",") if x.strip()], n=n,
                          n_controls=n_controls, model=model, concurrency=concurrency, max_seconds=max_seconds)))


@claim_app.command("report")
def claimcheck_report(model: str | None = typer.Option(None)) -> None:
    """Shares of supported, unsupported and contradicted statements per system, with paired intervals."""
    from scisum.claimcheck import report

    _echo(report(get_settings(), model=model))


@claim_app.command("export")
def claimcheck_export(systems: str = typer.Option("teacher,C,C_prefcc"), per_system: int = typer.Option(15),
                      model: str | None = typer.Option(None)) -> None:
    """A blinded sample of flagged and supported statements, with passages, for a person to verify."""
    from scisum.claimcheck import export_for_people

    out, key = export_for_people(get_settings(), [x.strip() for x in systems.split(",") if x.strip()],
                                 model=model, per_system=per_system)
    _echo({"written": str(out), "key": str(key)})


@imp_app.command("pool")
def improve_pool() -> None:
    """Choose the papers (validation papers outside the monitoring subset) and store their prompts."""
    from scisum.improve import select_pool

    _echo({"papers": len(select_pool(get_settings()))})


@imp_app.command("sample")
def improve_sample(max_seconds: float | None = typer.Option(None)) -> None:
    """K samples per pool paper from SCISUM_INIT_ADAPTER (GPU). Resumable; exits 3 if unfinished."""
    from scisum.improve import run_sample

    out = run_sample(get_settings(), max_seconds=max_seconds)
    _echo(out)
    if out["remaining"]:
        raise typer.Exit(3)


@imp_app.command("reward")
def improve_reward(max_seconds: float = typer.Option(3000), concurrency: int = typer.Option(24)) -> None:
    """Questions, reference readers and candidate readers on the judge endpoint (no GPU)."""
    from scisum.improve import run_reward

    out = asyncio.run(run_reward(get_settings(), max_seconds=max_seconds, concurrency=concurrency))
    _echo(out)
    # 0 = nothing left; 4 = more to do and this run moved it; 5 = more to do but this run could not
    # move it (it is waiting for samples), so the job script should wait before coming back
    if out["remaining"]:
        raise typer.Exit(4 if (out["questions"] or out["references"] or out["candidates"]) else 5)


@imp_app.command("claims")
def improve_claims(max_seconds: float = typer.Option(3000), concurrency: int = typer.Option(24),
                   model: str | None = typer.Option(None, help="default: imp_claim_model")) -> None:
    """Round 3: claim-check every scored candidate of the selection against its paper (no GPU). Resumable."""
    from scisum.improve import run_claims

    out = asyncio.run(run_claims(get_settings(), max_seconds=max_seconds, concurrency=concurrency, model=model))
    _echo(out)
    if out["remaining"]:
        raise typer.Exit(4)


@imp_app.command("select")
def improve_select() -> None:
    """Score every candidate, choose each paper's target, and write the statistics of the round."""
    from scisum.improve import run_select

    _echo(run_select(get_settings()))


@imp_app.command("build")
def improve_build(arm: str = typer.Option(..., help="ctl (teacher targets), rft (selected targets) or pref (pairs)")) -> None:
    """Tokenise one arm's training set into data/<model>/tokens/<imp_dir>-<arm>."""
    from scisum.improve import build_arm

    _echo(build_arm(get_settings(), arm))


@imp_app.command("synth")
def improve_synth() -> None:
    """Smoke test: write known-defective candidates in place of samples (no GPU)."""
    from scisum.improve import write_synthetic_samples

    _echo({"papers": write_synthetic_samples(get_settings())})


@imp_app.command("synth-report")
def improve_synth_report() -> None:
    """Smoke test: the reward per kind of defect. The teacher and its copy should lead."""
    from scisum.improve import synth_report

    _echo(synth_report(get_settings()))


@imp_app.command("status")
def improve_status() -> None:
    """Where the round stands: sampled, questions, references, graded, finished."""
    from scisum.improve import progress

    _echo(progress(get_settings()))


@imp_app.command("eval-ids")
def improve_eval_ids(first: int = typer.Option(500, help="also the first N test papers")) -> None:
    """Write state/improve/eval_ids.json: the fact-test papers plus the first N test papers."""
    from scisum.improve import write_eval_ids

    _echo(write_eval_ids(get_settings(), first))


@imp_app.command("validate")
def improve_validate(max_papers: int = typer.Option(250), max_seconds: float = typer.Option(3000),
                     concurrency: int = typer.Option(24)) -> None:
    """Re-score teacher / chosen sample / typical sample with the evaluation's instrument (endpoint only)."""
    from scisum.improve import run_validate

    _echo(asyncio.run(run_validate(get_settings(), max_papers, max_seconds, concurrency)))


@imp_app.command("validate-report")
def improve_validate_report() -> None:
    """Is the chosen sample better than a typical one on questions the selection never saw?"""
    from scisum.improve import validation_report

    _echo(validation_report(get_settings()))


@imp_app.command("validate-pairs")
def improve_validate_pairs(max_papers: int = typer.Option(200), max_seconds: float = typer.Option(3000),
                           concurrency: int = typer.Option(24)) -> None:
    """Are the preference pairs ordered correctly on independent questions? (endpoint only)"""
    from scisum.improve import pair_validation_report, run_validate_pairs

    _echo(asyncio.run(run_validate_pairs(get_settings(), max_papers, max_seconds, concurrency)))
    _echo(pair_validation_report(get_settings()))


if __name__ == "__main__":
    app()
