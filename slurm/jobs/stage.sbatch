#!/usr/bin/env bash
#SBATCH --job-name=sum-stage
#SBATCH --output=runs/logs/%x_%j.log
#
# One stage of the pipeline that is not a training chain or a grader: data preparation, model download,
# tokenisation, test-split generation and scoring, candidate sampling, pair building, benchmarks, the report.
# Header only; resources come from the site at submission. The body runs `scripts/pipeline.py exec`, which
# runs the stage's commands in order, skips the ones already done, and writes the done/failed markers.
exec bash "${SUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/jobs/stage_body.sh"
