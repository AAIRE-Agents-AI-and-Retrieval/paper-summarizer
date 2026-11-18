#!/usr/bin/env bash
#SBATCH --job-name=sum-grader
#SBATCH --output=runs/logs/%x_%j.log
#
# One grader model (SUM_GRADER = qwen38-27b or gemma4-31b-it) and the grading work that needs it.
# Header only; resources come from the site ("grader" kind: 4 GCDs on LUMI, tensor parallel).
#
# serve mode (LUMI default): start vLLM with the model under its served name, wait until /v1/models answers,
# write state/graders/<model>.json, then work through every ready grading stage of every student
# (scripts/pipeline.py work), talking to the server on localhost. Resubmits itself while work is left.
# external mode (a cluster with a shared endpoint in .env): no server; a CPU job that runs the same work.
# SUM_GRADER_SERVE_ONLY=1: serve until the walltime for other clients (they wait with
# `scripts/pipeline.py wait-grader --model <name>` and resolve the endpoint from state/graders/<name>.json).
exec bash "${SUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/jobs/grader_body.sh"
