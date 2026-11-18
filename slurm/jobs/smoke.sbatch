#!/usr/bin/env bash
#SBATCH --job-name=sum-smoke
#SBATCH --output=runs/logs/%x_%j.log
#
# The runtime smoke test (containers/smoke_test.py) on one GPU of the site:
#   bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch
#   SUM_SMOKE_ARGS="--model Qwen/Qwen3.5-4B" bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch
# Prints one JSON report and exits non-zero if a required check fails.
set -uo pipefail

main() {
  source slurm/common.sh || exit 1
  # shellcheck disable=SC2086
  $PY containers/smoke_test.py ${SUM_SMOKE_ARGS:-}
}

main "$@"; exit $?
