#!/usr/bin/env bash
# Submit any job script with a site's resources for one job kind, outside the pipeline:
#   bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch
#   SUM_STUDENT=k2h-7b SUM_RECIPE=sft50k bash slurm/submit.sh --site lumi gen my_probe.sbatch
# Kinds: train gen grader cpu cpu-big driver (slurm/site/<site>.sh).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
[ "${1:-}" = "--site" ] || { echo "usage: slurm/submit.sh --site <lumi|cuda> <kind> <script> [sbatch args]"; exit 2; }
SUM_SITE=$2 kind=$3 script=$4; shift 4
REPO=$PWD
# shellcheck disable=SC1091
[ -f slurm/site/local.sh ] && source slurm/site/local.sh
# shellcheck disable=SC1090
source "slurm/site/$SUM_SITE.sh"
flags=$(site_sbatch_flags "$kind")
case "$flags" in *MISSING_*) echo "slurm/site/$SUM_SITE.sh: $flags -- set the missing value in slurm/site/local.sh"; exit 1 ;; esac
export SUM_SITE SUM_REPO=$REPO SUM_SBATCH_FLAGS="$flags"
mkdir -p runs/logs
# shellcheck disable=SC2086
exec sbatch --output=runs/logs/%x_%j.log $flags "$@" --export=ALL "$script"
