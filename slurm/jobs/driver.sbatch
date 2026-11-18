#!/usr/bin/env bash
#SBATCH --job-name=sum-pipeline-driver
#SBATCH --output=runs/logs/%x_%j.log
#
# Hands-off operation: a small CPU job that runs `scripts/pipeline.py next` (submit every runnable stage)
# and resubmits itself SUM_DRIVER_EVERY_MIN minutes later, until every stage is done or
# state/pipeline/driver.stop exists. Submit it with `scripts/pipeline.py --site <site> driver`.
set -uo pipefail

main() {
  source slurm/common.sh || exit 1
  source slurm/lib.sh
  [ -e state/pipeline/driver.stop ] && { echo "state/pipeline/driver.stop exists; exiting"; exit 0; }
  local hp
  hp=$(site_host_python)
  # shellcheck disable=SC2086
  "$hp" scripts/pipeline.py --site "$SUM_SITE" next --with-report ${SUM_DRIVER_ARGS:-}
  "$hp" scripts/pipeline.py --site "$SUM_SITE" status
  if "$hp" scripts/pipeline.py --site "$SUM_SITE" status --json | "$hp" -c \
      'import json, sys; d = json.load(sys.stdin); sys.exit(0 if all(v["state"] == "done" for g in d.values() for v in g.values()) else 1)'; then
    echo "== every stage is done; the driver stops"; exit 0
  fi
  submit_like_me "${SLURM_JOB_NAME:-sum-pipeline-driver}" --begin=now+"${SUM_DRIVER_EVERY_MIN:-30}"minutes "$REPO/slurm/jobs/driver.sbatch" >/dev/null \
    && echo "== next run in ${SUM_DRIVER_EVERY_MIN:-30} min"
}

main "$@"; exit $?
