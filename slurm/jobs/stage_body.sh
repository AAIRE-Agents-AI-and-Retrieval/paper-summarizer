#!/usr/bin/env bash
# The body of slurm/jobs/stage.sbatch. Exit codes of `pipeline.py exec`: 0 done, 3 stopped on the time limit
# with work left (this job resubmits itself, at most SUM_MAX_RESUBMITS times in a row), 10 inputs not done yet
# (nothing to do; `pipeline.py next` or the driver submits it again later), anything else failed (marker in
# state/pipeline/<student>/<stage>.failed, the log is runs/logs/<job name>_<job id>.log).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1

main() {
  # downloads need the network; every other stage reads the cache only
  case "${SUM_STAGE:-}" in fetch|anchor-fetch) export SUM_HF_OFFLINE=0 ;; esac
  source slurm/common.sh || exit 1
  source slurm/lib.sh
  : "${SUM_STAGE:?SUM_STAGE is the stage to run (scripts/pipeline.py sets it)}"
  [ -e state/pipeline/stop ] && { echo "state/pipeline/stop exists; exiting"; exit 0; }
  local who=${SUM_STUDENT:-_global} counter rc left
  counter=state/pipeline/$who/$SUM_STAGE.resubmits
  mkdir -p "state/pipeline/$who"
  left=$(seconds_left 600)
  t0=$(date +%s)
  $PY scripts/pipeline.py exec ${SUM_STUDENT:+--student "$SUM_STUDENT"} --stage "$SUM_STAGE" --max-seconds "$left"
  rc=$?
  case "$rc" in
    0)  rm -f "$counter"; echo "== stage $SUM_STAGE of $who done" ;;
    3)  n=$(( $(cat "$counter" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$counter"
        if [ "$n" -gt "${SUM_MAX_RESUBMITS:-20}" ]; then
          echo "== $n resubmissions in a row without finishing; giving up (rm $counter, then resubmit)"
          ledger_event gave_up "\"resubmits\": $n"; exit 1
        fi
        # a job that stopped within minutes made (almost) no progress: wait before the next one, so a
        # too-short time limit or a stalled input cannot turn into a storm of one-second jobs
        begin=(); [ $(( $(date +%s) - t0 )) -lt 600 ] && begin=(--begin=now+10minutes)
        echo "== unfinished; resubmitting ($n) ${begin[*]}"
        submit_like_me "${SLURM_JOB_NAME:-sum-stage}" "${begin[@]}" "$REPO/slurm/jobs/stage.sbatch" ;;
    10) echo "== inputs not ready; nothing done" ;;
    *)  echo "== stage $SUM_STAGE of $who failed (rc=$rc); see state/pipeline/$who/$SUM_STAGE.failed"; exit "$rc" ;;
  esac
}

main "$@"; exit $?
