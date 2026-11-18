#!/usr/bin/env bash
# The body of slurm/jobs/train.sbatch: one chunk of a training chain, on however many GPUs the allocation has.
# Everything lives in main(), called on the last line, so replacing this file (with mv) never disturbs a job
# that is already running it.
#
# Robustness, all inherited from the runs this recipe was developed on:
# - a successor is queued at the start (afterany), so the chain survives a hard kill (node failure, scancel,
#   out-of-memory killer) that never reaches the end of this script;
# - the trainer resumes from the last checkpoint (at most save_every_examples papers back);
# - a crash keeps the allocation and retries (twice at once, then with backoff, at most SUM_RETRY_BUDGET_S of
#   idle time without progress), because a new process re-reads scisum/*.py and a fix made meanwhile is used;
# - a watchdog turns a hang (no new line in metrics.jsonl for SUM_STALL_S) into a crash the retry recovers;
# - three chunks in a row without progress stop the chain (state/train-<run>.stop) instead of cycling;
# - the self-check proves the launcher, the model, the data and the memory ceiling on the four longest
#   examples before the first real step.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1   # the repository root, wherever this was launched from

STOPPING=""

stop_training() {
  STOPPING=$1
  echo "== $(date -u +%FT%TZ) $1 received: asking the trainer to checkpoint and stop"
  kill -USR1 "${TRAIN_PID:-0}" 2>/dev/null
  pkill -USR1 -P "${TRAIN_PID:-0}" 2>/dev/null
}

# A hang is not a crash: once the trainer blocked for 8.7 hours in a file-system metadata stall with the
# GPUs idle and the job still RUNNING. The watchdog kills the trainer when runs/<run>/metrics.jsonl has not
# moved for SUM_STALL_S (default 20 min; rank 0 writes it every few steps, and a held-out evaluation takes
# a few minutes at most), and the retry loop resumes from the last checkpoint.
watchdog() {
  local pid=$1 started=$2 stall=${SUM_STALL_S:-1200} f="runs/$RUN/metrics.jsonl" mt age
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    mt=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$mt" -lt "$started" ] && mt=$started          # a stale file from an earlier chunk is not progress
    age=$(( $(date +%s) - mt ))
    if [ "$age" -gt "$stall" ]; then
      echo "== $(date -u +%FT%TZ) watchdog: no training metrics for ${age}s; killing the trainer so"
      echo "   the retry can resume from the last checkpoint"
      ledger_event watchdog "\"stalled_s\": $age"
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
      return
    fi
  done
}

# The self-check (once per model, recipe, site and GPU count) and one training attempt. Returns the
# trainer's exit code, or 9 if no configuration of GPUs and gradient checkpointing works at all.
attempt() {
  if [ ! -e "$SELFCHECK_OK" ]; then
    echo "== self-check on the longest examples in the split"
    run_selfcheck "$NGPU_REQ" || return 9
  fi
  read -r NGPU CKPT < "$SELFCHECK_OK"
  # 8 examples (pairs) per optimiser step on every site: the recipe's SCISUM_TOTAL_BATCH over the GPUs
  export SCISUM_GRADIENT_CHECKPOINTING=$CKPT SCISUM_GRAD_ACCUM=$(( ${SCISUM_TOTAL_BATCH:-8} / NGPU ))
  local maxs rc
  maxs=$(seconds_left 1500)
  if [ "$maxs" -lt 900 ]; then echo "== only ${maxs}s of walltime left; stopping cleanly"; return 2; fi
  echo "== training: run=$RUN gpus=$NGPU checkpointing=$CKPT max_seconds=$maxs"
  launch "$NGPU" -v train --run "$RUN" --max-seconds "$maxs" &
  TRAIN_PID=$!
  watchdog "$TRAIN_PID" "$(date +%s)" &
  WATCH_PID=$!
  wait "$TRAIN_PID"; rc=$?
  [ "$rc" -gt 128 ] && { wait "$TRAIN_PID"; rc=$?; }   # the first wait returned because of a trap
  kill "$WATCH_PID" 2>/dev/null
  return $rc
}

main() {
  source slurm/common.sh || exit 1
  source slurm/lib.sh
  RUN=${SUM_RUN:?SUM_RUN is the run to train (scripts/pipeline.py sets it)}
  NAME=${SLURM_JOB_NAME:-sum-train-$RUN}
  NOPROG=state/train-$RUN.noprogress RUNSTOP=state/train-$RUN.stop
  if [ -e state/train.stop ]; then echo "state/train.stop exists; exiting"; exit 0; fi
  if [ -e "$RUNSTOP" ]; then echo "$RUNSTOP exists; exiting (remove it to resume)"; exit 0; fi

  # A successor is queued at the start of every chunk, so the chain must end somewhere: here.
  if [ "$(run_finished "$RUN")" = "1" ]; then
    echo "== run $RUN is already finished; nothing to continue"
    ledger_event finished "\"run\": \"$RUN\""
    $SCISUM status --run "$RUN" || true
    exit 0
  fi
  ledger_event started "\"run\": \"$RUN\""

  NGPU_REQ=$(site_gpu_count)
  [ "${NGPU_REQ:-0}" -ge 1 ] || { echo "== no GPU visible in this allocation"; exit 1; }
  if [ $(( ${SCISUM_TOTAL_BATCH:-8} % NGPU_REQ )) -ne 0 ]; then
    echo "== $NGPU_REQ GPUs do not divide SCISUM_TOTAL_BATCH=${SCISUM_TOTAL_BATCH:-8}: use 1, 2, 4 or 8"; exit 1
  fi
  SELFCHECK_OK=$(selfcheck_key "$NGPU_REQ").ok
  NEED_CKPT=state/selfcheck/${MODEL_TAG}-${SUM_RECIPE:-none}.need_checkpointing
  echo "== $NGPU_REQ GPU(s) allocated; ${SCISUM_TOTAL_BATCH:-8} examples per optimiser step whatever the count"

  if [ -z "${SUM_NO_CHAIN:-}" ]; then
    submit_like_me "$NAME" --dependency=afterany:"$SLURM_JOB_ID" "$REPO/slurm/jobs/train.sbatch" >/dev/null \
      && echo "== successor queued"
  fi
  # Milestone evaluations (SFT only): a GPU job of the site's "gen" kind that evaluates each milestone
  # checkpoint on the validation curve subset as it appears, and resubmits itself while training runs.
  if [ "${SUM_MILESTONE_EVAL:-0}" = "1" ] && [ -z "${SUM_NO_EVAL:-}" ] && ! squeue --me -h -n "$NAME-evalw" | grep -q .; then
    SUM_SBATCH_FLAGS="$(site_sbatch_flags gen)" submit_like_me "$NAME-evalw" "$REPO/slurm/jobs/eval_worker.sbatch" \
      && echo "== milestone evaluation worker queued"
  fi

  trap 'stop_training USR1' USR1
  trap 'stop_training TERM' TERM

  started=$(examples_done "$RUN"); seen=$started
  budget=${SUM_RETRY_BUDGET_S:-900}; spent=0; backoff=0; tries=0
  while :; do
    attempt; rc=$?
    now=$(examples_done "$RUN")
    if [ "$now" -gt "$seen" ]; then seen=$now; spent=0; backoff=0; fi   # progress earns new credit
    [ "$rc" = "0" ] && break
    [ "$rc" = "2" ] && break
    [ -n "$STOPPING" ] && { rc=2; break; }
    { [ -e state/train.stop ] || [ -e "$RUNSTOP" ]; } && { echo "== stop file"; rc=2; break; }
    tries=$((tries + 1))
    left=$(seconds_left 1500)
    if [ "$left" -lt $((backoff + 900)) ]; then echo "== not enough walltime left to retry"; rc=2; break; fi
    if [ "$spent" -ge "$budget" ]; then
      echo "== $spent s of retries without progress (budget $budget s): releasing the allocation"
      break
    fi
    echo "== attempt $tries failed with rc=$rc at $now examples."
    echo "   Keeping the allocation and retrying in ${backoff}s. A retry starts a new process, so a fix to"
    echo "   scisum/*.py made right now is picked up, and training resumes from the last checkpoint."
    echo "   'touch $RUNSTOP' to stop instead."
    ledger_event retry "\"rc\": $rc, \"examples\": $now"
    [ "$backoff" -gt 0 ] && sleep "$backoff"
    spent=$((spent + backoff))
    case "$backoff" in 0) backoff=$([ "$tries" -ge 2 ] && echo 60 || echo 0) ;; 60) backoff=180 ;; *) backoff=300 ;; esac
  done

  echo "== chunk finished: exit=$rc, examples $started -> $(examples_done "$RUN")"
  $SCISUM status --run "$RUN" || true
  $SCISUM curves --run "$RUN" || true     # learning curves after every chunk (runs/<run>/ and reports/)

  if [ "$(examples_done "$RUN")" -gt "$started" ]; then
    rm -f "$NOPROG"
  elif [ "$rc" != "0" ]; then
    n=$(( $(cat "$NOPROG" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$NOPROG"
    if [ "$n" -ge 3 ]; then
      echo "== $n chunks in a row made no progress; stopping this run's chain ($RUNSTOP)."
      echo "   Fix the cause, then: rm $RUNSTOP $NOPROG and resubmit the stage (scripts/pipeline.py run ...)"
      ledger_event stopped "\"why\": \"no progress in $n chunks\""
      touch "$RUNSTOP"
    fi
  fi

  if [ "$rc" = "0" ]; then
    echo "== run complete"
    ledger_event finished "\"run\": \"$RUN\""
    # The allocation is held anyway: spend what is left on pending milestone evaluations (SFT).
    left=$(seconds_left 900)
    if [ "${SUM_MILESTONE_EVAL:-0}" = "1" ] && [ "$left" -gt 3600 ]; then
      echo "== ${left}s of walltime left: running the pending milestone evaluations here"
      $SCISUM -v eval worker --run "$RUN" --once --max-seconds $((left - 1800)) || true
      $SCISUM curves --run "$RUN" || true
      $SCISUM report --run "$RUN" || true
    fi
  else
    echo "== the successor queued at start continues from the last checkpoint"
  fi
}

main "$@"; exit $?
