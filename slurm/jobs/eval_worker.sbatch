#!/usr/bin/env bash
#SBATCH --job-name=sum-evalw
#SBATCH --output=runs/logs/%x_%j.log
#
# Milestone evaluations of an SFT run (queued by its training chain as <training job name>-evalw, with the
# site's "gen" resources): the base model once, then every milestone checkpoint training has written,
# generated and scored on the validation curve subset (1,000 papers: ROUGE, BERTScore, JSON validity,
# NLI faithfulness on 300). Then the learning curves are redrawn. If training is still running it resubmits
# itself for later, so the GPU is not held while nothing is pending. Every step is idempotent: a killed
# worker redoes only the checkpoint it was on.
set -uo pipefail

main() {
  source slurm/common.sh || exit 1
  source slurm/lib.sh
  RUN=${SUM_RUN:?SUM_RUN is the run whose milestones to evaluate}
  NAME=${SLURM_JOB_NAME:-sum-evalw}
  TRAIN_NAME=${NAME%-evalw}
  if [ -e state/eval.stop ]; then echo "state/eval.stop exists; exiting"; exit 0; fi

  # The base model is the zero point of every curve. An untrained reasoning model writes to the token cap,
  # so it gets the time limit too; SUM_SKIP_BASE=1 skips it.
  g=0
  if [ "${SUM_SKIP_BASE:-0}" != "1" ]; then
    $SCISUM eval generate --run "$RUN" --ckpt base --split val --max-seconds "$(seconds_left 900)"; g=$?
    if [ "$g" = "0" ]; then
      $SCISUM eval score --run "$RUN" --ckpt base --split val || echo "WARNING: base scoring failed"
    elif [ "$g" = "3" ]; then
      echo "== base generation reached the time limit; the successor continues"
    else
      echo "WARNING: base generation failed (rc=$g)"
    fi
  fi
  if [ "$g" = "3" ]; then
    rc=3
  else
    $SCISUM -v eval worker --run "$RUN" --once --max-seconds "$(seconds_left 600)"
    rc=$?
  fi
  $SCISUM curves --run "$RUN" || true
  $SCISUM report --run "$RUN" || true

  read -r finished started pending < <($PY - "$RUN" <<'PY'
import json, sys
from pathlib import Path
from scisum.config import get_settings
from scisum.evaluate import pending_checkpoints
s, run = get_settings(), sys.argv[1]
p = Path(f"runs/{run}/state.json")
state = json.loads(p.read_text()) if p.exists() else {}
print(int(bool(state.get("finished"))), int(state.get("examples", 0) > 0), len(pending_checkpoints(s, run)))
PY
)
  train_jobs=$(squeue --me -h -n "$TRAIN_NAME" | grep -c . || true)
  echo "== worker exit=$rc finished=$finished started=$started pending=$pending train_jobs=$train_jobs"
  if [ "${pending:-1}" != "0" ]; then
    echo "checkpoints still pending; coming back"
    submit_like_me "$NAME" --begin=now+20minutes "$REPO/slurm/jobs/eval_worker.sbatch"
  elif [ "$finished" = "1" ]; then
    echo "every checkpoint is evaluated and the run is finished"
    ledger_event milestones_done "\"run\": \"$RUN\""
  elif [ "$train_jobs" = "0" ]; then
    echo "no training job in the queue; nothing to come back for"
  else
    echo "training in flight ($train_jobs job(s)); coming back later"
    submit_like_me "$NAME" --begin=now+45minutes "$REPO/slurm/jobs/eval_worker.sbatch"
  fi
}

main "$@"; exit $?
