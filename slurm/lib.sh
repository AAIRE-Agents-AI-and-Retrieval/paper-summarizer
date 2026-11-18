# Helpers shared by the job bodies. Sourced after slurm/common.sh.
#
# Every failure that once cost a GPU allocation lived in a few lines like these, not in the model code:
# `torchrun` missing from PATH (rc 127), `timeout` asked to run a shell function (rc 127 again), `scontrol
# show job 0` dumping every job so the walltime came back as zero, a successor queued without its parent's
# card, time or memory. tests/test_slurm_lib.py exercises them on a login node in seconds.

: "${MODEL_TAG:=unknown}"

# The job's own sbatch flags (SUM_SBATCH_FLAGS, set by scripts/pipeline.py from the site), plus this job's
# time limit and memory: a successor that fell back to defaults once waited hours longer in the queue, and
# one that lost its card constraint could land on a GPU too small for it.
submit_like_me() {
  local name=$1; shift
  local -a extra=() flags=()
  # shellcheck disable=SC2206
  [ -n "${SUM_SBATCH_FLAGS:-}" ] && flags=($SUM_SBATCH_FLAGS)
  local tl=""
  [ -n "${SLURM_JOB_ID:-}" ] && tl=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null)
  [ -n "$tl" ] && extra+=(--time="$tl")
  [ -n "${SLURM_MEM_PER_NODE:-}" ] && extra+=(--mem="${SLURM_MEM_PER_NODE}M")
  sbatch -J "$name" --export=ALL "${flags[@]}" "${extra[@]}" "$@"
}

# Seconds left in this allocation, minus a margin for the final checkpoint and the report.
seconds_left() {
  local margin=${1:-1500} end ts left
  # Outside a job, or if scontrol says anything unexpected, fall back to a conservative 10.75 h:
  # stopping early costs a resume, while trusting a bad number could mean no work at all.
  if [ -z "${SLURM_JOB_ID:-}" ]; then echo $((40200 - margin)); return; fi
  end=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -m1 '^EndTime=' | cut -d= -f2)
  if [ -z "$end" ] || [ "$end" = "Unknown" ]; then echo $((40200 - margin)); return; fi
  ts=$(date -d "$end" +%s 2>/dev/null) || ts=""
  case "$ts" in ''|*[!0-9]*) echo $((40200 - margin)); return ;; esac   # `date -d ""` is "now"
  left=$(( ts - $(date +%s) - margin ))
  [ "$left" -lt 0 ] && left=0
  echo "$left"
}

# `torchrun` without depending on PATH, and the single-process path when there is one GPU.
# LAUNCH_TIMEOUT wraps the real binary: `timeout` is a program and cannot run a shell function.
launch() {
  local n=$1; shift
  local -a pre=()
  [ -n "${LAUNCH_TIMEOUT:-}" ] && pre=(timeout "$LAUNCH_TIMEOUT")
  if [ "$n" -gt 1 ]; then
    "${pre[@]}" $PY -m torch.distributed.run --nproc_per_node="$n" \
      --master_port=$((20000 + RANDOM % 10000)) -m scisum.cli "$@"
  else
    "${pre[@]}" $PY -m scisum.cli "$@"
  fi
}

# Papers (or pairs) trained so far, as the run itself records them. The progress signal of the retry loop.
examples_done() {
  $PY - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
p = Path(f"runs/{sys.argv[1]}/state.json")
print(json.loads(p.read_text()).get("examples", 0) if p.exists() else 0)
PY
}

# 1 when the run has reached its cap, 0 otherwise.
run_finished() {
  $PY - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
p = Path(f"runs/{sys.argv[1]}/state.json")
print(int(bool(json.loads(p.read_text()).get("finished"))) if p.exists() else 0)
PY
}

# One JSON line per event in state/pipeline/ledger.jsonl (scripts/pipeline.py writes the same format).
ledger_event() {
  local event=$1; shift
  local extra=${1:-}
  mkdir -p state/pipeline
  printf '{"t": "%s", "event": "%s", "student": "%s", "stage": "%s", "job": "%s", "name": "%s", "node": "%s"%s}\n' \
    "$(date +%Y-%m-%dT%H:%M:%S%z)" "$event" "${SUM_STUDENT:-_global}" "${SUM_STAGE:-}" "${SLURM_JOB_ID:-}" \
    "${SLURM_JOB_NAME:-}" "$(hostname)" "${extra:+, $extra}" >> state/pipeline/ledger.jsonl
}

# ---- training self-check ---------------------------------------------------------------------------------
# Which GPU count and gradient-checkpointing setting work for this model, recipe, site and allocation size:
# proven once on the four longest examples (about five minutes), then reused by every chunk.
selfcheck_key() { echo "state/selfcheck/${MODEL_TAG}-${SUM_RECIPE:-none}-${SUM_SITE:-site}-g${1:-0}"; }
: "${SELFCHECK_OK:=}"
: "${NEED_CKPT:=}"

# Each job's self-check has its own run directory: jobs that start together must not delete each other's.
selfcheck_run() { echo "selfcheck${SLURM_JOB_ID:+-$SLURM_JOB_ID}"; }

selfcheck_once() {
  local n=$1 ckpt=$2
  rm -rf "runs/$(selfcheck_run)"
  export SCISUM_GRADIENT_CHECKPOINTING=$ckpt SCISUM_LOSS_EVAL_N=4 SCISUM_GRAD_ACCUM=$(( ${SCISUM_TOTAL_BATCH:-8} / n ))
  local longest=--longest
  [ "${SUM_PREFLIGHT_LONGEST:-1}" = "1" ] || longest=""
  echo "-- self-check: $n GPU(s), gradient_checkpointing=$ckpt ${longest:-(average-length examples)}"
  LAUNCH_TIMEOUT=1800 launch "$n" train --run "$(selfcheck_run)" --smoke $((4 * n)) $longest --no-resume
}

# Configurations to try, best first. Gradient checkpointing is ~1.4x slower but at 32k tokens it is
# normally the only one that fits; once a run proves that (rc 3 = out of memory), the marker skips the test.
candidates() {
  local n=$1
  if [ -e "$NEED_CKPT" ]; then
    echo "$n true"
    [ "$n" -gt 1 ] && echo "1 true"
  else
    echo "$n false"
    echo "$n true"
    [ "$n" -gt 1 ] && echo "1 true"
  fi
  return 0
}

# Writes $SELFCHECK_OK ("<gpus> <checkpointing>") and returns 0 as soon as one configuration works.
run_selfcheck() {
  local n=$1 ok="" rc
  mkdir -p "$(dirname "$SELFCHECK_OK")"
  while read -r gpus ckpt; do
    [ -z "$gpus" ] && continue
    selfcheck_once "$gpus" "$ckpt"; rc=$?
    if [ "$rc" = "0" ]; then ok="$gpus $ckpt"; break; fi
    echo "-- self-check failed with rc=$rc"
    [ "$rc" = "3" ] && touch "$NEED_CKPT"
  done <<< "$(candidates "$n")"
  unset SCISUM_LOSS_EVAL_N
  [ -z "$ok" ] && return 1
  rm -rf "runs/$(selfcheck_run)"
  echo "$ok" > "$SELFCHECK_OK"
  echo "== self-check passed with: $ok"
  return 0
}

# ---- graders ---------------------------------------------------------------------------------------------
# The served name (what the client asks for) -> the Hugging Face weights. Override in slurm/site/local.sh.
grader_hf_id() {
  case "$1" in
    qwen38-27b)    echo "${SUM_GRADER_HF_QWEN:-Qwen/Qwen3.8-27B}" ;;
    gemma4-31b-it) echo "${SUM_GRADER_HF_GEMMA:-google/gemma-4-31B-it}" ;;
    *) echo "grader_hf_id: unknown grader $1" >&2; return 1 ;;
  esac
}
# Extra vLLM arguments per grader. qwen38-27b is a reasoning model: with a reasoning parser its thinking goes
# to `reasoning_content` and the JSON answer stays in `content`, as the client expects (docs/GRADERS.md).
grader_vllm_args() {
  case "$1" in
    qwen38-27b)    echo "${SUM_GRADER_ARGS_QWEN---reasoning-parser qwen3}" ;;
    gemma4-31b-it) echo "${SUM_GRADER_ARGS_GEMMA-}" ;;
  esac
}
