#!/usr/bin/env bash
# The body of slurm/jobs/grader_server.sbatch; see that file.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1

SERVER_PID=""
cleanup() {
  [ -n "${INFO:-}" ] && rm -f "$INFO"
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null; sleep 5; kill -9 "$SERVER_PID" 2>/dev/null
    pkill -9 -P "$SERVER_PID" 2>/dev/null
  fi
}

serve() {
  local hf_id port ngpu deadline log
  hf_id=$(grader_hf_id "$MODEL") || return 1
  ngpu=$(site_gpu_count)
  port=$(( 8100 + ${SLURM_JOB_ID:-$$} % 800 ))
  log=runs/logs/${SLURM_JOB_NAME:-sum-grader}_${SLURM_JOB_ID:-local}.vllm.log
  # The first start of a grader downloads its weights (~60 GB); afterwards the cache is enough.
  if ! $PY -c "import sys; from huggingface_hub import try_to_load_from_cache as t; sys.exit(0 if isinstance(t('$hf_id', 'config.json'), str) else 1)" 2>/dev/null; then
    echo "== $hf_id is not in the cache: allowing this job to download it"
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
  fi
  echo "== serving $hf_id as $MODEL on port $port, tensor parallel $ngpu (vLLM log: $log)"
  # shellcheck disable=SC2046
  $PY -m vllm.entrypoints.openai.api_server --model "$hf_id" --served-model-name "$MODEL" \
      --tensor-parallel-size "$ngpu" --max-model-len "${SUM_GRADER_MAX_LEN:-40960}" \
      --gpu-memory-utilization "${SUM_GRADER_GPU_UTIL:-0.90}" --host 0.0.0.0 --port "$port" \
      $(grader_vllm_args "$MODEL") ${SUM_GRADER_VLLM_EXTRA:-} > "$log" 2>&1 &
  SERVER_PID=$!
  deadline=$(( $(date +%s) + ${SUM_GRADER_START_S:-2400} ))
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "== the vLLM server exited during start-up:"; tail -40 "$log"; return 1; fi
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "== the vLLM server did not answer within ${SUM_GRADER_START_S:-2400}s"; tail -40 "$log"; return 1; fi
    sleep 20
  done
  mkdir -p state/graders
  INFO=state/graders/$MODEL.json
  printf '{"model": "%s", "hf_id": "%s", "base_url": "http://%s:%s/v1", "job": "%s", "started": "%s", "tensor_parallel": %s}\n' \
    "$MODEL" "$hf_id" "$(hostname)" "$port" "${SLURM_JOB_ID:-}" "$(date +%Y-%m-%dT%H:%M:%S%z)" "$ngpu" > "$INFO.tmp" && mv "$INFO.tmp" "$INFO"
  echo "== serving: $(cat "$INFO")"
  # the work below runs on this node and talks to the server directly (contract: SCISUM_GRADER_ENDPOINTS first)
  export SCISUM_GRADER_ENDPOINTS="{\"$MODEL\": \"http://127.0.0.1:$port/v1\"}"
  export JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
  ledger_event grader_up "\"model\": \"$MODEL\", \"base_url\": \"http://$(hostname):$port/v1\""
}

main() {
  source slurm/common.sh || exit 1
  source slurm/lib.sh
  MODEL=${SUM_GRADER:?SUM_GRADER is the grader model (qwen38-27b or gemma4-31b-it)}
  NAME=${SLURM_JOB_NAME:-sum-grader-$MODEL}
  [ -e state/pipeline/stop ] && { echo "state/pipeline/stop exists; exiting"; exit 0; }
  trap cleanup EXIT
  trap 'echo "== signal: stopping"; exit 0' TERM USR1
  if [ "${SUM_GRADER_MODE:-serve}" = "serve" ]; then
    serve || { ledger_event grader_failed "\"model\": \"$MODEL\""; exit 1; }
  else
    echo "== external grader endpoint (SUM_GRADER_MODE=external): no server in this job"
  fi
  # the health gate: is an endpoint configured for this model, does it answer, does it serve that name?
  if ! $SCISUM graders check --models "$MODEL"; then
    echo "== grader $MODEL is not usable (see the JSON above); no grading in this job"
    ledger_event grader_failed "\"model\": \"$MODEL\", \"why\": \"graders check\""
    exit 1
  fi
  if [ -n "${SUM_GRADER_SERVE_ONLY:-}" ]; then
    left=$(seconds_left 300); echo "== serve-only: up for ${left}s"; sleep "$left"; exit 0
  fi
  left=$(seconds_left 600)
  t0=$(date +%s)
  $PY scripts/pipeline.py work --grader "$MODEL" --max-seconds "$left"
  rc=$?
  if [ "$rc" = "3" ]; then
    begin=(); [ $(( $(date +%s) - t0 )) -lt 600 ] && begin=(--begin=now+15minutes)
    echo "== grading work left at the time limit; resubmitting ${begin[*]}"
    submit_like_me "$NAME" "${begin[@]}" "$REPO/slurm/jobs/grader_server.sbatch"
  else
    echo "== no grading work ready for $MODEL (rc=$rc); \`pipeline.py next\` (or the driver) submits a grader again when there is"
  fi
}

main "$@"; exit $?
