# Site: a CUDA Slurm cluster, with the Python environment in a venv inside the repository (.venv).
# Sourced by slurm/common.sh (in jobs) and by scripts/pipeline.py (on the login node, for flags and paths).
# Override anything below in slurm/site/local.sh (git-ignored), which is sourced first, e.g.
#   SUM_ACCOUNT=myproject SUM_GPU_PARTITION=gpu SUM_GPU_CONSTRAINT=gpumem.48gib SUM_TRAIN_GPUS=2
SUM_SITE_NAME=cuda
PY=${SUM_PY:-$REPO/.venv/bin/python}
# scripts/pipeline.py needs only the standard library; the venv's interpreter is fine for it here
SUM_HOST_PY=${SUM_HOST_PY:-$PY}
# compute nodes of some clusters mount /home read-only: keep HOME inside the cache
: "${SUM_OVERRIDE_HOME:=1}"

# Graders: "external" when an endpoint is configured (SCISUM_GRADER_ENDPOINTS, or SCISUM_JUDGE_BASE in the
# environment or .env), so the grader job is a CPU job calling it; otherwise "serve" them with vLLM on GPUs.
if [ -z "${SUM_GRADER_MODE:-}" ]; then
  if [ -n "${SCISUM_GRADER_ENDPOINTS:-}${SCISUM_JUDGE_BASE:-}" ] || grep -qs '^SCISUM_JUDGE_BASE=.' "$REPO/.env"; then
    SUM_GRADER_MODE=external
  else
    SUM_GRADER_MODE=serve
  fi
fi

site_gpu_count() { nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true; }
site_gpu_info() {
  local out   # nvidia-smi prints its "failed" message on stdout on a CPU node: show only a real answer
  command -v nvidia-smi >/dev/null && out=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null) && echo "$out"
  true
}
site_host_python() { echo "$SUM_HOST_PY"; }

_cuda_common() {
  local a=""
  [ -n "${SUM_ACCOUNT:-}" ] && a="--account=$SUM_ACCOUNT"
  echo "$a"
}

# sbatch flags per job kind: train (a training chain), gen (generation, scoring, sampling, benchmarks),
# grader (a served grader model, or a CPU client of an external endpoint), cpu, cpu-big (tokenising), driver.
site_sbatch_flags() {
  local a c
  a=$(_cuda_common)
  c=${SUM_GPU_CONSTRAINT:+--constraint=$SUM_GPU_CONSTRAINT}
  case "$1" in
    train)  echo "$a --partition=${SUM_GPU_PARTITION:-gpu} --gres=gpu:${SUM_TRAIN_GPUS:-2} $c --cpus-per-task=${SUM_TRAIN_CPUS:-16} --mem=${SUM_TRAIN_MEM:-192G} --time=${SUM_TRAIN_TIME:-24:00:00}" ;;
    gen)    echo "$a --partition=${SUM_GPU_PARTITION:-gpu} --gres=gpu:${SUM_GEN_GPUS:-1} $c --cpus-per-task=8 --mem=${SUM_GEN_MEM:-96G} --time=${SUM_GEN_TIME:-06:00:00}" ;;
    grader)
      if [ "$SUM_GRADER_MODE" = "external" ]; then
        echo "$a --partition=${SUM_CPU_PARTITION:-short} --cpus-per-task=4 --mem=16G --time=${SUM_CPU_TIME:-00:55:00}"
      else
        echo "$a --partition=${SUM_GPU_PARTITION:-gpu} --gres=gpu:${SUM_GRADER_GPUS:-2} $c --cpus-per-task=16 --mem=${SUM_GRADER_MEM:-192G} --time=${SUM_GRADER_TIME:-12:00:00}"
      fi ;;
    cpu)     echo "$a --partition=${SUM_CPU_PARTITION:-short} --cpus-per-task=4 --mem=24G --time=${SUM_CPU_TIME:-00:55:00}" ;;
    cpu-big) echo "$a --partition=${SUM_CPU_BIG_PARTITION:-general} --cpus-per-task=${SUM_CPU_BIG_CPUS:-64} --mem=${SUM_CPU_BIG_MEM:-192G} --time=04:00:00" ;;
    driver)  echo "$a --partition=${SUM_CPU_PARTITION:-short} --cpus-per-task=1 --mem=2G --time=00:20:00" ;;
    *) echo "site_sbatch_flags: unknown kind $1" >&2; return 1 ;;
  esac
}

# shellcheck disable=SC1091
source "$REPO/slurm/site/_caches.sh"
