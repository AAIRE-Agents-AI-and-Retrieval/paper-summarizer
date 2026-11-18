# Shared environment for every job. Sourced, never executed.
#
# Order: the site (how to run Python, which caches, which sbatch flags), then the student's profile
# (configs/students/$SUM_STUDENT.env), then the recipe (configs/recipes/$SUM_RECIPE.env), then the stage's
# own extra environment ($SUM_STAGE_ENV, written by scripts/pipeline.py under state/pipeline/). Later files
# win. Nothing here names a machine: everything site-specific is in slurm/site/<site>.sh and the
# git-ignored slurm/site/local.sh (account, partitions, container paths).
REPO=${SUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/pyproject.toml" ] || [ ! -d "$REPO/scisum" ]; then
  echo "common.sh: $REPO is not the repository root; submit from it or set SUM_REPO" >&2
  return 1 2>/dev/null || exit 1
fi
cd "$REPO" || exit 1
export SUM_REPO=$REPO
if [ -z "${SUM_SITE:-}" ]; then
  echo "common.sh: SUM_SITE is not set (lumi | cuda); scripts/pipeline.py --site sets it" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
[ -f slurm/site/local.sh ] && source slurm/site/local.sh
# shellcheck disable=SC1090
source "slurm/site/$SUM_SITE.sh" || { echo "common.sh: slurm/site/$SUM_SITE.sh failed" >&2; return 1 2>/dev/null || exit 1; }

# Caches (paths come from the site via slurm/site/_caches.sh). A compute node may have a read-only home.
[ "${SUM_OVERRIDE_HOME:-0}" = "1" ] && export HOME=$SUM_CACHE_ROOT/home
mkdir -p "$SUM_CACHE_ROOT" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR" runs/logs state/pipeline 2>/dev/null
[ "${SUM_OVERRIDE_HOME:-0}" = "1" ] && mkdir -p "$HOME"
# Jobs never depend on the network for models: the fetch stages download, everything else reads the cache.
export HF_HUB_OFFLINE=${SUM_HF_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=$HF_HUB_OFFLINE
# vLLM forks its engine core; the parent has the GPU runtime initialised by then, which fork cannot carry
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export TQDM_DISABLE=1          # every vLLM generate call passes use_tqdm=False; this silences the rest
export PYTHONUNBUFFERED=1
type site_job_setup >/dev/null 2>&1 && site_job_setup

# The student, the recipe and the stage. Everything downstream (token shards, prompts, caches) is keyed by
# the base model, so two students never share tokenised data.
if [ -n "${SUM_STUDENT:-}" ]; then
  # shellcheck disable=SC1090
  source "configs/students/$SUM_STUDENT.env" || { echo "no profile configs/students/$SUM_STUDENT.env" >&2; return 1 2>/dev/null || exit 1; }
fi
if [ -n "${SUM_RECIPE:-}" ]; then
  # shellcheck disable=SC1090
  source "configs/recipes/$SUM_RECIPE.env" || { echo "no recipe configs/recipes/$SUM_RECIPE.env" >&2; return 1 2>/dev/null || exit 1; }
fi
if [ -n "${SUM_STAGE_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$SUM_STAGE_ENV" || { echo "cannot read $SUM_STAGE_ENV" >&2; return 1 2>/dev/null || exit 1; }
fi
MODEL_TAG=$(printf '%s' "${SCISUM_BASE_MODEL:-none}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
export MODEL_TAG
# PY comes from the site (a venv's python, or slurm/site/lumi-python which runs python in the container)
export PY
export SCISUM="$PY -m scisum.cli"
echo "== $(date -u +%FT%TZ) node=$(hostname) job=${SLURM_JOB_ID:-none} name=${SLURM_JOB_NAME:-none} site=$SUM_SITE" \
     "student=${SUM_STUDENT:-none} recipe=${SUM_RECIPE:-none} stage=${SUM_STAGE:-none} run=${SUM_RUN:-none}" \
     "model=${SCISUM_BASE_MODEL:-none}@${SCISUM_BASE_MODEL_REVISION:-main}"
type site_gpu_info >/dev/null 2>&1 && site_gpu_info
true   # sourcing must always succeed when everything above did
