# Site: LUMI (LUMI-G: 4 AMD MI250X per node = 8 GCDs of 64 GB, each GCD one "GPU" to Slurm and PyTorch).
# Python runs inside a Singularity container (SUM_SIF) with an overlay venv (SUM_VENV); see docs/LUMI.md and
# containers/lumi/. Sourced by slurm/common.sh (in jobs) and by scripts/pipeline.py (login node).
#
# Put your values in slurm/site/local.sh (git-ignored, sourced first), for example:
#   LUMI_PROJECT=project_XXXXXXXXX                     # your LUMI project; SUM_ACCOUNT defaults to it
#   SUM_SIF=/scratch/$LUMI_PROJECT/containers/scisum-rocm.sif
#   SUM_VENV=/scratch/$LUMI_PROJECT/containers/scisum-venv
#   SCISUM_DATA_ROOT=/scratch/$LUMI_PROJECT/scisum-data   # only if data/ is not inside the repository
# Partitions and limits are LUMI's at the time of writing; check `sinfo` and the LUMI docs before relying on them.
SUM_SITE_NAME=lumi
: "${SUM_ACCOUNT:=${LUMI_PROJECT:-}}"
PY=$REPO/slurm/site/lumi-python
# LUMI's login nodes have an old system python3; scripts/pipeline.py needs >= 3.9 (`module load cray-python`)
SUM_HOST_PY=${SUM_HOST_PY:-python3}
: "${SUM_OVERRIDE_HOME:=0}"           # home is writable on LUMI compute nodes (but only 20 GB: caches go elsewhere)
: "${SUM_GRADER_MODE:=serve}"         # graders are served with vLLM inside the grader job
CPUS_PER_GCD=${SUM_CPUS_PER_GCD:-7}   # 56 usable cores / 8 GCDs
MEM_PER_GCD=${SUM_MEM_PER_GCD:-60}    # GB; ~480 GB usable / 8 GCDs

site_job_setup() {
  # MIOpen keeps a lock-heavy database in $HOME by default; on a shared file system that stalls or
  # corrupts. One private, node-local directory per job.
  export MIOPEN_USER_DB_PATH="/tmp/${USER:-user}-miopen-${SLURM_JOB_ID:-$$}"
  export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
  mkdir -p "$MIOPEN_USER_DB_PATH"
  # single-node jobs only: RCCL needs no network settings. Multi-node would need NCCL_SOCKET_IFNAME=hsn0,...
}

site_gpu_count() {
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then echo "$SLURM_GPUS_ON_NODE"
  elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
  else "$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0
  fi
}
site_gpu_info() {
  [ -n "${SLURM_GPUS_ON_NODE:-}${ROCR_VISIBLE_DEVICES:-}" ] && echo "== GCDs: ${SLURM_GPUS_ON_NODE:-?} (ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset})"
  true
}
site_host_python() {
  # the driver job runs pipeline.py on the host (it calls sbatch/squeue, which are not inside the container)
  if ! "$SUM_HOST_PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    type module >/dev/null 2>&1 && module load cray-python >/dev/null 2>&1
  fi
  echo "$SUM_HOST_PY"
}

_lumi_gpu() {  # partition gpus time -> flags for one node with that many GCDs
  local part=$1 g=$2 t=$3
  echo "--account=${SUM_ACCOUNT:-MISSING_SUM_ACCOUNT} --partition=$part --nodes=1 --ntasks=1 --gpus-per-node=$g" \
       "--cpus-per-task=$((CPUS_PER_GCD * g)) --mem=$((MEM_PER_GCD * g))G --time=$t"
}

# sbatch flags per job kind (see slurm/site/cuda.sh for the kinds). small-g bills what you request;
# standard-g allocates (and bills) whole nodes, so it is the default only for 8-GCD training.
site_sbatch_flags() {
  local g p
  case "$1" in
    train)
      g=${SUM_TRAIN_GPUS:-8}
      p=${SUM_TRAIN_PARTITION:-$([ "$g" -ge 8 ] && echo standard-g || echo small-g)}
      _lumi_gpu "$p" "$g" "${SUM_TRAIN_TIME:-24:00:00}" ;;
    gen)    _lumi_gpu "${SUM_GEN_PARTITION:-small-g}" "${SUM_GEN_GPUS:-1}" "${SUM_GEN_TIME:-06:00:00}" ;;
    grader) _lumi_gpu "${SUM_GRADER_PARTITION:-small-g}" "${SUM_GRADER_GPUS:-4}" "${SUM_GRADER_TIME:-12:00:00}" ;;
    cpu)     echo "--account=${SUM_ACCOUNT:-MISSING_SUM_ACCOUNT} --partition=${SUM_CPU_PARTITION:-small} --ntasks=1 --cpus-per-task=8 --mem=32G --time=${SUM_CPU_TIME:-02:00:00}" ;;
    cpu-big) echo "--account=${SUM_ACCOUNT:-MISSING_SUM_ACCOUNT} --partition=${SUM_CPU_PARTITION:-small} --ntasks=1 --cpus-per-task=${SUM_CPU_BIG_CPUS:-64} --mem=${SUM_CPU_BIG_MEM:-128G} --time=04:00:00" ;;
    driver)  echo "--account=${SUM_ACCOUNT:-MISSING_SUM_ACCOUNT} --partition=${SUM_CPU_PARTITION:-small} --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:20:00" ;;
    *) echo "site_sbatch_flags: unknown kind $1" >&2; return 1 ;;
  esac
}

# shellcheck disable=SC1091
source "$REPO/slurm/site/_caches.sh"
