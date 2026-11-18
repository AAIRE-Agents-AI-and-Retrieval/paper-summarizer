# Cache locations, sourced at the end of every site file. Sets variables only (no mkdir, no output), so
# scripts/pipeline.py can source a site on the login node to learn its paths and flags.
# SUM_CACHE_ROOT: everything below lives here unless set already (the default is inside the repository,
# which on LUMI should itself be on /scratch).
: "${SUM_CACHE_ROOT:=$REPO/.cache}"
export SUM_CACHE_ROOT
export HF_HOME=${HF_HOME:-$SUM_CACHE_ROOT/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}   # remote code (K2-Horizon) is cached here too
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SUM_CACHE_ROOT}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SUM_CACHE_ROOT/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$SUM_CACHE_ROOT/inductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SUM_CACHE_ROOT/vllm}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SUM_CACHE_ROOT/pip}
