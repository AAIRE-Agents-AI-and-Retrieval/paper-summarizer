#!/usr/bin/env bash
# Build the LUMI runtime once: a copy of a base image on scratch, and an overlay venv with this repository.
#
#   LUMI_PROJECT=project_XXXXXXXXX SUM_BASE_SIF=/appl/local/containers/sif-images/<image>.sif \
#     bash containers/lumi/setup_runtime.sh
#
# Optional: SUM_SIF (where the copy goes), SUM_VENV (where the venv goes). Run from the repository root on a
# login node (installing packages is light work; if your project prefers, run it in an interactive job:
# `srun --account=$LUMI_PROJECT --partition=small --cpus-per-task=8 --mem=16G --time=1:00:00 --pty bash`).
# Untested on LUMI: read containers/lumi/README.md first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
: "${LUMI_PROJECT:?set LUMI_PROJECT=project_XXXXXXXXX}"
if [ -z "${SUM_BASE_SIF:-}" ]; then
  echo "Set SUM_BASE_SIF to a ROCm image with PyTorch and vLLM. LUMI's images:"
  ls /appl/local/containers/sif-images/ 2>/dev/null | grep -i -E 'pytorch|vllm' || echo "  (none found under /appl/local/containers/sif-images/)"
  exit 2
fi
BASE=/scratch/$LUMI_PROJECT/containers
SUM_SIF=${SUM_SIF:-$BASE/$(basename "$SUM_BASE_SIF")}
SUM_VENV=${SUM_VENV:-$BASE/scisum-venv}
mkdir -p "$(dirname "$SUM_SIF")" "$(dirname "$SUM_VENV")"
if [ ! -f "$SUM_SIF" ]; then
  echo "== copying $SUM_BASE_SIF -> $SUM_SIF"
  cp "$SUM_BASE_SIF" "$SUM_SIF"
fi
binds=""
for b in /pfs /scratch /project /projappl /flash /appl; do [ -e "$b" ] && binds="${binds:+$binds,}$b"; done
run() { singularity exec -B "$binds" "$SUM_SIF" bash -c 'if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi; '"$1"; }

echo "== what the image provides"
run 'python - <<PY
import importlib
for m in ("torch", "vllm", "triton", "transformers", "peft"):
    try:
        mod = importlib.import_module(m); print(f"{m:13s} {mod.__version__}")
    except Exception as e:
        print(f"{m:13s} MISSING ({type(e).__name__})")
import torch; print("hip", torch.version.hip)
PY'
if ! run 'python -c "import vllm"' 2>/dev/null; then
  echo "!! this image has no vLLM: generation, sampling, benchmarks and the graders need it (containers/lumi/README.md, step 1)"
  exit 1
fi

echo "== overlay venv $SUM_VENV"
export PIP_CACHE_DIR=/scratch/$LUMI_PROJECT/.pip-cache
run "[ -x '$SUM_VENV/bin/python' ] || python -m venv --system-site-packages '$SUM_VENV'
     source '$SUM_VENV/bin/activate'
     pip install -q --upgrade pip
     pip install -r containers/requirements-overlay.txt
     pip install --no-deps -e .
     pip check || echo '!! pip check reports conflicts: see containers/requirements-overlay.txt (transformers vs vLLM)'
     python -c 'import scisum.cli; print(\"scisum imports\")'"

cat <<EOT

== done. Put these lines in slurm/site/local.sh:
LUMI_PROJECT=$LUMI_PROJECT
SUM_SIF=$SUM_SIF
SUM_VENV=$SUM_VENV

Then run the smoke test on one GCD:
  bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch
EOT
