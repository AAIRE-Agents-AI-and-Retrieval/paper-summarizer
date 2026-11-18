# The LUMI runtime: one container image plus an overlay venv

**Untested on LUMI.** This recipe was developed on NVIDIA GPUs. Everything below follows LUMI's documentation
as we understand it; the smoke test (`containers/smoke_test.py`) is the first thing to run, and its report
says which parts work on the MI250X. Version numbers and image names are examples: list what is installed
when you start.

## What is needed

| piece | why | where it comes from |
|---|---|---|
| PyTorch for ROCm | training (LoRA, DDP over up to 8 GCDs) | the base image |
| vLLM for ROCm, built for `gfx90a` (MI250X) | test-split generation, candidate sampling, benchmarks, the two grader servers | the base image, or built on top of a PyTorch image |
| Triton for ROCm | Qwen3.5's linear-attention kernels (`flash-linear-attention`), vLLM's kernels | comes with PyTorch/vLLM in the image |
| the packages of `containers/requirements-overlay.txt` and this repository | everything else | the overlay venv (`SUM_VENV`) |

`bitsandbytes` is not needed (no quantised training). `flash-attn` is not needed: attention goes through
PyTorch's `scaled_dot_product_attention`, whose ROCm build has memory-efficient kernels; the smoke test checks
that one of them works at 32,768 tokens, because the plain math kernel cannot (it would need tens of GB per head).

## Steps (on a LUMI login node, from the repository root on /scratch)

1. **Pick a base image.** LUMI provides AI images under `/appl/local/containers/sif-images/` (PyTorch with
   ROCm; some versions also ship vLLM). List them:
   ```bash
   ls /appl/local/containers/sif-images/ | grep -i -E 'pytorch|vllm'
   singularity exec <image>.sif bash -c '$WITH_CONDA; python -c "import torch, sys; print(torch.__version__, torch.version.hip); import vllm; print(vllm.__version__)"'
   ```
   Take an image where both imports work. If no image has vLLM: either pull an AMD vLLM image that supports
   `gfx90a` (`singularity pull docker://rocm/vllm:<tag>`; check that the tag's release notes list MI200/MI250),
   or build vLLM from source inside a PyTorch image (`PYTORCH_ROCM_ARCH=gfx90a`, vLLM's ROCm build guide). Both
   are untested here.
2. **Make the runtime** (copies the image to scratch, so a LUMI update does not change it under a running
   project, and creates the overlay venv):
   ```bash
   LUMI_PROJECT=project_XXXXXXXXX SUM_BASE_SIF=/appl/local/containers/sif-images/<image>.sif \
     bash containers/lumi/setup_runtime.sh
   ```
   It prints the three lines to put in `slurm/site/local.sh` (`LUMI_PROJECT`, `SUM_SIF`, `SUM_VENV`).
3. **Smoke test** on one GCD: `bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch`, then read
   `runs/logs/sum-smoke_<job>.log`. Required checks must be `ok`; `fla_gated_delta` must be `ok` before
   training `qwen35-4b`.
4. **Optional, faster start-up:** a venv is thousands of small files, which Lustre serves slowly when many
   jobs start. Pack it once (`mksquashfs $SUM_VENV scisum-venv.sqsh`), then in `slurm/site/local.sh`:
   `SUM_SINGULARITY_ARGS="-B /scratch/$LUMI_PROJECT/containers/scisum-venv.sqsh:/user-software:image-src=/"` and
   `SUM_VENV=/user-software`. Rebuild the squashfs after changing the venv.

## How jobs use it

`slurm/site/lumi.sh` sets `PY=slurm/site/lumi-python`. That wrapper runs `singularity exec` with LUMI's file
systems bound (`/pfs,/scratch,/project,/projappl,/flash,/appl`, whichever exist), evaluates the image's
`$WITH_CONDA` if it defines one, activates `$SUM_VENV`, and runs `python`. Called again from inside the
container (a stage command that runs `$PY`), it runs `python` directly. Slurm commands (`sbatch`, `squeue`)
are not available inside the container, so the job scripts call them on the host, and
`scripts/pipeline.py` itself runs on the host with LUMI's `cray-python` module.
