#!/usr/bin/env bash
#SBATCH --job-name=sum-train
#SBATCH --signal=B:USR1@600
#SBATCH --output=runs/logs/%x_%j.log
#
# A training chain (SFT or DPO): this file is only the Slurm header. Resources (partition, GPUs, time,
# account) come from the site at submission (scripts/pipeline.py -> slurm/site/<site>.sh), so the same header
# serves LUMI and a CUDA cluster. Slurm copies a batch script when it is submitted, so the body lives on
# shared storage and is read when the job starts: a fix made while the job waits is picked up. Replace the
# body with `mv`, never by editing it in place.
#
#   python3 scripts/pipeline.py --site lumi run --student k2h-7b --stage sft      # submit
#   touch state/train-k2h-7b-sft50k.stop                                          # stop after this chunk
#   python3 scripts/pipeline.py --site lumi cancel --student k2h-7b --stage sft   # stop now (resumes later)
exec bash "${SUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/jobs/train_body.sh"
