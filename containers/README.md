# Runtimes

The pipeline needs one Python environment that has PyTorch, vLLM (generation, sampling, the grader servers),
transformers, PEFT and the packages in `requirements-overlay.txt`, plus this repository installed in editable
mode. Training, generation, grading and the analysis all run in it.

| site | how Python runs | how to build it | where it is configured |
|---|---|---|---|
| CUDA cluster | `.venv/bin/python` in the repository | `python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps -e .` (the lock is the exact CUDA environment the recipe was developed in: torch 2.13, vLLM 0.31, transformers 5.17, PEFT 0.21.2) | `slurm/site/cuda.sh`, `slurm/site/local.sh` |
| LUMI (AMD MI250X, ROCm) | `slurm/site/lumi-python` = `singularity exec $SUM_SIF python` with an overlay venv `$SUM_VENV` | `containers/lumi/README.md` and `containers/lumi/setup_runtime.sh` | `slurm/site/lumi.sh`, `slurm/site/local.sh` |

After building either runtime, run the smoke test on one GPU before anything else:

```bash
bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch        # or --site cuda
tail -f runs/logs/sum-smoke_<job>.log                               # one JSON report; "ok": true
```

`containers/smoke_test.py` checks the GPU, bf16 speed, causal attention at 32,768 tokens with a memory-efficient
kernel, a LoRA backward pass with gradient checkpointing, the Triton kernel of Qwen3.5's linear attention, and
that `scisum` and vLLM import. `SUM_SMOKE_ARGS="--model <cached model id>"` adds a real vLLM generation.
