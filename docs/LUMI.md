# Running on LUMI

**Read this first: nothing in this repository has run on LUMI yet.** The recipe was developed on NVIDIA GPUs
(A100, H200, L40S) with CUDA. The LUMI layer here — `slurm/site/lumi.sh`, `slurm/site/lumi-python`,
`containers/lumi/` — follows LUMI's documentation as we know it and is tested only on a login node without
LUMI (syntax, flags, the wrapper's logic). The first day below is designed to find out quickly what works on
the MI250X: a smoke test on one GCD, then the smallest student end to end.

Sections marked *(check)* are facts from LUMI's documentation at the time of writing that you should confirm
against the current LUMI docs (docs.lumi-supercomputer.eu) and `sinfo`, `lumi-workspaces`, `lumi-allocations`.

## The machine, as this pipeline uses it

| item | value | used for |
|---|---|---|
| LUMI-G node | 4 × AMD MI250X; each MI250X is 2 GCDs; Slurm and PyTorch see **8 GPUs of 64 GB** per node *(check)* | one GCD = one "GPU" everywhere in this repository |
| CPU cores per GCD | 7 (56 of 64 cores usable; one per chiplet is reserved) *(check)* | `--cpus-per-task = 7 × GCDs` (`SUM_CPUS_PER_GCD`) |
| memory per GCD | ~60 GB (about 480 GB usable of 512 GB per node) *(check)* | `--mem = 60 G × GCDs` (`SUM_MEM_PER_GCD`) |
| partitions | `small-g` (shared nodes, billed for what you ask), `standard-g` (whole nodes), `dev-g` (short tests), `small` (CPU) *(check limits with `sinfo`)* | train with 8 GCDs → `standard-g`; fewer GCDs, generation, graders → `small-g`; CPU stages → `small` |
| billing | GPU-hours and CPU-core-hours from your project's allocation; whether a GPU-hour counts a GCD or a whole MI250X (2 GCDs), and how memory or cores above the per-GCD share are charged, is on LUMI's billing page *(check)* | the estimates in docs/PIPELINE.md are in **GCD-hours** |
| file systems | home `/users/<user>` (20 GB), `/project/<project>` (50 GB, persistent), `/scratch/<project>` (large; old files may be cleaned), `/flash/<project>` (fast, billed more) *(check quotas: `lumi-workspaces`)* | everything big on `/scratch`: the repository, `data/`, caches, the container and the venv |
| containers | Singularity on every node; no Docker; LUMI's AI images under `/appl/local/containers/sif-images/` *(check names)* | the runtime image (`SUM_SIF`) |
| network from compute nodes | LUMI documents outbound internet from compute nodes *(check)* | we do not rely on it: models are downloaded once (the `fetch` stages, or on the login node) and every other job runs with `HF_HUB_OFFLINE=1` |

## Layout on LUMI

| what | where (suggested) | size |
|---|---|---|
| this repository (incl. `runs/`, `state/`, `reports/`) | `/scratch/$LUMI_PROJECT/scisum-students` | adapters ~0.5 GB per run, evaluations a few GB in total |
| packaged data (`data/`) | inside the repository, or anywhere with `SCISUM_DATA_ROOT` | the archive's size (docs/DATA.md) + token shards ~1–3 GB per student |
| caches (HF models, Triton, vLLM) | `SUM_CACHE_ROOT` (default `<repo>/.cache`) | 5 students ~50 GB, graders ~115 GB, Qwen3.5-9B anchor ~19 GB (+ its text-only copy ~18 GB) |
| the runtime | `/scratch/$LUMI_PROJECT/containers/` (`SUM_SIF`, `SUM_VENV`) | image ~10–20 GB, venv ~1–2 GB |

Do not put caches or the venv in your home directory (20 GB) or in `/project` (50 GB).

## Your first day

1. **Clone and unpack** on scratch:
   ```bash
   cd /scratch/$LUMI_PROJECT && unzip scisum-students.zip && cd scisum-students   # or git clone + the data archive
   module load cray-python            # Python >= 3.9 for scripts/pipeline.py on the login node
   ```
2. **Runtime** (containers/lumi/README.md): pick an image with ROCm PyTorch and vLLM, then
   `LUMI_PROJECT=... SUM_BASE_SIF=... bash containers/lumi/setup_runtime.sh`.
3. **Site settings:** `cp slurm/site/local.sh.example slurm/site/local.sh`; set `LUMI_PROJECT`, `SUM_SIF`, `SUM_VENV`
   (and `SCISUM_DATA_ROOT`, `SUM_CACHE_ROOT` if not inside the repository).
4. **Check the plumbing without submitting:**
   ```bash
   python3 scripts/pipeline.py --site lumi status
   python3 scripts/pipeline.py --site lumi next --dry-run --student minicpm5-2b    # the exact sbatch lines
   ```
5. **Smoke test on one GCD** (a few minutes once it starts):
   `bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch`, then read `runs/logs/sum-smoke_<job>.log`.
   All required checks must say `"ok": true`. See "Known risks" for what to do if one fails.
6. **Download models** (optional on the login node, otherwise the `fetch` jobs do it):
   ```bash
   SUM_SITE=lumi slurm/site/lumi-python scripts/pipeline.py fetch-model --student minicpm5-2b
   ```
   (`HF_HOME` must point at the cache: source the site first, or set `HF_HOME=$SUM_CACHE_ROOT/hf`.)
7. **The smallest student end to end:** `python3 scripts/pipeline.py --site lumi next --student minicpm5-2b`, then
   `driver` or `next` every few hours. Watch the self-check and the first `train` lines of `metrics.jsonl`
   (docs/MONITORING.md); after the first 1,000 papers, `eta_s` tells you the real SFT time on LUMI — update the
   estimates for the other four before submitting them.
8. **The graders:** the first grader job downloads its model (~55–65 GB) and starts vLLM with tensor parallel 4.
   If it fails, read `runs/logs/sum-grader-<model>_<job>.vllm.log` (see "Known risks").

## Slurm flags the site generates

`slurm/site/lumi.sh` turns a job kind into flags (change them with `SUM_*` variables in `slurm/site/local.sh`):

| kind | default | variables |
|---|---|---|
| train | `--partition=standard-g --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=56 --mem=480G --time=24:00:00` (fewer GCDs → `small-g`) | `SUM_TRAIN_GPUS` (1, 2, 4, 8), `SUM_TRAIN_PARTITION`, `SUM_TRAIN_TIME` |
| gen | `small-g`, 1 GCD, 7 cores, 60 G, 6 h | `SUM_GEN_GPUS`, `SUM_GEN_TIME` |
| grader | `small-g`, 4 GCDs, 28 cores, 240 G, 12 h | `SUM_GRADER_GPUS`, `SUM_GRADER_TIME` |
| cpu / cpu-big / driver | `small`, 8 / 64 / 1 cores | `SUM_CPU_PARTITION`, `SUM_CPU_BIG_CPUS`, `SUM_CPU_BIG_MEM` |

All jobs use one node. Training uses DDP over the GCDs of that node (`torchrun`; 8 papers per optimiser step
whatever the GCD count, by gradient accumulation), so RCCL needs no network settings. Multi-node training is
not supported and not needed (one node finishes a student's SFT in about a day, estimate). For best
CPU-GCD affinity LUMI documents a per-task `--cpu-bind` mask; with one task that starts all ranks it is not
applied — an optimisation to try once things work, not a requirement.

The environment of every LUMI job also sets, per job, `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to a
directory in `/tmp`: MIOpen's database in the home directory is known to cause lock problems on shared file
systems.

## Known risks on the MI250X (ROCm), and what to do

| risk | how you see it | what to try |
|---|---|---|
| No memory-efficient attention kernel at 32k tokens on gfx90a | smoke `sdpa_32k` fails; or training self-check `rc=3` on the longest papers | a newer PyTorch ROCm image (its flash/efficient kernels for MI200); otherwise lower `SCISUM_MAX_SEQ_LEN` (e.g. 16,384) for every student alike — fewer papers fit whole (docs/WHAT_WORKED_ON_C.md: the long window was worth ~0.03 ROUGE-L on long papers) |
| Qwen3.5's linear attention kernels (`flash-linear-attention`, Triton) fail on ROCm | smoke `fla_gated_delta` fails | try the image's Triton version against fla's requirements; without the kernel, transformers' fallback is slow and ran out of memory at long context on CUDA. If it cannot be fixed, `qwen35-4b` is not trainable on this runtime — say so in the report rather than training it at a different length |
| vLLM missing or not built for gfx90a | smoke `vllm_import` fails | a different image (containers/lumi/README.md step 1); all generation and grading need vLLM |
| vLLM's ROCm support for a student architecture (K2-Horizon remote code, Qwen3.5 hybrid, Granite, Llama) | `sft-eval`/`sample` fail at model load | check vLLM's supported-model list for the image's version; `bench_inference.py --engine hf` measures speed without vLLM, but test-split generation needs vLLM |
| LoRA adapters in vLLM on ROCm | generation with an adapter fails or ignores it | the benchmark can serve a merged model (`scripts/analysis/merge_adapter.py`, docs/ANALYSIS.md); compare one output with and without the adapter (a silently ignored adapter scored the base model for days once on CUDA: docs/WHAT_WORKED_ON_C.md) |
| K2-Horizon remote code | errors loading the model with `HF_HUB_OFFLINE=1` | the code is cached under `HF_MODULES_CACHE` by the `fetch` stage; re-run `fetch` with network if it is missing; keep the revision pinned |
| Grader servers: memory, long prompts | vLLM log: out of memory, or prompts over `max_model_len` | `SUM_GRADER_GPUS=8`, or lower `SUM_GRADER_GPU_UTIL`; the claim check sends whole papers (up to ~25k tokens), so keep `SUM_GRADER_MAX_LEN` ≥ 32,768 |
| Reasoning output of qwen38-27b | grading answers parse badly | the server runs with `--reasoning-parser qwen3` so thinking goes to a separate field; check docs/GRADERS.md for what the client expects and adjust `SUM_GRADER_ARGS_QWEN` |
| Numbers differ from the earlier work | the anchors (C, prefcc, teacher) re-graded on LUMI differ from the published ones | expected to some degree: the graders are now self-hosted bf16 weights of the same model names, possibly not the same build or revision as the earlier hosted endpoint. Compare every new student with the **re-graded** anchors, never with the earlier published numbers |
| bitsandbytes | — | not used |

## What is verified, and what is not

| part | status |
|---|---|
| pipeline logic (stage graph, idempotent submission, status, task runner, markers) | unit-tested on a login node (`tests/test_pipeline.py`) |
| shell helpers (resubmission flags, time left, self-check keys, ledger) | unit-tested (`tests/test_slurm_lib.py`) |
| `slurm/site/lumi.sh` flags, `lumi-python` wrapper logic | unit-tested without LUMI (the wrapper's in-container path and its error without an image) |
| the job bodies (training retries, watchdog, self-check, resubmission) | the same code ran the earlier students on a CUDA cluster; the site layer around it is new |
| container image, ROCm kernels, vLLM on MI250X, RCCL, `singularity exec` binds, partitions and billing | **untested** — the smoke test and the first student are the test |

## Note from the benchmark smoke test (CUDA cluster, 2026-11-17)

On the original cluster's shared filesystem, importing vLLM and loading the engine took 1,150 s in the benchmark
smoke test, and 14 minutes in an earlier evaluation job. Python packages read from a network filesystem are slow to
import. On LUMI, keep the whole Python environment inside the container image, or in a squashfs overlay, rather than
in a venv on Lustre. The benchmark records `import_s` separately from `load_s` for this reason.
