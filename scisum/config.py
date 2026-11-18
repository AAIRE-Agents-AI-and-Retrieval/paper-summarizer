"""Settings and the on-disk layout.

Every path and hyper-parameter the pipeline uses lives here, so a run is described by one object
that is written into every manifest and log. Values come from defaults, then ``.env``, then
``SCISUM_*`` environment variables (secrets by their plain name, e.g. ``JUDGE_API_KEY``).

Layout (``data/`` is ``SCISUM_DATA_ROOT``, by default ``<repo>/data``; the rest is under the repository root)::

    data/corpus/index.jsonl.gz   eligible papers: metadata + teacher summary (packaged, read-only; docs/DATA.md)
    data/corpus/splits.json      train/val/test ids (seeded, stratified) + the fixed training order (packaged)
    data/corpus/bodies.sqlite    the parsed GROBID body of every eligible paper (packaged)
    data/<model>/tokens/<split>/part-NNN.{bin,idx.npy,meta.jsonl}   tokenised examples, in training order
    data/<model>/prompts/<split>.jsonl.gz                           prompt text + reference, for generation
    state/build-<model>.json     dataset manifest: counts, token statistics, provenance, code commit
    runs/<run>/              checkpoints (adapter only), logs, metrics.jsonl, run.json
    reports/                 curves (png/svg), results.json, REPORT.md
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "ft-grobid-v1"  # the teacher's prompt version; the student is trained on the same prompt


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCISUM_", env_file=".env", extra="ignore",
                                      protected_namespaces=())

    # ---- data sources -------------------------------------------------------------------------
    # The packaged corpus and everything built from it (token shards, prompts) live here. On a cluster
    # whose home or project space is small (LUMI: /project is 50 GB), point it at scratch.
    data_root: Path = REPO_ROOT / "data"
    repo_root: Path = REPO_ROOT
    teacher_prompt_version: str = PROMPT_VERSION

    # ---- model --------------------------------------------------------------------------------
    # Everything downstream of the tokenizer is per model: a different vocabulary and chat template
    # mean different token shards and different prompt strings, even for the same papers. The
    # splits, the paper order and the teacher summaries are shared, so two students are trained and
    # measured on exactly the same papers in exactly the same order.
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_model_revision: str = ""        # pin a commit for a model whose code we execute
    trust_remote_code: bool = False      # K2-Horizon ships its own modeling code; Qwen3 does not

    # ---- dataset ------------------------------------------------------------------------------
    seed: int = 20261014
    n_train: int = 70_000
    n_val: int = 15_000
    n_test: int = 15_000
    max_seq_len: int = 16_384          # prompt + target + EOS, per example
    gen_reserve_tokens: int = 3_072    # room kept for the summary, at training and at inference alike
    max_target_tokens: int = 6_500     # the teacher's output cap was 6000 tokens
    min_body_chars: int = 2_000        # a GROBID body shorter than this is not a usable full text
    max_body_chars: int = 160_000      # the teacher's input budget; the token budget cuts it further
    section_kinds: tuple[str, ...] = ("body", "captions")  # no references, no appendices

    # ---- training -----------------------------------------------------------------------------
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj")
    lr: float = 1e-4
    warmup_frac: float = 0.01          # warmup-stable-decay: comparable intermediate checkpoints
    decay_frac: float = 0.10
    decay_floor: float = 0.10
    grad_accum: int = 8                # examples per optimiser step, per rank
    max_grad_norm: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    use_liger: bool = False  # liger-kernel has no Python 3.13 wheel; the head-slice loss replaces it
    loss_impl: str = "head_slice"      # head_slice | full  (see scisum.train)
    gradient_checkpointing: bool = True
    log_every_steps: int = 5
    save_every_examples: int = 2_500   # rolling checkpoint, so a killed chunk loses at most this much
    milestones: tuple[int, ...] = (1_000, 5_000, 10_000, 20_000, 50_000, 70_000)

    # ---- evaluation ---------------------------------------------------------------------------
    loss_eval_every: int = 10_000      # examples between val-loss points on the curve
    loss_eval_n: int = 200             # papers in the val-loss subset (teacher-forced, ~2 min)
    curve_eval_n: int = 1_000          # papers generated at every milestone
    faithfulness_n: int = 300          # papers scored with the NLI metric at every milestone
    gen_max_new_tokens: int = 3_000    # the longest reference target is 2 463 tokens; see docs/EVAL.md
    gen_temperature: float = 0.0       # greedy: the reference metric must be reproducible
    eval_tensor_parallel: int = 0      # vLLM GPUs per engine; 0 = every GPU the job was given
    serve_model: str = ""              # a local checkpoint vLLM serves instead of base_model, e.g.
                                       # the text-only copy of a VL model (scripts/make_text_only.py)
    nli_model: str = "microsoft/deberta-large-mnli"
    bertscore_model: str = "roberta-large"
    bertscore_layer: int = 17

    # ---- judge, fact check, claim check (an OpenAI-compatible endpoint: a third model family) --------
    # exclude=True: neither field may appear in model_dump(), because run.json, the reports and the
    # paper tables all serialise the settings. The key did, once, and reached four commits; the
    # endpoint is a private service. Both are set in .env: JUDGE_API_KEY and SCISUM_JUDGE_BASE.
    judge_api_key: str = Field(default="", alias="JUDGE_API_KEY", exclude=True, repr=False)
    judge_base: str = Field(default="", exclude=True, repr=False)
    judge_model: str = "qwen38-27b"      # reasoning model: give it room or `content` comes back empty
    judge_max_tokens: int = 1_024
    judge_thinking: bool = False         # see scisum.judge: reasoning answered only 1 paper in 8
    judge_n: int = 1_000                 # the whole curve subset: ~1.3 s per paper, so it is free
    fact_gen_model: str = "gemma4-31b-it"   # writes the questions: a different family from the answerer/grader
    fact_n_papers: int = 300
    fact_exclude_dir: str = ""           # state/<dir>/papers.json: papers an earlier question set used, left out of this one
    fact_dir: str = "factcheck"            # under state/; a smoke run uses its own so it cannot mix into the real one
    fact_pool: int = 3_000               # papers are drawn from the first 3 000 test papers, which have base-model runs
    fact_paper_chars: int = 90_000       # how much of the paper the question writer reads
    claim_dir: str = "claimcheck"        # under state/ (scisum.claimcheck); a smoke run uses its own
    # ---- reward-selected distillation (scisum.improve) ------------------------------------------
    init_adapter: str = ""               # start a new run from this adapter instead of a fresh one
    train_split: str = "train"           # which token-shard directory the trainer reads (tokens/<name>)
    objective: str = "sft"               # "sft" (scisum.train) or "preference" (scisum.pref_train)
    pref_beta: float = 0.1               # DPO temperature
    pref_sft_weight: float = 1.0         # weight of the chosen summary's likelihood next to the preference term
    pref_len_alpha: float = 0.0          # R-DPO length term per target token (Park et al. 2024); 0 = plain DPO (round 1)
    pref_order_seed: int = 0             # 0 = pairs in file order (rounds 1-2); else a seeded shuffle of the pair order
    imp_dir: str = "improve"             # under state/
    imp_pool: int = 3_000                # papers sampled and scored per round
    imp_max_prompt_tokens: int = 0       # 0 = no cap; the smoke test caps it so the 40 GB slice fits every example
    imp_k: int = 5                       # student samples per paper (the teacher is a further candidate)
    imp_temperature: float = 0.8
    imp_greedy: bool = False             # also take the policy's greedy output (its behaviour at test time) as a candidate
    imp_top_p: float = 0.95
    imp_n_questions: int = 14
    imp_lookahead: int = 300             # the reward stage works this many papers ahead of the sampler, no further
    imp_min_questions: int = 5           # a paper with fewer usable questions keeps its teacher target
    imp_writer: str = "qwen38-27b"       # writes the training questions; the evaluation's writer is gemma4-31b-it
    imp_grader: str = "gemma4-31b-it"    # answers and grades; the evaluation's is qwen38-27b
    imp_lambda: float = 0.15             # weight of imitation (ROUGE-L against the teacher) in the reward
    imp_beta_num: float = 0.30           # penalty for numbers in the summary that the paper does not contain
    imp_gamma_len: float = 1.00          # penalty for length beyond imp_len_free x the teacher's
    imp_len_free: float = 1.20
    imp_pair_gap: float = 0.25           # a preference pair needs the chosen summary to beat the rejected by this much
    imp_pair_rule: str = "lowest"        # the sample a pair rejects: "lowest" reward (round 1) or "closest_length" to the chosen
    imp_margin: float = 0.05             # a student sample must beat the teacher's reward by this to replace it
    imp_claim_weight: float = 0.0        # round 3: reward minus this x the claim check's unsupported-or-contradicted share
    imp_claim_model: str = "gemma4-31b-it"  # verifier of the reward's claim check; the evaluation's is qwen38-27b
    imp_pair_filter: str = ""            # round 4: keep a pair only if this expression holds (scisum.improve.compile_filter)
    imp_pairs_per_paper: int = 1         # round 4: up to this many rejected samples per paper, each a pair with the chosen summary
    judge_enabled: bool = False          # judging runs in the grader job (docs/PIPELINE.md), not in training
    judge_source_chars: int = 9_000      # the excerpt the judge sees; its view of the paper is partial

    @property
    def max_prompt_tokens(self) -> int:
        """The prompt budget, identical during training and inference.

        At inference there is no target to measure, so the body has to be fitted against a fixed
        reserve for the summary; using the same reserve while training keeps the two identical and
        keeps the reference length out of the prompt. Examples whose target needs more than the
        reserve (about 1%) shrink their body further, since the whole target must be learnable.
        """
        return self.max_seq_len - self.gen_reserve_tokens

    # ---- paths --------------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.data_root)

    @property
    def corpus_dir(self) -> Path:
        return Path(self.data_root) / "corpus"

    @property
    def state_dir(self) -> Path:
        return self.repo_root / "state"

    @property
    def runs_dir(self) -> Path:
        return self.repo_root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def model_tag(self) -> str:
        """A short name for the student, used to keep each model's tokenised data apart."""
        return self.base_model.split("/")[-1].lower()

    @property
    def model_data_dir(self) -> Path:
        return self.data_dir / self.model_tag

    def tokens_dir(self, split: str) -> Path:
        return self.model_data_dir / "tokens" / split

    def prompts_path(self, split: str) -> Path:
        return self.model_data_dir / "prompts" / f"{split}.jsonl.gz"

    def hf_kwargs(self) -> dict:
        kw: dict = {"trust_remote_code": self.trust_remote_code}
        if self.base_model_revision:
            kw["revision"] = self.base_model_revision
        return kw

    def run_dir(self, run: str) -> Path:
        return self.runs_dir / run

    def git_commit(self) -> str:
        try:
            out = subprocess.run(["git", "-C", str(self.repo_root), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            dirty = subprocess.run(["git", "-C", str(self.repo_root), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return (out.stdout.strip() or "unknown") + ("-dirty" if dirty else "")
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def provenance(self) -> dict:
        return {"git_commit": self.git_commit(), "base_model": self.base_model, "seed": self.seed,
                "max_seq_len": self.max_seq_len, "prompt_version": self.teacher_prompt_version,
                "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "host": os.uname().nodename}


PRIVATE_SETTINGS = ("judge_api_key", "judge_base")


def scrub(d):
    """A stored settings dump without the private fields. run.json files written before 2026-11-08 still
    carry the endpoint (``judge_base``), and the reports and paper tables copy run.json into git."""
    if isinstance(d, dict):
        return {k: scrub(v) for k, v in d.items() if k not in PRIVATE_SETTINGS}
    if isinstance(d, list):
        return [scrub(v) for v in d]
    return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
