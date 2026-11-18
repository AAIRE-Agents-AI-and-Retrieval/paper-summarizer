"""Loading Qwen3-4B-Instruct-2507 for LoRA training, and the loss.

Two things here are deliberate and worth reading before changing them.

**The loss is computed only where there are labels.** The prompt is 85% of a typical example
(a 13k-token paper, a 2k-token summary) and contributes nothing to the loss, so materialising
``[T, 151936]`` logits for the whole sequence wastes about 20 GB at 16k tokens and ~10% of the
forward FLOPs. :func:`head_slice_loss` runs the transformer, slices the hidden states at the
positions that predict a label, and applies the LM head only there. ``tests/test_loss.py`` asserts
it returns the same number as the standard path (``loss_impl="full"``) on a small random model.
(liger-kernel's fused cross-entropy would be the alternative, but it has no Python 3.13 wheel.)

**Attention is PyTorch SDPA, not FlashAttention-2.** With one sequence per micro-batch there is no
padding to skip, SDPA dispatches to its flash kernel for causal GQA, and it needs no compiled
dependency — flash-attn has no cp313 wheel either and building it on a compute node is an hour of
nvcc per driver change.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from scisum.config import Settings

log = logging.getLogger(__name__)


def load_tokenizer(s: Settings):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(s.base_model, **s.hf_kwargs())


def load_model(s: Settings, *, train: bool = True, adapter: str | None = None, device: str = "cuda"):
    """The base model in bf16 with a fresh LoRA adapter (or ``adapter`` loaded from disk)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    # `use_cache` belongs on the config, not in the constructor: from_pretrained forwards unknown
    # keywords to the model's __init__, and Qwen3.5 rejects it there ("unexpected keyword argument
    # use_cache"), while Qwen3 and K2-Horizon quietly absorbed it. Setting it afterwards works for
    # every architecture, and nested text configs need it too.
    model = AutoModelForCausalLM.from_pretrained(
        s.base_model, dtype=torch.bfloat16, attn_implementation="sdpa", **s.hf_kwargs())
    model.config.use_cache = False
    for nested in ("text_config", "language_model_config"):
        sub = getattr(model.config, nested, None)
        if sub is not None:
            sub.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    else:
        model = get_peft_model(model, LoraConfig(
            r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
            target_modules=list(s.lora_targets), bias="none", task_type="CAUSAL_LM"))
    if train and s.gradient_checkpointing:
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    model.train(train)
    if s.use_liger:
        _apply_liger(model)
    return model


def _apply_liger(model) -> None:  # pragma: no cover - optional dependency
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(model=model.get_base_model())
        log.info("liger kernels applied")
    except ImportError:
        log.warning("use_liger is set but liger-kernel is not installed; using the head-slice loss")


def trainable_summary(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable,
            "trainable_pct": round(100 * trainable / max(total, 1), 3),
            "dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})}


def _autocast(input_ids: torch.Tensor):
    """bf16 autocast on GPU.

    The base weights are already bf16, but PEFT keeps the adapter in fp32 (which is what makes
    AdamW stable), and without autocast every LoRA branch would run in fp32 and upcast its
    activations. Autocast gives the standard mixed-precision arrangement: fp32 master weights,
    bf16 matmuls. On CPU (the tests) it is disabled, so the comparison stays exact.
    """
    return torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16,
                          enabled=input_ids.is_cuda)


def head_slice_loss(model, input_ids: torch.Tensor, prompt_len: int) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy over the target tokens only, and how many there were.

    ``input_ids`` is ``(1, T)``; ``input_ids[0, prompt_len:]`` are the labels, each predicted from
    the hidden state one position earlier.
    """
    base = model.get_base_model()
    labels = input_ids[:, prompt_len:]
    with _autocast(input_ids):
        hidden = base.model(input_ids=input_ids).last_hidden_state
        logits = base.lm_head(hidden[:, prompt_len - 1:-1, :])
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss, int(labels.numel())


def full_loss(model, input_ids: torch.Tensor, prompt_len: int) -> tuple[torch.Tensor, int]:
    """The standard path: labels masked with -100 over the prompt, loss from the model itself."""
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    with _autocast(input_ids):
        out = model(input_ids=input_ids, labels=labels)
    return out.loss.float(), int((labels != -100).sum())


def loss_fn(name: str):
    return {"head_slice": head_slice_loss, "full": full_loss}[name]


def seq_logp(model, input_ids: torch.Tensor, prompt_len: int) -> tuple[torch.Tensor, int]:
    """Summed log-probability of the target tokens (with gradient), and how many there were.

    The same head-slice computation as ``head_slice_loss``, but the sum and not the mean: a
    preference objective compares whole summaries.
    """
    base = model.get_base_model()
    labels = input_ids[:, prompt_len:]
    with _autocast(input_ids):
        hidden = base.model(input_ids=input_ids).last_hidden_state
        logits = base.lm_head(hidden[:, prompt_len - 1:-1, :])
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
    return -nll, int(labels.numel())


def preference_loss(model, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor, prompt_len: int,
                    ref_chosen: float, ref_rejected: float, beta: float,
                    sft_weight: float, len_alpha: float = 0.0) -> tuple[torch.Tensor, dict]:
    """DPO against a fixed reference plus ``sft_weight`` times the chosen summary's token-mean NLL.

    ``ref_*`` are the summed log-probabilities of the two summaries under the starting adapter
    (computed once, see ``scisum.pref_train``). Both sequences share the prompt, so both graphs are
    built before the single backward pass.

    ``len_alpha`` is the length term of R-DPO (Park et al. 2024, "Disentangling length from quality in
    direct preference optimization", eq. 9): ``len_alpha * (|chosen| - |rejected|)`` in target tokens is
    added to the logit, so a pair whose chosen summary is longer already counts as partly won and pushes
    the policy less, and one whose chosen summary is shorter pushes it more. The policy's implicit
    reward then excludes length. ``margin`` stays the policy's own term; ``logit`` includes the length.
    """
    lp_c, n_c = seq_logp(model, chosen_ids, prompt_len)
    lp_r, n_r = seq_logp(model, rejected_ids, prompt_len)
    margin = beta * ((lp_c - ref_chosen) - (lp_r - ref_rejected))
    logit = margin + len_alpha * (n_c - n_r)
    dpo = -F.logsigmoid(logit)
    nll = -lp_c / max(n_c, 1)
    return dpo + sft_weight * nll, {"dpo": float(dpo.detach()), "nll": float(nll.detach()),
                                    "margin": float(margin.detach()), "logit": float(logit.detach()),
                                    "acc": float(margin.detach() > 0)}


@torch.no_grad()
def preference_coefficients(model, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor, prompt_len: int,
                            ref_chosen: float, ref_rejected: float, beta: float, sft_weight: float,
                            len_alpha: float = 0.0) -> dict:
    """What a preference step needs to backpropagate the two summaries one at a time.

    ``preference_loss`` builds both sequences' graphs before one backward pass, so its memory is that of
    two long sequences, which needs an 80 GB card. The gradient of the same loss is a coefficient times
    the gradient of each summary's log-probability,

        dL/d lp_chosen   = -(beta * sigmoid(-logit) + sft_weight / n_chosen_tokens)
        dL/d lp_rejected = +beta * sigmoid(-logit)

    (``logit`` = margin + len_alpha * (n_chosen - n_rejected); the length term is a constant for a pair)
    and the coefficients depend only on the two log-probabilities, which can be computed without
    gradients. Backpropagating ``coef * lp`` for each sequence in turn gives the same gradient with the
    memory of one sequence, for two extra forward passes (about a quarter more compute).
    """
    lp_c, n_c = seq_logp(model, chosen_ids, prompt_len)
    lp_r, n_r = seq_logp(model, rejected_ids, prompt_len)
    margin = beta * ((lp_c - ref_chosen) - (lp_r - ref_rejected))
    logit = margin + len_alpha * (n_c - n_r)
    s = torch.sigmoid(-logit)
    nll = -lp_c / max(n_c, 1)
    return {"coef_chosen": float(-(beta * s + sft_weight / max(n_c, 1))), "coef_rejected": float(beta * s),
            "dpo": float(F.softplus(-logit)), "nll": float(nll), "margin": float(margin), "logit": float(logit),
            "acc": float(margin > 0), "len_diff": int(n_c - n_r)}
