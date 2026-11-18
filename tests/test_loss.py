"""The head-slice loss must equal the standard masked-label loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tiny_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config

    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    return model.eval()


def test_head_slice_matches_full_loss():
    from scisum.model import full_loss, head_slice_loss

    model = _tiny_model()
    ids = torch.randint(0, 512, (1, 64))
    prompt_len = 40
    with torch.no_grad():
        a, na = head_slice_loss(model, ids, prompt_len)
        b, nb = full_loss(model, ids, prompt_len)
    assert na == 64 - prompt_len
    assert nb == na
    assert torch.allclose(a, b, atol=1e-5), (a.item(), b.item())


def test_head_slice_gradients_flow_only_into_the_adapter():
    from scisum.model import head_slice_loss

    model = _tiny_model()
    model.train()
    ids = torch.randint(0, 512, (1, 48))
    loss, _ = head_slice_loss(model, ids, 30)
    loss.backward()
    grads = {n: (p.grad is not None) for n, p in model.named_parameters() if p.requires_grad}
    assert grads and all(grads.values())
    assert all("lora" in n for n in grads)
