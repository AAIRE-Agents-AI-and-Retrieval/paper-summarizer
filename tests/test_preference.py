"""The preference objective, on a tiny model: the arithmetic, the gradient path, and that it learns."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from scisum.improve import pick_pair  # noqa: E402


def _tiny_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config

    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM")).eval()


def _pair(prompt_len=24):
    torch.manual_seed(1)
    prompt = torch.randint(0, 512, (1, prompt_len))
    chosen = torch.cat([prompt, torch.randint(0, 512, (1, 20))], dim=1)
    rejected = torch.cat([prompt, torch.randint(0, 512, (1, 26))], dim=1)
    return chosen, rejected, prompt_len


def test_seq_logp_is_minus_the_summed_loss():
    from scisum.model import head_slice_loss, seq_logp

    model = _tiny_model()
    chosen, _, plen = _pair()
    with torch.no_grad():
        lp, n = seq_logp(model, chosen, plen)
        mean, n2 = head_slice_loss(model, chosen, plen)
    assert n == n2 == 20
    assert float(lp) == pytest.approx(-float(mean) * n, rel=1e-5)


def test_at_the_reference_the_preference_term_is_log_two_and_the_margin_is_zero():
    from scisum.model import preference_loss, seq_logp

    model = _tiny_model()
    chosen, rejected, plen = _pair()
    with torch.no_grad():
        ref_c, _ = seq_logp(model, chosen, plen)
        ref_r, _ = seq_logp(model, rejected, plen)
    loss, aux = preference_loss(model, chosen, rejected, plen, float(ref_c), float(ref_r), beta=0.1, sft_weight=0.0)
    assert aux["dpo"] == pytest.approx(math.log(2), abs=1e-5) and aux["margin"] == pytest.approx(0.0, abs=1e-5)
    assert float(loss) == pytest.approx(math.log(2), abs=1e-5)
    both, aux2 = preference_loss(model, chosen, rejected, plen, float(ref_c), float(ref_r), beta=0.1, sft_weight=1.0)
    assert float(both) == pytest.approx(math.log(2) + aux2["nll"], abs=1e-5)       # the likelihood term is added as is


def test_gradients_reach_only_the_adapter_through_both_sequences():
    from scisum.model import preference_loss

    model = _tiny_model()
    model.train()
    chosen, rejected, plen = _pair()
    loss, _ = preference_loss(model, chosen, rejected, plen, -50.0, -60.0, beta=0.1, sft_weight=1.0)
    loss.backward()
    grads = {n: p.grad is not None for n, p in model.named_parameters() if p.requires_grad}
    assert grads and all(grads.values()) and all("lora" in n for n in grads)


def test_a_few_steps_make_the_chosen_summary_more_likely_than_the_rejected_one():
    from scisum.model import preference_loss, seq_logp

    model = _tiny_model()
    model.train()
    chosen, rejected, plen = _pair()
    with torch.no_grad():
        ref_c, _ = seq_logp(model, chosen, plen)
        ref_r, _ = seq_logp(model, rejected, plen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, aux = preference_loss(model, chosen, rejected, plen, float(ref_c), float(ref_r), beta=0.5, sft_weight=0.1)
        first = first if first is not None else aux
        loss.backward()
        opt.step()
    assert aux["margin"] > first["margin"] + 0.5 and aux["acc"] == 1.0
    with torch.no_grad():
        lc, _ = seq_logp(model, chosen, plen)
        lr, _ = seq_logp(model, rejected, plen)
    assert float(lc) - float(ref_c) > float(lr) - float(ref_r)            # relative to the reference, chosen gained more


def test_pick_pair_rejects_only_a_clearly_worse_student_sample():
    # [teacher, s1, s2, s3]
    assert pick_pair([0.9, 0.5, 0.2, 0.6], chosen=0, gap=0.25) == 2           # the worst sample, 0.7 below
    assert pick_pair([0.9, 0.8, 0.7, 0.75], chosen=0, gap=0.25) is None       # nothing clearly worse
    assert pick_pair([0.4, 0.9, 0.2, 0.5], chosen=1, gap=0.25) == 2           # the teacher is never rejected
    assert pick_pair([0.9, None, 0.4, None], chosen=0, gap=0.25) == 2         # unscored samples are skipped
    assert pick_pair([None, 0.5, 0.1], chosen=0, gap=0.25) is None            # no reward for the chosen one
    assert pick_pair([0.9, 0.2], chosen=1, gap=0.25) is None                  # the only sample is the chosen one


def test_pick_pair_closest_length_matches_the_chosen_length_among_clearly_worse_samples():
    # [teacher, s1, s2, s3, s4]; lengths relative to the teacher
    r = [0.9, 0.3, 0.5, 0.85, 0.6]
    ln = [1.0, 0.80, 1.05, 1.0, 0.97]
    assert pick_pair(r, 0, 0.25, ln, "lowest") == 1                       # round 1: the worst, also the shortest
    assert pick_pair(r, 0, 0.25, ln, "closest_length") == 4               # 0.97 is closest to 1.0; s3 is not behind by 0.25
    assert pick_pair([0.9, 0.5, 0.3], 0, 0.25, [1.0, 1.1, 1.1], "closest_length") == 2       # a tie: the lower reward
    assert pick_pair([0.9, 0.8], 0, 0.25, [1.0, 1.0], "closest_length") is None
    with pytest.raises(ValueError):
        pick_pair(r, 0, 0.25, None, "closest_length")
    with pytest.raises(ValueError):
        pick_pair(r, 0, 0.25, ln, "longest")


def test_the_length_term_shifts_the_logit_by_alpha_times_the_length_difference():
    from scisum.model import preference_loss, seq_logp

    model = _tiny_model()
    chosen, rejected, plen = _pair()                    # chosen 20 target tokens, rejected 26
    with torch.no_grad():
        ref_c, _ = seq_logp(model, chosen, plen)
        ref_r, _ = seq_logp(model, rejected, plen)
    _, aux = preference_loss(model, chosen, rejected, plen, float(ref_c), float(ref_r), beta=0.1, sft_weight=0.0, len_alpha=0.05)
    assert aux["margin"] == pytest.approx(0.0, abs=1e-5) and aux["logit"] == pytest.approx(0.05 * (20 - 26), abs=1e-5)
    # a shorter chosen summary starts behind, so the pair pushes harder than under plain DPO
    assert aux["dpo"] == pytest.approx(math.log(1 + math.exp(0.3)), abs=1e-5) and aux["dpo"] > math.log(2)


@pytest.mark.parametrize("alpha", [0.0, 0.02])
def test_backpropagating_the_two_summaries_one_at_a_time_gives_the_same_gradient(alpha):
    """The memory-lean step (one sequence's graph at a time) must be the joint loss's gradient, not an approximation."""
    from scisum.model import preference_coefficients, preference_loss, seq_logp

    model = _tiny_model()
    model.train()
    chosen, rejected, plen = _pair()
    ref_c, ref_r, beta, w = -90.0, -80.0, 0.3, 0.7           # not the model's own values, so the margin is not zero
    loss, aux = preference_loss(model, chosen, rejected, plen, ref_c, ref_r, beta=beta, sft_weight=w, len_alpha=alpha)
    loss.backward()
    joint = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any(float(g.abs().sum()) > 0 for g in joint.values())
    model.zero_grad()

    co = preference_coefficients(model, chosen, rejected, plen, ref_c, ref_r, beta, w, alpha)
    assert all(p.grad is None for p in model.parameters()), "computing the coefficients must not build a graph"
    lp_c, _ = seq_logp(model, chosen, plen)
    (co["coef_chosen"] * lp_c).backward()
    lp_r, _ = seq_logp(model, rejected, plen)
    (co["coef_rejected"] * lp_r).backward()
    for n, g in joint.items():
        torch.testing.assert_close(dict(model.named_parameters())[n].grad, g, rtol=1e-4, atol=1e-7)
    assert co["dpo"] == pytest.approx(aux["dpo"], rel=1e-5) and co["nll"] == pytest.approx(aux["nll"], rel=1e-5)
    assert co["margin"] == pytest.approx(aux["margin"], rel=1e-5) and co["logit"] == pytest.approx(aux["logit"], rel=1e-5)
    assert co["len_diff"] == 20 - 26


class _FakeShards:
    """Pairs as consecutive examples, chosen first, like the real shards."""

    def __init__(self, n_pairs: int, prompt_len: int = 24):
        from types import SimpleNamespace

        import numpy as np

        g = torch.Generator().manual_seed(2)
        self.ex = [SimpleNamespace(tokens=np.asarray(torch.randint(0, 512, (prompt_len + 10 + i % 5,), generator=g)),
                                   prompt_len=prompt_len) for i in range(2 * n_pairs)]
        self.calls = 0

    def get(self, i):
        self.calls += 1
        return self.ex[i]


def test_the_reference_pass_computes_only_what_is_asked_and_resumes_from_its_saved_progress(tmp_path):
    import numpy as np

    from scisum.model import seq_logp
    from scisum.pref_train import reference_logps

    model, shards = _tiny_model(), _FakeShards(10)
    partial, beat = tmp_path / "ref.partial.npy", tmp_path / "metrics.jsonl"
    out = reference_logps(model, shards, [1, 4, 7], 10, "cpu", 1, 0, partial=partial, heartbeat=beat, chunk=2)
    assert shards.calls == 6 and beat.exists() and partial.exists()
    assert not np.isnan(out[[1, 4, 7]]).any() and np.isnan(out[[0, 2, 3, 5, 6, 8, 9]]).all()
    with torch.no_grad():
        ex = shards.ex[2 * 4 + 1]
        want, _ = seq_logp(model, torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0), ex.prompt_len)
    assert out[4, 1] == pytest.approx(float(want), rel=1e-6)
    # a restart reads the saved pairs back and computes only the new one
    shards.calls = 0
    again = reference_logps(model, shards, [1, 4, 7, 9], 10, "cpu", 1, 0, partial=partial, heartbeat=beat, chunk=2)
    assert shards.calls == 2
    np.testing.assert_array_equal(again[[1, 4, 7]], out[[1, 4, 7]])
    assert not np.isnan(again[9]).any()


def test_a_stored_reference_is_accepted_only_if_the_starting_adapter_reproduces_it(tmp_path):
    from scisum.pref_train import check_reference, reference_logps

    model, shards = _tiny_model(), _FakeShards(4)
    ref = reference_logps(model, shards, [0, 1, 2, 3], 4, "cpu", 1, 0)
    dc, dr = check_reference(model, shards, ref, 2, "cpu")
    assert dc < 1e-3 and dr < 1e-3
    wrong = ref.copy()
    wrong[2, 1] -= 25.0                 # another pair's or another adapter's value
    with pytest.raises(RuntimeError, match="belongs to other pairs"):
        check_reference(model, shards, wrong, 2, "cpu")
