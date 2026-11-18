"""Warmup-stable-decay: flat where the milestones are, decayed only at the end."""

from scisum.train import wsd_factor


def test_warmup_rises_from_near_zero_to_one():
    assert wsd_factor(0, 1000, 10, 100, 0.1) == 0.1
    assert wsd_factor(9, 1000, 10, 100, 0.1) == 1.0


def test_stable_phase_is_exactly_flat():
    assert all(wsd_factor(st, 1000, 10, 100, 0.1) == 1.0 for st in (10, 200, 500, 899))


def test_decay_reaches_the_floor_and_never_below():
    vals = [wsd_factor(st, 1000, 10, 100, 0.1) for st in range(900, 1000)]
    assert vals[0] == 1.0
    assert all(b <= a for a, b in zip(vals, vals[1:], strict=False))
    assert abs(vals[-1] - 0.1) < 0.01
    assert wsd_factor(5000, 1000, 10, 100, 0.1) >= 0.1


def test_no_decay_phase_is_allowed():
    assert wsd_factor(999, 1000, 10, 0, 0.1) == 1.0
