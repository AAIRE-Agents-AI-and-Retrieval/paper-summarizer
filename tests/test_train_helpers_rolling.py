"""Rolling checkpoints: a short continuation run must checkpoint too."""

from scisum.train import rolling_every


def test_a_short_run_checkpoints_at_least_eight_times_and_on_step_boundaries():
    assert rolling_every(2_500, 1_520, 8) == 184          # 1520 // 8 = 190, rounded down to whole steps
    assert rolling_every(2_500, 1_073, 8) == 128
    assert rolling_every(2_500, 70_000, 8) == 2_500       # a long run keeps the setting exactly
    assert rolling_every(2_500, 20, 8) == 8               # never less than one step


def test_the_condition_fires_once_per_interval():
    every, step = rolling_every(2_500, 1_520, 8), 8
    hits = [e for e in range(step, 1_521, step) if e % every < step]
    assert len(hits) == 1_520 // every and all(h % every == 0 for h in hits)
