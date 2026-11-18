import numpy as np
import pandas as pd

from flag_predictor.prediction.reading_noise import MIN_POOL, add_reading_noise, band_samples, residual_pool


def test_hour_zero_untouched_and_spread_added():
    rng = np.random.default_rng(1)
    pool = rng.normal(0, 0.01, 1000)
    samples = np.zeros((2000, 49))
    out = add_reading_noise(samples, pool)
    assert np.all(out[:, 0] == 0)
    # r_h - r_0: about sqrt(2) times the wobble at every lead
    assert 0.012 < out[:, 1:].std() < 0.016


def test_small_pool_does_nothing():
    samples = np.ones((10, 5))
    assert np.array_equal(add_reading_noise(samples, np.zeros(MIN_POOL - 1)), samples)


def test_pool_uses_only_residuals_known_at_t0():
    idx = pd.date_range('2026-10-01', periods=24 * 40, freq='1h')
    s = pd.Series(np.random.default_rng(0).normal(0, 0.01, len(idx)), index=idx)
    t0 = pd.Timestamp('2026-10-30')
    pool = residual_pool(s, t0)
    assert 0 < len(pool) <= 28 * 24
    s2 = s.copy()
    s2.loc[t0 + pd.Timedelta(hours=1):] += 0.05     # readings after t0 cannot matter
    assert np.allclose(pool, residual_pool(s2, t0))


def test_godstow_band_unchanged(tmp_path):
    samples = np.zeros((5, 3))
    assert band_samples('godstow', samples, pd.Timestamp('2026-11-18'), tmp_path) is samples
