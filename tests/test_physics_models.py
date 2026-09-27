"""
Structural guarantees of the physics-redesign hybrid.

These hold for any parameter values, so they are checked on randomly
initialised models with random inputs — no data or trained weights needed.

Run: python -m unittest discover tests   (or pytest)
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from flag_predictor.evaluation import dry_hours, recession_fraction  # noqa: E402
from flag_predictor.models.physics_data import fill_short_gaps  # noqa: E402
from flag_predictor.models.physics_models import (  # noqa: E402
    MonotoneRatingCurve,
    ReservoirHybrid,
    SmoothRatingCurve,
)

B, SEQ, F, H = 16, 100, 12, 240


def random_model(seed: int, **kwargs) -> ReservoirHybrid:
    torch.manual_seed(seed)
    model = ReservoirHybrid(F, d_lo=-0.15, d_hi=1.5, **kwargs)
    with torch.no_grad():
        # Scramble the physics parameters and the rating curve too.
        for p in model.params_raw.values():
            p.add_(torch.randn(()) * 1.5)
        model.split_logits.add_(torch.randn(3))
        model.rating.slope_raw.add_(torch.randn_like(model.rating.slope_raw))
    return model.eval()


def random_inputs(seed: int, rain_scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, SEQ, F, generator=g)
    rain = torch.rand(B, H, generator=g) ** 4 * 5 * rain_scale
    season = torch.cos(torch.linspace(0, 6, H)).expand(B, H)
    rain24 = torch.rand(B, generator=g) * 20 * rain_scale
    rain168 = rain24 + torch.rand(B, generator=g) * 40 * rain_scale
    d0 = torch.rand(B, generator=g) * 1.2 - 0.05
    return x, rain, season, rain24, rain168, d0


VARIANTS = [
    {},
    {'smooth_rating': True},
    {'smooth_rating': True, 'per_sample_params': True},
]


class TestHybridGuarantees(unittest.TestCase):

    def test_rating_pin_is_exact(self):
        """The forecast is anchored at d0 via g(g⁻¹(d0)) = d0."""
        for curve_cls in (MonotoneRatingCurve, SmoothRatingCurve):
            curve = curve_cls(-0.15, 1.5, math.log(1e-4), math.log(3.0))
            with torch.no_grad():
                curve.slope_raw.add_(torch.randn_like(curve.slope_raw))
            d = torch.linspace(-0.3, 2.5, 500)
            with torch.no_grad():
                back = curve(curve.inverse(d))
            self.assertLess(float((back - d).abs().max()), 1e-4, curve_cls.__name__)

    def test_rating_is_increasing(self):
        for curve_cls in (MonotoneRatingCurve, SmoothRatingCurve):
            curve = curve_cls(-0.15, 1.5, math.log(1e-4), math.log(3.0))
            with torch.no_grad():
                curve.slope_raw.add_(torch.randn_like(curve.slope_raw) * 2)
                d = curve(torch.linspace(-15, 5, 4000))
            self.assertTrue(bool((d[1:] > d[:-1]).all()), curve_cls.__name__)

    def test_smooth_rating_has_no_corners(self):
        curve = SmoothRatingCurve(-0.15, 1.5, math.log(1e-4), math.log(3.0))
        with torch.no_grad():
            curve.slope_raw.add_(torch.randn_like(curve.slope_raw))
            z = torch.linspace(-9.0, 1.0, 20001)
            slope = torch.diff(curve(z)) / (z[1] - z[0])
        # Continuous derivative: no jump bigger than a tiny step's worth.
        self.assertLess(float(torch.diff(slope).abs().max()), 1e-2)

    def test_no_rise_without_water(self):
        """No rain in the last 168h and none forecast: never rises."""
        for kw in VARIANTS:
            for seed in range(5):
                model = random_model(seed, **kw)
                x, rain, season, _, _, d0 = random_inputs(seed)
                zero = torch.zeros(B)
                with torch.no_grad():
                    d = model(x, torch.zeros_like(rain), season, zero, zero, d0)
                full = torch.cat([d0[:, None], d], 1)
                self.assertLessEqual(float(torch.diff(full, dim=1).max()), 1e-6, kw)

    def test_more_rain_never_lowers_forecast(self):
        for kw in VARIANTS:
            for seed in range(5):
                model = random_model(seed, **kw)
                x, rain, season, r24, r168, d0 = random_inputs(seed)
                with torch.no_grad():
                    base = model(x, rain, season, r24, r168, d0)
                    more = model(x, rain * 1.5 + 0.1, season, r24, r168, d0)
                self.assertGreaterEqual(float((more - base).min()), -1e-5, kw)

    def test_station_weights_keep_guarantees(self):
        model = random_model(0, smooth_rating=True, n_stations=5)
        with torch.no_grad():
            model.station_logits.add_(torch.randn(5) * 2)
        x, rain, season, r24, r168, d0 = random_inputs(0)
        g = torch.Generator().manual_seed(1)
        st = torch.rand(B, H, 5, generator=g) ** 4 * 5
        ok = torch.rand(B, H, 5, generator=g) > 0.2
        with torch.no_grad():
            base = model(x, rain, season, r24, r168, d0, rain_st=st, rain_st_ok=ok)
            more = model(x, rain, season, r24, r168, d0, rain_st=st * 2, rain_st_ok=ok)
        self.assertGreaterEqual(float((more - base).min()), -1e-5)


class TestEvaluationHelpers(unittest.TestCase):

    def test_dry_hours_counts_past_rain(self):
        past = np.zeros(72)
        past[-1] = 5.0            # a storm right at t0
        dry = dry_hours(past, np.zeros(240))
        self.assertFalse(dry[:71].any())   # still within 72h of the storm
        self.assertTrue(dry[71:].all())

    def test_recession_fraction(self):
        flat = np.zeros(241)
        self.assertEqual(recession_fraction(flat), 0.0)
        fast = -np.arange(241) * 0.01   # 24 cm/day
        self.assertEqual(recession_fraction(fast), 1.0)

    def test_fill_short_gaps(self):
        s = pd.Series([1.0, np.nan, 3.0] + [np.nan] * 30 + [5.0])
        out = fill_short_gaps(s, max_gap=24)
        self.assertAlmostEqual(out.iloc[1], 2.0)
        self.assertTrue(out.iloc[3:33].isna().all())


if __name__ == '__main__':
    unittest.main()
