"""Structural checks for the shared catchment model (models/shared_catchment.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from flag_predictor.models.shared_catchment import (FEEDS, GAUGES, STRETCHES, SUBCATCHMENTS,  # noqa: E402
                                                    SharedCatchmentHybrid, subcatchment_rain)


def model(seed=0):
    torch.manual_seed(seed)
    m = SharedCatchmentHybrid(12, d_lo=[-0.2, -0.2, -0.2], d_hi=[1.2, 1.2, 1.5], hidden=16)
    m.eval()
    return m


def inputs(B=4, H=240, rain_scale=1.0, recent=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    K = len(SUBCATCHMENTS)
    x = torch.randn(B, 100, 12, generator=g)
    rain = torch.rand(B, H, K, generator=g) * 0.6 * rain_scale
    season = torch.zeros(B, H)
    r24 = torch.rand(B, K, generator=g) * 5 * recent
    r168 = torch.rand(B, K, generator=g) * 20 * recent
    d0 = torch.tensor([[0.05, -0.04, 0.3]] * B)
    return x, rain, season, r24, r168, d0


class SharedCatchmentTests(unittest.TestCase):
    def test_starts_at_each_stretch_observation(self):
        m = model()
        x, rain, season, r24, r168, d0 = inputs(rain_scale=0.0, recent=0.0)
        with torch.no_grad():
            d = m(x, rain, season, r24, r168, d0)
        self.assertEqual(tuple(d.shape), (4, 240, 3))
        # The channels are pinned to each stretch's own reading: with no water in transit the first
        # hour is that reading (or a recession just below it).
        np.testing.assert_allclose(d[:, 0].numpy(), d0.numpy(), atol=0.02)
        self.assertLessEqual(float((d[:, 0] - d0).max()), 1e-5)

    def test_no_rise_without_water(self):
        m = model()
        x, rain, season, r24, r168, d0 = inputs(rain_scale=0.0, recent=0.0)
        with torch.no_grad():
            d = m(x, rain, season, r24, r168, d0)
        full = torch.cat([d0[:, None, :], d], 1)
        self.assertLessEqual(float((full[:, 1:] - full[:, :-1]).max()), 1e-5)

    def test_more_rain_never_lowers_any_stretch(self):
        m = model()
        x, rain, season, r24, r168, d0 = inputs()
        with torch.no_grad():
            d1 = m(x, rain, season, r24, r168, d0)
            d2 = m(x, rain * 1.5, season, r24, r168, d0)
        self.assertGreaterEqual(float((d2 - d1).min()), -1e-5)

    def test_godstow_ignores_cherwell_and_lower_rain(self):
        m = model()
        x, rain, season, r24, r168, d0 = inputs()
        more = rain.clone()
        more[:, :, 1:] *= 3.0                                  # Cherwell and lower sub-catchments only
        with torch.no_grad():
            a = m(x, rain, season, r24, r168, d0)
            b = m(x, more, season, r24, r168, d0)
        np.testing.assert_allclose(a[:, :, 0].numpy(), b[:, :, 0].numpy(), atol=1e-6)    # Godstow unchanged
        self.assertGreater(float((b[:, :, 1] - a[:, :, 1]).max()), 0.0)                  # the Isis rises
        self.assertGreater(float((b[:, :, 2] - a[:, :, 2]).max()), 0.0)                  # Wallingford rises

    def test_feeds_are_nested_downstream(self):
        self.assertEqual(list(STRETCHES), ['godstow', 'isis', 'wallingford'])
        self.assertTrue((FEEDS[1] >= FEEDS[0]).all() and (FEEDS[2] >= FEEDS[1]).all())
        all_g = [g for k in SUBCATCHMENTS for g in GAUGES[k]]
        self.assertEqual(len(all_g), len(set(all_g)))         # each gauge in one sub-catchment

    def test_subcatchment_rain_falls_back_to_all_gauges(self):
        idx = pd.date_range('2026-01-01', periods=3, freq='1h')
        rain = pd.DataFrame({'Worsham': [1.0, 2.0, np.nan], 'Abingdon': [3.0, np.nan, np.nan]}, index=idx)
        sr = subcatchment_rain(rain)
        self.assertEqual(sr.loc[idx[0], 'upper'], 1.0)
        self.assertEqual(sr.loc[idx[0], 'cherwell'], 2.0)      # no Cherwell gauge: mean of those reporting
        self.assertEqual(sr.loc[idx[0], 'lower'], 3.0)
        self.assertEqual(sr.loc[idx[2], 'upper'], 0.0)         # nothing reporting


if __name__ == '__main__':
    unittest.main()
