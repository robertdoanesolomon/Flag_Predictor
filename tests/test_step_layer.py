"""Checks for the lock-keeper step layer (src/flag_predictor/prediction/step_layer.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.config import get_flag_thresholds  # noqa: E402
from flag_predictor.prediction import step_layer as SL  # noqa: E402


def params(loc='isis'):
    return SL.StepParams.load(SL.params_path(ROOT / 'models', loc))


class StepLayerTests(unittest.TestCase):
    t0 = pd.Timestamp('2025-02-09 06:00', tz='UTC')   # a Sunday morning, GMT

    def test_triggers_see_a_jump_in_the_last_hour(self):
        trig, own = SL.issue_triggers('isis', np.r_[np.full(40, 0.3), np.full(9, 0.25)],
                                      {'godstow': np.r_[np.full(48, 0.5), 0.56]})
        self.assertEqual((trig[1], trig[-1], own[1], own[-1]), (1, 0, 0, 0))

    def test_hourly_jumps_finds_a_held_step(self):
        x = np.r_[np.full(10, 0.30), np.full(10, 0.40)]
        j = SL.hourly_jumps(x, 0.03)
        self.assertEqual(int(np.argmax(j)), 10)
        self.assertAlmostEqual(j[10], 0.10, places=6)
        self.assertEqual(int((j != 0).sum()), 1)

    def test_shape_start_and_tail(self):
        p = params()
        rng = np.random.default_rng(1)
        y = 0.30 + np.cumsum(rng.normal(0, 0.002, (40, 241)), axis=1)
        y[:, 0] = 0.30
        z = SL.simulate(p, np.full(49, 0.30), y, np.zeros(24), np.zeros(240), self.t0, rng=rng)
        self.assertEqual(z.shape, y.shape)
        np.testing.assert_allclose(z[:, 0], 0.30)
        np.testing.assert_allclose(z[:, SL.STEP_HORIZON_H + 1:], y[:, SL.STEP_HORIZON_H + 1:])
        self.assertTrue(np.isfinite(z).all())

    def test_flat_outcome_stays_close_to_flat(self):
        p = params()
        y = np.full((400, 241), 0.25)
        z = SL.simulate(p, np.full(49, 0.25), y, np.zeros(24), np.zeros(240), self.t0)
        # Keepers occasionally move the gates anyway, but most outcomes stay put and the average does not drift.
        self.assertGreater(np.mean(np.abs(z[:, 48] - 0.25) < 0.01), 0.7)
        self.assertLess(abs(np.mean(z[:, 48]) - 0.25), 0.01)

    def test_rise_happens_in_the_daytime_and_arrives(self):
        p = params()
        # Smooth rise of 25 cm spread over the next 36 hours.
        y = np.tile(0.30 + 0.25 * np.clip(np.arange(241) / 36, 0, 1), (400, 1))
        z = SL.simulate(p, np.full(49, 0.30), y, np.zeros(24), np.full(240, 0.3), self.t0,
                        rng=np.random.default_rng(0))
        dz = np.diff(z[:, :73], axis=1)
        local_h = pd.date_range(self.t0, periods=73, freq='1h').tz_convert('Europe/London').hour.to_numpy()[1:]
        big = dz > 0.03
        share_day = big[:, (local_h >= 8) & (local_h < 17)].sum() / max(big.sum(), 1)
        self.assertGreater(share_day, 0.5)   # vs 0.375 if jumps were spread evenly over the day
        # By 72 h nearly all of the rise has happened.
        self.assertGreater(np.median(z[:, 72]), 0.50)

    def test_outlook_isis(self):
        th = get_flag_thresholds('isis')
        amber_lo = th['amber'][0]
        y = np.tile(np.r_[0.40, np.full(240, amber_lo + 0.08)], (100, 1))
        out = SL.flag_outlook(y, th, self.t0, 0.40)
        self.assertEqual(out['current_flag'], 'dark_blue')
        self.assertEqual(out['up']['flag'], 'amber')
        self.assertEqual(out['down']['flag'], 'light_blue')
        self.assertAlmostEqual(out['windows']['today']['up'], 1.0)
        self.assertAlmostEqual(out['windows']['tomorrow']['down'], 0.0)

    def test_outlook_godstow_skips_placeholder_bands_and_wallingford_has_none(self):
        th = get_flag_thresholds('godstow')
        y = np.full((10, 241), 0.30)
        out = SL.flag_outlook(y, th, self.t0, 0.30)
        self.assertEqual(out['current_flag'], 'green')
        self.assertEqual(out['up']['flag'], 'amber')
        self.assertIsNone(out['down'])
        out = SL.apply_live('wallingford', np.full(49, 0.30), np.zeros(24), y[:2], np.zeros((2, 240)), None,
                            self.t0, get_flag_thresholds('wallingford'), ROOT / 'models')
        self.assertIsNone(out.outlook)
        self.assertIsNone(out.total)

    def test_day_windows_follow_uk_time(self):
        # 22:00 UTC on 1 July = 23:00 BST: no forecast hour left today, then 24 hours tomorrow.
        w = SL.day_windows(pd.Timestamp('2025-07-01 22:00', tz='UTC'), 96)
        self.assertEqual(w['today'], 0)
        self.assertEqual(w['tomorrow'], 24)
        w = SL.day_windows(pd.Timestamp('2025-01-01 22:00', tz='UTC'), 96)   # 22:00 GMT: 23:00 still today
        self.assertEqual(w['today'], 1)
        self.assertEqual(w['tomorrow'], 25)

    def test_apply_live_member_major_rain(self):
        th = get_flag_thresholds('isis')
        members = np.tile(np.linspace(0.30, 0.45, 241), (5, 1))
        total = np.repeat(members, 10, axis=0)
        out = SL.apply_live('isis', np.full(49, 0.30), np.zeros(24), members, np.zeros((5, 240)), total,
                            self.t0, th, ROOT / 'models',
                            neighbour_hists={'godstow': np.r_[np.full(46, 0.40), np.full(3, 0.48)]})
        self.assertEqual(out.triggers['neighbour_up'], 1)
        self.assertEqual(out.triggers['own_up'], 0)
        self.assertEqual(out.members.shape, members.shape)
        self.assertEqual(out.total.shape, total.shape)
        self.assertIn('windows', out.outlook)


if __name__ == '__main__':
    unittest.main()
