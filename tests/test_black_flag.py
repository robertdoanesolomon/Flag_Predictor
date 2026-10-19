"""Checks for the Black-flag forecast (prediction/black_flag.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.prediction import black_flag as BF  # noqa: E402


class BlackFlagTests(unittest.TestCase):
    t0 = pd.Timestamp('2024-01-04 06:00', tz='UTC')

    def level(self, values):
        return pd.Series(values, pd.date_range(self.t0 - pd.Timedelta(hours=len(values) - 1), self.t0, freq='1h'))

    def test_hysteresis_state(self):
        self.assertTrue(BF.black_state(pd.Series([0.2, 0.43, 0.3, 0.25])))
        self.assertFalse(BF.black_state(pd.Series([0.2, 0.43, 0.3, 0.21])))
        self.assertFalse(BF.black_state(pd.Series([0.2, 0.41, 0.41])))

    def test_rising_river_with_rain_vs_low_river(self):
        rising = self.level(np.linspace(0.25, 0.39, 30))
        wet = np.full((10, 240), 1.5)                      # 1.5 mm/h on every member
        hi = BF.black_forecast(rising, self.t0, 0.9, 1.1, np.full(72, 0.8), wet, ROOT / 'models')
        low = BF.black_forecast(self.level(np.full(30, 0.05)), self.t0, 0.3, 0.4, np.zeros(72), np.zeros((10, 240)), ROOT / 'models')
        self.assertGreater(hi['windows']['tomorrow'], 0.3)
        self.assertLessEqual(hi['windows']['today'], hi['windows']['tomorrow'])
        self.assertIsNotNone(hi['when'])
        self.assertLess(low['windows']['tomorrow'], 0.01)
        self.assertFalse(low['black_now'])

    def test_black_now_falling_comes_back_to_red(self):
        falling = self.level(np.r_[np.full(10, 0.46), np.linspace(0.46, 0.26, 20)])
        o = BF.black_forecast(falling, self.t0, 0.9, 1.0, np.zeros(72), np.zeros((10, 240)), ROOT / 'models')
        self.assertTrue(o['black_now'])
        self.assertGreater(o['windows']['tomorrow'], 0.3)

    def test_missing_godstow_uses_short_model(self):
        rising = self.level(np.linspace(0.25, 0.39, 30))
        o = BF.black_forecast(rising, self.t0, np.nan, np.nan, np.zeros(72), np.zeros((10, 240)), ROOT / 'models')
        self.assertTrue(0.0 <= o['windows']['tomorrow'] <= 1.0)
        self.assertIsNone(BF.black_forecast(self.level([0.3, 0.3]), self.t0, 0.9, 1.0, np.zeros(72), np.zeros((10, 240)), ROOT / 'models'))


if __name__ == '__main__':
    unittest.main()
