"""Checks for the official-flag model (prediction/official_flag.py) and the OURCs page parser."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.data.ourcs_flags import flag_series, parse_history_html  # noqa: E402
from flag_predictor.prediction import official_flag as OF  # noqa: E402

PAGE = """<h2 class="text-left">History</h2>
<table class="table"><thead><tr><th>Status</th><th>Start Date</th><th>End Date</th></tr></thead><tbody>
<tr>
  <td class="text-center">Amber</td>
  <td class="text-center">16/03/26 18:57</td>
  <td class="text-center"></td>
</tr>
<tr>
  <td class="text-center">Red</td>
  <td class="text-center">24/02/26 11:19</td>
  <td class="text-center">16/03/26 18:57</td>
</tr>
<tr>
  <td class="text-center">Dark Blue</td>
  <td class="text-center">05/07/25 10:00</td>
  <td class="text-center">24/02/26 11:19</td>
</tr>
</tbody></table>"""


def params():
    return OF.FlagParams.load(ROOT / 'models' / OF.PARAMS_FILE)


class OurcsPageTests(unittest.TestCase):
    def test_parse_history_uk_time_to_utc(self):
        h = parse_history_html(PAGE)
        self.assertEqual(list(h['flag']), ['dark_blue', 'red', 'amber'])
        # 05/07/25 10:00 is BST -> 09:00 UTC; 24/02/26 11:19 is GMT -> 11:19 UTC
        self.assertEqual(h['start'].iloc[0], pd.Timestamp('2025-07-05 09:00', tz='UTC'))
        self.assertEqual(h['start'].iloc[1], pd.Timestamp('2026-02-24 11:19', tz='UTC'))
        self.assertTrue(pd.isna(h['end'].iloc[2]))
        f = flag_series(h, pd.DatetimeIndex([pd.Timestamp('2026-03-01', tz='UTC'), pd.Timestamp('2020-01-01', tz='UTC')]))
        self.assertEqual(f.iloc[0], 'red')
        self.assertTrue(pd.isna(f.iloc[1]))


class OfficialFlagTests(unittest.TestCase):
    t0 = pd.Timestamp('2025-02-12 06:00', tz='UTC')     # Wednesday, 06:00 GMT

    def test_flag_comes_down_in_daylight_after_the_river_drops(self):
        names = OF.order('isis')
        red = names.index('red')
        hist = np.full(49, 0.60)
        paths = np.tile(np.r_[0.60, np.full(72, 0.45)], (300, 1))     # well into Amber from hour 1
        fl = OF.simulate(params(), 'isis', red, 48.0, hist, paths, self.t0, rng=np.random.default_rng(0))
        moved = (fl < red).any(axis=1)
        self.assertGreater(moved.mean(), 0.8)
        first = (fl < red).argmax(axis=1)[moved]
        hours = pd.date_range(self.t0, periods=73, freq='1h').tz_convert('Europe/London').hour.to_numpy()[first]
        self.assertGreater(np.mean((hours >= 8) & (hours <= 21)), 0.8)

    def test_flag_stays_when_river_stays_in_its_band(self):
        names = OF.order('godstow')
        amber = names.index('amber')
        paths = np.full((300, 73), 0.60)
        fl = OF.simulate(params(), 'godstow', amber, 24.0, np.full(49, 0.60), paths, self.t0)
        self.assertLess((fl != amber).any(axis=1).mean(), 0.15)

    def test_outlook_and_sync(self):
        names = OF.order('isis')
        fl = np.full((10, 73), names.index('red'))
        fl[:5, 30:] = names.index('amber')
        o = OF.outlook('isis', fl, names.index('red'), self.t0)
        self.assertEqual(o['current_flag'], 'red')
        self.assertAlmostEqual(o['windows']['tomorrow']['down'], 0.5)
        self.assertEqual(o['down']['flag'], 'amber')
        self.assertIsNone(o['up'])
        sy = OF.sync_note('isis', 'red', np.r_[np.full(40, 0.60), np.full(9, 0.48)])
        self.assertFalse(sy['in_sync'])
        self.assertEqual(sy['band'], 'amber')
        self.assertEqual(sy['hours_out_of_sync'], 9)
        self.assertFalse(sy['band_above_flag'])

    def test_black_indicator_hysteresis(self):
        idx = pd.date_range('2026-02-10', periods=6, freq='1h', tz='UTC')
        up = OF.black_indicator(pd.Series([0.2, 0.3, 0.45, 0.35, 0.30, 0.25], idx))
        self.assertTrue(up['black'])                  # went over 0.42, not yet below 0.22
        down = OF.black_indicator(pd.Series([0.2, 0.3, 0.45, 0.35, 0.25, 0.20], idx))
        self.assertFalse(down['black'])
        near = OF.black_indicator(pd.Series([0.2, 0.3, 0.36], idx[:3]))
        self.assertTrue(near['near'])

    def test_payload_without_status_or_flags(self):
        OF._STATUS['godstow'] = None
        self.assertEqual(OF.official_flag_payload('godstow', SimpleNamespace()), {})
        self.assertEqual(OF.official_flag_payload('wallingford', SimpleNamespace()), {})
        OF._STATUS.pop('godstow')


if __name__ == '__main__':
    unittest.main()
