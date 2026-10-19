"""Checks for the flag-alert rules (push/alerts.py)."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'push'))

from alerts import decide  # noqa: E402

NOW = datetime(2026, 2, 12, 7, 0, tzinfo=timezone.utc)


def bundle(flag='red', down_today=0.0, down_tom=0.0, stale=False, black=None, age_h=0.5):
    loc = {'stale': stale,
           'official_flag': {'flag': flag, 'label': flag.title()},
           'official_outlook': {'current_flag': flag,
                                'windows': {'today': {'up': 0.0, 'down': down_today}, 'tomorrow': {'up': 0.0, 'down': down_tom}},
                                'up': None,
                                'down': {'flag': 'amber', 'when': {'median_time': '2026-02-13T10:00:00+00:00'}}}}
    if black is not None:
        loc['black'] = black
    return {'generated_at': (NOW - timedelta(hours=age_h)).isoformat(), 'locations': {'isis': loc}}


class AlertTests(unittest.TestCase):
    def test_likely_change_once_then_today(self):
        msgs, st = decide(bundle(down_tom=0.7), {'isis': {'flag': 'red', 'sent': []}}, NOW)
        self.assertEqual(len(msgs), 1)
        self.assertIn('likely down to Amber by the end of tomorrow (70%)', msgs[0]['body'])
        self.assertIn('Probably mid-morning tomorrow', msgs[0]['body'])
        msgs, st = decide(bundle(down_tom=0.75), st, NOW)            # same forecast an hour later: quiet
        self.assertEqual(msgs, [])
        msgs, st = decide(bundle(down_today=0.65, down_tom=0.8), st, NOW)   # now today too: one more
        self.assertEqual(len(msgs), 1)
        self.assertIn('possibly down to Amber today (65%)', msgs[0]['body'])
        msgs, st = decide(bundle(down_today=0.7, down_tom=0.8), st, NOW)
        self.assertEqual(msgs, [])

    def test_reset_after_the_chance_drops(self):
        _, st = decide(bundle(down_tom=0.7), {'isis': {'flag': 'red', 'sent': []}}, NOW)
        _, st = decide(bundle(down_tom=0.5), st, NOW)                # hovering: no reset
        self.assertEqual(decide(bundle(down_tom=0.7), st, NOW)[0], [])
        _, st = decide(bundle(down_tom=0.1), st, NOW)                # dropped well below
        self.assertEqual(len(decide(bundle(down_tom=0.7), st, NOW)[0]), 1)

    def test_flag_change_and_first_run(self):
        msgs, st = decide(bundle(flag='red'), {}, NOW)               # first run: just remember
        self.assertEqual(msgs, [])
        _, st = decide(bundle(flag='red', down_tom=0.7), st, NOW)
        self.assertTrue(st['isis']['sent'])
        msgs, st = decide(bundle(flag='amber'), st, NOW)       # the change itself: sent by the Worker
        self.assertEqual(msgs, [])
        self.assertEqual(st['isis'], {'flag': 'amber', 'sent': [], 'up_top': None})

    def test_stale_or_old_bundle_sends_nothing_new(self):
        self.assertEqual(decide(bundle(down_tom=0.9, stale=True), {'isis': {'flag': 'red', 'sent': []}}, NOW)[0], [])
        self.assertEqual(decide(bundle(down_tom=0.9, age_h=5), {'isis': {'flag': 'red', 'sent': []}}, NOW)[0], [])

    def test_unclear_destination_gives_the_split(self):
        b = bundle(flag='dark_blue')
        oo = b['locations']['isis']['official_outlook']
        oo['windows']['tomorrow']['up'] = 0.7
        oo['down'] = None
        oo['up'] = {'flag': 'amber', 'when': None, 'split': {'amber': 0.4, 'red': 0.3}}
        msgs, _ = decide(b, {'isis': {'flag': 'dark_blue', 'sent': []}}, NOW)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]['body'], '⬆️ Isis Flag likely to go up by the end of tomorrow (70%): Amber 40%, Red 30%.')
        oo['up']['split'] = {'amber': 0.6, 'red': 0.1}                 # 86% agree on Amber: named
        msgs, _ = decide(b, {'isis': {'flag': 'dark_blue', 'sent': []}}, NOW)
        self.assertEqual(msgs[0]['body'], '⬆️ 🟠 Isis Flag likely up to Amber by the end of tomorrow (70%).')

    def test_escalation_to_a_higher_flag(self):
        b = bundle(flag='green')
        oo = b['locations']['isis']['official_outlook']
        oo['down'] = None
        oo['windows']['tomorrow']['up'] = 0.85
        oo['up'] = {'flag': 'light_blue', 'when': None,
                    'split': {'light_blue': 0.45, 'dark_blue': 0.2, 'amber': 0.1, 'red': 0.05}}
        msgs, st = decide(b, {'isis': {'flag': 'green', 'sent': []}}, NOW)
        self.assertIn('likely to go up by the end of tomorrow (85%): Light Blue 45%, Dark Blue 20%, Amber 10%, Red 5%', msgs[0]['body'])
        self.assertEqual(decide(b, st, NOW)[0], [])                       # same forecast: quiet
        oo['windows']['tomorrow']['up'] = 0.9
        oo['up']['split'] = {'light_blue': 0.1, 'dark_blue': 0.1, 'amber': 0.2, 'red': 0.5}   # Amber or higher 70%
        msgs, st = decide(b, st, NOW)
        self.assertEqual([m['body'] for m in msgs], ['⬆️ 🟠 Isis Flag now likely up to Amber by the end of tomorrow (70%).'])
        self.assertEqual(decide(b, st, NOW)[0], [])
        oo['up']['split'] = {'amber': 0.2, 'red': 0.7}
        msgs, st = decide(b, st, NOW)
        self.assertEqual([m['body'] for m in msgs], ['⬆️ 🔴 Isis Flag now likely up to Red by the end of tomorrow (70%).'])

    def test_black(self):
        bk = {'black_now': False, 'level_m': 0.39, 'windows': {'today': 0.2, 'tomorrow': 0.65},
              'when': {'median_time': '2026-02-12T11:00:00+00:00'}}
        msgs, st = decide(bundle(black=bk), {'isis': {'flag': 'red', 'sent': []}}, NOW)
        self.assertEqual(len(msgs), 1)
        self.assertIn('Isis Flag possibly to go Black by the end of tomorrow (65%)', msgs[0]['body'])
        low = dict(bk, windows={'today': 0.2, 'tomorrow': 0.55})              # 55%: below the 60% line
        self.assertEqual(decide(bundle(black=low), {'isis': {'flag': 'red', 'sent': []}}, NOW)[0], [])
        self.assertIn('0.39 m', msgs[0]['body'])
        self.assertIn('If so, probably mid-morning today', msgs[0]['body'])
        self.assertEqual(decide(bundle(black=bk), st, NOW)[0], [])


if __name__ == '__main__':
    unittest.main()
