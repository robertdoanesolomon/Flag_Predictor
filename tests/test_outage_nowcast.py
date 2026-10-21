"""Gauge outages: the differential estimated from the neighbouring stretch."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from flag_predictor.prediction import hybrid_live as hl  # noqa: E402


def _frames(down_h=72, now=pd.Timestamp('2026-10-20 12:00', tz='UTC')):
    idx = pd.date_range(now - pd.Timedelta(days=20), now, freq='1h')
    god = pd.Series(0.2 + 0.3 * np.clip((np.arange(len(idx)) - (len(idx) - 60)) / 40, 0, 1), idx)
    isis = 0.1 + 0.67 * (god - 0.2)
    last = now - pd.Timedelta(hours=down_h)
    hist = isis.loc[:last].tz_localize(None)
    merged = pd.DataFrame({'differential': hist, 'Osney': 0.0, 'flow_m3s_Farmoor': 20.0})
    return merged, god, isis, now


def _prime_cache(god, idx):
    hl._cache.clear()
    hl._cache['levels'] = {'godstow': god}
    hl._cache['api'] = pd.DataFrame({'Osney': 0.5}, index=idx)
    hl._cache['flow'] = pd.Series(25.0, idx)


def test_neighbour_nowcast_follows_the_neighbour():
    merged, god, isis, now = _frames()
    d = merged['differential'].tz_localize('UTC')
    est = hl.neighbour_nowcast(d, god, 0.67, now)
    assert est.index[0] == d.index[-1] + pd.Timedelta(hours=1) and est.index[-1] == now
    assert np.allclose(est.to_numpy(), isis.reindex(est.index).to_numpy(), atol=0.01)


def test_outage_nowcast_extends_history_to_now():
    merged, god, isis, now = _frames(down_h=72)
    _prime_cache(god, god.index)
    out, info = hl.outage_nowcast('isis', merged, now=now)
    assert info and info['from'] == 'godstow' and info['hours'] == 72 and info['confident']
    assert out.index[-1] == now.tz_localize(None) and out.index.tz is None
    assert abs(out['differential'].iloc[-1] - isis.iloc[-1]) < 0.01
    assert out['Osney'].iloc[-1] == 0.5 and out['flow_m3s_Farmoor'].iloc[-1] == 25.0
    assert 0 < info['sd'] < 0.2


def test_no_nowcast_when_fresh_or_neighbour_also_down():
    merged, god, isis, now = _frames(down_h=2)
    _prime_cache(god, god.index)
    assert hl.outage_nowcast('isis', merged, now=now)[1] is None
    merged, god, isis, now = _frames(down_h=72)
    _prime_cache(god.loc[:now - pd.Timedelta(hours=30)], god.index)
    assert hl.outage_nowcast('isis', merged, now=now)[1] is None
    assert hl.outage_nowcast('wallingford', merged, now=now)[1] is None


def test_error_grows_with_time_down_and_in_the_middle():
    a = hl.nowcast_error_sd('isis', 24, 0.05)
    assert hl.nowcast_error_sd('isis', 24 * 7, 0.05) > a
    assert hl.nowcast_error_sd('isis', 24, 0.4) > a


def test_mark_estimated_moves_estimated_hours_out_of_history():
    from generate_all_location_figures import mark_estimated
    payload = {'historical': {'times': ['2026-10-20T08:00:00', '2026-10-20T09:00:00', '2026-10-20T10:00:00'],
                              'differential': [0.1, 0.12, 0.13]}, 'stale': True}
    nowcast = {'from': 'godstow', 'last_reading': pd.Timestamp('2026-10-20 08:00', tz='UTC'), 'reading_time': None,
               'hours': 2, 'confident': True, 'level': 0.13, 'sd': 0.02}
    mark_estimated(payload, nowcast)
    assert payload['historical']['differential'] == [0.1, None, None]
    assert payload['estimated']['values'] == [0.1, 0.12, 0.13] and payload['stale'] is False


def test_alerts_note_or_skip_an_estimated_forecast():
    from push.alerts import decide
    now = pd.Timestamp('2026-10-20 12:00', tz='UTC').to_pydatetime()
    loc = {'official_flag': {'flag': 'green'},
           'official_outlook': {'windows': {'today': {'up': 0.1}, 'tomorrow': {'up': 0.8}},
                                'up': {'flag': 'light_blue', 'split': {'light_blue': 0.8}}},
           'estimated': {'from': 'godstow', 'confident': True, 'hours': 48}}
    bundle = {'generated_at': '2026-10-20T11:30:00Z', 'locations': {'isis': loc}}
    msgs, _ = decide(bundle, {}, now)
    assert len(msgs) == 1 and 'Estimated from Godstow' in msgs[0]['body']
    loc['estimated'].update(confident=False, hours=200)        # 8 days: still sent (70%+)
    assert len(decide(bundle, {}, now)[0]) == 1
    loc['estimated']['hours'] = 260                             # over 10 days: none
    assert decide(bundle, {}, now)[0] == []
