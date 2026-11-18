"""Gauge glitches screened per reading before the hourly step (data/api.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.data import api  # noqa: E402

IDX = pd.date_range('2026-11-17 00:00', '2026-11-18 12:45', freq='15min', tz='UTC')


def _isis(iffley):
    return {
        'osney_downstream': pd.DataFrame({'level': np.linspace(2.45, 2.356, len(IDX))}, index=IDX),
        'iffley_upstream': pd.DataFrame({'level': iffley}, index=IDX),
        'kings_mill_downstream': pd.DataFrame({'level': np.linspace(0.92, 0.855, len(IDX))}, index=IDX),
        'cherwell_botanic': pd.DataFrame({'level': np.linspace(4.86, 4.765, len(IDX))}, index=IDX),
    }


def test_newest_impossible_reading_drops_only_that_reading():
    """18 Nov 2026: Iffley upstream 0.479 at 12:45 (else ~0.22). The 12:00 hour keeps 12:30."""
    iff = pd.Series(0.22, index=IDX)
    iff.iloc[-1] = 0.479
    d = api.calculate_isis_differential(_isis(iff))
    last = d.index[-1]
    assert last == pd.Timestamp('2026-11-18 12:00', tz='UTC')
    assert d.attrs['reading_time'][last.tz_localize(None)] == pd.Timestamp('2026-11-18 12:30')
    assert d['differential'].iloc[-1] > -0.1
    assert d.attrs['diff_15min']['times'][-1] == '2026-11-18T12:30:00'


def test_glitch_inside_range_removed_once_it_comes_back():
    s = pd.Series(0.30, index=IDX[:20])
    s.iloc[10] = 0.48                       # +18 cm and straight back
    s.iloc[15:17] = 0.10                    # -20 cm for two readings, then back
    out = api.drop_level_glitches(s, verbose=False)
    assert len(out) == 17 and (out == 0.30).all()


def test_newest_jump_is_used_straight_away():
    """No hold: a jump in the newest reading counts at once (unless it is impossible)."""
    s = pd.Series(0.30, index=IDX[:20])
    s.iloc[-1] = 0.50
    assert len(api.drop_level_glitches(s, verbose=False)) == 20


def test_real_step_is_kept():
    s = pd.Series(0.30, index=IDX[:20])
    s.iloc[10:] = 0.50                      # weir change: stays up
    assert len(api.drop_level_glitches(s, verbose=False)) == 20
    s.iloc[10:] = np.r_[0.50, 0.50, 0.30 + np.zeros(8)]   # up for 3 readings is not a glitch
    s.iloc[10:13] = 0.50
    assert len(api.drop_level_glitches(s, verbose=False)) == 20


def test_godstow_and_wallingford_screened_too():
    lv = {
        'godstow_downstream': pd.DataFrame({'level': 1.70}, index=IDX),
        'osney_upstream': pd.DataFrame({'level': 0.03}, index=IDX),
        'benson_downstream': pd.DataFrame({'level': 2.30}, index=IDX),
        'cleeve_upstream': pd.DataFrame({'level': 0.20}, index=IDX),
    }
    lv['godstow_downstream'].iloc[-1] = -50.0
    lv['cleeve_upstream'].iloc[-1] = 5.0
    g = api.calculate_godstow_differential(lv)
    w = api.calculate_wallingford_differential(lv)
    for d in (g, w):
        assert d.attrs['reading_time'][d.index[-1].tz_localize(None)] == pd.Timestamp('2026-11-18 12:30')


def test_missing_gauge_is_held_for_a_few_hours_and_named():
    """Osney downstream silent after 14:30 (18 Nov 2026) while Iffley and King's Mill report to 16:00."""
    idx = pd.date_range('2026-11-17 00:00', '2026-11-18 16:00', freq='15min', tz='UTC')
    lv = _isis(pd.Series(0.22, index=idx))
    for k in lv:
        lv[k] = lv[k].reindex(idx).ffill().bfill()
    lv['osney_downstream'] = lv['osney_downstream'].loc[:'2026-11-18 14:30']
    d = api.calculate_isis_differential(lv)
    assert d.index[-1] == pd.Timestamp('2026-11-18 16:00', tz='UTC')           # held, not dropped
    assert d.attrs['reading_time'][d.index[-1].tz_localize(None)] == pd.Timestamp('2026-11-18 14:30')
    h = d.attrs['gauge_hold']
    assert [x['gauge'] for x in h] == ['osney_downstream'] and h[0]['since'] == '2026-11-18T14:30:00Z'
    assert d.attrs['diff_15min']['times'][-1] == '2026-11-18T16:00:00'           # chart and hourly agree


def test_hold_ends_after_three_hours():
    idx = pd.date_range('2026-11-17 00:00', '2026-11-18 20:00', freq='15min', tz='UTC')
    lv = _isis(pd.Series(0.22, index=idx))
    for k in lv:
        lv[k] = lv[k].reindex(idx).ffill().bfill()
    lv['osney_downstream'] = lv['osney_downstream'].loc[:'2026-11-18 14:30']
    d = api.calculate_isis_differential(lv)
    assert d.index[-1] <= pd.Timestamp('2026-11-18 17:00', tz='UTC')
    assert 'gauge_hold' not in d.attrs


def test_only_osney_downstream_is_held_longer():
    """Iffley silent after 14:30 while the others report: held 1 h, as before, not 3 h."""
    idx = pd.date_range('2026-11-17 00:00', '2026-11-18 16:00', freq='15min', tz='UTC')
    lv = _isis(pd.Series(0.22, index=idx))
    for k in lv:
        lv[k] = lv[k].reindex(idx).ffill().bfill()
    lv['iffley_upstream'] = lv['iffley_upstream'].loc[:'2026-11-18 14:30']
    d = api.calculate_isis_differential(lv)
    assert d.index[-1] == pd.Timestamp('2026-11-18 15:00', tz='UTC')
    assert 'gauge_hold' not in d.attrs


def test_godstow_gauge_not_held_longer():
    idx = pd.date_range('2026-11-17 00:00', '2026-11-18 16:00', freq='15min', tz='UTC')
    lv = {
        'godstow_downstream': pd.DataFrame({'level': 1.70}, index=idx),
        'osney_upstream': pd.DataFrame({'level': 0.03}, index=idx).loc[:'2026-11-18 14:30'],
    }
    d = api.calculate_godstow_differential(lv)
    assert d.index[-1] == pd.Timestamp('2026-11-18 15:00', tz='UTC') and 'gauge_hold' not in d.attrs


def test_nowcast_starts_when_a_core_gauge_stops_but_not_for_held_osney():
    idx = pd.date_range('2026-11-17 00:00', '2026-11-18 16:00', freq='15min', tz='UTC')
    lv = _isis(pd.Series(0.22, index=idx))
    for k in lv:
        lv[k] = lv[k].reindex(idx).ffill().bfill()
    lv['osney_downstream'] = lv['osney_downstream'].loc[:'2026-11-18 14:30']
    d = api.calculate_isis_differential(lv)
    core = d.attrs['reading_time_core']
    assert core[d.index[-1].tz_localize(None)] >= d.index[-1].tz_localize(None)    # Iffley/King's Mill current
