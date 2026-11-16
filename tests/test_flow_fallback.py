"""Farmoor flow fallbacks (16 Nov 2026): saved series when both EA feeds fail, carry-forward up to
FLOW_MAX_AGE_H, and the previous forecast held (marked stale) beyond it."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from flag_predictor.data import api  # noqa: E402
from flag_predictor.prediction import hybrid_live as hl  # noqa: E402


def _merged(last_flow_h_before_t0: float, t0='2026-11-16 10:00'):
    t0 = pd.Timestamp(t0, tz='UTC')
    idx = pd.date_range(t0 - pd.Timedelta(days=60), t0, freq='1h')
    flow = pd.Series(np.linspace(3.0, 4.0, len(idx)), index=idx)
    flow[idx > t0 - pd.Timedelta(hours=last_flow_h_before_t0)] = np.nan
    return pd.DataFrame({'differential': 0.0, 'flow_m3s_Farmoor': flow}, index=idx)


def test_fresh_flow_unchanged():
    m = _merged(0)
    f = hl._live_flow(m, m.index)
    assert f.notna().all() and f.iloc[-1] == pytest.approx(4.0)


def test_flow_carried_forward_within_limit():
    m = _merged(20)
    f = hl._live_flow(m, m.index)
    last = m['flow_m3s_Farmoor'].dropna().iloc[-1]
    assert f.notna().all()
    assert (f.iloc[-20:] == last).all()


def test_flow_too_old_raises():
    m = _merged(hl.FLOW_MAX_AGE_H + 2)
    with pytest.raises(hl.FlowUnavailable):
        hl._live_flow(m, m.index)
    with pytest.raises(hl.FlowUnavailable):
        hl._live_flow(m.drop(columns='flow_m3s_Farmoor'), m.index)


def test_both_feeds_fail_uses_saved(tmp_path, monkeypatch):
    path = tmp_path / 'farmoor_flow_recent.csv'
    monkeypatch.setattr(api, 'RECENT_FLOW_FILE', path)
    api._FLOW_CACHE.clear()
    idx = pd.date_range('2026-11-16 08:00', periods=9, freq='15min', tz='UTC')
    good = pd.DataFrame({'flow_m3s_Farmoor': np.arange(9.0)}, index=idx)
    monkeypatch.setattr(api, '_fetch_qualified_farmoor_flow', lambda verbose, days: pd.DataFrame())
    monkeypatch.setattr(api, '_fetch_live_farmoor_flow', lambda verbose=True, days=28: good.copy())
    api.fetch_flow_data(verbose=False)
    assert path.exists()
    saved_before = path.read_text()

    api._FLOW_CACHE.clear()
    monkeypatch.setattr(api, '_fetch_live_farmoor_flow', lambda verbose=True, days=28: pd.DataFrame())
    out = api.fetch_flow_data(verbose=False)
    assert out.attrs.get('source') == 'saved'
    assert out.index[-1] == pd.Timestamp('2026-11-16 10:00')
    assert api._FLOW_CACHE == {}          # a fallback is not cached: the next call retries the EA
    assert path.read_text() == saved_before


def test_held_payload_marked_stale(tmp_path, monkeypatch):
    import generate_all_location_figures as gen
    prev = {'generated_at': '2026-11-16T10:20:00Z', 'location_order': ['isis', 'godstow'],
            'locations': {'isis': {'key': 'isis', 'stale': False}, 'godstow': {'key': 'godstow', 'stale': False}}}
    (tmp_path / 'forecast_data.json').write_text(json.dumps(prev))
    monkeypatch.setattr(gen, 'HELD_REASONS', {'godstow': ('flow', 'no Farmoor flow')})
    fresh = {'isis': {'key': 'isis', 'stale': False}}
    gen.write_forecast_bundle(tmp_path, dict(fresh), ['godstow'])
    out = json.loads((tmp_path / 'forecast_data.json').read_text())['locations']
    assert out['isis'] == fresh['isis']
    assert out['godstow']['stale'] is True
    assert out['godstow']['held'] == {'kind': 'flow', 'detail': 'no Farmoor flow', 'since': '2026-11-16T10:20:00Z'}

    # Held again next run: keeps the original "since".
    gen.write_forecast_bundle(tmp_path, dict(fresh), ['godstow'])
    out = json.loads((tmp_path / 'forecast_data.json').read_text())['locations']
    assert out['godstow']['held']['since'] == '2026-11-16T10:20:00Z'
