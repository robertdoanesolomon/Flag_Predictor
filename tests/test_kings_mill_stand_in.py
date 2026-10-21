"""King's Mill offline: its level from the Botanic Gardens Cherwell gauge (data/api.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.data import api  # noqa: E402


def _levels(km_hours, cherwell_rise=0.0):
    idx = pd.date_range('2026-10-01', periods=4 * 24 * 10, freq='15min', tz='UTC')
    c = pd.Series(4.65, index=idx) + np.linspace(0, cherwell_rise, len(idx))
    k = (c - 3.944).iloc[:4 * km_hours]
    return {
        'osney_downstream': pd.DataFrame({'level': 2.20}, index=idx),
        'iffley_upstream': pd.DataFrame({'level': 0.10}, index=idx),
        'kings_mill_downstream': pd.DataFrame({'level': k}),
        'cherwell_botanic': pd.DataFrame({'level': c}),
    }


def test_reporting_gauge_is_left_alone():
    d = api.calculate_isis_differential(_levels(24 * 10))
    assert 'gauge_stand_in' not in d.attrs


def test_offline_king_mill_is_filled_from_botanic_gardens():
    full = api.calculate_isis_differential(_levels(24 * 10))
    part = api.calculate_isis_differential(_levels(24 * 3))
    assert part.attrs['gauge_stand_in']['gauge'] == "King's Mill"
    assert part.index[-1] == full.index[-1]
    assert np.allclose(part['differential'].to_numpy(), full['differential'].to_numpy(), atol=1e-6)


def test_no_stand_in_once_the_cherwell_has_moved_far():
    d = api.calculate_isis_differential(_levels(24 * 3, cherwell_rise=1.0))
    assert 'gauge_stand_in' not in d.attrs
    assert d.index[-1] < pd.Timestamp('2026-10-05', tz='UTC')


def test_no_king_mill_data_at_all_is_still_offline():
    # Nothing to anchor the offset or the Cherwell change on: the Isis is offline, as before.
    import pytest
    lv = _levels(24 * 10)
    lv['kings_mill_downstream'] = pd.DataFrame()
    with pytest.raises(RuntimeError):
        api.calculate_isis_differential(lv)
