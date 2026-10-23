"""Extra encoder inputs: soil moisture and upstream levels (models/extra_inputs.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.models import extra_inputs as ei  # noqa: E402


def test_soil_value_used_only_once_published():
    days = pd.date_range('2025-01-01', periods=10, freq='D', tz='UTC')
    vwc = pd.Series(np.arange(10, 20, dtype=float), index=days)
    idx = pd.date_range('2025-01-03', periods=48, freq='h', tz='UTC')
    f = ei.soil_features(vwc, idx)
    # 1 Jan's value is 10. Until 3 Jan 11:00 the newest published one is 1 Jan's; 2 Jan's (11) appears
    # at 2 Jan + 36 h = 3 Jan 12:00
    assert f.loc['2025-01-03 11:00', 'soil_vwc'] == 10.0
    assert f.loc['2025-01-03 12:00', 'soil_vwc'] == 11.0
    assert np.isnan(f.loc['2025-01-03 12:00', 'soil_vwc_d7'])        # no value a week before 2 Jan


def test_level_rise_and_changes():
    idx = pd.date_range('2025-01-01', periods=24 * 20, freq='h', tz='UTC')
    lv = pd.Series(1.0, index=idx)
    lv.iloc[-24:] = np.linspace(1.0, 1.5, 24)
    f = ei.level_features(pd.DataFrame({'1080TH': lv}), idx)
    assert abs(f['up_1080TH_rise'].iloc[-1] - 0.5) < 1e-9
    assert abs(f['up_1080TH_d24'].iloc[-1] - (1.5 - 1.0)) < 0.03
    assert f['up_1238TH_rise'].isna().all()                          # station missing -> NaN (the model's mean)


def test_frame_has_the_trained_columns():
    idx = pd.date_range('2025-01-01', periods=48, freq='h', tz='UTC')
    f = ei.extra_feature_frame(idx, pd.DataFrame(index=idx), None)
    assert list(f.columns) == ei.ALL_COLS and len(f) == 48


def test_cosmos_csv_parsing():
    text = ('timestamp,2026-10-20T16:21:34Z\nsite-id,CHIMN\nparameter-id,cosmos_vwc\n'
            '2026-10-18T00:00:00Z,24.1\n2026-10-19T00:00:00Z,23.9\n2026-10-20T00:00:00Z,\n')
    s = ei.parse_cosmos_csv(text)
    assert s.iloc[1] == 23.9 and np.isnan(s.iloc[2]) and str(s.index.tz) == 'UTC'
