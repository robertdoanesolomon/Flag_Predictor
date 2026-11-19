import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import flag_predictor.config as C
import flag_predictor.prediction.hybrid_live as HL


def test_variant_choice_and_offset():
    assert HL.pick_variant('hybrid_srh,hybrid_offset', 'isis') == 'hybrid_offset'
    assert HL.pick_variant('hybrid_offset', 'godstow') is None
    assert HL.input_offset('isis', 'hybrid_offset') == C.model_input_offset('isis')
    assert HL.input_offset('isis', 'hybrid_srh') == 0.0
    assert HL.input_offset('isis', None) == (C.model_input_offset('isis') if C.MODEL_INPUT_OFFSET_ON_MAIN else 0.0)


def test_model_sees_offset_and_paths_come_back(monkeypatch):
    t0 = pd.Timestamp('2026-11-19 09:00', tz='UTC')
    idx = pd.date_range(t0 - pd.Timedelta(days=3), t0, freq='1h')
    g = HL.TRAINING_GAUGES['isis'][0]
    frame = pd.DataFrame({'differential': 0.02, g: 0.0, 'flow_m3s_Farmoor': 10.0}, index=idx)
    fut = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HL.HORIZON, freq='1h')
    seen = []

    def predictor(fr, rain):
        seen.append(float(fr['differential'].iloc[-1]))
        return np.r_[fr['differential'].iloc[-1], np.full(HL.HORIZON, fr['differential'].iloc[-1] + 0.01)]

    monkeypatch.setattr(HL, 'live_frame', lambda *a, **k: frame)
    monkeypatch.setattr(HL, 'rain_forecast', lambda n: pd.DataFrame({f'{g}_member_0': 0.0}, index=fut))
    monkeypatch.setattr(HL, 'ensemble_predictor', lambda *a, **k: predictor)
    monkeypatch.setattr(HL, 'model_error_samples', lambda loc, fr, m, *a, **k: m.to_numpy().T.copy())
    monkeypatch.setattr(HL, 'STEP_LAYER', False)
    off = C.model_input_offset('isis')
    live = HL.forecast_with_uncertainty('isis', frame, ROOT, n_members=1, models_dir=ROOT / 'models', variant='hybrid_offset')
    assert np.isclose(seen[-1], 0.02 + off)                          # the model saw the raised differential
    assert np.isclose(live.members.iloc[0, 0], 0.02)                 # paths are back on the measured scale
    assert np.isclose(live.total_samples[0, 1], 0.03)
    assert np.isclose(live.level_hist[-1], 0.02)                     # flags and history stay measured
