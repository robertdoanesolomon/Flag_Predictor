"""
Live ensemble forecasts with the continuous-state hybrid (hybrid_cs).

The model needs the reservoir state at t0, which comes from simulating the
catchment through the last ~2 years of gauge rain (soil moisture and
groundwater remember that long). CI already caches ~800 days of EA qualified
rain CSVs in data/rainfall_training_data; the last weeks come from the recent
CSVs and the EA API, exactly as for the September hybrid (hybrid_live).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.loader import load_historical_rainfall
from ..models.continuous_train import (
    StateSimulation, continuous_predictor, load_forecast_model, simulate_states,
)
from .hybrid_live import TRAINING_GAUGES, _api_rain, _gauge_csv_rain, _short_gauge_name, _to_utc

SPINUP_DAYS = 760
_cache: Dict[str, object] = {}


def _long_csv_rain(project_root: Path) -> pd.DataFrame:
    if 'long' not in _cache:
        try:
            raw = load_historical_rainfall(rainfall_dir='data/rainfall_training_data/', project_root=project_root,
                                           location='wallingford', verbose=False)
            raw = _to_utc(raw).rename(columns=_short_gauge_name)
            raw = raw.loc[:, ~raw.columns.duplicated()].astype(float)
            raw = raw.loc[raw.index >= pd.Timestamp.utcnow() - pd.Timedelta(days=SPINUP_DAYS + 30)]
            _cache['long'] = raw.resample('1h').sum(min_count=1).clip(lower=0, upper=50)
        except Exception as exc:
            print(f"  [continuous] long rain CSVs unavailable ({exc})")
            _cache['long'] = pd.DataFrame()
    return _cache['long']


class SpinupError(RuntimeError):
    pass


def long_rain_series(location: str, t0: pd.Timestamp, project_root: Path,
                     min_coverage: float = 0.9) -> pd.Series:
    """Hourly mean-station rain over the location's gauges for the spin-up period ending at t0."""
    gauges = TRAINING_GAUGES[location]
    grid = pd.date_range(t0 - pd.Timedelta(days=SPINUP_DAYS), t0, freq='1h')
    parts = [df.reindex(index=grid, columns=gauges) for df in (_api_rain(), _gauge_csv_rain(project_root),
                                                                _long_csv_rain(project_root)) if len(df)]
    if not parts:
        raise RuntimeError('no rain history for the continuous state')
    rain = parts[0]
    for p in parts[1:]:
        rain = rain.combine_first(p)
    covered = rain.notna().any(axis=1).mean()
    print(f"  [continuous] {covered:.0%} of the {SPINUP_DAYS}-day spin-up has gauge rain")
    if covered < min_coverage:
        # Missing months would be simulated as dry weather and drain the soil.
        raise SpinupError(f'only {covered:.0%} of the spin-up period has gauge rain')
    return rain.clip(lower=0).mean(axis=1).fillna(0.0)


def simulation_features(location: str, frame: pd.DataFrame, project_root: Path, models_dir: Path,
                        core_name: str) -> pd.DataFrame:
    """Continuous-simulation encoder inputs (hybrid_simfeat) on the live frame's hours."""
    from ..models.continuous_train import SIM_FEATURES, load_core, location_et0, sim_features
    t0 = pd.Timestamp(frame.index[-1])
    t0 = t0.tz_localize('UTC') if t0.tzinfo is None else t0.tz_convert('UTC')
    core = load_core(core_name, models_dir)
    if core.et_mode == 'pet':
        raise NotImplementedError('live ET0 for the simulation is not wired up (the ET0 core was not adopted)')
    rain = long_rain_series(location, t0, project_root)
    sim = simulate_states(core, rain)
    idx = pd.DatetimeIndex(frame.index)
    idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    flow = frame['flow_m3s_Farmoor'].to_numpy(dtype=float) if 'flow_m3s_Farmoor' in frame else None
    return pd.DataFrame(sim_features(core, sim, idx, flow), index=frame.index, columns=SIM_FEATURES)


def state_simulations(names, location: str, t0: pd.Timestamp, project_root: Path,
                      models_dir: Path, et0: Optional[pd.Series] = None) -> Dict[str, StateSimulation]:
    rain = long_rain_series(location, t0, project_root)
    sims: Dict[str, StateSimulation] = {}
    by_core: Dict[object, StateSimulation] = {}
    for n in names:
        model, _, meta = load_forecast_model(n, location, models_dir)
        key = n if meta['cfg'].get('core_lr', 0) > 0 else meta['core']
        if key not in by_core:
            by_core[key] = simulate_states(model.core, rain, et0=et0)
        sims[n] = by_core[key]
    return sims


def continuous_live_predictor(names, location: str, frame: pd.DataFrame, project_root: Path,
                              models_dir: Path, et0: Optional[pd.Series] = None):
    """predict(frame, member_rain) -> 241 values, with the state simulated up to the frame's t0."""
    t0 = pd.Timestamp(frame.index[-1])
    t0 = t0.tz_localize('UTC') if t0.tzinfo is None else t0.tz_convert('UTC')
    sims = state_simulations(names, location, t0, project_root, models_dir, et0)
    return continuous_predictor(names, location, models_dir, sims, et0)
