"""
Live ensemble forecasts with the physics hybrid (the beta page).

The production history (`prepare_training_data`) only has rows where the
differential exists, so it has a months-long hole between the end of the
differential archive and the EA API's 4-week window. The hybrid's encoder
needs ~35 days of continuous rain, so this module builds its own hourly frame:

* rain: EA API readings (last ~4 weeks) on top of recent gauge CSVs
  (data/recent_rainfall, refreshed daily in CI; falls back to
  data/rainfall_training_data), for exactly the gauges the model trained on;
* flow and differential: the production frame's API values.

Forecast rain comes from the 19-gauge ECMWF AIFS ensemble so every training
gauge has a forecast. Downloads are cached per process, so running three
locations fetches them once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.api import fetch_rainfall_data, get_rainfall_forecast_ensemble
from ..data.loader import load_historical_rainfall
from ..models.candidates import ensemble_predictor

HYBRID_MODELS = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
HORIZON = 240
HISTORY_DAYS = 60

# Gauges each location's hybrid was trained on (the column order of the
# training frame; the encoder's rain features sum over exactly these).
_ALL_GAUGES = [
    'Aylesbury', 'Worsham', 'Osney', 'Bourton', 'Abingdon', 'St', 'Swindon', 'Eynsham',
    'Chipping', 'Stanford', 'Rapsgate', 'Wheatley', 'Stowell', 'Cleeve', 'Shorncote',
    'Benson', 'Grimsbury',
]
TRAINING_GAUGES = {
    'isis': _ALL_GAUGES,
    'wallingford': _ALL_GAUGES,
    'godstow': [g for g in _ALL_GAUGES if g != 'Grimsbury'],
}

_cache: Dict[str, object] = {}


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    return df


def _short_gauge_name(col: str) -> str:
    after = col.split('mm_', 1)[1] if 'mm_' in col else col
    return after.split('-')[0]


def _gauge_csv_rain(project_root: Path) -> pd.DataFrame:
    if 'csv' not in _cache:
        rain_dir = 'data/recent_rainfall/'
        if not list((project_root / rain_dir).glob('*.csv')):
            rain_dir = 'data/rainfall_training_data/'
        raw = load_historical_rainfall(rainfall_dir=rain_dir, project_root=project_root,
                                       location='wallingford', verbose=False)
        raw = _to_utc(raw).rename(columns=_short_gauge_name)
        raw = raw.loc[:, ~raw.columns.duplicated()].astype(float)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=HISTORY_DAYS + 10)
        raw = raw.loc[raw.index >= cutoff]
        _cache['csv'] = raw.resample('1h').sum(min_count=1).clip(lower=0, upper=50)
    return _cache['csv']


def _api_rain() -> pd.DataFrame:
    if 'api' not in _cache:
        try:
            api = fetch_rainfall_data(location='wallingford', verbose=False)
            api = _to_utc(api).astype(float)
            _cache['api'] = api.resample('1h').sum(min_count=1).clip(lower=0, upper=50)
        except Exception as exc:  # the gauge CSVs still cover most of the window
            print(f"  [beta] API rain fetch failed ({exc}); using gauge CSVs only")
            _cache['api'] = pd.DataFrame()
    return _cache['api']


def rain_forecast(n_members: int) -> pd.DataFrame:
    """19-gauge ensemble forecast, fetched once per process."""
    key = f'forecast_{n_members}'
    if key not in _cache:
        _cache[key] = get_rainfall_forecast_ensemble(location='wallingford', n_members=n_members)
    return _cache[key]


def live_frame(location: str, merged_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Continuous hourly history ending at the production frame's last hour."""
    merged = _to_utc(merged_df)
    t0 = merged.index[-1]
    grid = pd.date_range(t0 - pd.Timedelta(days=HISTORY_DAYS), t0, freq='1h')
    gauges = TRAINING_GAUGES[location]
    csv = _gauge_csv_rain(project_root).reindex(index=grid, columns=gauges)
    api = _api_rain()
    rain = api.reindex(index=grid, columns=gauges).combine_first(csv) if len(api) else csv
    flow = merged['flow_m3s_Farmoor'].reindex(grid).ffill(limit=6) if 'flow_m3s_Farmoor' in merged else np.nan
    diff = merged['differential'].reindex(grid)
    frame = pd.concat([diff.rename('differential'), rain], axis=1)
    frame['flow_m3s_Farmoor'] = flow
    missing = rain.isna().all(axis=1).loc[t0 - pd.Timedelta(days=35):].mean()
    if missing > 0.05:
        print(f"  [beta] warning: {missing:.0%} of the last 35 days have no gauge rain")
    return frame


def forecast_ensemble(
    location: str,
    merged_df: pd.DataFrame,
    project_root: Path,
    n_members: int = 50,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """(241, members) hybrid-ensemble forecast from the last observed hour."""
    frame = live_frame(location, merged_df, project_root)
    t0 = frame.index[-1]
    forecast = _to_utc(rain_forecast(n_members))
    fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
    gauges = TRAINING_GAUGES[location]
    kwargs = {'models_dir': models_dir} if models_dir else {}
    predict = ensemble_predictor(HYBRID_MODELS, location, **kwargs)
    members: Dict[str, np.ndarray] = {}
    for i in range(n_members):
        cols = {f'{g}_member_{i}': g for g in gauges if f'{g}_member_{i}' in forecast.columns}
        if not cols:
            continue
        member_rain = forecast[list(cols)].rename(columns=cols).reindex(fut_idx).fillna(0.0)
        members[f'member_{i}'] = predict(frame, member_rain)
    if not members:
        raise RuntimeError('no forecast-rain members available for the hybrid')
    return pd.DataFrame(members, index=pd.date_range(t0, periods=HORIZON + 1, freq='1h'))


def forecast_rain_gauges(location: str) -> List[str]:
    return list(TRAINING_GAUGES[location])
