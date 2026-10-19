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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.api import fetch_rainfall_data, get_rainfall_forecast_ensemble
from ..data.loader import load_historical_rainfall
from ..models.candidates import ensemble_predictor

HYBRID_MODELS = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
MODEL_TAG = 'hybrid_c1_ps'
# Optional per-location override with the hybrid that also gets continuous-
# simulation encoder inputs (hybrid_simfeat; see the October section of
# EXPERIMENTS.md). Off: all three locations use the plain hybrid above.
# Example: {'isis': {'tag': 'hybrid_simfeat', 'core': 'cs_cos',
#                    'names': ['hybrid_simfeat'] + [f'hybrid_simfeat_s{i}' for i in range(1, 5)]}}
SIMFEAT_MODELS: dict = {}
# Model-error library (build_error_library.py); missing file = rain spread only.
ERROR_LIBRARY = 'error_library_{tag}_{location}.npz'
HORIZON = 240
HISTORY_DAYS = 60
# Lock-keeper step layer (step_layer.py): a stepped copy of every outcome (held
# levels and daytime jumps) plus the flag-change box, for the preview page.
# The main page keeps the smooth forecast. Off, or a missing
# models/step_layer_<location>.json, = no stepped copy.
STEP_LAYER = True

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


@dataclass
class LiveForecast:
    members: pd.DataFrame                  # (241, members) rain-member trajectories, t0 first
    member_rain: pd.DataFrame              # (240, members) mean-station forecast rain (mm/h)
    total_samples: Optional[np.ndarray]    # (S, 241) members + borrowed model errors, or None
    model_name: str
    note: Optional[str] = None             # why a preferred model was not used, if it wasn't
    # The previous live model (plain hybrid) on the same rain, when the new forecast
    # used a different model; None means the members above are also the previous model's.
    previous_members: Optional[pd.DataFrame] = None
    # Stepped copies (step_layer.py) for the preview page; None if the step layer is off or failed.
    stepped_members: Optional[pd.DataFrame] = None
    stepped_total: Optional[np.ndarray] = None
    # Chance of a flag-band change today / by the end of tomorrow (step_layer.flag_outlook).
    flag_outlook: Optional[dict] = None
    step_triggers: Optional[dict] = None
    level_hist: Optional[np.ndarray] = None       # hourly observed differential, last 49 h to t0


def forecast_ensemble(
    location: str,
    merged_df: pd.DataFrame,
    project_root: Path,
    n_members: int = 50,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """(241, members) hybrid-ensemble forecast from the last observed hour."""
    return forecast_with_uncertainty(location, merged_df, project_root, n_members, models_dir).members


def model_error_samples(location: str, frame: pd.DataFrame, members: pd.DataFrame,
                        member_rain: pd.DataFrame, models_dir: Path, tag: str = MODEL_TAG,
                        seed: int = 0) -> Optional[np.ndarray]:
    """Rain members plus errors borrowed from the model's past forecasts in similar situations."""
    from .model_error import ErrorLibrary, start_features, total_uncertainty
    path = models_dir / ERROR_LIBRARY.format(tag=tag, location=location)
    if not path.exists():
        print(f"  [model error] no library at {path.name}; showing rain spread only")
        return None
    lib = ErrorLibrary.load(path)
    t0 = frame.index[-1]
    diff = frame['differential']
    d0 = float(diff.iloc[-1])
    near = diff.loc[t0 - pd.Timedelta(hours=26):t0 - pd.Timedelta(hours=22)].dropna()
    d24 = float(near.iloc[len(near) // 2]) if len(near) else float('nan')
    gauges = [g for g in TRAINING_GAUGES[location] if g in frame.columns]
    past = frame[gauges].clip(lower=0).mean(axis=1).fillna(0.0)
    past168 = float(past.iloc[-168:].sum())
    feats = np.stack([
        start_features(d0, d24, member_rain[c].to_numpy(), past168, t0) for c in members.columns
    ])
    samples = total_uncertainty(members.to_numpy().T, feats, lib, seed=seed)
    print(f"  [model error] {samples.shape[0]} outcomes from {len(lib.t0)} past forecasts ({lib.model})")
    return samples


def forecast_with_uncertainty(
    location: str,
    merged_df: pd.DataFrame,
    project_root: Path,
    n_members: int = 50,
    models_dir: Optional[Path] = None,
    with_previous: bool = True,
) -> LiveForecast:
    """Hybrid-ensemble forecast plus the model-error outcome cloud.

    With ``with_previous``, a location whose new model differs from the plain
    hybrid also gets the plain hybrid's members on the same rain, for the
    website's "previous model" view.
    """
    models_dir = models_dir or (project_root / 'models')
    frame = live_frame(location, merged_df, project_root)
    t0 = frame.index[-1]
    forecast = _to_utc(rain_forecast(n_members))
    fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
    gauges = TRAINING_GAUGES[location]
    names, tag = HYBRID_MODELS, MODEL_TAG
    extra = None
    note = None
    simfeat = SIMFEAT_MODELS.get(location)
    if simfeat:
        try:
            from .continuous_live import simulation_features
            extra = simulation_features(location, frame, project_root, models_dir, simfeat['core'])
            names, tag = simfeat['names'], simfeat['tag']
        except Exception as exc:
            note = f"{simfeat['tag']} unavailable: {type(exc).__name__}: {exc}"[:300]
            print(f"  [simfeat] {note}; using {MODEL_TAG}")
    predict = ensemble_predictor(names, location, models_dir=models_dir, extra_features=extra)
    predict_previous = (ensemble_predictor(HYBRID_MODELS, location, models_dir=models_dir)
                        if with_previous and tag != MODEL_TAG else None)
    members: Dict[str, np.ndarray] = {}
    previous: Dict[str, np.ndarray] = {}
    rain_means: Dict[str, np.ndarray] = {}
    for i in range(n_members):
        cols = {f'{g}_member_{i}': g for g in gauges if f'{g}_member_{i}' in forecast.columns}
        if not cols:
            continue
        member_rain = forecast[list(cols)].rename(columns=cols).reindex(fut_idx).fillna(0.0)
        members[f'member_{i}'] = predict(frame, member_rain)
        if predict_previous is not None:
            previous[f'member_{i}'] = predict_previous(frame, member_rain)
        rain_means[f'member_{i}'] = member_rain.clip(lower=0).mean(axis=1).to_numpy()
    if not members:
        raise RuntimeError('no forecast-rain members available for the hybrid')
    index = pd.date_range(t0, periods=HORIZON + 1, freq='1h')
    members_df = pd.DataFrame(members, index=index)
    previous_df = pd.DataFrame(previous, index=index) if previous else None
    rain_df = pd.DataFrame(rain_means, index=fut_idx)
    try:
        total = model_error_samples(location, frame, members_df, rain_df, models_dir, tag=tag)
    except Exception as exc:   # the spread is extra; never lose the forecast over it
        print(f"  [model error] failed ({exc}); showing rain spread only")
        total = None
    stepped = (None, None, None, None)
    if STEP_LAYER:
        try:
            stepped = step_forecast(location, frame, members_df, rain_df, total, models_dir)
        except Exception as exc:   # never lose the forecast over the step layer either
            print(f"  [step layer] failed ({type(exc).__name__}: {exc}); no stepped copy")
    return LiveForecast(members=members_df, member_rain=rain_df, total_samples=total, model_name=tag, note=note,
                        previous_members=previous_df, stepped_members=stepped[0], stepped_total=stepped[1],
                        flag_outlook=stepped[2], step_triggers=stepped[3],
                        level_hist=frame['differential'].loc[:t0].to_numpy()[-49:])


def _all_differentials() -> Dict[str, pd.Series]:
    """Hourly differential of every stretch from one EA fetch (cached per process), for the jump triggers."""
    if 'levels' not in _cache:
        from ..data.api import (calculate_godstow_differential, calculate_isis_differential,
                                calculate_wallingford_differential, fetch_river_level_data)
        out: Dict[str, pd.Series] = {}
        try:
            levels = fetch_river_level_data(verbose=False)
            iu = levels.get('iffley_upstream')
            if iu is not None and not iu.empty and 'level' in iu.columns:
                _cache['iffley_upstream'] = _to_utc(iu[['level']])['level']
            for loc, fn in (('isis', calculate_isis_differential), ('godstow', calculate_godstow_differential),
                            ('wallingford', calculate_wallingford_differential)):
                try:
                    out[loc] = _to_utc(fn(levels))['differential']
                except Exception as exc:
                    print(f"  [step layer] no {loc} differential for the jump triggers ({exc})")
        except Exception as exc:
            print(f"  [step layer] river-level fetch for the jump triggers failed ({exc})")
        _cache['levels'] = out
    return _cache['levels']


def iffley_upstream_level() -> Optional[pd.Series]:
    """Iffley Lock upstream stage (m), last ~4 weeks, for the Black-flag indicator."""
    _all_differentials()
    return _cache.get('iffley_upstream')


def step_forecast(location: str, frame: pd.DataFrame, members_df: pd.DataFrame, rain_df: pd.DataFrame,
                  total: Optional[np.ndarray], models_dir: Path):
    """(stepped members, stepped outcomes, flag-band outlook, triggers seen); Nones without parameters."""
    from ..config import get_flag_thresholds
    from .step_layer import apply_live
    t0 = members_df.index[0]
    gauges = [g for g in TRAINING_GAUGES[location] if g in frame.columns]
    level_hist = frame['differential'].loc[:t0].to_numpy()[-49:]
    rain_hist = frame[gauges].clip(lower=0).mean(axis=1).fillna(0.0).loc[:t0].to_numpy()[-24:]
    grid = pd.date_range(t0 - pd.Timedelta(hours=11), t0, freq='1h')
    neighbours = {loc: ser.reindex(grid).to_numpy() for loc, ser in _all_differentials().items() if loc != location}
    out = apply_live(location, level_hist, rain_hist, members_df.to_numpy().T, rain_df.to_numpy().T, total, t0,
                     get_flag_thresholds(location), models_dir, neighbour_hists=neighbours)
    if out is None:
        print(f"  [step layer] no parameters for {location}; no stepped copy")
        return None, None, None, None
    members_out = pd.DataFrame(out.members.T, index=members_df.index, columns=members_df.columns)
    o = out.outlook
    print(f"  [step layer] jumps in the last 3 h: {out.triggers}")
    if o:
        w = o['windows']
        print(f"  [step layer] flag now {o['current_flag']}; change up today {w['today']['up']:.0%} / by tomorrow "
              f"{w['tomorrow']['up']:.0%}, down {w['today']['down']:.0%} / {w['tomorrow']['down']:.0%}")
    return members_out, out.total, o, out.triggers


def forecast_rain_gauges(location: str) -> List[str]:
    return list(TRAINING_GAUGES[location])
