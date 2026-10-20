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
from ..models.physics_models import M3S_PER_MM_H

HYBRID_MODELS = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
MODEL_TAG = 'hybrid_c1_ps'
# Optional per-location override with the hybrid that also gets continuous-
# simulation encoder inputs (hybrid_simfeat; see the October section of
# EXPERIMENTS.md). Off: all three locations use the plain hybrid above.
# Example: {'isis': {'tag': 'hybrid_simfeat', 'core': 'cs_cos',
#                    'names': ['hybrid_simfeat'] + [f'hybrid_simfeat_s{i}' for i in range(1, 5)]}}
SIMFEAT_MODELS: dict = {}
# Trial models with extra encoder inputs, for the preview page only (generate_all_location_figures
# --preview-model). hybrid_su: measured soil moisture and upstream EA levels (models/extra_inputs.py).
VARIANTS: dict = {
    'hybrid_su': {'names': ['hybrid_su'] + [f'hybrid_su_s{i}' for i in range(1, 5)], 'extra': 'soil_upstream'},
}


def variant_extra(variant: str, index: pd.DatetimeIndex, project_root: Path) -> pd.DataFrame:
    key = f'extra_{variant}_{index[0]}_{index[-1]}'
    if key not in _cache:
        from ..models.extra_inputs import live_extra
        _cache[key] = live_extra(index, project_root)
    return _cache[key]
# Model-error library (build_error_library.py); missing file = rain spread only.
ERROR_LIBRARY = 'error_library_{tag}_{location}.npz'
HORIZON = 240
HISTORY_DAYS = 60
# Lock-keeper step layer (step_layer.py): a stepped copy of every outcome (held
# levels and daytime jumps) plus the flag-change box, for the preview page.
# The main page keeps the smooth forecast. Off, or a missing
# models/step_layer_<location>.json, = no stepped copy.
STEP_LAYER = True
# Gauge outages up to this long, once the gauge is back, are bridged before the encoder sees
# them (fill_outage; gap test in EXPERIMENTS.md "Gauge outages")...
OUTAGE_FILL_MAX_H = 14 * 24
# ...but only for the first hours after it is back: by then the bridge helps the first two
# days a lot (Isis, 7-12 day outage, 1 h back: 48 h error 2.0 cm vs 3.0-3.3 cm unbridged,
# 1.9 cm with no outage); a day after, the unbridged forecast is as good or better.
OUTAGE_BRIDGE_FOR_H = 12
# Neighbouring stretch for each, with the 2019-2026 slope of 24 h changes (this on neighbour);
# the slope used is this, pulled towards the slope fitted on the two weeks before the outage
# when the neighbour moved enough then to show it.
OUTAGE_NEIGHBOUR = {'isis': ('godstow', 0.67), 'godstow': ('isis', 0.96), 'wallingford': ('isis', 1.18)}
# A reading older than this = the gauge is offline (the page's "stale" label).
STALE_H = 4

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


def fill_outage(diff: pd.Series, neighbour: Optional[pd.Series] = None, slope: float = 1.0,
                max_hours: int = OUTAGE_FILL_MAX_H, fit_hours: int = 14 * 24) -> pd.Series:
    """Bridge gauge outages (runs of missing hours between two readings).

    The encoder only interpolates gaps up to a day itself; a longer hole left as NaN
    inside its 100 h window skews the forecast once the gauge is back. A hole longer
    than a day is shaped by a neighbouring stretch's differential when that has data
    through the hole (d = b * neighbour + a residual drawn straight between the
    readings either side; b is ``slope`` pulled towards the slope fitted on the
    ``fit_hours`` before the hole when the neighbour moved enough then), otherwise
    bridged with a straight line. Holes at the start of the frame (before the API
    window) and runs longer than ``max_hours`` stay NaN.
    """
    na = diff.isna().to_numpy()
    if not na.any() or (~na).sum() < 2:
        return diff
    out = diff.copy()
    straight = diff.interpolate(method='time', limit_area='inside')
    nb = neighbour.reindex(diff.index) if neighbour is not None else None
    edges = np.flatnonzero(np.diff(np.r_[0, na.astype(int), 0]))
    shaped = bridged = 0
    for i0, i1 in zip(edges[::2], edges[1::2]):          # missing run diff[i0:i1]
        if i0 == 0 or i1 == len(diff) or i1 - i0 > max_hours:
            continue
        seg = straight.iloc[i0:i1]
        if nb is not None and i1 - i0 > 24 and nb.iloc[i0:i1].notna().mean() > 0.8:
            pre = pd.DataFrame({'d': diff.iloc[max(0, i0 - fit_hours):i0],
                                'n': nb.iloc[max(0, i0 - fit_hours):i0]}).dropna()
            ends = nb.iloc[[i0 - 1, i1]]
            if len(pre) >= 48 and ends.notna().all():
                var = float(pre['n'].var())
                fit = float(np.polyfit(pre['n'], pre['d'], 1)[0]) if var > 0 else slope
                w = var / (var + 0.03 ** 2)
                b = float(np.clip(w * fit + (1 - w) * slope, 0.0, 2.5))
                n_seg = nb.iloc[i0 - 1:i1 + 1].interpolate(limit_area='inside')
                r = (diff.iloc[i0 - 1:i1 + 1] - b * n_seg).interpolate(method='time', limit_area='inside')
                seg = (r + b * n_seg).iloc[1:-1]
                shaped += i1 - i0
        out.iloc[i0:i1] = seg.to_numpy()
        bridged += i1 - i0
    if bridged:
        print(f"  [beta] bridged {bridged} missing differential hours (gauge outage); "
              f"{shaped} shaped by the neighbouring stretch, the rest a straight line")
    return out


def blend_slope(pre_d: pd.Series, pre_n: pd.Series, slope: float) -> float:
    """The long-run neighbour slope, pulled towards the slope fitted on the weeks before an outage
    when the neighbour moved enough then to show it (as fill_outage)."""
    pre = pd.DataFrame({'d': pre_d, 'n': pre_n}).dropna()
    if len(pre) < 48:
        return slope
    var = float(pre['n'].var())
    fit = float(np.polyfit(pre['n'], pre['d'], 1)[0]) if var > 0 else slope
    w = var / (var + 0.03 ** 2)
    return float(np.clip(w * fit + (1 - w) * slope, 0.0, 2.5))


def neighbour_nowcast(diff: pd.Series, neighbour: pd.Series, slope: float, until: pd.Timestamp,
                      fit_hours: int = 14 * 24) -> Optional[pd.Series]:
    """This stretch's differential after its last reading, estimated from the neighbour while the
    gauge is offline: d(t) = d_last + b * (n(t) - n(t_last)), hourly to ``until``.

    ``diff`` ends at (or holds NaN after) the last reading; ``neighbour`` must have data at the
    last reading and through to ``until`` (short gaps are interpolated). None if it hasn't.
    """
    d = diff.dropna()
    if len(d) == 0:
        return None
    t_last = d.index[-1]
    if until <= t_last:
        return None
    grid = pd.date_range(t_last, until, freq='1h')
    n = neighbour.reindex(neighbour.index.union(grid)).interpolate(method='time', limit=6,
                                                                   limit_area='inside').reindex(grid)
    if not np.isfinite(n.iloc[0]) or n.iloc[1:].isna().mean() > 0.2 or not np.isfinite(n.iloc[-1]):
        return None
    n = n.ffill()
    pre = slice(t_last - pd.Timedelta(hours=fit_hours), t_last)
    b = blend_slope(d.loc[pre], neighbour.reindex(d.loc[pre].index), slope)
    est = float(d.iloc[-1]) + b * (n - float(n.iloc[0]))
    return est.iloc[1:]


# Error of neighbour_nowcast (Isis <-> Godstow, 2019-2026, backtest_outage_nowcast.py): its standard
# deviation grows about as the square root of the time the gauge has been down, larger in the
# middle of the range where the flags change. sd = a * sqrt(days) (x mid factor in the band given).
NOWCAST_ERROR = {
    'isis': {'a': 0.026, 'mid': (0.215, 0.535), 'mid_factor': 1.38},
    'godstow': {'a': 0.022, 'mid': (0.30, 1.0), 'mid_factor': 1.61},
}
# Beyond this long (5 days) the estimate is too loose to warn on: the page marks it low confidence and
# no "likely" alerts are sent for that stretch.
NOWCAST_CONFIDENT_H = 5 * 24


def nowcast_error_sd(location: str, hours: float, level: float) -> float:
    p = NOWCAST_ERROR.get(location)
    if not p:
        return 0.0
    sd = p['a'] * np.sqrt(max(hours, 1.0) / 24.0)
    lo, hi = p['mid']
    return float(sd * (p['mid_factor'] if lo <= level < hi else 1.0))


def recent_gauge_rain(stations: List[str]) -> Optional[pd.Series]:
    """Hourly mean rain (mm) over these gauges from the EA API, UTC-naive index, or None.
    For the page's rain history while a stretch's river gauges are offline."""
    api = _api_rain()
    cols = [c for c in stations if c in api.columns]
    if not cols:
        return None
    out = api[cols].clip(lower=0).mean(axis=1)
    out.index = out.index.tz_convert('UTC').tz_localize(None)
    return out


def _recent_flow() -> Optional[pd.Series]:
    if 'flow' not in _cache:
        try:
            from ..data.api import fetch_flow_data
            f = fetch_flow_data(verbose=False)   # same request as the data pipeline (90 days)
            _cache['flow'] = _to_utc(f)['flow_m3s_Farmoor'].resample('1h').mean() if len(f) else None
        except Exception as exc:
            print(f"  [nowcast] no Farmoor flow ({exc})")
            _cache['flow'] = None
    return _cache['flow']


def outage_nowcast(location: str, merged_df: pd.DataFrame, now: Optional[pd.Timestamp] = None):
    """(history, info): while this stretch's gauges are offline (last reading more than
    STALE_H old) and its neighbour's are not, the history extended hourly to the neighbour's
    last reading with the differential estimated from it (neighbour_nowcast), so the forecast
    starts now instead of at the last reading. info = {'from', 'last_reading', 'hours', ...};
    (merged_df, None) when there is no outage or nothing to estimate from.
    """
    if location not in NOWCAST_ERROR or len(merged_df) == 0:
        return merged_df, None
    merged = _to_utc(merged_df)
    now = (pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)).floor('h')
    now = now.tz_localize('UTC') if now.tz is None else now.tz_convert('UTC')
    diff = merged['differential']
    t_last = diff.last_valid_index()
    if t_last is None or now - t_last <= pd.Timedelta(hours=STALE_H):
        return merged_df, None
    other, slope = OUTAGE_NEIGHBOUR[location]
    nb = _all_differentials().get(other)
    if nb is None or nb.dropna().empty:
        print(f"  [nowcast] {location}: gauges offline, no {other} differential to estimate from")
        return merged_df, None
    nb = nb.resample('1h').mean()
    until = nb.last_valid_index()
    if until is None or now - until > pd.Timedelta(hours=STALE_H) or until <= t_last:
        print(f"  [nowcast] {location}: gauges offline and {other}'s too; no estimate")
        return merged_df, None
    if until - t_last > pd.Timedelta(hours=OUTAGE_FILL_MAX_H):
        print(f"  [nowcast] {location}: offline for more than {OUTAGE_FILL_MAX_H // 24} days; no estimate")
        return merged_df, None
    est = neighbour_nowcast(diff.loc[:t_last], nb, slope, until)
    if est is None:
        print(f"  [nowcast] {location}: {other} has no reading at the last {location} reading; no estimate")
        return merged_df, None
    ext = pd.DataFrame(index=est.index, columns=merged.columns, dtype=float)
    ext['differential'] = est.to_numpy()
    # The frame stopped at the last reading for every column: carry the gauge rain on too, so the
    # page's rain history and totals keep going through the outage.
    try:
        api = _api_rain()
        shared = [c for c in ext.columns if c in api.columns]
        if shared:
            ext[shared] = api.reindex(index=ext.index, columns=shared).to_numpy()
    except Exception as exc:
        print(f"  [nowcast] no gauge rain for the outage hours ({exc})")
    if 'flow_m3s_Farmoor' in ext.columns:
        flow = _recent_flow()
        known = merged['flow_m3s_Farmoor']
        if flow is not None:
            known = known.combine_first(flow)
        else:
            print("  [nowcast] no recent Farmoor flow; the encoder sees it as missing")
        # Carried forward as the data pipeline does for every stretch (merge_and_clean_data): the
        # EA publishes the qualified Farmoor flow days late, so its last value usually stands in.
        ext['flow_m3s_Farmoor'] = known.reindex(known.index.union(ext.index)).ffill().reindex(ext.index)
    out = pd.concat([merged, ext])
    out.attrs = dict(merged_df.attrs)
    tz = merged_df.index.tz
    out.index = out.index.tz_convert(tz) if tz is not None else out.index.tz_localize(None)
    hours = int((until - t_last) / pd.Timedelta(hours=1))
    # The EA time of that last reading, when the frame kept it (the hourly row can be later).
    reading = (merged_df.attrs.get('reading_time') or {}).get(t_last.tz_convert('UTC').tz_localize(None))
    info = {'from': other, 'last_reading': t_last, 'reading_time': reading, 'until': until, 'hours': hours,
            'confident': hours <= NOWCAST_CONFIDENT_H, 'level': float(est.iloc[-1]),
            'sd': nowcast_error_sd(location, hours, float(est.iloc[-1]))}
    print(f"  [nowcast] {location}: gauges offline {hours} h; differential now estimated from {other} "
          f"as {info['level']:.3f} m (sd {info['sd']:.3f} m)")
    return out, info


def _hours_since_outage(diff: pd.Series) -> Optional[int]:
    """Hours of readings since the last run of more than a day of missing hours, or None."""
    na = diff.isna().to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, na.astype(int), 0]))
    ends = [i1 for i0, i1 in zip(edges[::2], edges[1::2]) if i0 > 0 and i1 < len(na) and i1 - i0 > 24]
    return len(na) - ends[-1] if ends else None


def _neighbour_differential(location: str, grid: pd.DatetimeIndex) -> Optional[pd.Series]:
    """The best-correlated other stretch's hourly differential (EA, ~4 weeks), or None."""
    other = OUTAGE_NEIGHBOUR.get(location, (None, 1.0))[0]
    ser = _all_differentials().get(other) if other else None
    if ser is None or len(ser) == 0:
        return None
    return ser.resample('1h').mean().reindex(grid)


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
    back = _hours_since_outage(diff)
    if back is not None and back <= OUTAGE_BRIDGE_FOR_H:
        diff = fill_outage(diff, _neighbour_differential(location, grid), OUTAGE_NEIGHBOUR[location][1])
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
    rain_hist: Optional[np.ndarray] = None        # hourly mean-station rain (mm), last 72 h to t0
    model_state: Optional[dict] = None            # the model's stores now and ahead (model_state_payload)


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
    start_sd: float = 0.0,
    variant: Optional[str] = None,
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
    if variant:
        # A trial model with extra encoder inputs (VARIANTS); errors propagate, so a failed trial
        # never quietly shows the deployed model under the trial's name.
        v = VARIANTS[variant]
        names = [n for n in v['names'] if (models_dir / f'redesign_{n}_{location}.pt').exists()]
        if not names:
            raise FileNotFoundError(f'no {variant} models for {location}')
        extra = variant_extra(variant, frame.index, project_root)
        tag, simfeat, with_previous = variant, None, False
        print(f"  [{variant}] {len(names)} trained copies with {extra.shape[1]} extra inputs")
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
    state_runs: list = []
    for i in range(n_members):
        cols = {f'{g}_member_{i}': g for g in gauges if f'{g}_member_{i}' in forecast.columns}
        if not cols:
            continue
        member_rain = forecast[list(cols)].rename(columns=cols).reindex(fut_idx).fillna(0.0)
        if hasattr(predict, 'with_states'):
            d_i, q_i, st_i = predict.with_states(frame, member_rain)
            members[f'member_{i}'] = d_i
            state_runs.append((q_i, st_i))
        else:
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
    if start_sd > 0:
        # Starting level estimated (gauges offline, outage_nowcast): every outcome is shifted by a
        # draw of that estimate's error, so the likely range and flag chances carry it.
        base = total if total is not None else members_df.to_numpy().T
        rng = np.random.default_rng(1)
        total = base + rng.normal(0.0, start_sd, size=(base.shape[0], 1))
        print(f"  [nowcast] outcomes widened by the starting estimate's error (sd {start_sd:.3f} m)")
    stepped = (None, None, None, None)
    if STEP_LAYER:
        try:
            stepped = step_forecast(location, frame, members_df, rain_df, total, models_dir)
        except Exception as exc:   # never lose the forecast over the step layer either
            print(f"  [step layer] failed ({type(exc).__name__}: {exc}); no stepped copy")
    model_state = None
    try:
        if state_runs:
            model_state = model_state_payload(location, names, models_dir, index, state_runs, frame)
            if variant and extra is not None and 'soil_vwc' in extra:
                sv = extra['soil_vwc'].dropna()
                if len(sv):
                    model_state['measured_soil'] = {'value': round(float(sv.iloc[-1]), 1), 'site': 'Chimney Meadows',
                                                    'source': 'COSMOS-UK'}
            if variant:
                model_state['variant'] = variant
    except Exception as exc:   # extra, for the preview page; never lose the forecast over it
        print(f"  [model state] skipped ({type(exc).__name__}: {exc})")
    return LiveForecast(members=members_df, member_rain=rain_df, total_samples=total, model_name=tag, note=note,
                        model_state=model_state,
                        previous_members=previous_df, stepped_members=stepped[0], stepped_total=stepped[1],
                        flag_outlook=stepped[2], step_triggers=stepped[3],
                        level_hist=frame['differential'].loc[:t0].to_numpy()[-49:],
                        rain_hist=frame[[g for g in gauges if g in frame.columns]].clip(lower=0).mean(axis=1)
                        .fillna(0.0).loc[:t0].to_numpy()[-72:])


STATE_NAMES = ('soil', 'quick', 'medium', 'slow', 'channel')
STATE_EVERY_H = 3


def model_state_payload(location: str, names: List[str], models_dir: Path, index: pd.DatetimeIndex,
                        runs: list, frame: pd.DataFrame) -> dict:
    """The hybrid's internal stores (mean of the trained copies), now and over the forecast.

    runs: per rain member, (outflow mm/h (240,), states (241, 5)). Stores are mm of water over the
    catchment; soil is the wetness fraction (0-1); flow is the channel's outflow in m³/s.
    Series every STATE_EVERY_H hours: mean and 10th/90th percentiles over the rain members.
    """
    from ..models.physics_train import load_candidate
    st = np.stack([r[1] for r in runs])                     # (M, 241, 5)
    q = np.stack([r[0] for r in runs]) * M3S_PER_MM_H       # (M, 240) m³/s
    flow = np.concatenate([q[:, :1], q], axis=1)            # t0 shown as the first hour's outflow
    keep = np.arange(0, st.shape[1], STATE_EVERY_H)
    series = {}
    for j, name in enumerate(STATE_NAMES + ('flow',)):
        v = flow if name == 'flow' else st[:, :, j]
        if name == 'soil':
            v = v * 100.0
        v = v[:, keep]
        series[name] = {'mean': np.round(v.mean(0), 2).tolist(),
                        'p10': np.round(np.percentile(v, 10, axis=0), 2).tolist(),
                        'p90': np.round(np.percentile(v, 90, axis=0), 2).tolist()}
    now = {name: series[name]['mean'][0] for name in series}
    params = {}
    for n in names:
        for k, v in load_candidate(n, location, models_dir)[0].describe().items():
            params.setdefault(k, []).append(v)
    params = {k: round(float(np.mean(v)), 4) for k, v in params.items()}
    out = {'times': [t.tz_convert('UTC').tz_localize(None).isoformat() for t in index[keep]],
           'now': now, 'series': series, 'params': params, 'every_h': STATE_EVERY_H}
    # The last real Farmoor measurement (the frame's flow is carried forward over the EA's delay).
    f = _recent_flow()
    f = f.dropna() if f is not None else pd.Series(dtype=float)
    if not len(f) and 'flow_m3s_Farmoor' in frame:
        f = frame['flow_m3s_Farmoor'].loc[:index[0]].dropna()
    if len(f):
        out['farmoor_flow'] = {'value': round(float(f.iloc[-1]), 1),
                               'time': f.index[-1].tz_convert('UTC').tz_localize(None).isoformat()}
    print(f"  [model state] soil {now['soil']:.0f}% wet; stores quick {now['quick']:.1f}, medium "
          f"{now['medium']:.1f}, slow {now['slow']:.1f}, channel {now['channel']:.1f} mm; flow {now['flow']:.1f} m³/s")
    return out


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
