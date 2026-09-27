"""
Shared evaluation for differential forecasters (physics-redesign branch).

Every candidate is a callable ``predict(history, future_rain) -> np.ndarray``
returning 241 values: index 0 is t0 (the last observed hour) and 1..240 are
the forecast hours. ``history`` is the merged frame up to and including t0;
``future_rain`` holds the station rainfall columns for the 240 forecast hours.

All forecasts use clairvoyant (observed) rainfall unless perturbed, and the
realism metrics are computed on whatever the candidate returns, so candidates
that should be judged "raw" must be built with their clamps switched off.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PHYSICAL_CONSTRAINTS
from .processing.features import rain_station_columns

HORIZON = 240
MAX_DROP_PER_H = PHYSICAL_CONSTRAINTS['max_recession_m_per_day'] / 24.0

# Rain affects dD/dt for ~48–60h after it falls (rain→dD/dt cross-correlation
# peaks at 6h and stays above a quarter of the peak to 48–60h at all three
# locations). A forecast hour is "dry" when mean station rain over the
# preceding DRY_LOOKBACK_H hours (observed past + forecast) is below
# DRY_RAIN_MM; the forecast should not climb during such hours.
DRY_LOOKBACK_H = 72
DRY_RAIN_MM = 2.0
# Tolerances (m) so float noise does not count as a violation.
CLIMB_TOL = 0.001
MONO_TOL = 0.005
REVERSAL_DEADBAND = 0.0005

LEAD_BINS = {'mae_1_24h': (1, 24), 'mae_25_72h': (25, 72), 'mae_73_240h': (73, 240)}
RAIN_SCALES = (0.0, 0.5, 1.0, 1.5)

Predictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def load_cleaned_differential(location: str, project_root: Path) -> pd.DataFrame:
    """The cleaned merged frame the September backtest uses (rows only where
    the differential exists; rain/flow there were zero-/forward-filled)."""
    cache = project_root / 'data' / f'backtest_merged_{location}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    from .pipeline import prepare_training_data
    df, _, _ = prepare_training_data(
        location=location, project_root=project_root, include_api=False, verbose=False
    )
    df.to_pickle(cache)
    return df


def load_merged(location: str, project_root: Path) -> pd.DataFrame:
    """
    Continuous hourly frame: the cleaned differential (NaN in its gaps) with
    the FULL rainfall and Farmoor flow records alongside.

    The September merged frame only has rows where the differential exists, so
    on a regular grid the rain and flow vanish during differential gaps (14% of
    hours at Isis, 67% of 2019) and the feature code then treats that missing
    rain as dry weather. Here a station that is not reporting stays NaN rather
    than zero; flow gaps stay NaN.
    """
    cache = project_root / 'data' / f'full_frame_{location}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    from .data.loader import load_historical_flow, load_historical_rainfall

    diff_df = load_cleaned_differential(location, project_root)
    stations = rain_station_columns(diff_df)
    stations = [c for c in stations if not c.startswith('flow_m3s_')]
    start = diff_df.index.min() - pd.Timedelta(days=60)
    end = diff_df.index.max()
    grid = pd.date_range(start.floor('h'), end.ceil('h'), freq='1h', tz=diff_df.index.tz)

    raw_rain = load_historical_rainfall(project_root=project_root, location=location, verbose=False)
    if raw_rain.index.tz is None:
        raw_rain = raw_rain.tz_localize('UTC')
    renamed = {}
    for col in raw_rain.columns:
        after = col.split('mm_', 1)[1] if 'mm_' in col else col
        renamed[col] = after.split('-')[0]
    raw_rain = raw_rain.rename(columns=renamed)
    raw_rain = raw_rain.loc[:, ~raw_rain.columns.duplicated()]
    rain = raw_rain.reindex(columns=stations).astype(float)
    rain = rain.resample('1h').sum(min_count=1).clip(lower=0, upper=50).reindex(grid)

    flow = load_historical_flow(project_root=project_root)
    if flow.index.tz is None:
        flow = flow.tz_localize('UTC')
    flow = flow[['flow_m3s_Farmoor']].astype(float).resample('1h').mean().clip(lower=0).reindex(grid)

    df = pd.concat([diff_df['differential'].reindex(grid), rain, flow], axis=1)
    df.index.name = None
    df.to_pickle(cache)
    return df


def forecast_starts(merged: pd.DataFrame, window: str) -> List[pd.Timestamp]:
    """Weekly t0s. 'test' matches the May/June/Sept backtest; 'val' is 2023."""
    ranges = {
        'test': ('2024-11-01', '2026-01-05'),
        'val': ('2023-01-02', '2023-12-25'),
    }
    lo, hi = ranges[window]
    actual = merged['differential']
    t0s = []
    for t in pd.date_range(lo, hi, freq='7D', tz=merged.index.tz):
        if t not in merged.index or pd.isna(actual.get(t)):
            continue
        if actual.loc[t:t + pd.Timedelta(hours=HORIZON)].count() > 200:
            t0s.append(t)
    return t0s


def mean_station_rain(df: pd.DataFrame) -> pd.Series:
    """Mean rain over the stations reporting that hour (mm/h); 0 if none."""
    cols = rain_station_columns(df)
    return df[cols].clip(lower=0).mean(axis=1).fillna(0.0)


def future_rain_frame(merged: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Station rain for the 240 forecast hours; missing stations stay NaN."""
    cols = rain_station_columns(merged)
    idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
    return merged[cols].reindex(idx)


def perturb_rain(rain: pd.DataFrame, shift_h: int, rng: np.random.Generator) -> pd.DataFrame:
    """Imperfect forecast: shift in time and multiply by 6-hourly lognormal noise."""
    vals = rain.to_numpy(dtype=float)
    shifted = np.zeros_like(vals)
    if shift_h >= 0:
        shifted[shift_h:] = vals[: len(vals) - shift_h]
    else:
        shifted[: len(vals) + shift_h] = vals[-shift_h:]
    sigma = 0.6
    blocks = int(np.ceil(len(vals) / 6))
    factors = np.exp(sigma * rng.standard_normal(blocks) - sigma ** 2 / 2)
    shifted *= np.repeat(factors, 6)[: len(vals), None]
    return pd.DataFrame(shifted, index=rain.index, columns=rain.columns)


def dry_hours(past_rain: np.ndarray, future_rain: np.ndarray) -> np.ndarray:
    """Boolean (240,) — forecast hour h (1..240) has < DRY_RAIN_MM in prior window."""
    series = np.concatenate([past_rain[-DRY_LOOKBACK_H:], future_rain])
    csum = np.concatenate([[0.0], np.cumsum(series)])
    n_past = len(past_rain[-DRY_LOOKBACK_H:])
    out = np.zeros(len(future_rain), dtype=bool)
    for h in range(len(future_rain)):
        end = n_past + h + 1
        start = max(0, end - DRY_LOOKBACK_H)
        out[h] = (csum[end] - csum[start]) < DRY_RAIN_MM
    return out


def trajectory_realism(pred: np.ndarray, dry: np.ndarray) -> Dict[str, float]:
    """Realism of one 241-value trajectory (pred[0] is t0)."""
    step = np.diff(pred)                      # (240,) change into hour h
    curv = np.diff(step)                      # (239,)
    climb = np.clip(step - CLIMB_TOL / 10, 0, None)
    moving = step[np.abs(step) > REVERSAL_DEADBAND]
    reversals = int(np.sum(np.sign(moving[1:]) != np.sign(moving[:-1]))) if len(moving) > 1 else 0
    dry_climb = float(np.sum(climb[dry]))
    return {
        'dry_climb_cm': dry_climb * 100,
        'dry_climb_any': float(dry_climb > 0.01),
        'roughness_mm': float(np.mean(np.abs(curv)) * 1000),
        'max_kink_mm': float(np.max(np.abs(curv)) * 1000),
        'reversals': reversals,
        'recession_viol_frac': recession_fraction(pred),
    }


def recession_fraction(values: np.ndarray) -> float:
    """Share of hours whose next-24h fall beats 3 in/day by >10% (NaN-aware)."""
    fall24 = values[24:] - values[:-24]
    ok = np.isfinite(fall24)
    if not ok.any():
        return float('nan')
    return float(np.mean(fall24[ok] < -1.1 * PHYSICAL_CONSTRAINTS['max_recession_m_per_day']))


@dataclass
class EvalConfig:
    location: str
    window: str = 'test'
    scaling: bool = True
    perturb: bool = True
    n_perturb: int = 4
    seed: int = 0


def evaluate(
    predict: Predictor,
    merged: pd.DataFrame,
    cfg: EvalConfig,
    trajectories: Optional[Dict[str, list]] = None,
) -> pd.DataFrame:
    """One row per forecast start with accuracy, realism and stress-test metrics.

    If ``trajectories`` is given, the clairvoyant and zero-rain forecasts, the
    observations and the forecast-period rain are appended to it per start.
    """
    actual = merged['differential']
    rain_mean = mean_station_rain(merged)
    rows = []
    for t0 in forecast_starts(merged, cfg.window):
        history = merged.loc[:t0]
        fut = future_rain_frame(merged, t0)
        act = actual.reindex(pd.date_range(t0, periods=HORIZON + 1, freq='1h')).to_numpy()
        past_rain = rain_mean.loc[:t0].to_numpy()[-DRY_LOOKBACK_H:]
        fut_mean = fut.clip(lower=0).mean(axis=1).fillna(0.0).to_numpy()

        pred = np.asarray(predict(history, fut), dtype=float)
        err = np.abs(pred - act)
        row = {
            't0': t0,
            'winter': t0.month in (11, 12, 1, 2, 3),
            'current': float(act[0]),
            'mae': float(np.nanmean(err[1:])),
            'bias': float(np.nanmean((pred - act)[1:])),
        }
        for name, (lo, hi) in LEAD_BINS.items():
            row[name] = float(np.nanmean(err[lo:hi + 1]))
        row.update(trajectory_realism(pred, dry_hours(past_rain, fut_mean)))
        row['obs_recession_frac'] = recession_fraction(act)
        zero = None

        if cfg.scaling:
            runs = {}
            for s in RAIN_SCALES:
                runs[s] = pred if s == 1.0 else np.asarray(predict(history, fut * s), dtype=float)
            viol = []
            for lo_s, hi_s in zip(RAIN_SCALES[:-1], RAIN_SCALES[1:]):
                viol.append(np.mean(runs[hi_s][1:] < runs[lo_s][1:] - MONO_TOL))
            row['mono_viol_frac'] = float(np.mean(viol))
            zero = runs[0.0]
            zero_dry = dry_hours(past_rain, np.zeros(HORIZON))
            z = trajectory_realism(zero, zero_dry)
            row['zero_rain_dry_climb_cm'] = z['dry_climb_cm']
            row['zero_rain_reversals'] = z['reversals']
            # After 72h any water already in transit should have arrived:
            # with no future rain the river should only fall or hold.
            running_min = np.minimum.accumulate(zero[DRY_LOOKBACK_H:])
            row['zero_rain_late_climb_cm'] = float(np.max(zero[DRY_LOOKBACK_H:] - running_min) * 100)
            row['mono_viol_frac_pos'] = float(np.mean([
                np.mean(runs[b][1:] < runs[a][1:] - MONO_TOL)
                for a, b in zip(RAIN_SCALES[1:-1], RAIN_SCALES[2:])
            ]))

        if cfg.perturb:
            rng = np.random.default_rng(cfg.seed + int(t0.value // 3_600_000_000_000) % 100_000)
            shifts = [-24, -12, 12, 24][: cfg.n_perturb]
            maes = []
            for sh in shifts:
                p = np.asarray(predict(history, perturb_rain(fut, sh, rng)), dtype=float)
                maes.append(np.nanmean(np.abs(p - act)[1:]))
            row['mae_perturbed'] = float(np.mean(maes))
        rows.append(row)
        if trajectories is not None:
            for key, value in (('t0', t0), ('pred', pred), ('actual', act),
                               ('zero_rain', zero), ('rain', fut_mean)):
                trajectories.setdefault(key, []).append(value)
    return pd.DataFrame(rows)


SUMMARY_COLS = [
    'mae', 'mae_1_24h', 'mae_25_72h', 'mae_73_240h', 'bias', 'mae_winter',
    'mae_perturbed', 'dry_climb_cm', 'dry_climb_any', 'roughness_mm', 'max_kink_mm',
    'reversals', 'recession_viol_frac', 'obs_recession_frac', 'mono_viol_frac', 'mono_viol_frac_pos', 'zero_rain_dry_climb_cm',
    'zero_rain_late_climb_cm', 'zero_rain_reversals',
]


def summarise(results: pd.DataFrame) -> pd.Series:
    out = results.drop(columns=['t0', 'winter'], errors='ignore').mean(numeric_only=True)
    out['mae_winter'] = results.loc[results['winter'], 'mae'].mean()
    out['n_starts'] = len(results)
    cols = [c for c in SUMMARY_COLS if c in out.index] + ['n_starts']
    return out[cols]
