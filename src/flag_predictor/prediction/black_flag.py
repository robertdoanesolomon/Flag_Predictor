"""
Black flag (Isis): a forecast of the Iffley Lock upstream level.

The official Isis flag goes Black once the Iffley upstream stage (EA 1501TH) reaches about
BLACK_UP_M and back to Red once it falls below about BLACK_DOWN_M (official_flag.py; the rule
matches the flag in 98% of Red/Black hours since 2019). The differential cannot tell Red from
Black: it saturates near 0.95.

Iffley moves slowly (24 h changes have a spread of about 6 cm when it is high) and is fed by
the Cherwell, which none of the differentials see. Forecast differentials add nothing
(fit_black_flag.py), so the level h hours ahead (h = 1..48) comes from a linear model per lead
of: the level now and its 6 h and 24 h change, the Isis and Godstow differentials now, the
rain of the last 24 h and 72 h and the forecast rain over the h hours (mean over the hybrid's
gauges, mm). Each rain member gives a mean path; a random walk with the model's spread at
each lead is added, and the hysteresis rule is run along every path.

It cannot see a sudden rise after Iffley has been drawn down ahead of a flood: before Storm
Bert (25 Nov 2024) Iffley fell 12 cm to 0.01 m, then rose to 0.76 m within two days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .official_flag import BLACK_DOWN_M, BLACK_UP_M, LOCAL_TZ

HORIZON = 48
FULL = ['u0', 'du6', 'du24', 'I0', 'G0', 'rp24', 'rp72', 'rf']
SHORT = ['u0', 'du6', 'du24', 'rp24', 'rp72', 'rf']       # without the differentials (a gauge offline)
FIT_ABOVE_M = 0.15            # fitted on hours with Iffley above this (run at every level)
PARAMS_FILE = 'black_flag.json'
PATHS_PER_MEMBER = 20


@dataclass
class BlackParams:
    models: Dict[str, dict]       # name -> {'features', 'coef' (H, F), 'intercept' (H,), 'sd' (H,)}
    notes: Dict

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            'models': {k: {'features': v['features'], 'coef': np.round(np.asarray(v['coef']), 6).tolist(),
                           'intercept': np.round(np.asarray(v['intercept']), 6).tolist(),
                           'sd': np.round(np.asarray(v['sd']), 6).tolist()} for k, v in self.models.items()},
            'notes': self.notes}, indent=1))

    @classmethod
    def load(cls, path: Path) -> 'BlackParams':
        j = json.loads(Path(path).read_text())
        return cls({k: {'features': v['features'], 'coef': np.asarray(v['coef'], float),
                        'intercept': np.asarray(v['intercept'], float), 'sd': np.asarray(v['sd'], float)}
                    for k, v in j['models'].items()}, j.get('notes', {}))


def black_state(level: pd.Series) -> bool:
    """Hysteresis state at the end of an hourly Iffley series."""
    state = False
    for x in level.dropna().to_numpy():
        if not state and x >= BLACK_UP_M:
            state = True
        elif state and x < BLACK_DOWN_M:
            state = False
    return state


def start_features(level: pd.Series, t0: pd.Timestamp, I0: float, G0: float,
                   rain_hist: np.ndarray) -> Optional[dict]:
    """Inputs that do not depend on the lead. level: hourly Iffley stage (m), UTC index.
    rain_hist: hourly mean-station rain (mm), the 72 h ending at t0. None if Iffley has no
    reading within 3 h of t0 or at the 6 h / 24 h lags."""
    lv = level.sort_index()

    def at(t):
        s = lv.loc[t - pd.Timedelta(hours=3):t].dropna()
        return float(s.iloc[-1]) if len(s) else np.nan
    u0, u6, u24 = at(t0), at(t0 - pd.Timedelta(hours=6)), at(t0 - pd.Timedelta(hours=24))
    if not np.isfinite([u0, u6, u24]).all():
        return None
    r = np.nan_to_num(np.asarray(rain_hist, float), nan=0.0).clip(min=0)
    return {'u0': u0, 'du6': u0 - u6, 'du24': u0 - u24, 'I0': I0, 'G0': G0,
            'rp24': float(r[-24:].sum()), 'rp72': float(r[-72:].sum())}


def mean_paths(params: BlackParams, f: dict, rain_future: np.ndarray) -> np.ndarray:
    """(M, H+1) mean Iffley path per rain member (index 0 = now). rain_future: (M, >=H) mm/h."""
    name = 'full' if np.isfinite([f['I0'], f['G0']]).all() else 'short'
    m = params.models[name]
    rf = np.cumsum(np.nan_to_num(np.asarray(rain_future, float)[:, :HORIZON], nan=0.0).clip(min=0), axis=1)
    M = rf.shape[0]
    out = np.empty((M, HORIZON + 1))
    out[:, 0] = f['u0']
    for h in range(1, HORIZON + 1):
        X = np.stack([np.full(M, f[c]) if c != 'rf' else rf[:, h - 1] for c in m['features']], axis=1)
        out[:, h] = f['u0'] + X @ m['coef'][h - 1] + m['intercept'][h - 1]
    return out


def simulate(params: BlackParams, f: dict, rain_future: np.ndarray, rng: Optional[np.random.Generator] = None,
             n: int = PATHS_PER_MEMBER) -> np.ndarray:
    """(M*n, H+1) Iffley paths: member means plus a random walk with the model's spread at each lead."""
    rng = rng or np.random.default_rng(0)
    mu = mean_paths(params, f, rain_future)
    name = 'full' if np.isfinite([f['I0'], f['G0']]).all() else 'short'
    sd = params.models[name]['sd']
    inc_sd = np.sqrt(np.maximum(np.diff(np.r_[0.0, sd ** 2]), 0.0))
    noise = np.cumsum(rng.standard_normal((mu.shape[0] * n, HORIZON)) * inc_sd, axis=1)
    paths = np.repeat(mu, n, axis=0)
    paths[:, 1:] += noise
    return paths


def outlook(paths: np.ndarray, black0: bool, t0: pd.Timestamp) -> dict:
    """Chance the Iffley rule changes state (Red -> Black, or Black -> Red) by the end of today / tomorrow."""
    from .step_layer import day_windows, likely_time
    hit = paths[:, 1:] < BLACK_DOWN_M if black0 else paths[:, 1:] >= BLACK_UP_M
    win = {k: min(v, HORIZON) for k, v in day_windows(t0, HORIZON).items()}
    res = {'black_now': bool(black0), 'level_m': round(float(paths[0, 0]), 3), 'up_m': BLACK_UP_M, 'down_m': BLACK_DOWN_M,
           'windows': {k: round(float(hit[:, :last].any(axis=1).mean()), 3) if last >= 1 else 0.0 for k, last in win.items()},
           'when': None}
    seg = hit[:, :win['tomorrow']]
    ok = seg.any(axis=1)
    if ok.any():
        local = pd.date_range(pd.Timestamp(t0), periods=HORIZON + 1, freq='1h')
        local = (local.tz_localize('UTC') if local.tz is None else local).tz_convert(LOCAL_TZ)
        res['when'] = {'median_time': likely_time(seg.argmax(axis=1)[ok] + 1, local)}
    return res


def black_forecast(level: pd.Series, t0: pd.Timestamp, I0: float, G0: float, rain_hist: np.ndarray,
                   rain_future: np.ndarray, models_dir: Path, seed: int = 0) -> Optional[dict]:
    """Live: the Black outlook, or None (no parameters or no recent Iffley reading). Run at every
    level: fitted on high water, it still scores better than "no chance" from low water (fit_black_flag.py)."""
    pfile = Path(models_dir) / PARAMS_FILE
    if not pfile.exists() or level is None:
        return None
    lv = level.copy()
    lv.index = pd.DatetimeIndex(lv.index)
    lv.index = lv.index.tz_localize('UTC') if lv.index.tz is None else lv.index.tz_convert('UTC')
    lv = lv.resample('1h').mean()
    t0 = pd.Timestamp(t0)
    t0 = t0.tz_localize('UTC') if t0.tz is None else t0.tz_convert('UTC')
    f = start_features(lv, t0, I0, G0, rain_hist)
    if f is None:
        return None
    black0 = black_state(lv.loc[:t0])
    paths = simulate(BlackParams.load(pfile), f, rain_future, rng=np.random.default_rng(seed))
    return outlook(paths, black0, t0)
