"""
Reading noise for the likely-range band (the cone).

The model and its error library work in hourly means of the differential,
but the page starts every forecast from a single gauge reading and plots
single readings (the last in each hour). Those wobble around the hourly
level by 1–2 cm on the Isis: held steps of a few hours, lock and weir
movements, and one-reading spikes. The borrowed model errors don't carry
that wobble, so in the first day the band was far too narrow (13 Oct –
18 Nov 2026 archive: Isis 80 % band covered ~31 % of readings at 1–6 h).

Fix: each outcome in the band gets

    noise(h) = r_h − r_0

where r_0 (one draw per outcome, held over all leads) is the unknown wobble
in the starting reading and r_h (a fresh draw each hour) is the wobble in
the reading at lead h. Both are drawn from the stretch's own recent live
record: hourly readings minus their centred 25 h running median, over the
last 28 days (only residuals fully known at t0 are used). Its weight is
fixed with lead, so it dominates the first hours and fades in relative
terms as the model error grows.

Only the plotted band uses this. The flag-band chances and the step layer
keep the outcomes without it, since their calibration was fitted on those.

Per stretch (walk-forward on the archive, 80 % band coverage at
1 / 3 / 6 / 12 / 24 h, before → after):

    Isis         31 / 31 / 32 / 38 / 55 %  →  81 / 72 / 74 / 75 / 80 %
    Wallingford  67 / 53 / 39 / 38 / 34 %  →  90 / 71 / 63 / 49 / 39 %
    Godstow      82 / 87 / 84 / 77 / 83 %     off: already covers

Wallingford's remaining shortfall is a high bias at low flow
(band-calibration findings), which widening would only hide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOCATIONS = ('isis', 'wallingford')
WINDOW_H = 25          # centred running median the readings wobble around
POOL_DAYS = 28         # recent record the wobble is drawn from
MIN_POOL = 200         # fewer residuals than this (e.g. after an outage): no noise
MAX_ABS = 0.10         # m; bigger departures are real moves or glitches, not wobble


def residual_pool(observed: pd.Series, t0: pd.Timestamp, window_h: int = WINDOW_H,
                  days: int = POOL_DAYS) -> np.ndarray:
    """Recent hourly readings minus their centred running median, known by t0."""
    s = observed.astype(float).sort_index()
    s = s[~s.index.duplicated()]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='1h'))
    r = s - s.rolling(window_h, center=True, min_periods=window_h).median()
    r = r[r.abs() < MAX_ABS].dropna()
    t0 = pd.Timestamp(t0)
    return r.loc[t0 - pd.Timedelta(days=days):t0 - pd.Timedelta(hours=window_h // 2)].to_numpy()


def load_observed(archive_dir: Path, location: str) -> Optional[pd.Series]:
    path = Path(archive_dir) / 'observed.json'
    try:
        enc = json.loads(path.read_text())[location]
    except Exception:
        return None
    vals = enc.get('values') or []
    idx = pd.date_range(enc['start'], periods=len(vals), freq=f"{float(enc.get('step_h', 1.0))}h")
    return pd.Series(pd.to_numeric(vals, errors='coerce'), index=idx, dtype=float)


def add_reading_noise(samples: np.ndarray, pool: np.ndarray, seed: int = 0) -> np.ndarray:
    """(S, H+1) outcomes plus r_h − r_0; hour 0 (the observation) is left alone."""
    out = np.array(samples, dtype=float, copy=True)
    if pool is None or len(pool) < MIN_POOL:
        return out
    rng = np.random.default_rng(seed)
    S, H1 = out.shape
    r0 = rng.choice(pool, size=(S, 1))
    out[:, 1:] += rng.choice(pool, size=(S, H1 - 1)) - r0
    return out


def band_samples(location: str, samples: np.ndarray, t0: pd.Timestamp, archive_dir: Path) -> np.ndarray:
    """The outcomes the plotted band is drawn from: with reading noise where it is on."""
    location = location.lower()
    if location not in LOCATIONS:
        return samples
    observed = load_observed(archive_dir, location)
    if observed is None:
        return samples
    t0 = pd.Timestamp(t0)
    t0 = t0.tz_convert('UTC').tz_localize(None) if t0.tzinfo else t0
    pool = residual_pool(observed, t0)
    if len(pool) < MIN_POOL:
        print(f"  [reading noise] {location}: only {len(pool)} recent readings; band without it")
        return samples
    print(f"  [reading noise] {location}: band widened by recent reading wobble "
          f"(10–90% {np.percentile(pool, 10) * 100:+.1f} / {np.percentile(pool, 90) * 100:+.1f} cm, n={len(pool)})")
    return add_reading_noise(samples, pool)
