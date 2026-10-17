"""
Hourly reference evapotranspiration (FAO-56 ET0) for the Upper Thames.

ET0 comes from the Open-Meteo archive (ERA5 / ERA5-Land reanalysis) for the
historical record and from the Open-Meteo forecast API for the last few days
(the archive lags real time by ~5 days) and the 10-day forecast. Values are
averaged over a few points spread across the catchment above Farmoor and the
Thame/Cherwell (the reanalysis grid is ~25 km, so a handful of points covers it).

Usage (one-off backfill, run where open-meteo.com is reachable, e.g. CI):

    python src/flag_predictor/data/download_et.py --start 2000-01-01 --out data/era5/et0_hourly.csv.gz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Cotswolds (Churn/Coln/Windrush), Evenlode/Cherwell, the Vale around Oxford,
# and the Thame (for Wallingford).
ET_POINTS = {
    'cotswolds_w': (51.75, -1.90),
    'evenlode': (51.90, -1.55),
    'oxford_vale': (51.70, -1.30),
    'thame': (51.80, -0.95),
}
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
HOURLY_VARS = [
    'et0_fao_evapotranspiration', 'temperature_2m',
    'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
]


def _get(url: str, params: dict, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 65 * (attempt + 1)
                print(f'  rate limited; waiting {wait}s')
                time.sleep(wait)
                continue
            print(f'  HTTP {r.status_code}: {r.text[:200]}')
        except requests.RequestException as exc:
            print(f'  request failed: {exc}')
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'giving up on {url} {params}')


def _frame(js: dict, variables) -> pd.DataFrame:
    h = js['hourly']
    idx = pd.to_datetime(h['time'], utc=True)
    return pd.DataFrame({v: pd.to_numeric(pd.Series(h.get(v, [np.nan] * len(idx))), errors='coerce').to_numpy()
                         for v in variables}, index=idx)


def fetch_archive(start: str, end: str, variables=HOURLY_VARS, chunk_years: int = 1,
                  pause_s: float = 4.0) -> pd.DataFrame:
    """Catchment-mean hourly variables from the reanalysis archive."""
    per_point = []
    for name, (lat, lon) in ET_POINTS.items():
        parts = []
        lo = pd.Timestamp(start)
        hi_all = pd.Timestamp(end)
        while lo <= hi_all:
            hi = min(hi_all, lo + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1))
            print(f'{name}: {lo.date()} -> {hi.date()}')
            js = _get(ARCHIVE_URL, {
                'latitude': lat, 'longitude': lon,
                'start_date': lo.strftime('%Y-%m-%d'), 'end_date': hi.strftime('%Y-%m-%d'),
                'hourly': ','.join(variables), 'timezone': 'UTC',
            })
            parts.append(_frame(js, variables))
            lo = hi + pd.Timedelta(days=1)
            time.sleep(pause_s)
        per_point.append(pd.concat(parts).sort_index())
    stacked = pd.concat(per_point, keys=list(ET_POINTS), axis=1)
    return stacked.T.groupby(level=1).mean().T[list(variables)]


def fetch_recent(past_days: int = 14, forecast_days: int = 11,
                 variables=('et0_fao_evapotranspiration', 'temperature_2m')) -> pd.DataFrame:
    """Catchment-mean hourly ET0 for the last past_days and the next forecast_days."""
    frames = []
    for lat, lon in ET_POINTS.values():
        js = _get(FORECAST_URL, {
            'latitude': lat, 'longitude': lon, 'hourly': ','.join(variables),
            'past_days': past_days, 'forecast_days': forecast_days, 'timezone': 'UTC',
        })
        frames.append(_frame(js, variables))
    return sum(frames) / len(frames)


def load_et0(path: Path) -> Optional[pd.Series]:
    """Hourly catchment-mean ET0 (mm/h) saved by this module, or None."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    return df['et0_fao_evapotranspiration'].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2000-01-01')
    ap.add_argument('--end', default=None, help='default: 6 days ago')
    ap.add_argument('--out', default='data/era5/et0_hourly.csv.gz')
    ap.add_argument('--append', action='store_true',
                    help='only fetch from the end of an existing --out file')
    args = ap.parse_args()
    end = args.end or (pd.Timestamp.utcnow() - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = args.start
    old = None
    if args.append and out.exists():
        old = pd.read_csv(out, index_col=0, parse_dates=True)
        start = (pd.DatetimeIndex(old.index).max() + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
    df = fetch_archive(start, end)
    if old is not None:
        old.index = pd.DatetimeIndex(old.index).tz_convert('UTC') if pd.DatetimeIndex(old.index).tz else pd.DatetimeIndex(old.index).tz_localize('UTC')
        df = pd.concat([old, df]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
    df.round(4).to_csv(out, index_label='time')
    print(f'wrote {out}: {len(df)} hours, {df.index.min()} -> {df.index.max()}')
    print(df.describe().T)


if __name__ == '__main__':
    main()
