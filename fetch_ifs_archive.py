"""
What ECMWF IFS (0.25 deg) actually forecast for each rain station, 0-7 days ahead, over a long period: the
Open-Meteo Previous Runs API (``precipitation_previous_dayN``: the value for that hour from the run N days
earlier). One file per month in data/verification/ifs_runs/ (time, station, day0..day7, mm/h). For backtesting
the 10-day model with the rain that was forecast rather than the rain that fell (rise_backtest.py).

    python fetch_ifs_archive.py --start 2024-10-01 --end 2026-01-31
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
import importlib.util  # noqa: E402  (config.py alone: the package needs torch)
_spec = importlib.util.spec_from_file_location('fp_config', ROOT / 'src' / 'flag_predictor' / 'config.py')
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
RAINFALL_STATION_COORDINATES = _cfg.RAINFALL_STATION_COORDINATES
WALLINGFORD_RAINFALL_STATION_COORDINATES = _cfg.WALLINGFORD_RAINFALL_STATION_COORDINATES

OUT = ROOT / 'data' / 'verification' / 'ifs_runs'
URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-03-01')
    ap.add_argument('--end', default='2026-01-31')
    ap.add_argument('--model', default='ecmwf_ifs025')
    a = ap.parse_args()
    stations = {**RAINFALL_STATION_COORDINATES, **WALLINGFORD_RAINFALL_STATION_COORDINATES}
    names = list(stations)
    hourly = ['precipitation'] + [f'precipitation_previous_day{n}' for n in range(1, 8)]
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for m0 in pd.date_range(a.start, a.end, freq='MS'):
        target = OUT / f'{m0:%Y-%m}.csv'
        if target.exists():
            continue
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(a.end))
        frames = []
        for attempt in range(4):                 # all stations in one call; one call per month
            try:
                r = requests.get(URL, timeout=90, params={
                    'latitude': ','.join(str(stations[n]['latitude']) for n in names),
                    'longitude': ','.join(str(stations[n]['longitude']) for n in names),
                    'hourly': ','.join(hourly), 'models': a.model, 'start_date': f'{m0:%Y-%m-%d}',
                    'end_date': f'{m1:%Y-%m-%d}', 'timezone': 'GMT'})
                js = r.json()
                if isinstance(js, dict) and js.get('error'):
                    raise RuntimeError(js.get('reason'))
                js = js if isinstance(js, list) else [js]
                frames = [pd.DataFrame({'time': j['hourly']['time'], 'station': n, 'day0': j['hourly']['precipitation'],
                                        **{f'day{k}': j['hourly'][f'precipitation_previous_day{k}'] for k in range(1, 8)}})
                          for n, j in zip(names, js)]
                break
            except Exception as e:               # noqa: BLE001
                print(f'  {m0:%Y-%m}: {e} (attempt {attempt + 1})', flush=True)
                time.sleep(15 * (attempt + 1))
        if len(frames) == len(names):
            df = pd.concat(frames)
            if df.drop(columns=['time', 'station']).notna().mean().max() < 0.05:
                print(f'{m0:%Y-%m}: no forecasts archived for this month (the archive starts later), skipped', flush=True)
                continue
            df.to_csv(target, index=False)
            cover = df.drop(columns=['time', 'station']).notna().mean().round(2).to_dict()
            print(f'{target.name}: {len(df)} rows; share with data {cover}', flush=True)
        else:
            failed += 1
            print(f'{m0:%Y-%m}: incomplete, not saved', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
