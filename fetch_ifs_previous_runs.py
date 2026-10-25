"""
For verification only: what ECMWF IFS actually forecast for each rain station in Feb 2025, 1-7 days ahead.

Open-Meteo's Previous Runs API keeps each model's hourly forecast as issued N days before the valid time
(``precipitation_previous_dayN``, N = 1..7; ``precipitation`` is the latest run, day 0). Writes
data/verification/ifs_previous_runs_2025.csv (time, station, day0..day7, mm/h). Runs in GitHub Actions
(.github/workflows/verification_rain.yml) because the API has a daily request limit per address.

    python fetch_ifs_previous_runs.py [--start 2025-01-10 --end 2025-03-10 --model ecmwf_ifs025]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
# config.py on its own (the package __init__ pulls in torch, which this job does not need)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('fp_config', ROOT / 'src' / 'flag_predictor' / 'config.py')
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
RAINFALL_STATION_COORDINATES = _cfg.RAINFALL_STATION_COORDINATES
WALLINGFORD_RAINFALL_STATION_COORDINATES = _cfg.WALLINGFORD_RAINFALL_STATION_COORDINATES

OUT = ROOT / 'data' / 'verification' / 'ifs_previous_runs_2025.csv'
URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-01-10')
    ap.add_argument('--end', default='2025-03-10')
    ap.add_argument('--model', default='ecmwf_ifs025')
    a = ap.parse_args()
    stations = {**RAINFALL_STATION_COORDINATES, **WALLINGFORD_RAINFALL_STATION_COORDINATES}
    names = list(stations)
    hourly = ['precipitation'] + [f'precipitation_previous_day{n}' for n in range(1, 8)]
    r = requests.get(URL, timeout=180, params={
        'latitude': ','.join(str(stations[n]['latitude']) for n in names),
        'longitude': ','.join(str(stations[n]['longitude']) for n in names),
        'hourly': ','.join(hourly), 'models': a.model, 'start_date': a.start, 'end_date': a.end, 'timezone': 'GMT'})
    js = r.json()
    if isinstance(js, dict) and js.get('error'):
        print('Open-Meteo:', js.get('reason'))
        return 1
    js = js if isinstance(js, list) else [js]
    frames = []
    for name, j in zip(names, js):
        h = j['hourly']
        df = pd.DataFrame({'time': h['time'], 'station': name, 'day0': h['precipitation'],
                           **{f'day{n}': h[f'precipitation_previous_day{n}'] for n in range(1, 8)}})
        frames.append(df)
    out = pd.concat(frames)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    cover = out.drop(columns=['time', 'station']).notna().mean().round(2).to_dict()
    print(f'{OUT}: {len(out)} rows, {len(names)} stations, {a.model}; share of hours with data: {cover}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
