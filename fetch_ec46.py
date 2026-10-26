"""
ECMWF extended-range (EC46) re-forecasts from the S2S archive on the ECMWF Data Store (ECDS), for testing
whether weeks 1-6 of rain help the long-range outlook. Daily total precipitation, control + 10 members,
the 20 past years ("hdate") ECMWF re-runs for each model-version date, over the upper Thames.

Runs in GitHub Actions (ec46_download.yml) with the CDS_API_KEY secret (CDS keys work on ECDS; the account
must have accepted the ECDS terms and the TIGGE/S2S licence). Writes data/ec46/raw/<date>.grib and a log.

    python fetch_ec46.py --probe            # one small request, to check access and keys
    python fetch_ec46.py                    # every Monday/Thursday model date for a year (skips files on disk)
"""
import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'ec46' / 'raw'
LOG = ROOT / 'data' / 'ec46' / 'log.txt'
STEPS = [str(h) for h in range(0, 1105, 24)]                # daily accumulations to day 46
BUDGET_MIN = 300


def request(date: dt.date, kind: str, years: int = 20) -> dict:
    hdates = [f'{date.year - k}-{date:%m-%d}' for k in range(1, years + 1)]          # lists: '/' strings get lost
    r = {'class': 's2', 'dataset': 's2s', 'date': f'{date:%Y-%m-%d}', 'hdate': hdates, 'expver': 'prod',
         'levtype': 'sfc', 'model': 'glob', 'origin': 'ecmf', 'param': '228228', 'step': STEPS, 'stream': 'enfh',
         'time': '00:00:00', 'type': kind, 'grid': '1.5/1.5', 'area': '53/-3/50.5/0', 'data_format': 'grib'}
    if kind == 'pf':
        r['number'] = [str(n) for n in range(1, 11)]
    return r


def model_dates(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() in (0, 3):
            yield d
        d += dt.timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--start', default='2025-06-30')
    ap.add_argument('--end', default='2026-06-29')
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    lines = [f'--- {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC ({"probe" if a.probe else "all"})']

    def log(m):
        print(m, flush=True)
        lines.append(m)
    try:
        if not os.environ.get('CDSAPI_KEY'):
            log('CDSAPI_KEY is not set')
            return 1
        import cdsapi
        c = cdsapi.Client(url='https://ecds.ecmwf.int/api', key=os.environ['CDSAPI_KEY'], quiet=True, progress=False)
        dates = [dt.date(2025, 11, 3)] if a.probe else list(model_dates(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)))
        t_start = time.time()
        for d in dates:
            for kind in ('cf', 'pf'):
                target = RAW / f'{d:%Y-%m-%d}_{kind}.grib'
                if target.exists():
                    continue
                if (time.time() - t_start) / 60 > BUDGET_MIN:
                    log('time budget reached; rerun to continue')
                    return 0
                t0 = time.time()
                try:
                    c.retrieve('s2s-reforecasts', request(d, kind, 2 if a.probe else 20), str(target))
                    log(f'{target.name}: ok ({target.stat().st_size // 1024} kB, {time.time() - t0:.0f} s)')
                except Exception as e:      # noqa: BLE001
                    log(f'{target.name}: FAILED {type(e).__name__}: {" | ".join(str(e).splitlines()[:8])[:1500]}')
                    if a.probe:
                        return 1
        log(f'done; files: {len(list(RAW.glob("*.grib")))}')
        return 0
    except Exception:                       # noqa: BLE001
        log(traceback.format_exc()[-2000:])
        return 1
    finally:
        with LOG.open('a') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.exit(main())
