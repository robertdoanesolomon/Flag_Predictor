"""
Download C3S seasonal forecasts (ECMWF SEAS5) for the upper Thames, for the termly outlook.

Dataset "seasonal-monthly-single-levels" on the Copernicus Climate Data Store: monthly means of total
precipitation and 2 m temperature, every ensemble member, lead months 1-6, every start month, over a
small box around the catchment above Farmoor (grid points 51-52 N, 2-1 W).

  hindcasts   1993-2016, SEAS5.1 (system 51; the same model as today's forecasts, rerun for past years)
  forecasts   2017 - Oct 2022 SEAS5 (system 5), Nov 2022 onwards SEAS5.1 (system 51)

Runs in GitHub Actions (.github/workflows/seasonal_download.yml) with the repository secret CDSAPI_KEY
(the CDS personal access token; never stored in the repo). One request per (system, start month,
block of years); a file already downloaded is not fetched again, so reruns only fill gaps. Writes
data/seasonal/raw/*.nc and a log, data/seasonal/download_log.txt.

    python download_seasonal.py            # everything not yet downloaded
    python download_seasonal.py --latest   # only this month's forecast (the monthly schedule)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'seasonal' / 'raw'
LOG = ROOT / 'data' / 'seasonal' / 'download_log.txt'
DATASET = 'seasonal-monthly-single-levels'
AREA = [52.5, -2.5, 50.5, -0.5]          # N, W, S, E: grid points 51-52 N, 2-1 W
VARIABLES = ['total_precipitation', '2m_temperature']
LEADS = [str(i) for i in range(1, 7)]
BUDGET_MIN = 300                          # stop starting new requests after this (job limit 330)


def requests_wanted(today: dt.date, latest_only: bool) -> list[dict]:
    """[{name, system, years, month}] for every block not yet on disk."""
    out = []
    if latest_only:
        months = [(today.year, today.month)]
        if today.day < 6:                 # the new month's forecast is out on the 5th
            prev = today.replace(day=1) - dt.timedelta(days=1)
            months = [(prev.year, prev.month)]
        for y, m in months:
            out.append({'name': f'ecmwf_s51_fc_{y}_{m:02d}', 'system': '51', 'years': [y], 'month': m})
    else:
        for m in range(1, 13):
            out.append({'name': f'ecmwf_s51_hc_{m:02d}', 'system': '51', 'years': list(range(1993, 2017)), 'month': m})
            yrs5 = [y for y in range(2017, 2023) if (y, m) <= (2022, 10)]
            out.append({'name': f'ecmwf_s5_fc_{m:02d}', 'system': '5', 'years': yrs5, 'month': m})
            yrs51 = [y for y in range(2022, today.year + 1) if (2022, 11) <= (y, m) <= (today.year, today.month)
                     and not (y == today.year and m == today.month and today.day < 6)]
            out.append({'name': f'ecmwf_s51_fc_{m:02d}', 'system': '51', 'years': yrs51, 'month': m})
            # Met Office GloSea hindcasts (1993-2016) from the newest system that has them, for comparison
            out.append({'name': f'ukmo_hc_{m:02d}', 'centre': 'ukmo', 'systems': ['604', '603', '602', '601', '600'],
                        'years': list(range(1993, 2017)), 'month': m})
    return [r for r in out if r['years'] and not (RAW / f"{r['name']}.nc").exists()]


def retrieve(client, req: dict) -> str:
    target = RAW / f"{req['name']}.nc"
    tmp = target.with_suffix('.part')
    body = {
        'originating_centre': req.get('centre', 'ecmwf'),
        'system': req.get('system'),
        'variable': VARIABLES,
        'product_type': ['monthly_mean'],
        'year': [str(y) for y in req['years']],
        'month': [f"{req['month']:02d}"],
        'leadtime_month': LEADS,
        'data_format': 'netcdf',
        'area': AREA,
    }
    t0 = time.time()
    systems = req.get('systems') or [req['system']]
    for i, system in enumerate(systems):
        body['system'] = system
        try:
            client.retrieve(DATASET, body, str(tmp))
            break
        except Exception:                 # noqa: BLE001 - try the next (older) system
            if i == len(systems) - 1:
                raise
    req = req | {'name': req['name'], 'used_system': body['system']}
    # Several variables can come back as a zip of NetCDF files: keep each one.
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            names = [n for n in z.namelist() if n.endswith('.nc')]
            for i, n in enumerate(names):
                dest = target if i == 0 else target.with_name(f"{target.stem}__{i}.nc")
                dest.write_bytes(z.read(n))
        tmp.unlink()
    else:
        tmp.rename(target)
    return f"{req['name']}: ok (system {req.get('used_system')}, {len(req['years'])} years, {time.time() - t0:.0f} s)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--latest', action='store_true')
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    lines = [f'--- {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC ({"latest" if args.latest else "all"})']

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    try:
        if not os.environ.get('CDSAPI_KEY'):
            log('CDSAPI_KEY is not set: add the repository secret (Settings > Secrets and variables > Actions).')
            return 1
        import cdsapi
        client = cdsapi.Client(url=os.environ.get('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api'),
                               key=os.environ['CDSAPI_KEY'], quiet=True, progress=False)
        todo = requests_wanted(dt.date.today(), args.latest)
        log(f'{len(todo)} requests to make')
        start = time.time()
        failed = 0
        with ThreadPoolExecutor(max_workers=1) as pool:   # the CDS rejects parallel requests per dataset
            futs = {}
            for req in todo:
                futs[pool.submit(lambda r=req: retrieve(client, r) if (time.time() - start) / 60 < BUDGET_MIN
                                 else f"{r['name']}: skipped (time budget)")] = req
            for f in as_completed(futs):
                try:
                    log(f.result())
                except Exception as e:          # noqa: BLE001 - log every failure and carry on
                    failed += 1
                    msg = str(e).strip().splitlines()
                    log(f"{futs[f]['name']}: FAILED {type(e).__name__}: {' | '.join(msg[:6])[:800]}")
        log(f'done: {len(todo) - failed} ok, {failed} failed; files now: {len(list(RAW.glob("*.nc")))}')
        return 0
    except Exception:                           # noqa: BLE001
        log(traceback.format_exc()[-2000:])
        return 1
    finally:
        with LOG.open('a') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.exit(main())
