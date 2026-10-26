"""Monthly rain over the catchment above Farmoor from CRU TS (UEA, 0.5 deg; the six cells 51.75-52.25 N,
2.25-1.25 W), for the rainfall normals on the long-range page: data/termly/cru_monthly_rain.csv.
Rerun when a new CRU TS version is out (yearly):  python build_cru_rain.py --version 4.10 --dir cruts.2604091129.v4.10"""
import argparse, gzip, shutil, tempfile, urllib.request
from pathlib import Path
import pandas as pd, xarray as xr
ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--version', default='4.10'); ap.add_argument('--dir', default='cruts.2604091129.v4.10')
ap.add_argument('--end', type=int, default=2025)
a = ap.parse_args()
base = f'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_{a.version}/{a.dir}/pre/'
parts = []
with tempfile.TemporaryDirectory() as tmp:
    for d0 in (1991, 2001, 2011, 2021):
        name = f'cru_ts{a.version}.{d0}.{min(d0 + 9, a.end) if d0 < 2021 else a.end}.pre.dat.nc'
        gz = Path(tmp) / (name + '.gz')
        urllib.request.urlretrieve(base + name + '.gz', gz)
        with gzip.open(gz) as fi, open(Path(tmp) / name, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        d = xr.open_dataset(Path(tmp) / name)['pre'].sel(lat=slice(51.5, 52.3), lon=slice(-2.3, -1.2))
        parts.append(d.mean(['lat', 'lon']).to_series())
s = pd.concat(parts)
s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
s = s[s.index.year >= 1993].round(1)
s.index.name = 'month'
s.rename('rain_mm').to_csv(ROOT / 'data' / 'termly' / 'cru_monthly_rain.csv')
print(f'{len(s)} months to {s.index[-1]:%Y-%m}')
