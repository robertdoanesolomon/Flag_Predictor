"""
ECMWF SEAS5 monthly rainfall over the upper Thames, from the raw C3S files (download_seasonal.py), into one
table: data/seasonal/seas5_upper_thames.csv (centre, init month, member, lead month 1-6, valid month, mm).
Met Office GloSea hindcasts (ukmo_*.nc), where downloaded, are included with centre 'ukmo'.

Box: grid points within 51-52.5 N, 2.5-1.0 W (SEAS5.1 files are on a half-degree-offset grid, SEAS5 on
whole degrees). Members 0-24 only, so hindcasts (25 members) and forecasts (51) are alike.

    python process_seasonal.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'seasonal' / 'raw'
OUT = ROOT / 'data' / 'seasonal' / 'seas5_upper_thames.csv'


def main() -> None:
    rows = {}
    for f in sorted(RAW.glob('*.nc')):
        centre = f.name.split('_')[0]
        d = xr.open_dataset(f)
        if 'indexing_time' in d.dims:      # Met Office: lagged ensemble, indexed by its nominal start
            d = d.rename({'indexing_time': 'forecast_reference_time'})
        d = d.where((d.latitude >= 51) & (d.latitude <= 52.5) & (d.longitude >= -2.5) & (d.longitude <= -1.0), drop=True)
        pr = (d.tprate * 86400 * 1000).mean(['latitude', 'longitude'])          # mm/day
        if centre == 'ecmwf':
            pr = pr.sel(number=[n for n in pr.number.values if n < 25])
        for t in pr.forecast_reference_time.values:
            init = pd.Timestamp(t).to_period('M')
            for lead in pr.forecastMonth.values:
                valid = init + int(lead) - 1
                v = pr.sel(forecast_reference_time=t, forecastMonth=lead).values
                for m, x in enumerate(v):
                    if np.isfinite(x):
                        rows[(centre, str(init), m, int(lead))] = (str(valid), round(float(x) * valid.days_in_month, 2))
    df = pd.DataFrame([(*k, *v) for k, v in rows.items()],
                      columns=['centre', 'init', 'member', 'lead', 'valid', 'rain_mm']).sort_values(['centre', 'init', 'lead', 'member'])
    df.to_csv(OUT, index=False)
    print(f'{OUT}: {df.init.nunique()} runs ({df.init.min()} to {df.init.max()}), {len(df)} rows')


if __name__ == '__main__':
    main()
