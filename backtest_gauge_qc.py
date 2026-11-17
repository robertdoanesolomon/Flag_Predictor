"""
Backtest the faulty rain gauge check (src/flag_predictor/data/gauge_qc.py) on the EA's
qualified daily rain for the 19 gauges, and compare its flags with the EA's own quality
codes (added months later). Prints the share of gauge-days flagged and each flagged run.

    python backtest_gauge_qc.py [--since 2015-01-01] [--refit-ratios]
"""

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from flag_predictor.data.download_rainfall import RAINFALL_HYDROLOGY_MEASURES  # noqa: E402
from flag_predictor.data import gauge_qc  # noqa: E402

SHORT = {'Chipping-Norton': 'Chipping', 'Osney-Lock': 'Osney', 'St-Johns': 'St', 'Stowell-Park': 'Stowell'}
URL = 'https://environment.data.gov.uk/hydrology/id/measures/{m}-rainfall-t-86400-mm-qualified/readings.csv'


def fetch(item, since):
    name, m = item
    r = requests.get(URL.format(m=m), params={'min-date': since, '_limit': 200000}, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.drop_duplicates('date').set_index(pd.to_datetime(df.drop_duplicates('date')['date']))
    return SHORT.get(name, name), df['value'].astype(float), df.get('quality')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2015-01-01')
    ap.add_argument('--refit-ratios', action='store_true')
    a = ap.parse_args()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda it: fetch(it, a.since), RAINFALL_HYDROLOGY_MEASURES.items()))
    d = pd.DataFrame({n: s for n, s, _ in res}).sort_index()
    q = pd.DataFrame({n: qq for n, _, qq in res if qq is not None}).reindex(d.index)
    gauges = list(d.columns)
    if a.refit_ratios:
        ok = d.where(q.isin(['Good', 'Estimated', 'Unchecked']))
        print('RATIO =', {g: round(float(ok[g].where(ok[gauge_qc.neighbours(g, gauges)].notna().all(axis=1)).sum()
                          / ok[gauge_qc.neighbours(g, gauges)].mean(axis=1).where(ok[g].notna()).sum()), 2)
                          for g in gauges})
    rows, rate = [], {}
    for g in gauges:
        nbs = gauge_qc.neighbours(g, gauges)
        nb = d[nbs].where(d[g].notna())
        caught, flag, _ = gauge_qc._check_gauge(d[g].to_numpy(float), nb.to_numpy(float), gauge_qc.RATIO.get(g, 1.0))
        f = pd.Series(flag, index=d.index)
        rate[g] = round(float(f[np.isfinite(caught)].mean()), 3)
        run = (f != f.shift()).cumsum()
        for _, s in f[f].groupby(run[f]):
            x, y = s.index[0], s.index[-1]
            qs = q.loc[x:y, g].value_counts(normalize=True) if g in q else pd.Series(dtype=float)
            rows.append((g, x.date(), y.date(), (y - x).days + 1, round(float(d.loc[x:y, g].sum()), 1),
                         round(float(qs.reindex(['Suspect', 'Missing', 'Estimated']).fillna(0).sum()), 2)))
    runs = pd.DataFrame(rows, columns=['gauge', 'start', 'end', 'days', 'gauge_mm', 'EA_suspect_share'])
    print('share of gauge-days flagged:', rate)
    print(f'{len(runs)} flagged runs; median {runs.days.median():.0f} days; '
          f'EA later marked {runs.EA_suspect_share.mean():.0%} of flagged days suspect/missing/estimated '
          f'(vs {q.isin(["Suspect", "Missing", "Estimated"]).to_numpy().mean():.0%} of all days)')
    print(runs.sort_values('start').to_string(index=False))


if __name__ == '__main__':
    main()
