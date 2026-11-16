"""
Fit the map from each stretch's differential to Farmoor flow, used to estimate Farmoor flow while
its gauge is offline (prediction/hybrid_live.estimate_flow).

    python fit_flow_from_differential.py          # writes models/flow_from_differential.json

Daily mean differential (data/<stretch>_flag_model_data.csv) against daily Farmoor flow
(data/termly/farmoor_daily.csv), isotonic in log flow. The live estimate is anchored to the last real
flow reading and moved by the change in this map (see estimate_flow), so only the map's slope matters.

Backtest (fit to 2023, test 2024-Jan 2026), RMSE in log flow after a gap of
1 / 3 / 7 / 14 days: carrying the last reading forward 0.21 / 0.39 / 0.58 / 0.70; anchored estimate
from all three stretches 0.22 / 0.30 / 0.31 / 0.35.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
COLS = {'isis': 'jameson_isis_differential', 'godstow': 'jameson_godstow_differential',
        'wallingford': 'jameson_wallingford_differential'}
OUT = ROOT / 'models' / 'flow_from_differential.json'


def main() -> None:
    q = pd.read_csv(ROOT / 'data' / 'termly' / 'farmoor_daily.csv', index_col=0, parse_dates=True)['flow_m3s']
    lq = np.log(q[q > 0.05])
    out = {}
    for loc, col in COLS.items():
        d = pd.read_csv(ROOT / 'data' / f'{loc}_flag_model_data.csv', index_col=0, parse_dates=True)[col]
        d.index = d.index.tz_convert(None) if d.index.tz is not None else d.index
        df = pd.concat([d.resample('1D').mean().rename('d'), lq.rename('lq')], axis=1).dropna()
        iso = IsotonicRegression(out_of_bounds='clip').fit(df['d'], df['lq'])
        grid = np.round(np.linspace(df['d'].quantile(0.002), df['d'].quantile(0.998), 121), 4)
        out[loc] = {'differential': grid.tolist(), 'log_flow': np.round(iso.predict(grid), 4).tolist(),
                    'n_days': int(len(df)), 'period': [str(df.index[0].date()), str(df.index[-1].date())]}
        print(f"{loc}: {len(df)} days, D {grid[0]:.2f}..{grid[-1]:.2f} -> Q "
              f"{np.exp(out[loc]['log_flow'][0]):.1f}..{np.exp(out[loc]['log_flow'][-1]):.1f} m3/s")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
