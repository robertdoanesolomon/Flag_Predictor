"""
Rule of thumb: "when Hannington Bridge goes under 4 m, the Isis / Godstow are usually 3-5 days off Red".

For every time the Thames at Hannington Bridge (0569TH) falls below 4.0 m after being above it
(debounced: above for 24 h before, below for 6 h after), how long until
  * the differential falls below the Red line (Isis 0.535 m, Godstow 0.75 m), 2019-2026;
  * the official flag leaves Red/Black (Isis from Nov 2021, Godstow from Sep 2024);
and how well the deployed hybrid (rain that fell) forecast that time from the crossing.

    python test_hannington_rule.py <hannington.csv>
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RED = {'isis': 0.535, 'godstow': 0.75}
FLAG_FROM = {'isis': '2021-11-01', 'godstow': '2024-09-01'}
LEVEL = 4.0


def crossings(h: pd.Series) -> list:
    below = h < LEVEL
    out = []
    for t in h.index[1:]:
        if below.get(t) and not below.get(t - pd.Timedelta(hours=1), True):
            before = h.loc[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1)]
            after = h.loc[t:t + pd.Timedelta(hours=6)]
            if len(before) >= 20 and (before >= LEVEL).all() and (after < LEVEL).all():
                if not out or t - out[-1] > pd.Timedelta(days=3):
                    out.append(t)
    return out


def main(path: str) -> None:
    import torch
    from build_error_library import predict_hybrid
    from flag_predictor.models.physics_data import HORIZON, load_location_data
    from flag_predictor.prediction.hybrid_live import HYBRID_MODELS
    torch.set_num_threads(1)
    h = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    h.index = pd.to_datetime(h.index, utc=True)
    h = h.interpolate(limit=6)
    events = crossings(h)
    print(f'{len(events)} times Hannington fell below {LEVEL} m (2019-2026)')
    for loc in ('isis', 'godstow'):
        data = load_location_data(loc, ROOT)
        idx = data.index.tz_convert('UTC')
        d = pd.Series(np.asarray(data.diff, float), idx)
        fh = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv')
        fh['start'] = pd.to_datetime(fh['start'], utc=True)
        fh['end'] = pd.to_datetime(fh['end'], utc=True)
        rows = []
        for t in events:
            t = t.floor('h')
            if t not in d.index or not np.isfinite(d.get(t, np.nan)):
                continue
            d0 = d.loc[t]
            fut = d.loc[t:t + pd.Timedelta(days=20)]
            below = fut[fut < RED[loc]]
            obs_days = (below.index[0] - t).total_seconds() / 86400 if len(below) else np.nan
            if d0 < RED[loc]:
                obs_days = 0.0
            flag_days = np.nan
            cur = fh[(fh.start <= t) & (fh.end > t)]
            flag_now = cur.iloc[-1].flag if len(cur) else None
            if t >= pd.Timestamp(FLAG_FROM[loc], tz='UTC') and flag_now in ('red', 'black'):
                nxt = fh[(fh.start > t) & ~fh.flag.isin(['red', 'black'])]
                if len(nxt):
                    flag_days = (nxt.iloc[0].start - t).total_seconds() / 86400
            # model forecast from the crossing (rain that fell): when does it go below the line?
            pos = idx.get_loc(t)
            pred_days = np.nan
            if pos + HORIZON < len(idx):
                try:
                    p = predict_hybrid(HYBRID_MODELS, loc, data, np.array([pos]))[0]
                    hit = np.where(p < RED[loc])[0]
                    pred_days = (hit[0] + 1) / 24 if len(hit) else np.inf
                    if d0 < RED[loc]:
                        pred_days = 0.0
                except Exception:
                    pass
            rows.append({'t': t, 'd0': d0, 'flag': flag_now, 'diff_below_red_days': obs_days,
                         'flag_off_red_days': flag_days, 'model_days': pred_days})
        df = pd.DataFrame(rows)
        print(f'\n=== {loc} (Red line {RED[loc]} m): {len(df)} crossings with a reading')
        print(df.assign(t=df.t.dt.strftime('%Y-%m-%d %H:%M')).round(2).to_string(index=False))
        red = df[df.d0 >= RED[loc]]
        q = red.diff_below_red_days.dropna()
        print(f'  differential in Red at the crossing: {len(red)}; days until below the Red line: '
              f'median {q.median():.1f}, middle half {q.quantile(.25):.1f}-{q.quantile(.75):.1f}, '
              f'range {q.min():.1f}-{q.max():.1f}; within 3-5 days: {((q >= 3) & (q <= 5)).mean():.0%}')
        f = df.flag_off_red_days.dropna()
        if len(f):
            print(f'  official flag Red/Black at the crossing: {len(f)}; days until it came off: median {f.median():.1f}, '
                  f'middle half {f.quantile(.25):.1f}-{f.quantile(.75):.1f}, range {f.min():.1f}-{f.max():.1f}; '
                  f'within 3-5 days: {((f >= 3) & (f <= 5)).mean():.0%}')
        m = red.dropna(subset=['diff_below_red_days', 'model_days'])
        m = m[np.isfinite(m.model_days)]
        if len(m):
            err = m.model_days - m.diff_below_red_days
            print(f'  model (rain that fell) timing of leaving Red: mean error {err.mean():+.1f} d, '
                  f'mean |error| {err.abs().mean():.1f} d (n={len(m)}); rule "4 days": mean |error| '
                  f'{(4 - m.diff_below_red_days).abs().mean():.1f} d')


if __name__ == '__main__':
    main(sys.argv[1])
