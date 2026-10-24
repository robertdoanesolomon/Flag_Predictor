"""
The first big rise of each winter half-year: did the forecast see it coming?

Event: the first time after 1 September that the differential crosses the first flag line (Isis Light
Blue 0.215 m, Godstow Amber 0.45 m; Wallingford, no flags: 0.25 m above its summer level) after at
least 30 days below it. For forecasts issued 3 days, 2 days and 1 day before the crossing (the deployed
hybrid, 5 copies, rain that fell): when does it forecast the crossing, and how high the first peak
(highest level in the 3 days after the crossing)?

    python test_first_rise.py [--trial hybrid_srh]   # the trial model (soil moisture + upstream gauges)
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
import torch  # noqa: E402

from build_error_library import predict_hybrid  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402

LINE = {'isis': 0.215, 'godstow': 0.45}
HELD_OUT = lambda t: t.year == 2023 or t >= pd.Timestamp('2024-10-22', tz='UTC')  # noqa: E731


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default=None)
    args = ap.parse_args()
    names = HYBRID_MODELS
    torch.set_num_threads(2)
    for loc in ('isis', 'godstow', 'wallingford'):
        data = load_location_data(loc, ROOT)
        if args.trial:
            from train_extra_models import data_with_extra
            names = [args.trial] + [f'{args.trial}_s{i}' for i in range(1, 5)]
            data = data_with_extra(loc, 'soil_rise_han')
        idx = data.index.tz_convert('UTC')
        d = pd.Series(np.asarray(data.diff, float), idx)
        enc_ok = np.isfinite(data.enc).all(axis=1)
        ok = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(enc_ok)] == SEQ_LEN
        print(f'\n=== {loc} ===')
        rows = []
        for year in range(2019, 2026):
            w = d.loc[f'{year}-09-01':f'{year + 1}-03-31']
            if w.notna().mean() < 0.5:
                continue
            line = LINE.get(loc)
            if line is None:
                base = d.loc[f'{year}-08-01':f'{year}-09-30'].median()
                line = base + 0.25
            cross = None
            for t in w.index[w > line]:
                prev = d.loc[t - pd.Timedelta(days=30):t - pd.Timedelta(hours=1)].dropna()
                if len(prev) > 24 * 15 and (prev < line).all():
                    cross = t
                    break
            if cross is None:
                continue
            obs_peak = d.loc[cross:cross + pd.Timedelta(days=3)].max()
            row = {'year': f'{year}/{str(year + 1)[2:]}', 'crossed': cross.strftime('%d %b %H:%M'),
                   'held_out': HELD_OUT(cross), 'peak': round(float(obs_peak), 3)}
            for k in (72, 48, 24):
                t0 = cross - pd.Timedelta(hours=k)
                if t0 not in idx:
                    continue
                pos = idx.get_loc(t0)
                if not ok[pos] or not np.isfinite(d.iloc[pos]) or pos + HORIZON >= len(idx):
                    continue
                p = predict_hybrid(names, loc, data, np.array([pos]))[0]
                ptimes = idx[pos + 1:pos + HORIZON + 1]
                hit = np.flatnonzero(p > line)
                row[f'{k}h: lead'] = round((ptimes[hit[0]] - t0).total_seconds() / 3600) if len(hit) else None
                sel = (ptimes >= cross) & (ptimes <= cross + pd.Timedelta(days=3))
                row[f'{k}h: peak'] = round(float(p[sel].max()), 3) if sel.any() else None
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f'line {LINE.get(loc, "summer level + 0.25")} m; "lead": hours from issue to the forecast crossing '
              f'(true: 72 / 48 / 24); "peak": highest level in the 3 days after the real crossing')
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
