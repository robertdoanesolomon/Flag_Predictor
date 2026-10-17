#!/usr/bin/env python
"""
Quick batch comparison of forecast models on 6-hourly starts (observed rain).

    python compare_batch.py --hybrid hybrid_c1_ps --continuous hybrid_cs_ft --period test

Each argument is a '+'-separated list of model names (seeds) that are averaged.
Prints MAE by lead time overall and for named episodes, e.g. the first storms
after the dry summer of 2025.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import build_error_library as B  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, load_location_data  # noqa: E402

EPISODES = {
    'autumn_2025_first_storms': ('2025-10-20', '2025-12-15'),
    'winter_2024_25': ('2024-11-15', '2025-03-01'),
    'summer_2025_dry': ('2025-06-01', '2025-09-30'),
}
LEADS = {'1-24h': (0, 24), '25-72h': (24, 72), '73-240h': (72, 240), 'all': (0, 240)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hybrid', action='append', default=[], help='+-separated hybrid model names')
    ap.add_argument('--continuous', action='append', default=[], help='+-separated continuous model names')
    ap.add_argument('--simfeat', action='append', default=[], help='NAMES@CORE: hybrid + simulation features')
    ap.add_argument('--period', default='test', choices=['test', 'val2023'])
    ap.add_argument('--locations', default='isis,godstow,wallingford')
    args = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    per_start = {}
    for loc in args.locations.split(','):
        data = load_location_data(loc, ROOT)
        t_idx = B.start_positions(data)[args.period]
        obs = np.stack([data.diff[t + 1:t + 1 + HORIZON] for t in t_idx])
        t0 = data.index[t_idx]
        models = ([('hybrid', m) for m in args.hybrid] + [('continuous', m) for m in args.continuous]
                  + [('simfeat', m) for m in args.simfeat])
        for fam, names in models:
            if fam == 'simfeat':
                names, core = names.split('@')
                pred = B.predict_simfeat(names.split('+'), loc, data, t_idx, core)
            else:
                fn = B.predict_hybrid if fam == 'hybrid' else B.predict_continuous
                pred = fn(names.split('+'), loc, data, t_idx)
            err = np.abs(pred - obs)
            label = names.split('+')[0] + (f' x{len(names.split("+"))}' if '+' in names else '')
            per_start[f'{loc}|{label}|t0'] = t0.tz_convert('UTC').tz_localize(None).to_numpy().astype('datetime64[s]').astype('int64')
            per_start[f'{loc}|{label}|mae'] = np.nanmean(err, axis=1)
            per_start[f'{loc}|{label}|mae_1_72'] = np.nanmean(err[:, :72], axis=1)
            sets = {'all': np.ones(len(t_idx), bool)}
            for ep, (a, b) in EPISODES.items():
                m = (t0 >= pd.Timestamp(a, tz='UTC')) & (t0 < pd.Timestamp(b, tz='UTC'))
                if m.any():
                    sets[ep] = np.asarray(m)
            for sname, m in sets.items():
                row = {'location': loc, 'model': names.split('+')[0] + (f' x{len(names.split("+"))}' if '+' in names else ''),
                       'subset': sname, 'n': int(m.sum())}
                for lname, (a, b) in LEADS.items():
                    row[lname] = float(np.nanmean(err[m, a:b]))
                rows.append(row)
    df = pd.DataFrame(rows)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(df.round(4).to_string(index=False))
    out = ROOT / 'figures' / f'compare_batch_{args.period}.csv'
    df.to_csv(out, index=False)
    np.savez_compressed(ROOT / 'figures' / f'compare_batch_{args.period}_per_start.npz', **per_start)
    paired_bootstrap(per_start)


def paired_bootstrap(per_start: dict, block_days: int = 7, n_boot: int = 2000, seed: int = 0) -> None:
    """95% intervals for the MAE difference between each model and the first, resampling whole weeks."""
    rng = np.random.default_rng(seed)
    locs = sorted({k.split('|')[0] for k in per_start})
    for loc in locs:
        labels = [k.split('|')[1] for k in per_start if k.startswith(loc + '|') and k.endswith('|mae')]
        if len(labels) < 2:
            continue
        base = labels[0]
        t0 = per_start[f'{loc}|{base}|t0']
        week = (t0 - t0.min()) // (block_days * 86400)
        weeks = np.unique(week)
        for other in labels[1:]:
            for metric in ('mae', 'mae_1_72'):
                d = per_start[f'{loc}|{other}|{metric}'] - per_start[f'{loc}|{base}|{metric}']
                rel = []
                for _ in range(n_boot):
                    pick = rng.choice(weeks, size=len(weeks), replace=True)
                    idx = np.concatenate([np.where(week == w)[0] for w in pick])
                    rel.append(np.nanmean(d[idx]) / np.nanmean(per_start[f'{loc}|{base}|{metric}'][idx]))
                lo, hi = np.percentile(rel, [2.5, 97.5])
                mean = np.nanmean(d) / np.nanmean(per_start[f'{loc}|{base}|{metric}'])
                print(f'{loc:12s} {other} vs {base} [{metric}]: {mean:+.1%} (95% CI {lo:+.1%} to {hi:+.1%})')


if __name__ == '__main__':
    main()
