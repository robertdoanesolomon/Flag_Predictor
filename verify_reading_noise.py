#!/usr/bin/env python3
"""
Walk-forward check of the reading-noise term (src/flag_predictor/prediction/reading_noise.py)
against the live forecast archive: 80 % band coverage of the hourly readings, before and after.

Each archived issue keeps only the band's 10/25/50/75/90 % quantiles, so its outcomes are
re-drawn from those (piecewise-linear), the noise is added with the pool known at that issue's
t0, and the 10–90 % range is taken again. Isis issues during the 13–27 Oct 2026 outage are left out.

    python verify_reading_noise.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from flag_predictor.prediction.reading_noise import (  # noqa: E402
    LOCATIONS, MIN_POOL, add_reading_noise, load_observed, residual_pool,
)

LEADS = [1, 3, 6, 12, 24, 72, 120]
P = np.r_[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def redraw(q, n, rng):
    x = np.r_[q[0] - 0.8 * (q[1] - q[0]), q, q[4] + 0.8 * (q[4] - q[3])]
    return np.interp(rng.uniform(size=n), P, x)


def main(n=2000):
    archive = ROOT / 'docs' / 'archive'
    rng = np.random.default_rng(0)
    obs = {loc: load_observed(archive, loc) for loc in ('isis', 'godstow', 'wallingford')}
    hits = {(loc, k): {L: [] for L in LEADS} for loc in obs for k in ('before', 'after')}
    for f in sorted(glob.glob(str(archive / 'issues' / '*.json'))):
        for loc, v in json.load(open(f))['locations'].items():
            if 'total_p10' not in v or obs.get(loc) is None:
                continue
            t0 = pd.Timestamp(v['t0'])
            if loc == 'isis' and pd.Timestamp('2026-10-13') <= t0 <= pd.Timestamp('2026-10-28'):
                continue
            Q = np.array([v[f'total_p{q}'] for q in (10, 25, 50, 75, 90)], dtype=float)
            pool = residual_pool(obs[loc], t0) if loc in LOCATIONS else np.array([])
            leads = [L for L in LEADS if L < Q.shape[1] and t0 + pd.Timedelta(hours=L) in obs[loc].index]
            if not leads:
                continue
            S = np.stack([redraw(Q[:, L], n, rng) for L in leads], axis=1)
            S = add_reading_noise(np.c_[np.zeros(n), S], pool, seed=int(rng.integers(1 << 30)))[:, 1:]
            for j, L in enumerate(leads):
                o = obs[loc][t0 + pd.Timedelta(hours=L)]
                if not np.isfinite(o):
                    continue
                lo, hi = np.percentile(S[:, j], [10, 90])
                hits[(loc, 'before')][L].append(Q[0, L] <= o <= Q[4, L])
                hits[(loc, 'after')][L].append(lo <= o <= hi)
    print('80% band coverage at ' + ' / '.join(f'{L} h' for L in LEADS))
    for (loc, k), h in hits.items():
        row = ' '.join(f'{100 * np.mean(h[L]):4.0f}%' if h[L] else '    -' for L in LEADS)
        print(f'  {loc:12s} {k:7s} {row}   (n={len(h[1])})')


if __name__ == '__main__':
    main()
