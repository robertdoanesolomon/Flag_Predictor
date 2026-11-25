"""Bias-correct the long-range page's lower / usual / higher odds from the leave-one-year-out hindcasts written
by verify_sentences.py (the Torpids chance is no longer corrected: see verify_torpids.py). For each lead group:
logit p' = a + b logit p, fitted on all years, checked leave-one-year-out. Writes data/termly/calibration.json (read by termly.calibrate_*).

    python verify_sentences.py && python calibrate_termly.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from flag_predictor import termly as T  # noqa: E402

EV = ROOT / 'figures' / 'eval'
df = pd.read_csv(EV / 'termly_sentences.csv')


def lg(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def fit(p, y):
    """Shift only (slope fixed at 1): corrects the overall bias without re-stretching the odds, which the
    leave-one-year-out check showed over-fits with only ~33 independent years."""
    x = lg(np.asarray(p))
    y = np.asarray(y, float)
    f = lambda a: -np.sum(y * (a[0] + x) - np.log1p(np.exp(a[0] + x)))  # noqa: E731
    return np.array([minimize(f, np.array([0.0])).x[0], 1.0])


def apply(b, p):
    return 1 / (1 + np.exp(-(b[0] + b[1] * lg(np.asarray(p)))))


df['group'] = df.lead.map(T.calibration_group)
out = {'terms': {}, 'check': {}}


def rps3(lo, hi, cat):
    c = np.c_[lo, 1 - hi]
    o = np.c_[cat == 0, cat <= 1].astype(float)
    return ((c - o) ** 2).sum(1)


for g, d in df.groupby('group'):
    out['terms'][g] = {'lower': list(fit(d.p_lo, d.cat == 0)), 'higher': list(fit(d.p_hi, d.cat == 2))}
    lo2, hi2 = np.zeros(len(d)), np.zeros(len(d))
    for Y in d.Y.unique():          # leave-one-year-out check
        tr, te = d[d.Y != Y], (d.Y == Y).to_numpy()
        lo, hi = apply(fit(tr.p_lo, tr.cat == 0), d.p_lo[te]), apply(fit(tr.p_hi, tr.cat == 2), d.p_hi[te])
        s = np.maximum(1, lo + hi)
        lo2[te], hi2[te] = lo / s, hi / s
    clim = rps3(np.full(len(d), 1 / 3), np.full(len(d), 1 / 3), d.cat.to_numpy()).mean()
    out['check'][f'terms {g}'] = {'skill_before': round(100 * (1 - rps3(d.p_lo, d.p_hi, d.cat.to_numpy()).mean() / clim), 1),
                                  'skill_after': round(100 * (1 - rps3(lo2, hi2, d.cat.to_numpy()).mean() / clim), 1),
                                  'n': len(d)}
# keep a correction only where the leave-one-year-out check shows it helps
keep = lambda k: out['check'][k]['skill_after'] > out['check'][k]['skill_before']  # noqa: E731
out['terms'] = {g: {k: [round(float(v), 3) for v in b] for k, b in d.items()} for g, d in out['terms'].items()
                if keep(f'terms {g}')}
(ROOT / T.CALIBRATION_FILE).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
