"""Bias-correct the long-range page's percentages (lower / usual / higher odds and the Torpids chance) from the
leave-one-year-out hindcasts written by verify_sentences.py. For each lead group: logit p' = a + b logit p, fitted
on all years, checked leave-one-year-out. Writes data/termly/calibration.json (read by termly.calibrate_*).

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
tp = pd.read_csv(EV / 'termly_sentences_torpids.csv')


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
tp['group'] = tp.before.map(T.calibration_group_torpids)
out = {'terms': {}, 'torpids': {}, 'check': {}}


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
for g, d in tp.groupby('group'):
    out['torpids'][g] = list(fit(d.go, d.raced))
    p2 = np.zeros(len(d))
    for Y in d.Y.unique():
        tr, te = d[d.Y != Y], (d.Y == Y).to_numpy()
        p2[te] = apply(fit(tr.go, tr.raced), d.go[te])
    clim = ((d.raced.mean() - d.raced) ** 2).mean()
    out['check'][f'torpids {g}'] = {'skill_before': round(100 * (1 - ((d.go - d.raced) ** 2).mean() / clim), 1),
                                    'skill_after': round(100 * (1 - ((p2 - d.raced) ** 2).mean() / clim), 1),
                                    'mean_forecast_before': round(float(d.go.mean()), 2), 'mean_after': round(float(p2.mean()), 2),
                                    'raced': round(float(d.raced.mean()), 2), 'n': len(d)}
# keep a correction only where the leave-one-year-out check shows it helps; where even the corrected Torpids
# chance is worse than the record, the page shows the record ('too early to tell')
keep = lambda k: out['check'][k]['skill_after'] > out['check'][k]['skill_before']  # noqa: E731
out['terms'] = {g: {k: [round(float(v), 3) for v in b] for k, b in d.items()} for g, d in out['terms'].items()
                if keep(f'terms {g}')}
out['torpids_record_only'] = [g for g in out['torpids'] if max(out['check'][f'torpids {g}']['skill_after'],
                                                               out['check'][f'torpids {g}']['skill_before']) < 0]
out['torpids'] = {g: [round(float(v), 3) for v in b] for g, b in out['torpids'].items() if keep(f'torpids {g}')}
(ROOT / T.CALIBRATION_FILE).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
