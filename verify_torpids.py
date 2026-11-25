"""Torpids chance: leave-one-year-out hindcasts scored on 'pseudo-Torpids' as well as the real ones, and the fit of
the rain-since-January adjustment (termly.TORPIDS_RAIN_COEF).

Why pseudo-Torpids: only 32 Torpids since 1994 (2021 not held), and a forecast of a 4-day mean over 42 m3/s is
judged on whether a few years landed just over or under the line. Every 4-day window starting 16 Feb, 18 Feb, ...
6 Mar (where Torpids falls) in every year gives ten times the cases with the same meaning. Scores are Brier skill
against an average year for the same dates (the share of the other years over 42 m3/s on them).

Also checks whether the real Torpids weeks were wetter than the weeks around them in the same winters (dates
shifted by 1-7 days): the reason the record (31% lost) is above an average year for the dates (about 21%).

    python verify_torpids.py            # ~1 minute
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
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

q = T.load_flow(ROOT)
rain = pd.read_csv(ROOT / T.ERA5_DAILY, index_col=0, parse_dates=True).iloc[:, 0]
LAST = 2025
YEARS = [Y for Y in range(T.FIRST_YEAR, LAST + 1) if Y + 1 != 2021]
OVER = lambda s: float(q.loc[s:s + pd.Timedelta(days=3)].mean() >= T.TORPIDS_FLOW)  # noqa: E731
lg = lambda p: logit(np.clip(p, 0.02, 0.98))  # noqa: E731


def windows(Y):
    w = [('pseudo', pd.Timestamp(Y + 1, 2, 16) + pd.Timedelta(days=2 * k)) for k in range(10)]
    return w + ([('torpids', T.race_days('torpids', Y)[0])] if Y + 1 != 2021 else [])


def usual(Y, s):
    """Share of the other years over the line on these calendar dates."""
    return np.mean([q.reindex([T._same_date(s + pd.Timedelta(days=i), Y - y) for i in range(4)]).mean() >= T.TORPIDS_FLOW
                    for y in YEARS if y != Y])


def rain_z(Y, t0):
    j1 = pd.Timestamp(Y + 1, 1, 1)
    if t0 < j1:
        return np.nan, 0
    n = (t0 - j1).days + 1
    o = [rain.loc[pd.Timestamp(y, 1, 1):pd.Timestamp(y, 1, 1) + pd.Timedelta(days=n - 1)].mean() for y in range(1994, LAST + 2) if y != Y + 1]
    return (rain.loc[j1:t0].mean() - np.mean(o)) / np.std(o), n


# ---- hindcasts: issued every 3 days from 1 Sep, the chance as the page works it out (river only)
rows = []
for Y in range(T.FIRST_YEAR, LAST + 1):
    model = T.Outlook(q, exclude=Y)
    ref = {s: usual(Y, s) for _, s in windows(Y)}
    for t0 in pd.date_range(f'{Y}-09-01', f'{Y + 1}-03-08', freq='3D'):
        try:
            ens = model.ensemble(t0)
        except ValueError:
            continue
        z, n = rain_z(Y, t0)
        for kind, s in windows(Y):
            lead = (s - t0).days
            if lead <= 0:
                continue
            race = pd.date_range(s, periods=4)
            samples, wts = T.mix(ens, lead)
            p = float(np.sum(wts * (T.race_flow(samples, ens.days, race) >= T.TORPIDS_FLOW)))
            rows.append({'Y': Y, 'kind': kind, 'lead': lead, 'p': p, 'clim': float((T.race_flow(ens.climatology, ens.days, race)
                                                                                     >= T.TORPIDS_FLOW).mean()),
                         'z': z, 'n': n, 'obs': OVER(s), 'ref': ref[s]})
    print(Y, flush=True)
d = pd.DataFrame(rows)
d['zn'] = d.z.fillna(0) * d.n / 60
pseudo = d.kind == 'pseudo'


def fit(x, y):
    X = np.c_[np.ones(len(x)), x]
    f = lambda b: -np.sum(y * (X @ b) - np.log1p(np.exp(X @ b))) + 1e-3 * np.sum(b ** 2)  # noqa: E731
    return minimize(f, np.r_[0, 1, np.zeros(X.shape[1] - 2)]).x


# ---- rain since 1 January: fitted on the pseudo weeks in the last TORPIDS_RAIN_DAYS, from 1 January
m = pseudo & d.z.notna() & (d.lead <= T.TORPIDS_RAIN_DAYS)
coef = fit(np.c_[lg(d.p[m]), d.zn[m]], d.obs[m].to_numpy())
print('\nTORPIDS_RAIN_COEF (all years):', tuple(round(float(c), 2) for c in coef))
p_rain = d.p.to_numpy().copy()
for Y in d.Y.unique():                       # leave-one-year-out for the scores
    b = fit(np.c_[lg(d.p[m & (d.Y != Y)]), d.zn[m & (d.Y != Y)]], d.obs[m & (d.Y != Y)].to_numpy())
    te = ((d.Y == Y) & d.z.notna()).to_numpy()
    p_rain[te] = expit(b[0] + b[1] * lg(d.p[te]) + b[2] * d.zn[te])
d['p_rain'] = [T.torpids_rain_blend(p, r, l) if np.isfinite(z) else p for p, r, l, z in zip(d.p, p_rain, d.lead, d.z)]

# ---- scores
buckets = ((0, 7, 'last week'), (7, 21, '1-3 weeks'), (21, 35, '3-5 weeks'), (35, 63, '5-9 weeks'),
           (63, T.TORPIDS_TOO_EARLY, '9 weeks - 3 months'), (T.TORPIDS_TOO_EARLY, 400, 'further'))
bss = lambda g, col: round(100 * (1 - ((g[col] - g.obs) ** 2).mean() / ((g.ref - g.obs) ** 2).mean()), 1)  # noqa: E731
for kind, label in (('pseudo', 'pseudo-Torpids (16 Feb - 6 Mar windows)'), ('torpids', 'actual Torpids')):
    g0 = d[d.kind == kind]
    print(f'\nBrier skill (%) vs an average year for the dates, {label}; n = {len(g0)}, '
          f'over 42 m3/s {g0.obs.mean():.1%}, forecast {g0.p.mean():.1%}, average year {g0.ref.mean():.1%}')
    t = pd.DataFrame({lab: {'river only': bss(g, 'p'), 'with rain': bss(g, 'p_rain'), 'n': len(g)}
                      for a, b, lab in buckets if len(g := g0[(g0.lead > a) & (g0.lead <= b)])}).T
    print(t.to_string())

# ---- were the actual Torpids weeks wetter than the weeks around them?
st = {Y: T.race_days('torpids', Y)[0] for Y in YEARS}
print('\nOver the line (4-day mean), actual Torpids dates vs the same winters with the dates shifted 1-7 days:')
for thr in (35, 42, 50):
    over = lambda s: q.loc[s:s + pd.Timedelta(days=3)].mean() >= thr  # noqa: E731
    shifted = np.mean([sum(over(st[Y] + pd.Timedelta(days=o)) for Y in YEARS) for o in range(-7, 8) if o])
    print(f'  {thr} m3/s: actual {sum(over(st[Y]) for Y in YEARS)} of {len(YEARS)}, shifted {shifted:.1f}')
