"""Does weighting the member years by the site's 10-day forecast help the long-range page? And how do the odds for
the rest of a period (now on the page) compare with those for the whole period (before)?

Leave-one-year-out hindcasts 2016/17-2025/26 (the years with a Godstow differential record), issued every 3 days
September-June. Only about two months of real archived 10-day forecasts exist (a dry autumn), so the 'forecast' is
synthetic, as in verify_sentences.py's EC46 check: the 10-day mean Godstow noon differential that was measured,
plus an error of a given size. Error sizes tried: 0.05 m (what the archive shows, Oct-Nov 2026), 0.10 m, and
0.04 + 0.15 x the differential (bigger when the river is up). 'Over-confident' = the weighting is told 0.05 m but
the real error is the proportional one. The curve flow -> differential is refitted without the year being tested.

Scored by RPS skill against a usual year (1/3 each) for periods under way ('during') and starting within a month.

    python test_flow_forecast.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

q = T.load_flow(ROOT)
dg = pd.read_csv(ROOT / 'data' / 'termly' / f'differential_noon_{T.FLOW_FC_LOC}.csv', index_col=0,
                 parse_dates=True).iloc[:, 0]
FIRST, LAST = 2016, 2025
rng = np.random.default_rng(1)
# name: (true error sd as a function of the measured 10-day mean, sd the weighting is told)
CASES = {
    'err 0.05': (lambda d: 0.05, lambda d: 0.05),
    'err 0.10': (lambda d: 0.10, lambda d: 0.10),
    'err prop': (lambda d: 0.04 + 0.15 * d, lambda d: 0.04 + 0.15 * d),
    'over-confident': (lambda d: 0.04 + 0.15 * d, lambda d: 0.05),
}


def rps(p, cat):
    o = np.zeros(3)
    o[cat] = 1
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2))


def thirds(period, Y, off, n):
    v = []
    for y in range(T.FIRST_YEAR, LAST + 2):
        if y == Y:
            continue
        a0 = T.term_window(period, y)[0] + pd.Timedelta(days=off)
        v.append(float(q.loc[a0:a0 + pd.Timedelta(days=n - 1)].mean()))
    return np.nanpercentile(v, [100 / 3, 200 / 3])


def cat_of(x, lo, hi):
    return 0 if x < lo else (2 if x > hi else 1)


def probs(pm, wts, lo, hi):
    p = np.array([np.sum(wts * (pm < lo)), 0, np.sum(wts * (pm > hi))])
    p[1] = 1 - p[0] - p[2]
    return p


rows = []
for Y in range(FIRST, LAST + 1):
    model = T.Outlook(q, exclude=Y)
    curve = T.differential_curve(ROOT, q, exclude=Y)
    whole = {p: thirds(p, Y, 0, (T.term_window(p, Y)[1] - T.term_window(p, Y)[0]).days + 1) for p in T.PERIODS}
    for t0 in pd.date_range(f'{Y}-09-01', f'{Y + 1}-06-30', freq='3D'):
        fc_days = pd.date_range(t0 + pd.Timedelta(days=1), periods=10)
        truth = dg.reindex(fc_days)
        if truth.notna().sum() < 8 or not np.isfinite(dg.get(t0, np.nan)):
            continue
        try:
            ens = model.ensemble(t0)
        except ValueError:
            continue
        d_true = float(truth.mean())
        lws = {}
        for name, (err, told) in CASES.items():
            fc = d_true + err(d_true) * rng.standard_normal()
            lws[name] = T.flow_forecast_lw(ens, curve, fc_days, fc, told(d_true), float(dg[t0]), float(q[t0]))
        for p in T.PERIODS:
            w0, w1 = T.term_window(p, Y)
            lead = max(0, (w0 - t0).days)
            if w1 <= t0 + pd.Timedelta(days=1) or lead > 31:
                continue
            s_ = max(w0, t0 + pd.Timedelta(days=1))
            sel = (ens.days >= s_) & (ens.days <= w1)
            past = q.loc[w0:t0].to_numpy() if w0 <= t0 else np.array([])
            lo_w, hi_w = whole[p]
            lo_r, hi_r = thirds(p, Y, (s_ - w0).days, (w1 - s_).days + 1)
            c_w = cat_of(float(q.loc[w0:w1].mean()), lo_w, hi_w)
            c_r = cat_of(float(q.loc[s_:w1].mean()), lo_r, hi_r)
            rec = {'Y': Y, 't0': t0, 'period': p, 'lead': lead, 'days_left': int(sel.sum()),
                   'clim_w': rps(np.full(3, 1 / 3), c_w), 'clim_r': rps(np.full(3, 1 / 3), c_r)}
            for name, lw in [('river', None)] + list(lws.items()):
                ens.year_w = None if lw is None else T.flatten_weights(lw)[0]
                samples, wts = T.mix(ens, lead, slow=True)
                rest = samples[:, sel].mean(1)
                whole_m = (samples[:, sel].sum(1) + past.sum()) / (sel.sum() + len(past))
                rec[f'w {name}'] = rps(probs(whole_m, wts, lo_w, hi_w), c_w)
                rec[f'r {name}'] = rps(probs(rest, wts, lo_r, hi_r), c_r)
            ens.year_w = None
            rows.append(rec)
    print(Y, len(rows), flush=True)

df = pd.DataFrame(rows)
out = ROOT / 'figures' / 'eval'
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'termly_flow_forecast.csv', index=False)
df['when'] = np.where(df.lead == 0, 'during', 'up to 1 month')
df.loc[(df.lead == 0) & (df.days_left <= 21), 'when'] = 'during, last 3 weeks'
names = ['river'] + list(CASES)


def skill(g, target):
    c = g[f'clim_{target}'].mean()
    return {n: round(100 * (1 - g[f'{target} {n}'].mean() / c), 1) for n in names}


def boot(g, target, a, b, n=1000):
    """95% interval of skill(b) - skill(a), resampling whole years."""
    ys = g.Y.unique()
    d = []
    for _ in range(n):
        s = pd.concat([g[g.Y == y] for y in rng.choice(ys, len(ys))])
        c = s[f'clim_{target}'].mean()
        d.append(100 * (s[f'{target} {a}'].mean() - s[f'{target} {b}'].mean()) / c)
    return np.round(np.percentile(d, [2.5, 97.5]), 1)


for target, label in (('w', 'Whole period (as before)'), ('r', 'Rest of period (new)')):
    print(f'\n{label}: RPS skill vs a usual year (%)')
    t = pd.DataFrame({w: skill(g, target) for w, g in df.groupby('when')}).T
    t['n'] = df.groupby('when').size()
    print(t.to_string())
    for w, g in df.groupby('when'):
        print(f'  {w}: err prop - river, 95% CI {boot(g, target, "river", "err prop")}')
