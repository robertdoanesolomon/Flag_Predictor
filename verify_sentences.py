"""How useful is the long-range page? Leave-one-year-out hindcasts 1993/94-2025/26, issued on the 1st and 15th of
every month September-June: for each sentence the page would have shown, what actually happened; the hit rate of
the most likely category; Torpids reliability. Also a method check for the (not yet tested) EC46 weighting: the
rain that actually fell, degraded to the skill EC46 is assumed to have, used to weight the member years.

    python verify_sentences.py
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
from scipy.special import ndtri  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

q = T.load_flow(ROOT)
rain = pd.read_csv(ROOT / T.ERA5_DAILY, index_col=0, parse_dates=True).iloc[:, 0]
LAST = 2025
rng = np.random.default_rng(0)


def sentence(lo, hi):
    if lo >= 0.6: return 'Very likely lower'
    if hi >= 0.6: return 'Very likely higher'
    if lo >= 0.5: return 'Likely lower'
    if hi >= 0.5: return 'Likely higher'
    if lo >= 0.4 and hi <= 0.2: return 'Leaning lower'
    if hi >= 0.4 and lo <= 0.2: return 'Leaning higher'
    return 'No clear signal'


def week_total(a, years_back):
    b = T._same_date(a, years_back)
    v = rain.reindex(pd.date_range(b, periods=7))
    return float(v.sum()) if v.notna().all() else np.nan


def synthetic_weights(ens, Y):
    """Year weights as forecast_year_weights, with the 'forecast' = what fell in year Y degraded to skill r."""
    acad0 = T.academic_year(ens.t0)
    years = np.array(ens.years)
    lw = np.zeros(len(years))
    for k, r in enumerate(T.EC46_SKILL):
        a = ens.t0 + pd.Timedelta(days=1 + 7 * k)
        clim = np.array([week_total(a, acad0 - y) for y in range(T.FIRST_YEAR, LAST + 1) if y != Y])
        clim = clim[np.isfinite(clim)]
        z = lambda x: float(ndtri(np.clip(((clim < x).sum() + 0.5 * (clim == x).sum() + 0.5) / (len(clim) + 1), .01, .99)))  # noqa: E731
        truth = week_total(a, 0)
        if not np.isfinite(truth):
            return None
        zf = r * z(truth) + np.sqrt(1 - r * r) * rng.standard_normal()
        zy = np.array([z(v) if np.isfinite(v := week_total(a, acad0 - y)) else 0.0 for y in years])
        lw += -0.5 * (zy - r * zf) ** 2 / (1 - r * r) + 0.5 * zy ** 2
    lam = 1.0
    for _ in range(30):
        w = np.exp(lam * (lw - lw.max()))
        w /= w.sum()
        if 1 / np.sum(w ** 2) >= T.MIN_EFFECTIVE_YEARS:
            break
        lam *= 0.8
    return w


def rps(p, cat):
    o = np.zeros(3)
    o[cat] = 1
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2))


rows, torp = [], []
for Y in range(T.FIRST_YEAR, LAST + 1):
    model = T.Outlook(q, exclude=Y)
    usual = {p: np.percentile([float(q.loc[slice(*T.term_window(p, y))].mean()) for y in range(T.FIRST_YEAR, LAST + 1)
                               if y != Y], [100 / 3, 200 / 3]) for p in T.PERIODS}
    for m in (9, 10, 11, 12, 1, 2, 3, 4, 5, 6):
        for d in (1, 15):
            t0 = pd.Timestamp(Y if m >= 9 else Y + 1, m, d)
            try:
                ens = model.ensemble(t0)
            except ValueError:
                continue
            sw = synthetic_weights(ens, Y)
            for p in T.PERIODS:
                w0, w1 = T.term_window(p, Y)
                if w1 <= t0:
                    continue
                lo_t, hi_t = usual[p]
                ob = float(q.loc[w0:w1].mean())
                cat = 0 if ob < lo_t else (2 if ob > hi_t else 1)
                s_ = max(w0, t0 + pd.Timedelta(days=1))
                sel = (ens.days >= s_) & (ens.days <= w1)
                past = q.loc[w0:t0].to_numpy() if w0 <= t0 else np.array([])
                lead = max(0, (w0 - t0).days)
                rec = {'Y': Y, 't0': t0, 'period': p, 'lead': lead, 'cat': cat}
                for name, yw in (('river', None), ('syn', sw)):
                    if name == 'syn' and yw is None:
                        continue
                    ens.year_w = yw
                    samples, wts = T.mix(ens, lead)
                    pm = (samples[:, sel].sum(1) + past.sum()) / (sel.sum() + len(past))
                    pr = np.array([np.sum(wts * (pm < lo_t)), 0, np.sum(wts * (pm > hi_t))])
                    pr[1] = 1 - pr[0] - pr[2]
                    rec[f'rps_{name}'] = rps(pr, cat)
                    if name == 'river':
                        rec |= {'p_lo': pr[0], 'p_us': pr[1], 'p_hi': pr[2], 'say': sentence(pr[0], pr[2])}
                ens.year_w = None
                rec['rps_clim'] = rps(np.full(3, 1 / 3), cat)
                rows.append(rec)
            race = T.race_days('torpids', Y)
            if race[0] > t0 and Y + 1 != 2021:
                samples, wts = T.mix(ens, (race[0] - t0).days)
                pl = float(np.sum(wts * (T.race_flow(samples, ens.days, race) >= T.TORPIDS_FLOW)))
                torp.append({'Y': Y, 't0': t0, 'before': (race[0] - t0).days, 'go': 1 - pl,
                             'raced': float(float(q.reindex(race).mean()) < T.TORPIDS_FLOW)})
    print(Y, flush=True)

df, tp = pd.DataFrame(rows), pd.DataFrame(torp)
out = ROOT / 'figures' / 'eval'
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'termly_sentences.csv', index=False)
tp.to_csv(out / 'termly_sentences_torpids.csv', index=False)
df['when'] = pd.cut(df.lead, [-1, 0, 31, 92, 1000], labels=['during', 'up to 1 month', '1-3 months', '3+ months'])
order = ['Very likely lower', 'Likely lower', 'Leaning lower', 'No clear signal', 'Leaning higher', 'Likely higher', 'Very likely higher']
names = {0: 'lower', 1: 'usual', 2: 'higher'}
print('\nWhat happened after each sentence (%), all leads:')
t = df.groupby('say').cat.value_counts(normalize=True).unstack(fill_value=0).rename(columns=names).mul(100).round()
t['n'] = df.groupby('say').size()
print(t.reindex(order).to_string())
for w, g in df.groupby('when'):
    t = g.groupby('say').cat.value_counts(normalize=True).unstack(fill_value=0).rename(columns=names).mul(100).round()
    t['n'] = g.groupby('say').size()
    print(f'\n{w}:')
    print(t.reindex([o for o in order if o in t.index]).to_string())
df['top'] = df[['p_lo', 'p_us', 'p_hi']].to_numpy().argmax(1)
print('\nMost likely category right (%) [chance 33]:')
print(df.groupby('when').apply(lambda g: round(100 * (g.top == g.cat).mean())).to_string())
print('\nShare of forecasts saying something other than "No clear signal" (%):')
print(df.groupby('when').apply(lambda g: round(100 * (g.say != 'No clear signal').mean())).to_string())
print('\nRPS skill vs a usual year (%), river only vs with a synthetic EC46-skill forecast:')
print(df.groupby('when').apply(lambda g: pd.Series({'river': round(100 * (1 - g.rps_river.mean() / g.rps_clim.mean()), 1),
                                                     'with_syn_ec46': round(100 * (1 - g.rps_syn.mean() / g.rps_clim.mean()), 1)})).to_string())
tp['bin'] = pd.cut(tp.go, [0, .5, .7, .8, .9, 1.0])
tp['when'] = pd.cut(tp.before, [0, 35, 92, 400], labels=['last 5 weeks', '5 weeks-3 months', '3+ months'])
print('\nTorpids: forecast "goes ahead" vs how often it did:')
print(tp.groupby(['when', 'bin']).agg(n=('raced', 'size'), forecast=('go', 'mean'), raced=('raced', 'mean')).round(2).dropna().to_string())
