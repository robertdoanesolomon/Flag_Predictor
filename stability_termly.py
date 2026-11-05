"""How much does the long-range page move from one day to the next, and what does steadying it cost?

Leave-one-year-out hindcasts 1993/94-2025/26 issued EVERY day Sep-Jun (verify_termly.py and verify_sentences.py
issue twice a month, which hides day-to-day changes). For each period, the lower/usual/higher odds; for
Torpids, the chance of a cancellation-level river. River only (no EC46 weighting, no calibration). Compared:

  now        the 7-day and 90-day mean scores (termly.PREDICTORS), for every lead
  blend      7-day -> 30-day mean as the period gets further off (termly.SLOW_RAMP; mix(..., slow=True))
  smoothed   the 'now' odds smoothed over time: exponential smoothing of the log-odds, half-life h days.
             This is the steady-state Kalman filter of a random walk seen through noise; fitted to the
             hindcasts (lag-1 covariance of daily changes) its gain is 1 at every lead, because daily
             changes do not reverse the next day (lag-1 correlation 0.0 to 0.4), so it is shown at fixed h.

Scores: RPS skill against a usual year (1/3 each) and Brier skill against the record for Torpids; the mean
day-to-day change (|d lower| + |d higher|, percentage points) and changes of the page's sentence per 30 days.

    python stability_termly.py           # ~4 min on 2 cores
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

LAST = 2025
q = T.load_flow(ROOT)


def one_year(Y: int) -> list:
    model = T.Outlook(q, exclude=Y)
    usual = {p: np.percentile([float(q.loc[slice(*T.term_window(p, y))].mean()) for y in range(T.FIRST_YEAR, LAST + 1)
                               if y != Y], [100 / 3, 200 / 3]) for p in T.PERIODS}
    race = T.race_days('torpids', Y)
    raced_flow = float(q.reindex(race).mean())
    out = []
    for t0 in pd.date_range(f'{Y}-09-01', f'{Y + 1}-06-30'):
        try:
            ens = model.ensemble(t0)
        except ValueError:
            continue
        for p in T.PERIODS:
            w0, w1 = T.term_window(p, Y)
            if w1 <= t0 or w1 > ens.days[-1]:
                continue
            lo_t, hi_t = usual[p]
            ob = float(q.loc[w0:w1].mean())
            cat = 0 if ob < lo_t else (2 if ob > hi_t else 1)
            lead = max(0, (w0 - t0).days)
            sel = (ens.days >= max(w0, t0 + pd.Timedelta(days=1))) & (ens.days <= w1)
            past = q.loc[w0:t0].to_numpy() if w0 <= t0 else np.array([])
            for name, slow in (('now', False), ('blend', True)):
                samples, wts = T.mix(ens, lead, slow=slow)
                pm = (samples[:, sel].sum(1) + past.sum()) / (sel.sum() + len(past))
                out.append((name, Y, t0, p, lead, cat, float(np.sum(wts * (pm < lo_t))), float(np.sum(wts * (pm > hi_t)))))
        if race[0] > t0 and np.isfinite(raced_flow) and Y + 1 != 2021:
            tb = (race[0] - t0).days
            samples, wts = T.mix(ens, tb)
            pc = float(np.sum(wts * (T.race_flow(samples, ens.days, race) >= T.TORPIDS_FLOW)))
            out.append(('now', Y, t0, 'Torpids', tb, int(raced_flow >= T.TORPIDS_FLOW), pc, np.nan))
    return out


def sentence(lo, hi):
    s = np.full(len(lo), 'none', dtype=object)
    s[(lo >= 0.4) & (hi <= 0.2)], s[(hi >= 0.4) & (lo <= 0.2)] = 'lean lower', 'lean higher'
    s[lo >= 0.5], s[hi >= 0.5] = 'lower', 'higher'
    s[lo >= 0.6], s[hi >= 0.6] = 'very lower', 'very higher'
    return s


def smooth(df: pd.DataFrame, h: float) -> pd.DataFrame:
    """Exponential smoothing (half-life h days) of log(lower/usual) and log(higher/usual), which keeps the three
    odds summing to one; Torpids on the log-odds."""
    a = 1 - 0.5 ** (1 / h)
    df = df.sort_values('t0').copy()
    t = df.period != 'Torpids'
    lo, hi = df.p_lo.clip(.01, .98), df.p_hi.clip(.01, .98)
    us = (1 - lo - hi).clip(.01, 1)
    x = pd.DataFrame({'a': np.log(lo / us), 'b': np.log(hi / us)})
    x.loc[~t, 'a'] = np.log(df.p_lo[~t].clip(.005, .995) / (1 - df.p_lo[~t].clip(.005, .995)))
    x = x.groupby([df.Y, df.period]).transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
    e1, e2 = np.exp(x.a), np.exp(x.b)
    df.loc[t, 'p_lo'], df.loc[t, 'p_hi'] = (e1 / (1 + e1 + e2))[t], (e2 / (1 + e1 + e2))[t]
    df.loc[~t, 'p_lo'] = (1 / (1 + np.exp(-x.a)))[~t]
    return df


def score(df: pd.DataFrame) -> pd.Series:
    t = df[df.period != 'Torpids'].sort_values('t0').copy()
    o0, o1 = (t.cat == 0) * 1., (t.cat <= 1) * 1.
    t['rps'] = (t.p_lo - o0) ** 2 + ((1 - t.p_hi) - o1) ** 2
    t['rps_c'] = (1 / 3 - o0) ** 2 + (2 / 3 - o1) ** 2
    g = t.groupby(['Y', 'period'])
    t['jump'] = g.p_lo.diff().abs() + g.p_hi.diff().abs()
    t['say'] = sentence(t.p_lo.to_numpy(), t.p_hi.to_numpy())
    t['flip'] = (t.say != g.say.shift()) & g.say.shift().notna()
    t['lb'] = pd.cut(t.lead, [-1, 0, 30, 90, 1000], labels=['during', 'up to 1 month', '1-3 months', '3+ months'])
    out = {}
    for k, x in t.groupby('lb', observed=True):
        out[(k, 'skill %')] = 100 * (1 - x.rps.mean() / x.rps_c.mean())
        out[(k, 'change/day')] = 100 * x.jump.mean()
        out[(k, 'flips/30d')] = 30 * x.flip.mean()
    tp = df[df.period == 'Torpids'].sort_values('t0').copy()
    if len(tp):
        rec = tp.groupby('Y').cat.first().mean()
        tp['b'], tp['bc'] = (tp.p_lo - tp.cat) ** 2, (rec - tp.cat) ** 2
        tp['jump'] = tp.groupby('Y').p_lo.diff().abs()
        tp['lb'] = pd.cut(tp.lead, [-1, 35, 90, 1000], labels=['Torpids last 5 weeks', 'Torpids 5 weeks-3 months', 'Torpids 3+ months'])
        for k, x in tp.groupby('lb', observed=True):
            out[(k, 'skill %')] = 100 * (1 - x.b.mean() / x.bc.mean())
            out[(k, 'change/day')] = 100 * x.jump.mean()
    return pd.Series(out)


def main() -> None:
    with Pool(2) as pool:
        res = pool.map(one_year, range(T.FIRST_YEAR, LAST + 1))
    df = pd.DataFrame([r for rr in res for r in rr], columns=['run', 'Y', 't0', 'period', 'lead', 'cat', 'p_lo', 'p_hi'])
    now, blend = df[df.run == 'now'], df[(df.run == 'blend') | (df.period == 'Torpids')]
    table = pd.DataFrame({'now': score(now), 'blend (page)': score(blend),
                          **{f'now smoothed h={h}d': score(smooth(now, h)) for h in (2, 4, 7, 14)}}).T
    pd.set_option('display.width', 250)
    for m in ('skill %', 'change/day', 'flips/30d'):
        print(f'\n{m}')
        print(table.xs(m, axis=1, level=1).round(1).to_string())


if __name__ == '__main__':
    main()
