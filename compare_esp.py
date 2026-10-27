"""Hydrological outlook (catchment model run through past years' weather, src/flag_predictor/esp.py) against the
current statistical outlook (termly.Outlook), leave-one-year-out 1993/94-2025/26, same issue dates and scores as
verify_termly.py, plus the lower/usual/higher odds the page shows (ranked probability skill).

    python compare_esp.py [--tau 60] [--last 2025]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor import esp, termly as T  # noqa: E402
from verify_termly import bucket, issue_dates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tau', type=float, default=60.0)
ap.add_argument('--last', type=int, default=2025)
ap.add_argument('--out', default='figures/eval/esp_compare.csv')
a = ap.parse_args()

q = T.load_flow(ROOT)
VARIANTS = ('stat', 'esp', 'pool')


def weights(name, ens_stat, ens_esp, lead):
    if name == 'stat':
        return T.mix(ens_stat, lead)
    if name == 'pool':          # half the weight on each outlook
        s1, w1 = T.mix(ens_stat, lead)
        n = len(ens_esp.members)
        return np.concatenate([s1, ens_esp.members]), np.r_[0.5 * w1, np.full(n, 0.5 / n)]
    n = len(ens_esp.members)
    return np.concatenate([ens_esp.members, ens_esp.climatology]), np.r_[np.full(n, 1 / n), np.zeros(n)]


def rps(p, cat):
    o = np.zeros(3)
    o[cat] = 1
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2))


rows, torp = [], []
for Y in range(T.FIRST_YEAR, a.last + 1):
    stat = T.Outlook(q, exclude=Y)
    hyd = esp.ESPOutlook(q, ROOT, exclude=Y, tau=a.tau)
    for t0, kind in issue_dates(Y):
        try:
            es = stat.ensemble(t0)
            eh = hyd.ensemble(t0)
        except ValueError:
            continue
        prev = float(q.get(t0, np.nan))
        for period in T.PERIODS:
            w0, w1 = T.term_window(period, Y)
            if w1 <= t0 or (kind.startswith('start:') and kind[6:] != period):
                continue
            s = max(w0, t0 + pd.Timedelta(days=1))
            obs_q = q.loc[s - pd.Timedelta(days=1):w1].to_numpy()
            lead = max(0, (w0 - t0).days)
            usual = [float(q.loc[slice(*T.term_window(period, y))].mean()) for y in range(T.FIRST_YEAR, a.last + 1) if y != Y]
            lo_t, hi_t = np.percentile(usual, [100 / 3, 200 / 3])
            obs_pm = float(q.loc[w0:w1].mean())
            cat = 0 if obs_pm < lo_t else (2 if obs_pm > hi_t else 1)
            sel = (es.days >= s) & (es.days <= w1)
            past = q.loc[w0:t0].to_numpy() if w0 <= t0 else np.array([])
            rec = {'Y': Y, 'kind': kind, 'period': period, 'lead': bucket((w0 - t0).days), 'cat': cat,
                   'rps_clim': rps(np.full(3, 1 / 3), cat)}
            for v in VARIANTS:
                samples, wts = weights(v, es, eh, lead)
                pm = (samples[:, sel].sum(1) + past.sum()) / (sel.sum() + len(past))
                p = np.array([np.sum(wts * (pm < lo_t)), 0, np.sum(wts * (pm > hi_t))])
                p[1] = 1 - p[0] - p[2]
                rec[f'rps_{v}'] = rps(p, cat)
                for loc in ('isis',):
                    ob = T.days_at_flags(obs_q[1:], pd.date_range(s, w1), s, w1, loc, prev_flow=obs_q[0])[0]
                    fx = T.days_at_flags(samples, es.days, s, w1, loc, prev_flow=prev)
                    rec[f'crps_{v}'] = float(np.mean([T.wcrps(fx[:, k], wts, ob[k]) for k in range(fx.shape[1])]))
                    if v == 'stat':
                        cl = fx[len(es.members):]
                        rec['crps_clim'] = float(np.mean([T.crps(cl[:, k], ob[k]) for k in range(fx.shape[1])]))
            rows.append(rec)
        race = T.race_days('torpids', Y)
        if race[0] > t0 and Y + 1 != 2021:
            ob = float(float(q.reindex(race).mean()) >= T.TORPIDS_FLOW)
            before = (race[0] - t0).days
            rec = {'Y': Y, 'kind': kind, 'lead': 'last week' if before <= 8 else bucket(before), 'obs': ob}
            for v in VARIANTS:
                samples, wts = weights(v, es, eh, before)
                rec[f'p_{v}'] = float(np.sum(wts * (T.race_flow(samples, es.days, race) >= T.TORPIDS_FLOW)))
            rec['p_clim'] = float((T.race_flow(es.climatology, es.days, race) >= T.TORPIDS_FLOW).mean())
            torp.append(rec)
    print(Y, flush=True)

df, tp = pd.DataFrame(rows), pd.DataFrame(torp)
Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / a.out, index=False)
tp.to_csv(ROOT / a.out.replace('.csv', '_torpids.csv'), index=False)
cal = df[df.kind == 'calendar']
order = ['during', 'up to 1 month', '1-2 months', '2-3 months', '3-4 months', '4+ months']
print('\nSkill against a usual year (%): lower/usual/higher odds (RPS) | Isis flag days (CRPS)')
tab = pd.DataFrame({f'{m}_{v}': cal.groupby('lead').apply(lambda g: round(100 * (1 - g[f'{m}_{v}'].mean() / g[f'{m}_clim'].mean()), 1))
                    for m in ('rps', 'crps') for v in VARIANTS}).reindex(order)
tab['n'] = cal.groupby('lead').size()
print(tab.to_string())
print('\nTorpids (Brier skill %, against how often it was lost):')
tp_c = tp[tp.kind == 'calendar']
print(pd.DataFrame({v: tp_c.groupby('lead').apply(lambda g: round(100 * (1 - ((g[f'p_{v}'] - g.obs) ** 2).mean() / ((g.p_clim - g.obs) ** 2).mean()), 1))
                    for v in VARIANTS}).to_string())
