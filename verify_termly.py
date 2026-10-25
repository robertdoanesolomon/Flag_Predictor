"""
Verification of the termly outlook (src/flag_predictor/termly.py), leave-one-year-out, 1993/94 - 2025/26.

Each academic year Y is forecast from a record with Y left out entirely, from the 1st and 15th of every
month Sep - Jun and from the day before each period starts. Scored against flags rebuilt from Farmoor
flow, so the score is the outlook's alone (how well rebuilt flags match the real ones is checked
separately).

  days at each flag   CRPS of the forecast as shown (blended by lead, termly.BLEND) against an average
                      year (all past years equally likely), as a skill score: 100% perfect, 0% no better
  Torpids             Brier score of the chance of a Wed-Sat mean >= 44 m3/s, against an average year

Writes docs/termly/verification.json (read by the page) and a copy in figures/eval/.

    python verify_termly.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

LAST_YEAR = 2025
LEADS = ['during', 'up to 1 month', '1-2 months', '2-3 months', '3-4 months', '4+ months']
OUTCOME = {1994: 'day 1 raced, then flood', 2000: 'cancelled', 2002: 'cancelled', 2007: 'cancelled',
           2014: 'cancelled', 2018: 'partly lost to snow', 2020: 'cancelled', 2021: 'not held (COVID)',
           2024: 'cancelled', 2026: 'cancelled'}


def bucket(days_before: int) -> str:
    for limit, name in ((0, 'during'), (31, 'up to 1 month'), (61, '1-2 months'), (92, '2-3 months'),
                        (123, '3-4 months')):
        if days_before <= limit:
            return name
    return '4+ months'


def issue_dates(Y: int) -> list[tuple[pd.Timestamp, str]]:
    out = [(pd.Timestamp(Y if m >= 9 else Y + 1, m, d), 'calendar')
           for m in (9, 10, 11, 12, 1, 2, 3, 4, 5, 6) for d in (1, 15)]
    out += [(T.term_window(p, Y)[0] - pd.Timedelta(days=1), f'start:{p}') for p in T.PERIODS]
    out.append((T.race_days('torpids', Y)[0] - pd.Timedelta(days=7), 'start:torpids-1wk'))
    return out


def main() -> None:
    q = T.load_flow(ROOT)
    rows, torp = [], []
    for Y in range(T.FIRST_YEAR, LAST_YEAR + 1):
        model = T.Outlook(q, exclude=Y)
        for t0, kind in issue_dates(Y):
            try:
                ens = model.ensemble(t0)
            except ValueError:
                continue
            prev = float(q.get(t0, np.nan))
            for period in T.PERIODS:
                w0, w1 = T.term_window(period, Y)
                if w1 <= t0 or (kind.startswith('start:') and kind[6:] != period):
                    continue
                s = max(w0, t0 + pd.Timedelta(days=1))
                obs_q = q.loc[s - pd.Timedelta(days=1):w1].to_numpy()
                samples, wts = T.mix(ens, max(0, (w0 - t0).days))
                for loc in ('isis', 'godstow'):
                    obs = T.days_at_flags(obs_q[1:], pd.date_range(s, w1), s, w1, loc, prev_flow=obs_q[0])[0]
                    fx = T.days_at_flags(samples, ens.days, s, w1, loc, prev_flow=prev)
                    cl = fx[len(ens.members):]
                    for k, flag in enumerate(T.NAMES[loc]):
                        rows.append({'Y': Y, 'kind': kind, 'period': period, 'loc': loc, 'flag': flag,
                                     'lead': bucket((w0 - t0).days), 'obs': int(obs[k]),
                                     'crps': T.wcrps(fx[:, k], wts, obs[k]), 'crps_clim': T.crps(cl[:, k], obs[k]),
                                     'med': T.wquantile(fx[:, k], wts, 50), 'med_clim': float(np.median(cl[:, k])),
                                     'p10': T.wquantile(fx[:, k], wts, 10), 'p90': T.wquantile(fx[:, k], wts, 90)})
            race = T.race_days('torpids', Y)
            if race[0] > t0 and Y + 1 != 2021:
                obs = float(q.reindex(race).mean()) >= T.TORPIDS_FLOW
                samples, wts = T.mix(ens, (race[0] - t0).days)
                torp.append({'Y': Y, 'kind': kind, 'before': (race[0] - t0).days, 'obs': int(obs),
                             'p': float(np.sum(wts * (T.race_flow(samples, ens.days, race) >= T.TORPIDS_FLOW))),
                             'p_clim': float((T.race_flow(ens.climatology, ens.days, race) >= T.TORPIDS_FLOW).mean())})
    df, tp = pd.DataFrame(rows), pd.DataFrame(torp)

    def score(g):
        c = g.crps_clim.mean()
        return {'skill': round(float(1 - g.crps.mean() / c), 3) if c > 0 else None,
                'mae': round(float((g.med - g.obs).abs().mean()), 1),
                'mae_clim': round(float((g.med_clim - g.obs).abs().mean()), 1),
                'in_range': round(float(((g.obs >= g.p10) & (g.obs <= g.p90)).mean()), 2), 'n': int(len(g))}

    skill = [{'loc': loc, 'period': p, 'flag': f, 'lead': l} | score(g)
             for (loc, p, f, l), g in df[df.kind == 'calendar'].groupby(['loc', 'period', 'flag', 'lead'])]
    at_start = {}
    for (loc, p, f), g in df[df.kind.str.startswith('start:')].groupby(['loc', 'period', 'flag']):
        at_start.setdefault(loc, {}).setdefault(p, {})[f] = score(g)

    def brier(g):
        b, bc = float(((g.p - g.obs) ** 2).mean()), float(((g.p_clim - g.obs) ** 2).mean())
        return {'skill': round(1 - b / bc, 3), 'n': int(len(g))}
    torp_skill = {'week before': brier(tp[tp.kind == 'start:torpids-1wk'])}
    cal = tp[tp.kind == 'calendar'].copy()
    cal['lead'] = cal.before.map(lambda d: 'last week' if d <= 8 else bucket(d))
    torp_skill |= {l: brier(g) for l, g in cal.groupby('lead')}

    agree = {}
    for loc in ('isis', 'godstow'):
        f = T.daily_flags(ROOT, loc)
        r = T.reconstructed_flags(q, loc).reindex(f.index)
        agree[loc] = round(float((f == r)[r.notna()].mean()), 3)
    record = []
    for Y in range(T.FIRST_YEAR - 1, LAST_YEAR + 1):
        d = T.race_days('torpids', Y)
        record.append({'year': Y + 1, 'flow': round(float(q.reindex(d).mean()), 1), 'outcome': OUTCOME.get(Y + 1, 'raced')})

    report = {'generated': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
              'years': f'{T.FIRST_YEAR}/{str(T.FIRST_YEAR + 1)[2:]}-{LAST_YEAR}/{str(LAST_YEAR + 1)[2:]}',
              'skill': skill, 'at_start': at_start, 'torpids_skill': torp_skill,
              'torpids_climatology': round(float(tp.groupby('Y').obs.first().mean()), 3),
              'torpids_record': record, 'flag_agreement': agree}
    for out in (ROOT / 'docs' / 'termly' / 'verification.json', ROOT / 'figures' / 'eval' / 'termly_verification.json'):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))

    sk = pd.DataFrame(skill)
    for loc in ('isis', 'godstow'):
        t = sk[sk['loc'] == loc].pivot_table(index=['period', 'flag'], columns='lead', values='skill', sort=False)
        print(f'\n=== {loc}: skill against an average year, by lead (rebuilt flags agree with real on {agree[loc]:.0%} of days)')
        print((100 * t[[c for c in LEADS if c in t.columns]]).round(0).to_string())
        print('from the day before each period: ' + '; '.join(
            f"{p} red {100 * v['red']['skill']:.0f}% ({v['red']['mae']} vs {v['red']['mae_clim']} d)"
            for p, v in at_start[loc].items() if v['red']['skill'] is not None))
    print('\nTorpids: ' + ', '.join(f"{k} {100 * v['skill']:.0f}%" for k, v in torp_skill.items()))


if __name__ == '__main__':
    main()
