"""
Verification of the termly outlook (src/flag_predictor/termly.py), leave-one-year-out, 1993/94 - 2025/26.

For every academic year Y, the record with year Y left out entirely (seasonal distribution, regression
and ensemble members) forecasts year Y from the 1st and 15th of every month Sep - Jun and from the day
before each term's 0th week. Scored against the flags rebuilt from Farmoor flow (the same lines as the
forecast), so the score is the outlook's skill alone; how well the rebuilt flags match the OURCs flags
is checked separately (2019 onwards).

  days at each flag   CRPS of the forecast as shown (the conditioned members blended with climatology by
                      lead, termly.BLEND; the blend weights were picked on these same hindcasts) against
                      climatology (every past year equally likely, whatever the river now), as a skill
                      score (100% perfect, 0 no better than an average year, negative worse); the
                      typical error of the middle forecast; how often the 10-90% range held the outcome
  Torpids             Brier score of the chance of a cancellation-level week (Wed-Sat mean >= 44 m3/s)
                      against climatology

Writes figures/eval/termly_verification.json and docs/termly/verification.json (for the page).

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

LAST_YEAR = 2025                   # last academic year with every term complete
LEAD_BUCKETS = [(-999, 0, 'during the term'), (0, 31, 'up to 1 month before'), (31, 61, '1-2 months before'),
                (61, 92, '2-3 months before'), (92, 123, '3-4 months before'), (123, 999, '4+ months before')]
OUTCOME = {1994: 'day 1 raced, then flood', 2000: 'cancelled', 2002: 'cancelled', 2007: 'cancelled',
           2014: 'cancelled', 2018: 'partly lost to snow', 2020: 'cancelled', 2021: 'not held (COVID)',
           2024: 'cancelled', 2026: 'cancelled'}


def issue_dates(Y: int) -> list[tuple[pd.Timestamp, str]]:
    out = []
    for m in (9, 10, 11, 12, 1, 2, 3, 4, 5, 6):
        for d in (1, 15):
            out.append((pd.Timestamp(Y if m >= 9 else Y + 1, m, d), 'calendar'))
    for term in T.TERMS:
        w0, _ = T.term_window(term, Y)
        out.append((w0 - pd.Timedelta(days=1), f'start:{term}'))
    out.append((T.race_days('torpids', Y)[0] - pd.Timedelta(days=7), 'start:torpids-1wk'))
    return out


def bucket(days_before: int) -> str:
    for lo, hi, name in LEAD_BUCKETS:
        if lo < days_before <= hi or (lo == -999 and days_before <= 0):
            return name
    return LEAD_BUCKETS[-1][2]


def reconstruction_check(q: pd.Series) -> dict:
    out = {}
    for loc in ('isis', 'godstow'):
        f = T.daily_flags(ROOT, loc)
        rec = T.reconstructed_flags(q, loc)
        both = pd.DataFrame({'f': f, 'r': rec}).dropna()
        both = both[both.index >= '2019-01-01']
        rows = []
        for Y in range(2018, LAST_YEAR + 2):
            for term in T.TERMS:
                w0, w1 = T.term_window(term, Y)
                if w1 > q.dropna().index.max():
                    continue
                days = pd.date_range(w0, w1)
                a = f.reindex(days)
                if a.notna().mean() < 0.8:
                    continue
                r = rec.reindex(days)
                rows.append({'term': f'{term} {w0.year}', 'missing': int(a.isna().sum()),
                             'actual': {n: int((a == n).sum()) for n in T.NAMES[loc]},
                             'rebuilt': {n: int((r == n).sum()) for n in T.NAMES[loc]}})
        out[loc] = {'daily_agreement': round(float((both.f == both.r).mean()), 3), 'days': int(len(both)),
                    'terms': rows}
    return out


def torpids_record(q: pd.Series) -> list[dict]:
    rows = []
    for Y in range(T.FIRST_YEAR - 1, LAST_YEAR + 1):
        d = T.race_days('torpids', Y)
        if d[-1] > q.dropna().index.max():
            continue
        rows.append({'year': Y + 1, 'first_day': f'{d[0]:%Y-%m-%d}', 'flow': round(float(q.reindex(d).mean()), 1),
                     'outcome': OUTCOME.get(Y + 1, 'raced')})
    return rows


def main() -> None:
    q = T.load_flow(ROOT)
    rows, torp, starts = [], [], []
    for Y in range(T.FIRST_YEAR, LAST_YEAR + 1):
        model = T.Outlook(q, exclude=Y)
        for t0, kind in issue_dates(Y):
            try:
                ens = model.ensemble(t0)
            except ValueError:
                continue
            prev = float(q.get(t0, np.nan))
            for term in T.TERMS:
                w0, w1 = T.term_window(term, Y)
                if w1 <= t0:
                    continue
                s = max(w0, t0 + pd.Timedelta(days=1))
                obs_q = q.loc[s - pd.Timedelta(days=1):w1].to_numpy()
                samples, wts = T.mix(ens, max(0, (w0 - t0).days))
                for loc in ('isis', 'godstow'):
                    obs = T.days_at_flags(obs_q[1:], pd.date_range(s, w1), s, w1, loc, prev_flow=obs_q[0])[0]
                    fx = T.days_at_flags(samples, ens.days, s, w1, loc, prev_flow=prev)
                    cl = fx[len(ens.members):]
                    for k, name in enumerate(T.NAMES[loc]):
                        rows.append({'Y': Y, 't0': t0, 'kind': kind, 'term': term, 'loc': loc, 'flag': name,
                                     'before': (w0 - t0).days, 'ndays': (w1 - s).days + 1, 'obs': int(obs[k]),
                                     'crps': T.wcrps(fx[:, k], wts, obs[k]), 'crps_clim': T.crps(cl[:, k], obs[k]),
                                     'med': T.wquantile(fx[:, k], wts, 50), 'med_clim': float(np.median(cl[:, k])),
                                     'p10': T.wquantile(fx[:, k], wts, 10), 'p90': T.wquantile(fx[:, k], wts, 90),
                                     'p10_clim': float(np.percentile(cl[:, k], 10)),
                                     'p90_clim': float(np.percentile(cl[:, k], 90))})
            race = T.race_days('torpids', Y)
            if race[0] > t0 and Y + 1 != 2021:
                obs = float(q.reindex(race).mean()) >= T.TORPIDS_FLOW
                samples, wts = T.mix(ens, (race[0] - t0).days)
                p = float(np.sum(wts * (T.race_flow(samples, ens.days, race) >= T.TORPIDS_FLOW)))
                pc = float((T.race_flow(ens.climatology, ens.days, race) >= T.TORPIDS_FLOW).mean())
                torp.append({'Y': Y, 't0': t0, 'kind': kind, 'before': (race[0] - t0).days, 'obs': int(obs),
                             'p': p, 'p_clim': pc})
    df = pd.DataFrame(rows)
    tp = pd.DataFrame(torp)
    df['bucket'] = df.before.map(bucket)

    # --- skill by term, flag and how far ahead
    skill = []
    for (loc, term, flag, b), g in df[df.kind == 'calendar'].groupby(['loc', 'term', 'flag', 'bucket']):
        skill.append({'loc': loc, 'term': term, 'flag': flag, 'lead': b, 'n': int(len(g)),
                      'years': int(g.Y.nunique()),
                      'skill': round(float(1 - g.crps.mean() / g.crps_clim.mean()), 3) if g.crps_clim.mean() > 0 else None,
                      'mae': round(float((g.med - g.obs).abs().mean()), 1),
                      'mae_clim': round(float((g.med_clim - g.obs).abs().mean()), 1),
                      'in_range': round(float(((g.obs >= g.p10) & (g.obs <= g.p90)).mean()), 2),
                      'mean_obs': round(float(g.obs.mean()), 1)})
    order = {n: i for i, (_, _, n) in enumerate(LEAD_BUCKETS)}
    skill.sort(key=lambda r: (r['loc'], T.TERMS.index(r['term']), r['flag'], order[r['lead']]))

    # --- what it said the day before each term's 0th week, every year
    st = df[df.kind.str.startswith('start:') & (df.kind.str[6:] == df.term)]
    term_starts = []
    for (Y, term, loc), g in st.groupby(['Y', 'term', 'loc']):
        term_starts.append({'year': int(T.term_window(term, Y)[0].year), 'acad': int(Y), 'term': term, 'loc': loc,
                            'flags': {r.flag: {'obs': r.obs, 'med': round(r.med, 1), 'p10': round(r.p10, 1),
                                               'p90': round(r.p90, 1), 'clim_med': round(r.med_clim, 1)}
                                      for r in g.itertuples()}})
    st_skill = {}
    for (loc, term, flag), g in st.groupby(['loc', 'term', 'flag']):
        st_skill.setdefault(loc, {}).setdefault(term, {})[flag] = {
            'skill': round(float(1 - g.crps.mean() / g.crps_clim.mean()), 3) if g.crps_clim.mean() > 0 else None,
            'mae': round(float((g.med - g.obs).abs().mean()), 1), 'mae_clim': round(float((g.med_clim - g.obs).abs().mean()), 1),
            'in_range': round(float(((g.obs >= g.p10) & (g.obs <= g.p90)).mean()), 2), 'n': int(len(g))}

    # --- Torpids
    torp_skill = []
    tp['bucket'] = pd.cut(tp.before, [0, 8, 31, 61, 92, 123, 999],
                          labels=['last week', 'up to 1 month before', '1-2 months before', '2-3 months before',
                                  '3-4 months before', '4+ months before'])
    for b, g in tp[tp.kind != 'start:torpids-1wk'].groupby('bucket', observed=True):
        bs, bc = float(((g.p - g.obs) ** 2).mean()), float(((g.p_clim - g.obs) ** 2).mean())
        torp_skill.append({'lead': str(b), 'n': int(len(g)), 'brier': round(bs, 3), 'brier_clim': round(bc, 3),
                           'skill': round(1 - bs / bc, 3)})
    wk = tp[tp.kind == 'start:torpids-1wk']
    torp_skill.append({'lead': 'a week before (Wednesday of 6th week)', 'n': int(len(wk)),
                       'brier': round(float(((wk.p - wk.obs) ** 2).mean()), 3),
                       'brier_clim': round(float(((wk.p_clim - wk.obs) ** 2).mean()), 3),
                       'skill': round(1 - float(((wk.p - wk.obs) ** 2).mean()) / float(((wk.p_clim - wk.obs) ** 2).mean()), 3)})
    by_year = []
    for Y, g in tp.groupby('Y'):
        cell = {'year': int(Y + 1), 'obs': int(g.obs.iloc[0])}
        for label, (m, d) in {'1 Nov': (11, 1), '1 Dec': (12, 1), '1 Jan': (1, 1), '1 Feb': (2, 1), '15 Feb': (2, 15)}.items():
            r = g[g.t0 == pd.Timestamp(Y if m >= 9 else Y + 1, m, d)]
            cell[label] = round(float(r.p.iloc[0]), 2) if len(r) else None
        r = g[g.kind == 'start:torpids-1wk']
        cell['week before'] = round(float(r.p.iloc[0]), 2) if len(r) else None
        by_year.append(cell)

    report = {'generated': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
              'years': f'{T.FIRST_YEAR}/{str(T.FIRST_YEAR + 1)[2:]} - {LAST_YEAR}/{str(LAST_YEAR + 1)[2:]}',
              'n_years': LAST_YEAR - T.FIRST_YEAR + 1, 'lines': T.LINES, 'torpids_flow': T.TORPIDS_FLOW,
              'skill': skill, 'term_start_skill': st_skill, 'term_starts': term_starts,
              'torpids_skill': torp_skill, 'torpids_by_year': by_year,
              'torpids_climatology': round(float(tp.groupby('Y').obs.first().mean()), 3),
              'torpids_record': torpids_record(q), 'reconstruction': reconstruction_check(q)}
    for out in (ROOT / 'figures' / 'eval' / 'termly_verification.json', ROOT / 'docs' / 'termly' / 'verification.json'):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str))
    # --- print
    print(f"years {report['years']}; Torpids at cancellation level in {report['torpids_climatology']:.0%} of years")
    sk = pd.DataFrame(skill)
    for loc in ('isis', 'godstow'):
        t = sk[sk['loc'] == loc].pivot_table(index=['term', 'flag'], columns='lead', values='skill', sort=False)
        print(f'\n=== {loc}: skill against an average year (CRPS), by how far ahead')
        print((100 * t[[n for _, _, n in LEAD_BUCKETS if n in t.columns]]).round(0).to_string())
        print(f'--- from the day before 0th week: skill / typical error (average year) / in 10-90% range')
        for term in T.TERMS:
            print('  ' + term + ': ' + ', '.join(
                f"{f} {100 * v['skill']:.0f}% / {v['mae']} ({v['mae_clim']}) d / {v['in_range']:.0%}"
                for f, v in st_skill[loc][term].items() if v['skill'] is not None))
    print('\n=== Torpids: Brier skill against climatology')
    for r in torp_skill:
        print(f"  {r['lead']:40s} {100 * r['skill']:5.0f}%  (n={r['n']})")
    print(pd.DataFrame(by_year).to_string(index=False))


if __name__ == '__main__':
    main()
