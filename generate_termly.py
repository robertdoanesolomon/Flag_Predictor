"""
The termly outlook for flagpredictor.com/termly: writes docs/termly/termly.json.

Updates Farmoor's daily flow record (data/termly/farmoor_daily.csv) from the EA feeds, runs the
ensemble from the latest complete day (src/flag_predictor/termly.py), blends it with an average year
by lead (termly.BLEND), and summarises it for each term
still to come within the next ~9 months (days at each flag, so far this term + the rest), week by week,
and for Torpids. The skill figures shown next to each number come from docs/termly/verification.json
(verify_termly.py).

    python generate_termly.py              # fetch the latest flow, then forecast
    python generate_termly.py --no-fetch   # use the stored record
    python generate_termly.py --date 2025-11-01 --no-fetch   # as if issued on that day (for checks)
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flag_predictor import termly as T  # noqa: E402

OUT = ROOT / 'docs' / 'termly' / 'termly.json'
VERIFY = ROOT / 'docs' / 'termly' / 'verification.json'
WEEKS = 36
N_ANALOGS = 7


def _q(a, p) -> float:
    return round(float(np.percentile(a, p)), 1)


def _wq(a, w, p) -> float:
    return round(T.wquantile(a, w, p), 1)


def lead_bucket(days_before: int) -> str:
    from verify_termly import bucket
    return bucket(days_before)


def skill_lookup(ver: dict, loc: str, period: str, flag: str, lead: str):
    for r in ver.get('skill', []):
        if r['loc'] == loc and r['period'] == period and r['flag'] == flag and r['lead'] == lead:
            return r['skill']
    return None


def term_summary(q: pd.Series, ens: T.Ensemble, term: str, acad: int, loc: str, actual: pd.Series,
                 rebuilt_all: pd.Series, ver: dict, analogs: list) -> dict:
    t0 = ens.t0
    w0, w1 = T.term_window(term, acad)
    names = T.NAMES[loc]
    # so far: the OURCs flag, or the differential's band when no flag flies, else the flag rebuilt from flow
    so_far = np.zeros(len(names), int)
    if w0 <= t0:
        days = pd.date_range(w0, min(t0, w1))
        a = actual.reindex(days)
        r = rebuilt_all.reindex(days)
        f = a.where(a.notna(), r)
        so_far = np.array([(f == n).sum() for n in names])
    samples, wts = T.mix(ens, max(0, (w0 - t0).days))
    nc = len(ens.members)
    if w1 > t0:
        s = max(w0, t0 + pd.Timedelta(days=1))
        prev = float(q.get(s - pd.Timedelta(days=1), q.dropna().iloc[-1]))
        rest = T.days_at_flags(samples, ens.days, s, w1, loc, prev_flow=prev)
    else:
        rest = np.zeros((len(samples), len(names)), int)
    tot = so_far + rest
    tot_c = tot[nc:]
    # an average year: the whole term, rebuilt from flow, every year on record
    normal = []
    for y in range(T.FIRST_YEAR, acad):
        a0, a1 = T.term_window(term, y)
        r = rebuilt_all.reindex(pd.date_range(a0, a1))
        if r.notna().all():
            normal.append([(r == n).sum() for n in names])
    normal = np.array(normal)
    lead = lead_bucket((w0 - t0).days)
    flags = {}
    for k, n in enumerate(names):
        flags[n] = {'so_far': int(so_far[k]), 'median': _wq(tot[:, k], wts, 50), 'p10': _wq(tot[:, k], wts, 10),
                    'p90': _wq(tot[:, k], wts, 90), 'mean': round(float(np.sum(wts * tot[:, k])), 1),
                    'clim_median': _q(tot_c[:, k], 50), 'clim_p10': _q(tot_c[:, k], 10), 'clim_p90': _q(tot_c[:, k], 90),
                    'normal_mean': round(float(normal[:, k].mean()), 1),
                    'normal_median': _q(normal[:, k], 50), 'normal_p10': _q(normal[:, k], 10),
                    'normal_p90': _q(normal[:, k], 90),
                    'skill': skill_lookup(ver, loc, term, n, lead)}
    worst = tot[:, -1]
    # higher or lower than usual: mean flow over the period against the past years' terciles
    usual = []
    for y in range(T.FIRST_YEAR, acad):
        a0, a1 = T.term_window(term, y)
        usual.append(float(q.loc[a0:a1].mean()))
    lo_t, hi_t = np.percentile(usual, [100 / 3, 200 / 3])
    if w1 > t0:
        s_ = max(w0, t0 + pd.Timedelta(days=1))
        sel = (ens.days >= s_) & (ens.days <= w1)
        past = q.loc[w0:t0].to_numpy() if w0 <= t0 else np.array([])
        pm = (samples[:, sel].sum(1) + past.sum()) / (sel.sum() + len(past))
    else:
        pm = np.full(len(samples), float(q.loc[w0:w1].mean()))
    tercile = {'lower': round(float(np.sum(wts * (pm < lo_t))), 2), 'higher': round(float(np.sum(wts * (pm > hi_t))), 2)}
    tercile['usual'] = round(1 - tercile['lower'] - tercile['higher'], 2)
    return {'term': term, 'year': int(w0.year), 'label': f'{term} {w0.year}', 'start': f'{w0:%Y-%m-%d}',
            'end': f'{w1:%Y-%m-%d}', 'full_term_start': f'{T.term_start(term, acad):%Y-%m-%d}',
            'days': int((w1 - w0).days + 1), 'days_done': int(max(0, min((t0 - w0).days + 1, (w1 - w0).days + 1))),
            'tercile': tercile, 'lead': lead, 'trust': T.blend_weight(max(0, (w0 - t0).days)), 'flags': flags,
            'chance_red_week': round(float(np.sum(wts * (worst >= 7))), 2),
            'chance_red_week_normal': round(float((normal[:, -1] >= 7).mean()), 2),
            'analogs': analog_terms(rebuilt_all, term, acad, analogs, loc)}


def analog_terms(rebuilt_all: pd.Series, term: str, acad: int, analogs: list, loc: str) -> list:
    """How this term went in the past years whose last 90 days were most like now (flags rebuilt from flow)."""
    acad0 = T.academic_year(pd.Timestamp(analogs[0]['issued'])) if analogs else acad
    out = []
    for a in analogs:
        y = a['acad'] + (acad - acad0)
        w0, w1 = T.term_window(term, y)
        r = rebuilt_all.reindex(pd.date_range(w0, w1))
        if r.notna().all():
            out.append({'label': f'{term} {w0.year}', 'pct_normal_90d': a['pct_normal_90d'],
                        'days': {n: int((r == n).sum()) for n in T.NAMES[loc]}})
    return sorted(out, key=lambda x: x['days'][T.NAMES[loc][-1]])


def weekly(ens: T.Ensemble, loc: str, prev: float, weeks: int = WEEKS) -> dict:
    nc = len(ens.members)
    allm = np.concatenate([ens.members, ens.climatology])
    r = T.flag_ranks(np.concatenate([np.full((len(allm), 1), prev), allm], axis=1), loc)[:, 1:]
    start = ens.t0 + pd.Timedelta(days=(6 - ens.t0.dayofweek) % 7 or 7)     # the next Sunday
    out = {'week': [], 'chance': {n: [] for n in T.NAMES[loc]}, 'chance_clim': {n: [] for n in T.NAMES[loc]}}
    for w in range(weeks):
        a = start + pd.Timedelta(days=7 * w)
        sel = (ens.days >= a) & (ens.days < a + pd.Timedelta(days=7))
        if sel.sum() < 7:
            break
        _, wts = T.mix(ens, (a - ens.t0).days)
        out['week'].append(f'{a:%Y-%m-%d}')
        for k, n in enumerate(T.NAMES[loc]):
            frac = (r[:, sel] == k).mean(1)
            out['chance'][n].append(round(float(np.sum(wts * frac)), 3))
            out['chance_clim'][n].append(round(float(frac[nc:].mean()), 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--date', default=None)
    args = ap.parse_args()
    q = T.load_flow(ROOT) if args.no_fetch else T.update_flow(ROOT)
    t0 = pd.Timestamp(args.date) if args.date else q.dropna().index.max()
    qk = q[q.index <= t0]
    ver = json.loads(VERIFY.read_text()) if VERIFY.exists() else {}
    model = T.Outlook(qk)
    ens = model.ensemble(t0)
    # weather forecasts: weight the member years by how well their rain fits EC46's next three weeks
    fetched, weighting = None, None
    try:
        fetched = T.ec46_fetch()
        ens.year_w, weighting = T.forecast_year_weights(ens, ROOT, *fetched)
        print(f"  EC46 weighting: {weighting['effective_years']} effective years; most weight to {weighting['top_years']}")
        log = ROOT / 'data' / 'termly' / 'ec46_log.csv'      # kept to measure EC46's own skill here later
        rows = pd.DataFrame([{'issued': f'{t0:%Y-%m-%d}', 'run_first_day': f'{fetched[0][0]:%Y-%m-%d}', **w}
                             for w in weighting['weeks']])
        old_log = pd.read_csv(log) if log.exists() else pd.DataFrame()
        pd.concat([old_log, rows]).drop_duplicates(['issued', 'from'], keep='last').to_csv(log, index=False)
    except Exception as e:              # noqa: BLE001
        print(f'  EC46 weighting off: {e}')
    acad = T.academic_year(t0)
    actual = {loc: T.daily_flags(ROOT, loc) for loc in ('isis', 'godstow')}
    actual = {k: v[v.index <= t0] for k, v in actual.items()}
    rebuilt = {loc: T.reconstructed_flags(qk, loc) for loc in ('isis', 'godstow')}

    terms = []
    for y in (acad, acad + 1):
        for term in T.PERIODS:
            w0, w1 = T.term_window(term, y)
            if w1 > t0 and w1 <= ens.days[-1]:
                terms.append((term, y))
    terms = terms[:5]
    # the river now: mean flow over the last 7 and 90 days against the same days in every past year
    def mean_to(t, n):
        return float(qk.loc[t - pd.Timedelta(days=n - 1):t].mean())
    past7, past14, past90 = [], [], []
    for y in range(T.FIRST_YEAR, acad):
        t = T._same_date(t0, acad - y)
        past7.append(mean_to(t, 7))
        past90.append(mean_to(t, 90))
        past14.append(mean_to(t, 14))
    now7, now14, now90 = mean_to(t0, 7), mean_to(t0, 14), mean_to(t0, 90)
    med90 = float(np.median(past90))
    near = sorted(range(len(past90)), key=lambda i: abs(np.log(past90[i] / now90)))[:N_ANALOGS]
    analogs = [{'acad': T.FIRST_YEAR + i, 'issued': f'{t0:%Y-%m-%d}', 'pct_normal_90d': round(100 * past90[i] / med90)}
               for i in near]
    state = {'date': f'{t0:%Y-%m-%d}', 'flow': round(float(qk.iloc[-1]), 2), 'flow_7d': round(now7, 2),
             'normal': round(float(model.normal_on(pd.DatetimeIndex([t0]))[0]), 1),
             'pct_normal_7d': round(100 * now7 / float(np.median(past7)), 0),
             'pct_normal_90d': round(100 * now90 / float(np.median(past90)), 0),
             'wetter_years_90d': int(sum(v > now90 for v in past90)), 'n_years': len(past90),
             'pct_normal_14d': round(100 * now14 / float(np.median(past14)), 0),
             'wetter_years_14d': int(sum(v > now14 for v in past14)),
             'pct_normal_7d': round(100 * now7 / float(np.median(past7)), 0),
             'wetter_years_7d': int(sum(v > now7 for v in past7)),
             'score_7d': ens.state['7d'], 'score_90d': ens.state['90d']}

    prev = float(qk.dropna().iloc[-1])
    out_terms = {loc: [term_summary(qk, ens, term, y, loc, actual[loc], rebuilt[loc], ver, analogs) for term, y in terms]
                 for loc in ('isis', 'godstow')}
    if ens.year_w is not None:          # the same without the weather forecasts, to show what they changed
        w_keep, ens.year_w = ens.year_w, None
        for loc in out_terms:
            for t, (term, y) in zip(out_terms[loc], terms):
                t['tercile_river_only'] = term_summary(qk, ens, term, y, loc, actual[loc], rebuilt[loc], ver, analogs)['tercile']
        ens.year_w = w_keep

    # Torpids: the next one
    ty = acad if T.race_days('torpids', acad)[-1] > t0 else acad + 1
    race = T.race_days('torpids', ty)
    torpids = None
    if race[-1] <= ens.days[-1] and race[0] > t0:
        tb = (race[0] - t0).days
        samples, wts = T.mix(ens, tb)
        rf = T.race_flow(samples, ens.days, race)
        rfc = rf[len(ens.members):]
        lead = 'last week' if tb <= 8 else lead_bucket(tb)
        sk = (ver.get('torpids_skill') or {}).get(lead, {}).get('skill')
        torpids = {'year': int(race[0].year), 'first_day': f'{race[0]:%Y-%m-%d}', 'last_day': f'{race[-1]:%Y-%m-%d}',
                   'days_away': tb, 'chance': round(float(np.sum(wts * (rf >= T.TORPIDS_FLOW))), 2),
                   'chance_state_only': round(float((rf[:len(ens.members)] >= T.TORPIDS_FLOW).mean()), 2),
                   'chance_clim': round(float((rfc >= T.TORPIDS_FLOW).mean()), 2),
                   'chance_record': ver.get('torpids_climatology'), 'trust': T.blend_weight(tb),
                   'chance_river_only': round(float(np.sum(T.mix(T.Ensemble(**{**ens.__dict__, 'year_w': None}), tb)[1]
                                                    * (rf >= T.TORPIDS_FLOW))), 2),
                   'flow_median': _wq(rf, wts, 50), 'flow_p10': _wq(rf, wts, 10), 'flow_p90': _wq(rf, wts, 90),
                   'lead': lead, 'skill': sk}

    # ECMWF's seasonal run changes once a month: if the fetch fails, keep the last one
    seasonal = []
    for attempt in range(3):
        try:
            seasonal = T.seasonal_rain(ROOT)
            break
        except Exception as e:          # noqa: BLE001 - the outlook does not depend on it
            print(f'  seasonal forecast, attempt {attempt + 1}: {e}')
            time.sleep(10 * (attempt + 1))
    try:
        ec46 = T.ec46_rain(ROOT, fetched=fetched) if fetched is not None else T.ec46_rain(ROOT)
    except Exception as e:              # noqa: BLE001
        print(f'  EC46 unavailable: {e}')
        ec46 = json.loads(OUT.read_text()).get('ec46', []) if OUT.exists() else []
    if not seasonal and OUT.exists():
        seasonal = json.loads(OUT.read_text()).get('seasonal') or []
        print(f'  kept the previous seasonal forecast ({len(seasonal)} months)')
    report = {'generated': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC'), 'issued': f'{t0:%Y-%m-%d}',
              'members': len(ens.members), 'member_years': ens.years, 'state': state,
              'lines': T.LINES, 'names': T.NAMES, 'torpids_flow': T.TORPIDS_FLOW,
              'terms': out_terms, 'weekly': {loc: weekly(ens, loc, prev) for loc in ('isis', 'godstow')},
              'torpids': torpids, 'seasonal': seasonal, 'ec46': ec46,
              'forecast_weighting': weighting}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f'Termly outlook from {t0:%Y-%m-%d}: Farmoor {state["flow"]} m3/s, {state["pct_normal_7d"]:.0f}% of normal '
          f'(7 days), {state["pct_normal_90d"]:.0f}% (90 days)')
    for t in out_terms['isis']:
        print(f"  Isis {t['label']} ({t['lead']}): " + ', '.join(
            f"{n} {v['p10']:.0f}-{v['p90']:.0f} (average {v['normal_mean']:.0f})" for n, v in t['flags'].items()))
    for m in seasonal:
        print(f"  SEAS5 {m['month']}: {m['median']} mm (normal {m['normal']}), wet third {m['wet']:.0%}")
    if torpids:
        print(f"  Torpids {torpids['year']}: chance of a cancellation-level river {torpids['chance']:.0%} "
              f"(an average year {torpids['chance_clim']:.0%})")
    print(f'written {OUT}')


if __name__ == '__main__':
    main()
