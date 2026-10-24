"""
Verification numbers for the report page (figures/eval/verification.json).

1. Held-out backtest (2023 validation year + Nov 2024 - Jan 2026 test window, forecasts every 6 h, the
   rain that fell): mean error by lead hour for the deployed hybrid (5 copies), "no change"
   (persistence) and the trial model with soil moisture and upstream gauges (hybrid_srh); share of
   hours in the right flag band (Isis, Godstow); error at 24 h and 72 h by month.
2. Likely range: cross-validated coverage and CRPS by lead (build_error_library.py check files).
3. February 2025: the river hour by hour and what the forecast said 6, 24, 48 and 72 h before; the
   official flags.
4. Live forecasts since the archive began (real rain forecasts): error by lead against "no change",
   share inside the likely range.
5. Flag alerts (alert_replay_outage.py caches, current thresholds): by year; how reliable the chance of
   a flag change is (reliability diagram).
6. Gauge-outage estimates (neighbour estimate faded in over 12 h) by hours offline.

    python verification_report.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from build_error_library import predict_hybrid, start_positions  # noqa: E402
from flag_predictor.config import get_flag_thresholds  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, load_location_data  # noqa: E402
from flag_predictor.prediction import official_flag as of  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402

LOCS = ('isis', 'godstow', 'wallingford')
TRIAL = ['hybrid_srh'] + [f'hybrid_srh_s{i}' for i in range(1, 5)]
OUT = ROOT / 'figures' / 'eval' / 'verification.json'


def edges(loc):
    th = get_flag_thresholds(loc)
    return sorted({hi for lo, hi in th.values() if np.isfinite(hi) and hi > -1e8})


def r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def backtest() -> dict:
    from train_extra_models import data_with_extra
    out = {}
    for loc in LOCS:
        data = load_location_data(loc, ROOT)
        st = start_positions(data)
        t_all = np.concatenate([st['val2023'], st['test']])
        obs = np.stack([data.diff[t + 1:t + HORIZON + 1] for t in t_all])
        main = predict_hybrid(HYBRID_MODELS, loc, data, t_all)
        trial = predict_hybrid(TRIAL, loc, data_with_extra(loc, 'soil_rise_han'), t_all)
        pers = np.repeat(np.asarray(data.diff)[t_all][:, None], HORIZON, axis=1)
        test = np.isin(t_all, st['test'])
        res = {'n_starts': int(len(t_all)), 'n_test': int(test.sum()),
               'lead_h': list(range(1, HORIZON + 1))}
        for name, p in (('model', main), ('persistence', pers), ('trial', trial)):
            res[f'mae_{name}'] = [r(np.nanmean(np.abs(p[:, h] - obs[:, h]))) for h in range(HORIZON)]
            res[f'mae_{name}_test'] = [r(np.nanmean(np.abs(p[test, h] - obs[test, h]))) for h in range(HORIZON)]
        ed = edges(loc) if loc != 'wallingford' else []
        if ed:
            ok = np.isfinite(obs)
            bo = np.digitize(obs, ed)
            for name, p in (('model', main), ('persistence', pers), ('trial', trial)):
                hit = (np.digitize(p, ed) == bo) & ok
                res[f'band_{name}'] = [r(hit[:, h].sum() / max(ok[:, h].sum(), 1), 4) for h in range(HORIZON)]
            # how often the band changed at all over the lead (the hard cases)
            moved = np.digitize(obs, ed) != np.digitize(np.asarray(data.diff)[t_all], ed)[:, None]
            hitm = (np.digitize(main, ed) == bo) & ok & moved
            hitp = (np.digitize(pers, ed) == bo) & ok & moved
            res['band_when_changed'] = {
                'lead_h': [24, 48, 72, 120],
                'model': [r(hitm[:, h - 1].sum() / max((ok & moved)[:, h - 1].sum(), 1)) for h in (24, 48, 72, 120)],
                'persistence': [r(hitp[:, h - 1].sum() / max((ok & moved)[:, h - 1].sum(), 1)) for h in (24, 48, 72, 120)],
                'n': [int((ok & moved)[:, h - 1].sum()) for h in (24, 48, 72, 120)]}
        # by month at 24 h and 72 h
        months = pd.DatetimeIndex(data.index[t_all]).tz_convert('UTC').to_period('M').astype(str)
        bym = []
        for m in sorted(set(months)):
            sel = months == m
            bym.append({'month': m, 'n': int(sel.sum()),
                        'mae24': r(np.nanmean(np.abs(main[sel, 23] - obs[sel, 23]))),
                        'mae72': r(np.nanmean(np.abs(main[sel, 71] - obs[sel, 71]))),
                        'pers72': r(np.nanmean(np.abs(pers[sel, 71] - obs[sel, 71]))),
                        'level': r(np.nanmean(np.asarray(data.diff)[t_all][sel]), 3)})
        res['by_month'] = bym
        out[loc] = res
        print(f'{loc}: backtest done ({len(t_all)} starts)', flush=True)
    return out


def coverage() -> dict:
    out = {}
    for tag in ('hybrid_c1_ps', 'hybrid_srh'):
        p = ROOT / 'figures' / f'error_library_{tag}_check.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[tag] = {}
        for loc, v in d.items():
            if not isinstance(v, dict) or 'by_k' not in v:
                continue
            k = str(v.get('chosen_k') or v.get('k') or sorted(v['by_k'], key=int)[-1])
            if k not in v['by_k']:
                k = sorted(v['by_k'], key=int)[0]
            cal = v['by_k'][k]['calibrated']
            out[tag][loc] = {b: {'in_band': r(c['in_band_10_90'], 3), 'crps': r(c['crps']), 'mae': r(c['mae_point'])}
                             for b, c in cal.items()}
    return out


def feb2025() -> dict:
    lo_t, hi_t = pd.Timestamp('2025-01-20', tz='UTC'), pd.Timestamp('2025-02-28', tz='UTC')
    out = {}
    for loc in LOCS:
        data = load_location_data(loc, ROOT)
        idx = data.index.tz_convert('UTC')
        d = np.asarray(data.diff, float)
        enc_ok = np.isfinite(data.enc).all(axis=1)
        ok = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(enc_ok)] == SEQ_LEN
        pos = np.flatnonzero((idx >= lo_t - pd.Timedelta(days=4)) & (idx < hi_t) & ok & np.isfinite(d))
        pred = dict(zip(pos, predict_hybrid(HYBRID_MODELS, loc, data, pos)))
        win = np.flatnonzero((idx >= lo_t) & (idx < hi_t))
        res = {'times': [t.strftime('%Y-%m-%dT%H:%M') for t in idx[win]], 'observed': [r(d[i], 3) for i in win]}
        for lead in (6, 24, 48, 72):
            res[f'lead{lead}'] = [r(pred[i - lead][lead - 1], 3) if (i - lead) in pred else None for i in win]
        ed = edges(loc) if loc != 'wallingford' else []
        if ed:
            res['band_hit'] = {}
            for lead in (6, 24, 48, 72):
                f = np.array([np.nan if v is None else v for v in res[f'lead{lead}']], float)
                o = np.array([np.nan if v is None else v for v in res['observed']], float)
                m = np.isfinite(f) & np.isfinite(o)
                res['band_hit'][lead] = r((np.digitize(f[m], ed) == np.digitize(o[m], ed)).mean(), 3)
            res['mae'] = {lead: r(np.nanmean(np.abs(np.array([np.nan if v is None else v for v in res[f'lead{lead}']], float)
                                                    - np.array([np.nan if v is None else v for v in res['observed']], float))))
                          for lead in (6, 24, 48, 72)}
            fh = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv')
            fh['start'] = pd.to_datetime(fh['start'], utc=True)
            fh['end'] = pd.to_datetime(fh['end'], utc=True)
            sel = fh[(fh.end > lo_t) & (fh.start < hi_t)]
            res['flags'] = [{'flag': row.flag, 'start': max(row.start, lo_t).strftime('%Y-%m-%dT%H:%M'),
                             'end': min(row.end, hi_t).strftime('%Y-%m-%dT%H:%M')} for row in sel.itertuples()]
        else:
            res['mae'] = {lead: r(np.nanmean(np.abs(np.array([np.nan if v is None else v for v in res[f'lead{lead}']], float)
                                                    - np.array([np.nan if v is None else v for v in res['observed']], float))))
                          for lead in (6, 24, 48, 72)}
        res['lines'] = ed
        out[loc] = res
        print(f'{loc}: Feb 2025 done', flush=True)
    return out


def live() -> dict:
    o = json.loads((ROOT / 'docs' / 'archive' / 'observed.json').read_text())
    obs = {loc: pd.Series(v['values'], index=pd.date_range(v['start'], periods=len(v['values']),
                                                             freq=f"{v['step_h']}h"), dtype=float) for loc, v in o.items()}
    bins = [(1, 6), (7, 24), (25, 72), (73, 168), (169, 240)]
    rows = []
    issues = sorted(glob.glob(str(ROOT / 'docs' / 'archive' / 'issues' / '*.json')))
    for f in issues:
        iss = json.loads(Path(f).read_text())
        for loc, L in iss['locations'].items():
            t0 = pd.Timestamp(L['t0'])
            n = len(L['mean'])
            y = obs[loc].reindex(pd.date_range(t0, periods=n, freq='1h')).to_numpy()
            mean = np.array([np.nan if v is None else v for v in L['mean']], float)
            lo = L.get('total_p10')
            hi = L.get('total_p90')
            for h in range(1, n):
                if not np.isfinite(y[h]) or not np.isfinite(mean[h]):
                    continue
                inside = None
                if lo is not None and hi is not None and lo[h] is not None and hi[h] is not None:
                    inside = lo[h] <= y[h] <= hi[h]
                rows.append((loc, h, abs(mean[h] - y[h]), abs((L.get('d0') or mean[0]) - y[h]), inside))
    df = pd.DataFrame(rows, columns=['loc', 'h', 'err', 'pers', 'inside'])
    out = {'issues': len(issues), 'first': Path(issues[0]).stem, 'last': Path(issues[-1]).stem, 'bins': [f'{a}-{b} h' for a, b in bins]}
    for loc in LOCS:
        g = df[df['loc'] == loc]
        res = {'mae': [], 'pers': [], 'inside': [], 'n': []}
        for a, b in bins:
            s = g[(g.h >= a) & (g.h <= b)]
            ins = s.inside.dropna()
            res['mae'].append(r(s.err.mean()))
            res['pers'].append(r(s.pers.mean()))
            res['inside'].append(r(ins.astype(float).mean(), 3) if len(ins) else None)
            res['n'].append(int(len(s)))
        out[loc] = res
    print('live archive done', flush=True)
    return out


def alerts() -> dict:
    out = {}
    cfg = {'isis': ('alert_replay_outage_isis_2021-2026.pkl', '2021-11-01', 0.7),
           'godstow': ('alert_replay_outage_godstow_2019-2026.pkl', '2024-09-01', 0.8)}
    for loc, (pkl, frm, down) in cfg.items():
        runs = pickle.loads((ROOT / 'figures' / 'eval' / pkl).read_bytes())[0]
        runs = [x for x in runs if x[0] >= pd.Timestamp(frm, tz='UTC')]
        names = of.order(loc)
        h = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{loc}.csv')
        h['start'] = pd.to_datetime(h.start, utc=True)
        h['end'] = pd.to_datetime(h.end, utc=True)
        ch = []
        for k in range(1, len(h)):
            a, b = h.iloc[k - 1], h.iloc[k]
            if a.flag in names and b.flag in names and a.end == b.start and b.start >= pd.Timestamp(frm, tz='UTC'):
                ch.append((b.start, 'up' if names.index(b.flag) > names.index(a.flag) else 'down'))
        thr = {'up': 0.6, 'down': down}
        alerts_, sent, cur = [], set(), None
        rel = []
        for t0, f0, oo in runs:
            if f0 != cur:
                sent, cur = set(), f0
            w = oo['windows']
            wins = of.day_windows(t0, 72)
            for dr in ('up', 'down'):
                if not oo.get(dr):
                    continue
                until = t0 + pd.Timedelta(hours=wins['tomorrow'])
                happened = any(t0 < c <= until and dd == dr for c, dd in ch)
                rel.append((w['tomorrow'][dr], happened))
                pt, pm = w['today'][dr], w['tomorrow'][dr]
                if pm < 0.3:
                    sent -= {f'{dr}:today', f'{dr}:tomorrow'}
                    continue
                for key, p, last in ((f'{dr}:today', pt, wins['today']), (f'{dr}:tomorrow', pm, wins['tomorrow'])):
                    if p >= thr[dr] and key not in sent and not (key.endswith('tomorrow') and f'{dr}:today' in sent):
                        sent.add(key)
                        u = t0 + pd.Timedelta(hours=last)
                        alerts_.append((t0, dr, any(t0 < c <= u and dd == dr for c, dd in ch)))
                        break
        years = sorted({t.year for t, _, _ in alerts_} | {c.year for c, _ in ch})
        by_year = []
        for y in years:
            al = [a for a in alerts_ if a[0].year == y]
            cy = [c for c in ch if c[0].year == y]
            warned = sum(any(a[1] == c[1] and c[0] - pd.Timedelta(hours=48) <= a[0] < c[0] for a in alerts_) for c in cy)
            by_year.append({'year': y, 'alerts': len(al), 'correct': sum(a[2] for a in al),
                            'false': sum(not a[2] for a in al), 'changes': len(cy), 'warned': warned, 'missed': len(cy) - warned})
        p = np.array([x[0] for x in rel])
        o_ = np.array([x[1] for x in rel], float)
        bins = np.array([0, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0001])
        diag = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 5:
                diag.append({'p': r(p[m].mean(), 3), 'observed': r(o_[m].mean(), 3), 'n': int(m.sum())})
        out[loc] = {'from': frm, 'thresholds': thr, 'by_year': by_year, 'reliability': diag,
                    'total': {'alerts': len(alerts_), 'correct': sum(a[2] for a in alerts_), 'changes': len(ch),
                              'warned': sum(r_['warned'] for r_ in by_year)}}
    print('alerts done', flush=True)
    return out


def outages() -> dict:
    sys.argv = ['x']
    import backtest_outage_nowcast as b
    from flag_predictor.prediction.hybrid_live import OUTAGE_NEIGHBOUR, neighbour_nowcast
    S = {loc: b.series(loc) for loc in LOCS}
    rng = np.random.default_rng(3)
    hours = (1, 2, 4, 8, 12, 24, 48, 72, 120)
    out = {'hours': list(hours)}
    for loc in LOCS:
        other, slope = OUTAGE_NEIGHBOUR[loc]
        d, n = S[loc], S[other]
        idx = d.dropna().index
        idx = idx[(idx > idx[0] + pd.Timedelta(days=15)) & (idx < idx[-1] - pd.Timedelta(days=6))]
        rows = []
        for t in rng.choice(idx, 700, replace=False):
            t = pd.Timestamp(t)
            est = neighbour_nowcast(d.loc[t - pd.Timedelta(days=15):t], n, slope, t + pd.Timedelta(hours=max(hours)))
            if est is None:
                continue
            for hh in hours:
                tt = t + pd.Timedelta(hours=hh)
                if tt in d.index and np.isfinite(d.loc[tt]) and tt in est.index:
                    rows.append((hh, abs(d.loc[tt] - est.loc[tt]), abs(d.loc[tt] - d.loc[t]), d.loc[tt]))
        df = pd.DataFrame(rows, columns=['h', 'err', 'hold', 'level'])
        out[loc] = {'from': other,
                    'estimate': [r(df[df.h == hh].err.mean()) for hh in hours],
                    'hold': [r(df[df.h == hh].hold.mean()) for hh in hours],
                    'estimate_high': [r(df[(df.h == hh) & (df.level > 0.2)].err.mean()) for hh in hours]}
    print('outages done', flush=True)
    return out


def main() -> None:
    torch.set_num_threads(2)
    rep = {'generated': pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}
    rep['live'] = live()
    rep['alerts'] = alerts()
    rep['coverage'] = coverage()
    rep['outages'] = outages()
    rep['feb2025'] = feb2025()
    rep['backtest'] = backtest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, separators=(',', ':')))
    print(f'written {OUT} ({OUT.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
