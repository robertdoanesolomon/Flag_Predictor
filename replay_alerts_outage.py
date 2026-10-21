"""
Isis flag alerts during a gauge outage: how good are they when the forecast starts from the
Godstow estimate (hybrid_live.neighbour_nowcast) instead of a reading?

Replays 2025 (the months the Isis flag was monitored) with a forecast every 3 h:

* normal: the forecast starts from the Isis reading;
* down D days (2, 5, 10): the Isis readings of the last D days are replaced by the estimate from
  Godstow, the outcomes widened by its error (nowcast_error_sd), as the live page does.

Each forecast: the deployed hybrid (5 copies) on the rain that fell, plus the model-error library
(200 outcomes per copy), then the official-flag model (official_flag.simulate / outlook) from the
flag actually flying. Alerts follow push/alerts.py: "likely" = chance >= 0.6 today or by the end of
tomorrow, once per direction until the chance falls below 0.3 or the flag changes.

Scored against the OURCs flag history:
* an alert came true if the flag moved that way by the end of its window;
* a flag change was warned if an alert in its direction went out in the 48 h before it.

Rain that fell (no archive of the rain forecasts), and no lock-keeper step layer: the absolute
numbers are a little optimistic; the comparison between the start conditions is the point.

    python replay_alerts_outage.py [--every 3] [--days 2,5,10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from flag_predictor.evaluation import load_merged  # noqa: E402
from flag_predictor.models.candidates import ensemble_predictor  # noqa: E402
from flag_predictor.prediction import official_flag as of  # noqa: E402
from flag_predictor.prediction.hybrid_live import (  # noqa: E402
    ERROR_LIBRARY, HYBRID_MODELS, MODEL_TAG, OUTAGE_NEIGHBOUR, TRAINING_GAUGES,
    neighbour_nowcast, nowcast_error_sd,
)
from flag_predictor.prediction.model_error import ErrorLibrary, start_features, total_uncertainty  # noqa: E402

LOC = 'isis'
RESET = 0.3
H = 72


def flag_history() -> pd.DataFrame:
    h = pd.read_csv(ROOT / 'data' / 'flags' / f'history_{LOC}.csv')
    h['start'] = pd.to_datetime(h['start'], utc=True)
    h['end'] = pd.to_datetime(h['end'], utc=True)
    return h.sort_values('start').reset_index(drop=True)


def flag_at(h: pd.DataFrame, t: pd.Timestamp):
    row = h[(h.start <= t) & (h.end > t)]
    return (row.iloc[-1].flag, row.iloc[-1].start) if len(row) else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--days', default='2,5,10')
    ap.add_argument('--samples', type=int, default=40, help='model-error draws per copy')
    ap.add_argument('--out', default=None)
    ap.add_argument('--river', default='isis', choices=['isis', 'godstow'])
    ap.add_argument('--likely', default='0.6,0.7', help='alert thresholds to score')
    ap.add_argument('--cache', default=None, help='pickle of the forecasts: reuse if present')
    args = ap.parse_args()
    global LOC
    LOC = args.river
    torch.set_num_threads(2)
    downs = [0] + [int(x) for x in args.days.split(',') if x]
    out_path = Path(args.out or ROOT / 'figures' / 'eval' / f'alert_replay_outage_{LOC}.json')
    cache = Path(args.cache or ROOT / 'figures' / 'eval' / f'alert_replay_outage_{LOC}.pkl')

    F = load_merged(LOC, ROOT)
    F.index = F.index.tz_localize('UTC') if F.index.tz is None else F.index.tz_convert('UTC')
    other, slope = OUTAGE_NEIGHBOUR[LOC]
    N = load_merged(other, ROOT)['differential']
    N.index = N.index.tz_localize('UTC') if N.index.tz is None else N.index.tz_convert('UTC')
    d = F['differential']
    stations = [c for c in F.columns if c not in ('differential', 'flow_m3s_Farmoor')]
    gauges = [g for g in TRAINING_GAUGES[LOC] if g in F.columns]
    predict = ensemble_predictor(HYBRID_MODELS, LOC)
    lib = ErrorLibrary.load(ROOT / 'models' / ERROR_LIBRARY.format(tag=MODEL_TAG, location=LOC))
    params = of.FlagParams.load(ROOT / 'models' / of.PARAMS_FILE)
    names = of.order(LOC)
    hist_flags = flag_history()

    starts = pd.date_range('2025-01-01', '2025-12-31', freq=f'{args.every}h', tz='UTC')
    starts = [t for t in starts if t in d.index and np.isfinite(d.get(t, np.nan))
              and flag_at(hist_flags, t)[0] in names]
    print(f'{len(starts)} starts with a modelled flag flying, {len(downs)} start conditions', flush=True)

    import pickle
    runs = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [D for D in downs if D not in runs]
    for D in todo:
        runs[D] = []
    rng = np.random.default_rng(0)
    if not todo:
        starts = []
    t_clock = time.time()
    for i, t0 in enumerate(starts):
        fut = pd.date_range(t0 + pd.Timedelta(hours=1), periods=240, freq='1h')
        rain = F[stations].reindex(fut).fillna(0.0)
        rain_mean = rain[gauges].clip(lower=0).mean(axis=1).to_numpy()
        past168 = float(F[gauges].clip(lower=0).mean(axis=1).fillna(0.0).loc[:t0].iloc[-168:].sum())
        hist = F.loc[t0 - pd.Timedelta(days=30):t0].copy()
        flag0, set_at = flag_at(hist_flags, t0)
        since = max((t0 - set_at).total_seconds() / 3600, 0.0)
        for D in todo:
            h = hist
            sd = 0.0
            if D:
                t_last = t0 - pd.Timedelta(days=D)
                last = d.loc[:t_last].dropna()
                if not len(last):
                    continue
                est = neighbour_nowcast(last, N, slope, t0)
                if est is None or t0 not in est.index:
                    continue
                h = hist.copy()
                h.loc[est.index.intersection(h.index), 'differential'] = est.reindex(
                    est.index.intersection(h.index)).to_numpy()
                sd = nowcast_error_sd(LOC, (t0 - last.index[-1]).total_seconds() / 3600, float(est.loc[t0]))
            traj = np.asarray(predict(h, rain), float)[:H + 1]
            diff_h = h['differential']
            d0 = float(diff_h.iloc[-1])
            near = diff_h.loc[t0 - pd.Timedelta(hours=26):t0 - pd.Timedelta(hours=22)].dropna()
            d24 = float(near.iloc[len(near) // 2]) if len(near) else float('nan')
            feat = start_features(d0, d24, rain_mean, past168, t0)[None, :]
            members = traj[None, :]
            out = total_uncertainty(np.pad(members, ((0, 0), (0, 240 - H)), mode='edge'), feat, lib,
                                    n=args.samples * len(HYBRID_MODELS), seed=i)[:, :H + 1]
            if sd > 0:
                out = out + rng.normal(0.0, sd, size=(out.shape[0], 1))
            level_hist = diff_h.loc[t0 - pd.Timedelta(hours=96):t0].to_numpy()
            flags = of.simulate(params, LOC, names.index(flag0), since, level_hist, out, t0,
                                rng=np.random.default_rng(i))
            runs[D].append((t0, flag0, of.outlook(LOC, flags, names.index(flag0), t0)))
        if i % 100 == 0:
            print(f'  {i}/{len(starts)} {t0:%d %b %H:%M}  {(time.time() - t_clock) / 60:.1f} min', flush=True)

    cache.write_bytes(pickle.dumps(runs))

    # flag changes between modelled flags
    ch = []
    for k in range(1, len(hist_flags)):
        a, b = hist_flags.iloc[k - 1], hist_flags.iloc[k]
        if a.flag in names and b.flag in names and b.start.year == 2025 and a.end == b.start:
            ch.append({'t': b.start, 'from': a.flag, 'to': b.flag,
                       'dir': 'up' if names.index(b.flag) > names.index(a.flag) else 'down'})

    def first_change_after(t, direction, until):
        for c in ch:
            if t < c['t'] <= until and c['dir'] == direction:
                return c
        return None

    report = {}
    for likely in [float(x) for x in args.likely.split(',')]:
        print(f'\n{LOC}, alert at {likely:.0%}+ ({len(ch)} flag changes in 2025)')
        print(' offline | alerts | correct | false alarms | changes warned | changes missed | '
              'up: correct/false | down: correct/false')
        for D in downs:
            alerts = []
            sent, cur_flag = set(), None
            for t0, flag0, oo in runs[D]:
                if flag0 != cur_flag:
                    sent, cur_flag = set(), flag0
                win = oo['windows']
                wins = of.day_windows(t0, H)
                for dr in ('up', 'down'):
                    if not oo.get(dr):
                        continue
                    p_today, p_tom = win['today'][dr], win['tomorrow'][dr]
                    if p_tom < RESET:
                        sent -= {f'{dr}:today', f'{dr}:tomorrow'}
                        continue
                    for key, p, last in ((f'{dr}:today', p_today, wins['today']),
                                         (f'{dr}:tomorrow', p_tom, wins['tomorrow'])):
                        if p >= likely and key not in sent and not (key.endswith('tomorrow') and f'{dr}:today' in sent):
                            sent.add(key)
                            until = t0 + pd.Timedelta(hours=last)
                            alerts.append({'t': t0, 'dir': dr, 'p': p,
                                           'right': first_change_after(t0, dr, until) is not None})
                            break
            warned = sum(any(a['dir'] == c['dir'] and c['t'] - pd.Timedelta(hours=48) <= a['t'] < c['t']
                             for a in alerts) for c in ch)
            hits = sum(a['right'] for a in alerts)
            by = {dr: (sum(a['right'] for a in alerts if a['dir'] == dr),
                       sum(not a['right'] for a in alerts if a['dir'] == dr)) for dr in ('up', 'down')}
            report[f'{likely:.2f}/{D}d'] = {'alerts': len(alerts), 'correct': hits, 'false_alarms': len(alerts) - hits,
                                            'changes': len(ch), 'warned': warned, 'missed': len(ch) - warned,
                                            'by_direction': by, 'starts': len(runs[D])}
            print(f' {D:4d} d  | {len(alerts):6d} | {hits:3d} ({hits / max(len(alerts), 1):.0%}) | '
                  f'{len(alerts) - hits:3d} ({(len(alerts) - hits) / max(len(alerts), 1):.0%})    | '
                  f'{warned:3d} ({warned / max(len(ch), 1):.0%})      | {len(ch) - warned:3d}            | '
                  f'{by["up"][0]}/{by["up"][1]}               | {by["down"][0]}/{by["down"][1]}', flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f'written {out_path}')

if __name__ == '__main__':
    main()
