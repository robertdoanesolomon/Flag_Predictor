"""Gauge outages: how well one stretch's differential is estimated from its neighbour.

While a stretch's gauges are offline, the live forecast starts from
``hybrid_live.neighbour_nowcast`` (Isis from Godstow, Godstow from the Isis). This
script measures that estimate on 2019-2026: at random hours, the gauge "goes down",
and the estimate is compared with what was measured 1-14 days later.

Part 1 (default): error by days down and level, and the fit behind
``hybrid_live.NOWCAST_ERROR``.
Part 2 (--forecast): the hybrid forecast with rain that fell, from the true level,
the estimate, and the last reading held, at 40 starts per stretch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.models.physics_data import load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import (  # noqa: E402
    NOWCAST_ERROR, OUTAGE_NEIGHBOUR, neighbour_nowcast, nowcast_error_sd,
)

DAYS = (1, 2, 3, 5, 7, 10, 14)


def series(loc: str) -> pd.Series:
    data = load_location_data(loc, ROOT)
    s = pd.Series(np.asarray(data.diff, float), data.index.tz_convert('UTC'))
    return s[s.index >= '2019-01-01']


def part1(n_starts: int = 1500, seed: int = 0) -> None:
    diffs = {loc: series(loc) for loc in ('isis', 'godstow')}
    rng = np.random.default_rng(seed)
    for loc in ('isis', 'godstow'):
        other, slope = OUTAGE_NEIGHBOUR[loc]
        d, n = diffs[loc], diffs[other]
        idx = d.dropna().index
        idx = idx[(idx > idx[0] + pd.Timedelta(days=15)) & (idx < idx[-1] - pd.Timedelta(days=15))]
        starts = rng.choice(idx, n_starts, replace=False)
        rows = []
        for t in starts:
            t = pd.Timestamp(t)
            est = neighbour_nowcast(d.loc[t - pd.Timedelta(days=15):t], n, slope, t + pd.Timedelta(days=14))
            if est is None:
                continue
            for days in DAYS:
                tt = t + pd.Timedelta(days=days)
                if tt in d.index and np.isfinite(d.loc[tt]) and tt in est.index:
                    rows.append({'days': days, 'err': d.loc[tt] - est.loc[tt], 'held': d.loc[tt] - d.loc[t],
                                 'level': d.loc[tt], 'est': est.loc[tt]})
        df = pd.DataFrame(rows)
        lo, hi = NOWCAST_ERROR[loc]['mid']
        df['mid'] = (df.est >= lo) & (df.est < hi)
        print(f'\n{loc} from {other} (slope {slope}), {df.groupby("days").size().iloc[0]} starts')
        print(' days | MAE est (mid / other) | sd est (mid / other) | MAE held | sd model (mid / other) | in +-1.28 sd')
        a = []
        for days, g in df.groupby('days'):
            m, o = g[g.mid], g[~g.mid]
            sd_m = np.array([nowcast_error_sd(loc, days * 24, v) for v in g.est])
            cover = np.mean(np.abs(g.err) <= 1.2816 * sd_m)
            print(f' {days:4d} | {g.err.abs().mean():.3f} ({m.err.abs().mean():.3f} / {o.err.abs().mean():.3f}) | '
                  f'{g.err.std():.3f} ({m.err.std():.3f} / {o.err.std():.3f}) | {g.held.abs().mean():.3f} | '
                  f'{nowcast_error_sd(loc, days * 24, (lo + hi) / 2):.3f} / {nowcast_error_sd(loc, days * 24, -1):.3f} | '
                  f'{cover:.0%}')
            a.append((days, o.err.std(), m.err.std()))
        a = np.array(a)
        fit_o = np.sum(a[:, 1] * np.sqrt(a[:, 0])) / np.sum(a[:, 0])
        fit_m = np.sum(a[:, 2] * np.sqrt(a[:, 0])) / np.sum(a[:, 0])
        print(f' fitted sd = a sqrt(days): a = {fit_o:.3f} outside the middle, {fit_m:.3f} in it '
              f'(factor {fit_m / fit_o:.2f})')


def part2(n_starts: int = 40, seed: int = 0) -> None:
    from flag_predictor.config import get_flag_thresholds
    from flag_predictor.evaluation import load_merged
    from flag_predictor.models.candidates import ensemble_predictor
    from flag_predictor.prediction.hybrid_live import HYBRID_MODELS

    rng = np.random.default_rng(seed)
    frames = {loc: load_merged(loc, ROOT) for loc in ('isis', 'godstow')}
    for loc in ('isis', 'godstow'):
        other, slope = OUTAGE_NEIGHBOUR[loc]
        F, N = frames[loc], frames[other]['differential']
        stations = [c for c in F.columns if c not in ('differential', 'flow_m3s_Farmoor')]
        predict = ensemble_predictor(HYBRID_MODELS, loc)
        th = get_flag_thresholds(loc)
        edges = sorted({hi for lo, hi in th.values() if np.isfinite(hi)})
        d = F['differential']
        ok = d.notna() & N.reindex(d.index).notna()
        idx = d.index[ok & (((d.index >= '2023-01-15') & (d.index < '2024-01-01'))
                            | ((d.index >= '2024-11-15') & (d.index < '2026-01-01')))]
        starts = rng.choice(idx, n_starts, replace=False)
        print(f'\n{loc} ({other} as neighbour), {n_starts} starts, rain that fell; error over the 10 days after "now" (m)')
        print(' days down | from the true level 48 h / 10 d | from the estimate 48 h / 10 d | '
              'stale forecast (from last reading) 48 h / 10 d | right band 48 h: true / estimate / stale')
        for days in (2, 5, 10):
            res = {k: ([], [], []) for k in ('true', 'est', 'stale')}
            for t0 in starts:
                t0 = pd.Timestamp(t0)
                t_last = t0 - pd.Timedelta(days=days)
                if not np.isfinite(d.get(t_last, np.nan)):
                    continue
                fut = pd.date_range(t0 + pd.Timedelta(hours=1), periods=240, freq='1h')
                actual = d.reindex(fut).to_numpy()
                if np.isnan(actual).mean() > 0.2:
                    continue
                rain = F[stations].reindex(fut).fillna(0.0)
                hist = F.loc[t0 - pd.Timedelta(days=30):t0].copy()
                est = neighbour_nowcast(d.loc[:t_last], N, slope, t0)
                if est is None:
                    continue
                hist_est = hist.copy()
                hist_est.loc[est.index, 'differential'] = est.to_numpy()
                runs = {'true': predict(hist, rain)[1:], 'est': predict(hist_est, rain)[1:]}
                # what the page shows now: the forecast from the last reading, read off at the same hours
                fut_s = pd.date_range(t_last + pd.Timedelta(hours=1), periods=240, freq='1h')
                rain_s = F[stations].reindex(fut_s).fillna(0.0)
                stale = pd.Series(predict(F.loc[t_last - pd.Timedelta(days=30):t_last], rain_s)[1:], fut_s)
                # (beyond its 10 days it holds its last value)
                runs['stale'] = stale.reindex(fut_s.union(fut)).ffill().reindex(fut).to_numpy()
                for k, p in runs.items():
                    e = np.abs(p - actual)
                    res[k][0].append(np.nanmean(e[:48]))
                    res[k][1].append(np.nanmean(e))
                    v = np.isfinite(actual[:48]) & np.isfinite(p[:48])
                    res[k][2].append(np.mean(np.digitize(p[:48][v], edges) == np.digitize(actual[:48][v], edges)))
            m = {k: [np.nanmean(x) for x in v] for k, v in res.items()}
            print(f' {days:9d} | {m["true"][0]:.3f} / {m["true"][1]:.3f}                  | '
                  f'{m["est"][0]:.3f} / {m["est"][1]:.3f}                | '
                  f'{m["stale"][0]:.3f} / {m["stale"][1]:.3f}                                 | '
                  f'{m["true"][2]:.0%} / {m["est"][2]:.0%} / {m["stale"][2]:.0%}  (n={len(res["true"][0])})')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--starts', type=int, default=None)
    ap.add_argument('--forecast', action='store_true', help='part 2: the hybrid forecast from each start')
    args = ap.parse_args()
    if args.forecast:
        part2(args.starts or 40)
    else:
        part1(args.starts or 1500)


if __name__ == '__main__':
    main()
