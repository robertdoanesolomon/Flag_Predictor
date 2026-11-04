#!/usr/bin/env python
"""
Backtest of the lock-keeper step layer: one set of parameters against another.

Out of sample (2023 and Nov 2024 - Jan 2026), starts every 6 h, the deployed
hybrid forecast with the rain that fell as the smooth outcome; each start is
stepped N times. Scores, hours 1-48 and 1-72:

* CRPS of the stepped draws against the observed hourly differential (cm), and
  the smooth outcome's absolute error;
* Brier score of "flag band changes and holds 3 h, today / by the end of
  tomorrow" (Isis and Godstow);
* quiet water: starts whose smooth outcome moves < 2 cm over 72 h. Share of
  draws, and of the real river, that wander > 5 / > 10 cm from the start,
  and the size of jumps of at least the jump threshold (simulated vs real).

With --with-error the outcomes are scored as on the live site: the smooth
forecast plus model errors borrowed from the error library (never from the
same period as the start), stepped with the errors smoothed over a day and
their hour-scale wobble added back. Scores are then for those outcomes,
smooth against stepped.

    python backtest_step_layer.py --params models --compare /tmp/new_params [--with-error]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from flag_predictor.config import get_flag_thresholds  # noqa: E402
from flag_predictor.prediction import step_layer as SL  # noqa: E402
import fit_step_layer as F  # noqa: E402

H = 96
CACHE = ROOT / 'data' / 'verification' / 'step_backtest_cache'


def crps(draws: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """(H,) CRPS of S draws (S, H) against obs (H,), NaN where obs is missing."""
    a = np.abs(draws - obs[None, :]).mean(0)
    s = np.sort(draws, axis=0)
    n = s.shape[0]
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    b = (w * s).sum(0) / n ** 2
    return a - b


def starts_and_forecasts(location: str, levels: dict):
    """Start times, observed hourly paths (from the start, H+1), smooth outcomes (H+1), rain hist/future."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{location}.npz'
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, load_location_data
    data = load_location_data(location, ROOT)
    D_ser = levels[location]
    grid = pd.date_range(max(D_ser.index.min(), data.index.min()), min(D_ser.index.max(), data.index.max()), freq='1h')
    D = D_ser.reindex(grid).to_numpy()
    rain = pd.Series(data.rain, data.index).reindex(grid).fillna(0.0).to_numpy()
    nb = {o: levels[o].reindex(grid).to_numpy() for o in SL.LOCATIONS if o != location}
    enc_ok = np.isfinite(data.enc).all(axis=1)
    ok_count = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[: len(enc_ok)]
    idx = data.index
    cand = np.arange(SEQ_LEN, len(idx) - HORIZON - 1)
    tt = idx[cand]
    inp = (tt.year == F.VAL_YEAR) | ((tt >= F.TEST[0]) & (tt < F.TEST[1]))
    cand = cand[inp & (tt.hour % 6 == 0) & (ok_count[cand] == SEQ_LEN) & np.isfinite(data.diff[cand])]
    gpos = grid.get_indexer(idx[cand])
    keep = (gpos >= 60) & (gpos + H + 1 < len(grid))
    cand, gpos = cand[keep], gpos[keep]
    keep = np.isfinite(D[gpos])
    cand, gpos = cand[keep], gpos[keep]
    pred = F.forecasts(location, data, cand)                      # (N, 96) leads 1..96
    d0 = D[gpos]
    smooth = np.c_[d0, pred + (d0 - data.diff[cand])[:, None]]
    obs = np.stack([D[gpos + k] for k in range(H + 1)], 1)
    hist = np.stack([D[gpos - 48 + k] for k in range(49)], 1)
    rh = np.stack([rain[gpos - 24 + k] for k in range(24)], 1)
    rf = np.stack([rain[gpos + 1 + k] for k in range(H)], 1)
    nbh = {o: np.stack([v[gpos - 48 + k] for k in range(49)], 1) for o, v in nb.items()}
    out = dict(t0=np.array([str(t) for t in grid[gpos]]), obs=obs, smooth=smooth, hist=hist, rh=rh, rf=rf,
               **{f'nb_{o}': v for o, v in nbh.items()})
    np.savez_compressed(path, **out)
    return out


def borrowed_errors(location: str, b: dict, n: int, seed: int = 0) -> np.ndarray:
    """(N, n, H) model errors for each start, from library starts outside the start's own period."""
    from flag_predictor.models.physics_data import load_location_data
    from flag_predictor.prediction.model_error import ErrorLibrary, start_features
    lib = ErrorLibrary.load(ROOT / 'models' / f'error_library_hybrid_c1_ps_{location}.npz')
    data = load_location_data(location, ROOT)
    rain = pd.Series(data.rain, data.index)
    rain.index = rain.index.tz_localize('UTC') if rain.index.tz is None else rain.index
    c168 = rain.fillna(0).clip(lower=0).rolling(168, min_periods=1).sum()
    lt = pd.DatetimeIndex(lib.t0).tz_localize('UTC')
    in_val = lt.year == F.VAL_YEAR
    in_test = (lt >= F.TEST[0]) & (lt < F.TEST[1])
    rng = np.random.default_rng(seed)
    out = np.zeros((len(b['t0']), n, H))
    libs = {'val': lib.subset(~np.asarray(in_val)), 'test': lib.subset(~np.asarray(in_test))}
    for i, t in enumerate(b['t0']):
        t0 = pd.Timestamp(str(t))
        t0 = t0.tz_localize('UTC') if t0.tz is None else t0
        f = start_features(b['hist'][i][-1], b['hist'][i][-25], b['rf'][i], float(c168.asof(t0)), t0)
        e = libs['val' if t0.year == F.VAL_YEAR else 'test'].sample(f[None, :], n=n, rng=rng)[0, :, :H]
        out[i, :, :e.shape[1]] = e
        if e.shape[1] < H:
            out[i, :, e.shape[1]:] = e[:, -1:]
    return out


def run(location: str, params: SL.StepParams, b: dict, n_draws: int, seed: int = 0,
        errors: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    N = len(b['t0'])
    thresholds = None if location in SL.NO_FLAGS else get_flag_thresholds(location)
    flags = SL.ordered_flags(thresholds) if thresholds else None
    res = {k: [] for k in ('crps48', 'crps72', 'mae48', 'mae72')}
    brier = {(w, d): [] for w in ('today', 'tomorrow') for d in ('up', 'down')}
    quiet_dev, quiet_obs, sim_j, obs_j = [], [], [], []
    thr = SL.JUMP_THRESHOLD[location]
    for i in range(N):
        t0 = pd.Timestamp(str(b['t0'][i]))
        y = np.repeat(b['smooth'][i][None, :], n_draws, 0)
        nbh = {o: b[f'nb_{o}'][i] for o in SL.LOCATIONS if f'nb_{o}' in b}
        trig, own = SL.issue_triggers(location, b['hist'][i], nbh)
        o = b['obs'][i]
        if errors is None:
            z = SL.simulate(params, b['hist'][i], y, b['rh'][i], b['rf'][i], t0, trig, own, rng)
            e = np.abs(b['smooth'][i][1:] - o[1:])
        else:
            err = errors[i, :n_draws]
            total = y.copy()
            total[:, 1:] += err
            targets = y.copy()
            targets[:, 1:] += SL.smooth_errors(err)
            z = SL.simulate(params, b['hist'][i], targets, b['rh'][i], b['rf'][i], t0, trig, own, rng, final=total)
            w = np.clip((SL.STEP_HORIZON_H - np.arange(1, H + 1)) / (SL.STEP_HORIZON_H - SL.BLEND_START_H), 0, 1)
            z[:, 1:] += SL.hourly_wobble(err) * w
            e = crps(total[:, 1:], o[1:])          # 'smooth' score: CRPS of the unstepped outcomes
        c = crps(z[:, 1:], o[1:])
        for hh in (48, 72):
            m = np.isfinite(o[1:hh + 1])
            if m.mean() >= 0.8:
                res[f'crps{hh}'].append(np.nanmean(c[:hh][m[:hh]]))
                res[f'mae{hh}'].append(np.nanmean(e[:hh][m[:hh]]))
        if flags is not None and np.isfinite(o[:49]).mean() > 0.95:
            ol = SL.flag_outlook(z, thresholds, t0, float(o[0]))
            if ol is not None:
                win = SL.day_windows(t0, H)
                c0 = int(SL.flag_index(np.array([o[0]]), flags)[0])
                oo = pd.Series(o[1:]).ffill().to_numpy()
                cat = SL.flag_index(oo[None, :], flags)
                ch = {1: SL.held(cat > c0), -1: SL.held(cat < c0)}
                for w, last in win.items():
                    if last < 1:
                        continue
                    for d, s in (('up', 1), ('down', -1)):
                        brier[(w, d)].append((ol['windows'][w][d] - float(ch[s][0, :last].any())) ** 2)
        sm = b['smooth'][i][:73]
        if np.ptp(sm) < 0.02 and np.isfinite(o[:73]).all():
            quiet_dev.append(np.abs(z[:, :73] - z[:, :1]).max(1))
            quiet_obs.append(np.abs(o[:73] - o[0]).max())
            dz = np.diff(z[:, :73], axis=1)
            sim_j.append(np.abs(dz[np.abs(dz) >= thr]))
            do = np.diff(o[:73])
            obs_j.append(np.abs(do[np.abs(do) >= thr]))
    out = {k: float(np.mean(v)) * 100 for k, v in res.items()}
    out['n'] = N
    out['brier'] = {f'{w}_{d}': float(np.mean(v)) for (w, d), v in brier.items() if v}
    if quiet_dev:
        qd = np.concatenate(quiet_dev)
        qo = np.array(quiet_obs)
        sj = np.concatenate(sim_j) if any(len(x) for x in sim_j) else np.array([np.nan])
        oj = np.concatenate(obs_j) if any(len(x) for x in obs_j) else np.array([np.nan])
        out['quiet'] = {
            'starts': len(qo),
            'sim_gt5': float((qd > 0.05).mean()), 'obs_gt5': float((qo > 0.05).mean()),
            'sim_gt10': float((qd > 0.10).mean()), 'obs_gt10': float((qo > 0.10).mean()),
            'sim_jump_rate': float(len(sj) / (len(qd) * 72)), 'obs_jump_rate': float(len(oj) / (len(qo) * 72)),
            'sim_jump_p50_p90': np.round(np.nanpercentile(sj, [50, 90]) * 100, 1).tolist(),
            'obs_jump_p50_p90': np.round(np.nanpercentile(oj, [50, 90]) * 100, 1).tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--locations', default=','.join(SL.LOCATIONS))
    ap.add_argument('--params', default=str(ROOT / 'models'), help='directory with step_layer_<loc>.json')
    ap.add_argument('--compare', nargs='*', default=[], help='more parameter directories to score')
    ap.add_argument('--draws', type=int, default=50)
    ap.add_argument('--with-error', action='store_true', help='score full outcomes (smooth + borrowed model error)')
    args = ap.parse_args()
    levels = F.hourly_levels()
    for loc in args.locations.split(','):
        b = starts_and_forecasts(loc, levels)
        print(f'\n== {loc}: {len(b["t0"])} starts')
        errors = borrowed_errors(loc, b, args.draws) if args.with_error else None
        for d in [args.params] + args.compare:
            p = SL.StepParams.load(SL.params_path(Path(d), loc))
            r = run(loc, p, b, args.draws, errors=errors)
            label = 'smooth outcomes CRPS' if args.with_error else 'smooth MAE'
            print(f'  [{d}] CRPS 1-48 h {r["crps48"]:.2f} ({label} {r["mae48"]:.2f}), '
                  f'1-72 h {r["crps72"]:.2f} (smooth {r["mae72"]:.2f}) cm')
            if r['brier']:
                print('     Brier ' + ', '.join(f'{k} {v:.4f}' for k, v in r['brier'].items()))
            if 'quiet' in r:
                print(f'     quiet: {r["quiet"]}')


if __name__ == '__main__':
    main()
