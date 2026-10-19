#!/usr/bin/env python
"""
Fit the lock-keeper step layer (src/flag_predictor/prediction/step_layer.py).

Data: the 15-minute differential record (data/<location>_flag_model_data.csv)
turned into hourly values the way the live site builds them (last reading in
each hour), the hourly mean-gauge rain, and the deployed hybrid's forecasts
(with the rain that fell) from a start every 2 hours. Fitted on the training
years only (before 2024-10-21, without 2023), like the forecast models; 2023
and the Nov 2024 - Jan 2026 test window are only used for the printed checks.

Each forecast hour is one row, starting from the level the river was really at
the hour before (teacher forcing): did it jump up / down this hour, given how
far the forecast from the start says the river still has to go?

    python fit_step_layer.py            # all three locations -> models/step_layer_<loc>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.prediction.step_layer import (  # noqa: E402
    DAY_END_H, DAY_START_H, JUMP_THRESHOLD, LEADS, LOCAL_TZ, LOCATIONS, SINCE_CAP_H, StepParams,
    hazard_design, hourly_jumps, params_path,
)

TRAIN_END = pd.Timestamp('2024-10-21', tz='UTC')
VAL_YEAR = 2023
TEST = (pd.Timestamp('2024-11-01', tz='UTC'), pd.Timestamp('2026-01-10', tz='UTC'))
HYBRID = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]
H_FIT = 72
MAX_LEAD = 96


def load_15min(location: str) -> pd.Series:
    d = pd.read_csv(ROOT / f'data/{location}_flag_model_data.csv', parse_dates=['timestamp'], index_col='timestamp')
    s = d.iloc[:, 0].astype(float)
    s = s[~s.index.duplicated()].sort_index()
    if s.index.tz is None:
        s = s.tz_localize('UTC')
    g = s.reindex(pd.date_range(s.index.min().ceil('15min'), s.index.max(), freq='15min'))
    med = g.rolling(9, center=True, min_periods=5).median()
    return g.where((g - med).abs() < 0.25)          # drop single-reading glitches


def hourly_levels() -> dict:
    return {loc: load_15min(loc).resample('1h').last() for loc in LOCATIONS}


def seen_at_start(level: np.ndarray, thr: float, sign: int) -> np.ndarray:
    """(T,) 1 if, with data up to hour t only, a jump this way shows in hours t-2..t (as live)."""
    s = pd.Series(level)
    d1 = s.diff()
    before = s.rolling(3, min_periods=2).median().shift(1)
    afters = {1: s, 2: pd.concat([s, s.shift(-1)], axis=1).median(axis=1), 3: s[::-1].rolling(3, min_periods=2).median()[::-1]}
    flag = {}
    for k, after in afters.items():
        pj = after - before
        flag[k] = ((d1.abs() >= thr) & (pj.abs() >= 0.7 * thr) & (np.sign(pj) == np.sign(d1)) & (sign * pj > 0)).to_numpy()
    out = flag[1] | np.r_[False, flag[2][:-1]] | np.r_[False, False, flag[3][:-2]]
    return out.astype(float)


def forecasts(location: str, data, starts: np.ndarray) -> np.ndarray:
    from build_error_library import predict_hybrid
    return predict_hybrid(HYBRID, location, data, starts)[:, :MAX_LEAD]


def build_rows(location: str, levels: dict, data, period: str, stride: int) -> dict:
    """Teacher-forced rows (one per start and forecast hour 1..H_FIT)."""
    from flag_predictor.models.physics_data import HORIZON, SEQ_LEN
    D_ser = levels[location]
    grid = pd.date_range(max(D_ser.index.min(), data.index.min()), min(D_ser.index.max(), data.index.max()), freq='1h')
    D = D_ser.reindex(grid).to_numpy()
    J = hourly_jumps(D, JUMP_THRESHOLD[location])
    rain = pd.Series(data.rain, data.index).reindex(grid).fillna(0.0).to_numpy()
    crain = np.r_[0.0, np.cumsum(rain)]
    T = len(grid)
    local = grid.tz_convert(LOCAL_TZ)
    lhour, ldow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    since = {}
    for sign in (1, -1):
        ev = np.where(sign * J > 0, np.arange(T), -10 ** 6)
        last = np.r_[-10 ** 6, np.maximum.accumulate(ev)[:-1]]
        since[sign] = np.minimum(np.arange(T) - last, SINCE_CAP_H).astype(float)
    trig, own = {}, {}
    for sign in (1, -1):
        own[sign] = seen_at_start(D, JUMP_THRESHOLD[location], sign)
        others = [seen_at_start(levels[o].reindex(grid).to_numpy(), JUMP_THRESHOLD[o], sign)
                  for o in LOCATIONS if o != location]
        trig[sign] = np.maximum.reduce(others)

    # forecast starts: positions in data.index with a full encoder window, inside the period
    enc_ok = np.isfinite(data.enc).all(axis=1)
    ok_count = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[: len(enc_ok)]
    idx = data.index
    cand = np.arange(SEQ_LEN, len(idx) - HORIZON - 1)
    tt = idx[cand]
    if period == 'train':
        inp = (tt.year != VAL_YEAR) & (tt < TRAIN_END)
    elif period == 'val':
        inp = tt.year == VAL_YEAR
    else:
        inp = (tt >= TEST[0]) & (tt < TEST[1])
    cand = cand[inp & (tt.hour % stride == 0) & (ok_count[cand] == SEQ_LEN) & np.isfinite(data.diff[cand])]
    gpos = grid.get_indexer(idx[cand])
    keep = (gpos >= 30) & (gpos + H_FIT + 1 < T)
    cand, gpos = cand[keep], gpos[keep]
    keep = np.isfinite(D[gpos])
    cand, gpos = cand[keep], gpos[keep]
    pred = forecasts(location, data, cand)

    h = np.arange(1, H_FIT + 1)
    lead_idx = np.array(LEADS)
    cols = {k: [] for k in ('pend', 'trend', 'r24', 'su', 'sd', 'lh', 'dw', 'h', 'tu', 'td', 'ou', 'od', 'J', 'dD')}
    for i, p0 in enumerate(gpos):
        d0 = D[p0]
        m = np.r_[d0, pred[i] + (d0 - data.diff[cand[i]])]          # forecast from the start, leads 0..96
        cur = D[p0 + h - 1]
        ok = np.isfinite(cur) & np.isfinite(D[p0 + h]) & np.isfinite(D[p0 + h - 25])
        if ok.mean() < 0.7:
            continue
        hh = h[ok]
        tgt = m[np.minimum(hh[:, None] + lead_idx[None, :], MAX_LEAD)]
        cols['pend'].append(tgt - cur[ok][:, None])
        cols['trend'].append(cur[ok] - D[p0 + hh - 25])
        cols['r24'].append(crain[p0 + hh] - crain[p0 + hh - 24])
        cols['su'].append(since[1][p0 + hh]); cols['sd'].append(since[-1][p0 + hh])
        cols['lh'].append(lhour[p0 + hh]); cols['dw'].append(ldow[p0 + hh]); cols['h'].append(hh)
        n = len(hh)
        cols['tu'].append(np.full(n, trig[1][p0])); cols['td'].append(np.full(n, trig[-1][p0]))
        cols['ou'].append(np.full(n, own[1][p0])); cols['od'].append(np.full(n, own[-1][p0]))
        cols['J'].append(J[p0 + hh]); cols['dD'].append(D[p0 + hh] - cur[ok])
    return {k: np.concatenate(v) for k, v in cols.items()}


def design_for(r: dict, direction: str) -> np.ndarray:
    s = 1 if direction == 'up' else -1
    return hazard_design(np.maximum(s * r['pend'], 0), np.maximum(r['trend'], 0), np.maximum(-r['trend'], 0),
                         r['r24'], r['su'] if s > 0 else r['sd'], r['lh'], r['dw'], r['h'],
                         r['tu'] if s > 0 else r['td'], r['ou'] if s > 0 else r['od'])


def fit_location(location: str, levels: dict, verbose: bool = True) -> StepParams:
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import log_loss, roc_auc_score
    from flag_predictor.models.physics_data import load_location_data
    data = load_location_data(location, ROOT)
    tr = build_rows(location, levels, data, 'train', stride=2)
    checks = {p: build_rows(location, levels, data, p, stride=6) for p in ('val', 'test')}
    coef, intercept, size_w, size_b, size_q, notes = {}, {}, {}, {}, {}, {'training_rows': int(len(tr['J']))}
    for d, sign in (('up', 1), ('down', -1)):
        y = (sign * tr['J'] > 0).astype(int)
        m = LogisticRegression(C=1.0, max_iter=5000).fit(design_for(tr, d), y)
        coef[d], intercept[d] = m.coef_[0], float(m.intercept_[0])
        for p, rows in checks.items():
            yy = (sign * rows['J'] > 0).astype(int)
            pp = m.predict_proba(design_for(rows, d))[:, 1]
            notes[f'hazard_{d}_{p}'] = {
                'log_loss': round(float(log_loss(yy, pp)), 4),
                'log_loss_constant_rate': round(float(log_loss(yy, np.full(len(pp), y.mean()))), 4),
                'auc': round(float(roc_auc_score(yy, pp)), 3)}
        sel = y == 1
        pend = np.maximum(sign * tr['pend'][sel], 0)
        size = np.abs(tr['J'][sel])
        lr = LinearRegression(positive=True).fit(pend, size)
        fitted = np.maximum(lr.predict(pend), 0.005)
        size_w[d], size_b[d] = lr.coef_, float(lr.intercept_)
        size_q[d] = np.quantile(np.clip(size / fitted, 0.1, 4.0), np.linspace(0, 1, 21))
        notes[f'jumps_{d}'] = int(sel.sum())
    # Drift in hours without a jump: ridge (the pending moves are strongly correlated).
    no_jump = tr['J'] == 0
    day = (tr['lh'] >= DAY_START_H) & (tr['lh'] < DAY_END_H)
    drift = {}
    for name, mask in (('day', day), ('night', ~day)):
        sel = no_jump & mask
        rg = Ridge(alpha=float(sel.sum()) * 1e-4, fit_intercept=False).fit(tr['pend'][sel] / 0.1, tr['dD'][sel])
        drift[name] = rg.coef_ / 0.1
    params = StepParams(
        location=location, jump_threshold=JUMP_THRESHOLD[location], coef=coef, intercept=intercept,
        size_w=size_w, size_b=size_b, size_q=size_q, drift=drift,
        fitted_on=f"hybrid forecasts from starts before {TRAIN_END:%Y-%m-%d} (15-minute record from 2019), without {VAL_YEAR}",
        notes=notes)
    if verbose:
        print(f"{location}: {notes['training_rows']} rows; drift day {np.round(drift['day'], 3)} night {np.round(drift['night'], 3)}")
        for d in ('up', 'down'):
            print(f"   size {d}: {size_b[d]:.3f} + {np.round(size_w[d], 3)} . pending, factor q10/50/90 "
                  f"{np.round(np.interp([.1, .5, .9], np.linspace(0, 1, 21), size_q[d]), 2)}")
        for k, v in notes.items():
            if k.startswith('hazard'):
                print(f'   {k}: {v}')
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--locations', default=','.join(LOCATIONS))
    ap.add_argument('--out-dir', default=str(ROOT / 'models'))
    args = ap.parse_args()
    levels = hourly_levels()
    for loc in args.locations.split(','):
        p = fit_location(loc, levels)
        p.save(params_path(Path(args.out_dir), loc))


if __name__ == '__main__':
    main()
