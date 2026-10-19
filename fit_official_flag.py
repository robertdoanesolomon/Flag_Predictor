#!/usr/bin/env python
"""
Fit the official-flag model (src/flag_predictor/prediction/official_flag.py).

Data: the OURCs flag history of each river (data/flags/history_<river>.csv;
--refresh downloads it again from ourcs.co.uk) and the 15-minute differential
record (data/<river>_flag_model_data.csv, extended to October 2026 with
data/flags/differential_after_csv_<river>.csv.gz from flags.jamesonlee.com, the
same source), as hourly values. One hourly logistic
hazard for a change up and one for a change down, pooled over the Isis (from
November 2021, when the Light / Dark Blue flags started) and Godstow (from
September 2024, since when the flag has been set from this differential), with
a river term. Grey (not monitored) hours are left out.

Check: leave-one-season-out log loss against a constant rate.

    python fit_official_flag.py [--refresh]       # -> models/official_flag.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.data.ourcs_flags import fetch_history, flag_series  # noqa: E402
from flag_predictor.prediction.official_flag import (  # noqa: E402
    LOCAL_TZ, PARAMS_FILE, RUN_CAP_H, FlagParams, band_rank, design, lines, order,
)
from fit_step_layer import load_15min  # noqa: E402

# Isis: since the Light / Dark Blue flags started. Godstow: the flag has been set from this
# differential only since September 2024.
START = {'isis': pd.Timestamp('2021-11-01', tz='UTC'), 'godstow': pd.Timestamp('2024-09-01', tz='UTC')}


def history(river: str, refresh: bool) -> pd.DataFrame:
    path = ROOT / 'data' / 'flags' / f'history_{river}.csv'
    if refresh or not path.exists():
        h = fetch_history(river)
        if h is None:
            raise RuntimeError(f'could not download the {river} flag history')
        path.parent.mkdir(parents=True, exist_ok=True)
        h.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=['start', 'end'])


def load_differential(river: str) -> pd.Series:
    """15-minute differential, the training CSV plus its 2026 extension, glitches removed."""
    g = load_15min(river)
    ext_path = ROOT / 'data' / 'flags' / f'differential_after_csv_{river}.csv.gz'
    if ext_path.exists():
        e = pd.read_csv(ext_path, parse_dates=['timestamp']).set_index('timestamp')['differential'].astype(float)
        e = e[e.index > g.index.max()]
        e = e[~e.index.duplicated()].sort_index()
        e = e.reindex(pd.date_range(e.index.min().ceil('15min'), e.index.max(), freq='15min'))
        med = e.rolling(9, center=True, min_periods=5).median()
        g = pd.concat([g, e.where((e - med).abs() < 0.25)])
    return g


def rows(river: str, h: pd.DataFrame, exclude=None) -> dict:
    D = load_differential(river).resample('1h').last()
    D = D[D.index >= START[river]]
    if exclude is not None:
        D = D[(D.index < exclude[0]) | (D.index >= exclude[1])]
    names = order(river)
    rank = {k: i for i, k in enumerate(names)}
    F = flag_series(h, D.index).map(rank)
    starts = pd.DatetimeIndex(h['start'].sort_values())
    k = np.maximum(starts.searchsorted(D.index, side='right') - 1, 0)
    since = ((D.index - starts[k]).total_seconds() / 3600.0).to_numpy()
    L = lines(river)
    top = len(L) - 1
    d = D.ffill(limit=3).to_numpy()
    r = F.to_numpy(dtype=float)
    nxt = np.r_[r[1:], np.nan]
    ok = np.isfinite(r) & np.isfinite(d) & np.isfinite(nxt)
    rr = np.where(np.isfinite(r), r, 0).astype(int)
    lt = D.index.tz_convert(LOCAL_TZ)
    trend = d - np.r_[np.full(6, np.nan), d[:-6]]
    out = {}
    for direction in ('up', 'down'):
        if direction == 'up':
            margin = d - L[np.minimum(rr + 1, top)]
            valid = ok & (rr < top)
            y = nxt > r
        else:
            margin = L[rr] - d
            valid = ok & (rr > 0)
            y = nxt < r
        run = np.zeros(len(d))
        c = 0
        for i, past in enumerate(margin > 0):
            c = c + 1 if past else 0
            run[i] = min(c, RUN_CAP_H)
        X = design(margin, run, lt.hour.to_numpy(), lt.dayofweek.to_numpy(), since, trend, river)
        out[direction] = (X[valid], y[valid].astype(int), D.index[valid])
    return out


def check_black(h: pd.DataFrame) -> dict:
    """How well the Iffley-upstream hysteresis rule (official_flag.BLACK_UP_M / BLACK_DOWN_M)
    matches the official Black flag in Red/Black hours (data/flags/iffley_upstream_hourly.csv.gz,
    EA hydrology API, station 1501TH)."""
    from flag_predictor.prediction.official_flag import BLACK_DOWN_M, BLACK_UP_M
    path = ROOT / 'data' / 'flags' / 'iffley_upstream_hourly.csv.gz'
    if not path.exists():
        return {}
    iu = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')['iffley_upstream_m']
    f = flag_series(h, iu.index)
    df = pd.DataFrame({'iu': iu, 'f': f}).dropna()
    df = df[df.f.isin(['red', 'black'])]
    state, out = False, []
    for v in df.iu.to_numpy():
        if not state and v >= BLACK_UP_M:
            state = True
        elif state and v < BLACK_DOWN_M:
            state = False
        out.append(state)
    out = np.array(out)
    black = (df.f == 'black').to_numpy()
    return {'red_black_hours': int(len(df)), 'agreement': round(float(np.mean(out == black)), 3),
            'black_hours_caught': round(float(np.mean(out[black])), 3), 'false_black': round(float(np.mean(out[~black])), 3)}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refresh', action='store_true', help='download the OURCs flag histories again')
    ap.add_argument('--out-dir', default=str(ROOT / 'models'))
    args = ap.parse_args()
    per = {river: rows(river, history(river, args.refresh)) for river in ('isis', 'godstow')}
    coef, intercept, notes = {}, {}, {}
    for direction in ('up', 'down'):
        X = np.concatenate([per[r][direction][0] for r in per])
        y = np.concatenate([per[r][direction][1] for r in per])
        ix = pd.DatetimeIndex(np.concatenate([per[r][direction][2] for r in per]))
        river = np.concatenate([[r] * len(per[r][direction][1]) for r in per])
        season = (ix.year - (ix.month < 8)).to_numpy()
        p = np.full(len(y), np.nan)
        for s in np.unique(season):
            te = season == s
            if y[~te].sum() >= 5 and te.any():
                p[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        for r in per:
            k = (river == r) & np.isfinite(p)
            notes[f'{direction}_{r}'] = {
                'changes': int(y[river == r].sum()), 'hours': int((river == r).sum()),
                'cv_log_loss': round(float(log_loss(y[k], p[k])), 4),
                'cv_log_loss_constant_rate': round(float(log_loss(y[k], np.full(k.sum(), y[k].mean()))), 4)}
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        coef[direction], intercept[direction] = m.coef_[0], float(m.intercept_[0])
    notes['black_rule_isis'] = check_black(history('isis', False))
    params = FlagParams(coef, intercept, notes)
    params.save(Path(args.out_dir) / PARAMS_FILE)
    for k, v in notes.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
