#!/usr/bin/env python
"""
Fit the Black-flag model (src/flag_predictor/prediction/black_flag.py): the Iffley Lock upstream
level 1-48 h ahead, one linear model per lead, on hours with Iffley above FIT_ABOVE_M.

Data: data/flags/iffley_upstream_hourly.csv.gz (EA hydrology API, 1501TH), the Isis and Godstow
differentials (fit_official_flag.load_differential) and the hybrid's mean-station rain
(physics_data.load_location_data('isis')).

Check: fitted before 2023, tested from 2023 to January 2026 with the rain that fell (starts
every 3 h, every level): Brier of "Iffley reaches 0.42 m (Red -> Black) / falls below 0.22 m (Black -> Red)
within 24 h / 48 h" against the test-period rate, and the same against the official flag.
Also the same with forecast differentials in place of the observed ones (no gain), which is
why the model uses only the differentials now.

    python fit_black_flag.py          # -> models/black_flag.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.data.ourcs_flags import flag_series  # noqa: E402
from flag_predictor.prediction.black_flag import (  # noqa: E402
    FIT_ABOVE_M, FULL, HORIZON, PARAMS_FILE, SHORT, BlackParams, black_state, simulate,
)
from flag_predictor.prediction.official_flag import BLACK_DOWN_M, BLACK_UP_M  # noqa: E402
from fit_official_flag import load_differential  # noqa: E402

TEST_FROM = pd.Timestamp('2023-01-01', tz='UTC')


def hourly_frame() -> pd.DataFrame:
    from flag_predictor.models.physics_data import load_location_data
    iu = pd.read_csv(ROOT / 'data' / 'flags' / 'iffley_upstream_hourly.csv.gz',
                     parse_dates=['timestamp']).set_index('timestamp')['iffley_upstream_m']
    data = load_location_data('isis', ROOT)
    rain = pd.Series(data.rain, data.index)
    df = pd.DataFrame({'u': iu, 'I': load_differential('isis').resample('1h').mean(),
                       'G': load_differential('godstow').resample('1h').mean(), 'r': rain}).asfreq('1h')
    return df[df.index <= rain.index.max()]


def lead_table(df: pd.DataFrame, h: int) -> pd.DataFrame:
    cs = df.r.fillna(0).cumsum()
    X = pd.DataFrame({'u0': df.u, 'du6': df.u - df.u.shift(6), 'du24': df.u - df.u.shift(24), 'I0': df.I, 'G0': df.G,
                      'rp24': cs - cs.shift(24), 'rp72': cs - cs.shift(72), 'rf': cs.shift(-h) - cs,
                      'y': df.u.shift(-h) - df.u})
    return X[X.u0 > FIT_ABOVE_M]


def fit(df: pd.DataFrame, before=None) -> BlackParams:
    from sklearn.linear_model import LinearRegression
    models = {}
    for name, cols in (('full', FULL), ('short', SHORT)):
        coef, inter, sd = [], [], []
        for h in range(1, HORIZON + 1):
            X = lead_table(df, h).dropna(subset=cols + ['y'])
            if before is not None:
                X = X[X.index < before]
            m = LinearRegression().fit(X[cols], X.y)
            coef.append(m.coef_); inter.append(m.intercept_)
            sd.append(float((X.y - m.predict(X[cols])).std()))
        models[name] = {'features': cols, 'coef': np.array(coef), 'intercept': np.array(inter), 'sd': np.array(sd)}
    return BlackParams(models, {})


def check(df: pd.DataFrame, params: BlackParams) -> dict:
    """Out-of-sample Brier, rain that fell, starts every 3 h from TEST_FROM."""
    hist = pd.read_csv(ROOT / 'data' / 'flags' / 'history_isis.csv', parse_dates=['start', 'end'])
    rng = np.random.default_rng(0)
    cs = df.r.fillna(0).cumsum()
    rows = []
    for t0 in df.index[(df.index >= TEST_FROM) & (df.index.hour % 3 == 0)]:
        u = df.u.loc[:t0]
        if not np.isfinite(df.u.get(t0, np.nan)):
            continue
        black0 = black_state(u.iloc[-24 * 30:])
        u0 = float(df.u[t0])
        fut = df.u.loc[t0 + pd.Timedelta(hours=1):t0 + pd.Timedelta(hours=HORIZON)].to_numpy()
        if len(fut) < HORIZON or np.isnan(fut).mean() > 0.1:
            continue
        lag = lambda k: df.u.get(t0 - pd.Timedelta(hours=k), np.nan)
        f = {'u0': u0, 'du6': u0 - lag(6), 'du24': u0 - lag(24), 'I0': df.I.get(t0, np.nan), 'G0': df.G.get(t0, np.nan),
             'rp24': float(cs[t0] - cs.get(t0 - pd.Timedelta(hours=24), np.nan)),
             'rp72': float(cs[t0] - cs.get(t0 - pd.Timedelta(hours=72), np.nan))}
        if not np.isfinite([f['du6'], f['du24'], f['rp24'], f['rp72']]).all():
            continue
        rfut = df.r.loc[t0 + pd.Timedelta(hours=1):t0 + pd.Timedelta(hours=HORIZON)].fillna(0).to_numpy()[None, :]
        paths = simulate(params, f, rfut, rng=rng, n=200)
        hit = paths[:, 1:] < BLACK_DOWN_M if black0 else paths[:, 1:] >= BLACK_UP_M
        obs = pd.Series(fut).ffill().to_numpy()
        ohit = obs < BLACK_DOWN_M if black0 else obs >= BLACK_UP_M
        grid = pd.date_range(t0, periods=HORIZON + 1, freq='1h')
        fl = flag_series(hist, grid)
        row = {'t0': t0, 'black0': black0, 'flag0': fl.iloc[0]}
        for w in (24, 48):
            row[f'p{w}'] = float(hit[:, :w].any(axis=1).mean())
            row[f'o{w}'] = float(ohit[:w].any())
            if isinstance(fl.iloc[0], str) and fl.iloc[0] in ('red', 'black'):
                seg = fl.iloc[1:w + 1]
                row[f'f{w}'] = float((seg == ('red' if fl.iloc[0] == 'black' else 'black')).any()) if seg.notna().all() else np.nan
        rows.append(row)
    r = pd.DataFrame(rows)
    out = {}
    for state, sub in (('red_to_black', r[~r.black0]), ('black_to_red', r[r.black0])):
        for w in (24, 48):
            o = sub[f'o{w}']
            out[f'{state}_{w}h'] = {'starts': int(len(sub)), 'events': int(o.sum()),
                                    'brier': round(float(np.mean((sub[f'p{w}'] - o) ** 2)), 4),
                                    'brier_rate': round(float(np.mean((o.mean() - o) ** 2)), 4)}
            if f'f{w}' in sub:
                s = sub[sub[f'f{w}'].notna() & (sub.flag0 == ('black' if state == 'black_to_red' else 'red'))]
                if len(s):
                    out[f'{state}_{w}h_official_flag'] = {
                        'starts': int(len(s)), 'events': int(s[f'f{w}'].sum()),
                        'brier': round(float(np.mean((s[f'p{w}'] - s[f'f{w}']) ** 2)), 4),
                        'brier_rate': round(float(np.mean((s[f'f{w}'].mean() - s[f'f{w}']) ** 2)), 4)}
    reds = r[~r.black0]
    for w in (24, 48):
        c = pd.cut(reds[f'p{w}'], [0, 0.05, 0.2, 0.5, 1.01], right=False)
        t = reds.groupby(c, observed=True)[f'o{w}'].agg(['mean', 'count'])
        out[f'reliability_red_to_black_{w}h'] = {f'{iv.left:.2f}-{iv.right:.2f}': [round(float(x['mean']), 2), int(x['count'])]
                                                 for iv, x in t.iterrows()}
    return out


def main() -> None:
    df = hourly_frame()
    notes = check(df, fit(df, before=TEST_FROM))
    params = fit(df)
    params.notes = {'test_from_2023_rain_that_fell': notes, 'fitted_to': str(df.index.max())}
    params.save(ROOT / 'models' / PARAMS_FILE)
    for k, v in notes.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
