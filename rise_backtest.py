"""
Rises in the 10-day forecast: how high, and when, with the rain that fell and with the rain that was forecast.

Starts every day at 00 UTC over 2023 and Nov 2024 - Jan 2026 (the periods the hybrid never trained on). Future rain:

  observed   the rain that fell (mean of the location's stations), as in every earlier backtest
  ifs        what ECMWF IFS forecast for each station from that day's run, 0-7 days ahead
             (data/verification/ifs_runs/, fetch_ifs_archive.py); days 8-10 the month's average hourly rain

Rises: every peak in the observed differential at least RISE_MIN above the lowest point in the 5 days before it
(peak = highest point within 3 days either side, on a 7 h running median). For each rise and each start 6 h to
7 days before its peak, the forecast's highest point within 3 days of the observed peak gives the height error
(forecast - observed, m) and the time error (forecast - observed, h); the first time it reaches halfway up the
rise gives the onset error.

    python rise_backtest.py                 # all three locations, both rain sources, the deployed hybrid
    python rise_backtest.py --models ctl=hybrid_ctl_s0+hybrid_ctl_s1+hybrid_ctl_s2,peak=hybrid_pk1_s0+...
    python rise_backtest.py --models 'srh=hybrid_srh+hybrid_srh_s1@soil_rise_han,...'   # extra-input models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from flag_predictor.models.physics_data import HORIZON, WindowSampler, load_location_data  # noqa: E402

IFS_DIR = ROOT / 'data' / 'verification' / 'ifs_runs'
PERIODS = {'2023': ('2023-01-01', '2023-12-21'), 'test': ('2024-11-01', '2026-01-10')}
RISE_MIN = 0.12
HYBRID = ['hybrid_c1_ps'] + [f'hybrid_c1_ps_s{i}' for i in range(1, 5)]


def daily_starts(data) -> np.ndarray:
    """00 UTC starts with a full encoder window and at least 200 observed hours ahead."""
    from build_error_library import start_positions
    st = start_positions(data, stride_h=24)
    return np.concatenate([st['val2023'], st['test']])


def station_key(name: str) -> str:
    return name.split('-')[0].split('_')[0]


def load_ifs() -> pd.DataFrame | None:
    files = sorted(IFS_DIR.glob('*.csv'))
    if not files:
        return None
    df = pd.concat(pd.read_csv(f, parse_dates=['time']) for f in files)
    df['station'] = df.station.map(station_key)
    return df


def ifs_rain(data, t_idx: np.ndarray, ifs: pd.DataFrame, stations: list) -> np.ndarray:
    """(n, HORIZON) mean-station forecast rain (mm/h) from the run issued at each start (00 UTC)."""
    keys = [station_key(s) for s in stations]
    sub = ifs[ifs.station.isin(keys)]
    cols = [f'day{k}' for k in range(8)]
    mean = sub.groupby('time')[cols].mean()                  # mean over stations, per valid hour and lead day
    idx = data.index.tz_convert('UTC').tz_localize(None)
    clim = pd.Series(data.rain, idx).groupby(idx.month).mean()
    out = np.zeros((len(t_idx), HORIZON), np.float32)
    for i, t in enumerate(t_idx):
        t0 = idx[t]
        hrs = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='h')
        k = ((hrs.normalize() - t0.normalize()) / pd.Timedelta(days=1)).astype(int)
        m = mean.reindex(hrs)
        v = np.array([m[f'day{kk}'].iloc[j] if kk <= 7 else np.nan for j, kk in enumerate(k)], dtype=float)
        if np.isfinite(v[k <= 7]).mean() < 0.95:           # no archived run for this start
            out[i] = np.nan
            continue
        v = np.where(np.isfinite(v), v, clim.reindex(hrs.month).to_numpy())
        out[i] = v
    return out


@torch.no_grad()
def run_hybrid(loc: str, data, t_idx: np.ndarray, rain: np.ndarray | None = None, names=HYBRID) -> np.ndarray:
    """Five-seed mean forecast; ``rain`` (n, HORIZON) replaces the rain that fell."""
    from flag_predictor.models.physics_train import load_candidate
    preds = []
    for n in names:
        model, scaler, _ = load_candidate(n, loc, ROOT / 'models')
        s = WindowSampler(data, scaler, torch.device('cpu'))
        out = []
        for i in range(0, len(t_idx), 256):
            b = s.batch(t_idx[i:i + 256])
            r = b['rain'] if rain is None else torch.as_tensor(rain[i:i + 256], dtype=b['rain'].dtype)
            out.append(model(b['x'], r, b['season'], b['rain24'], b['rain168'], b['d0']).numpy())
        preds.append(np.concatenate(out))
    return np.mean(preds, axis=0)


def observed_series(data) -> pd.Series:
    idx = data.index.tz_convert('UTC').tz_localize(None)
    return pd.Series(data.diff, idx)


def find_rises(d: pd.Series, lo: str, hi: str, rise_min: float = RISE_MIN) -> pd.DataFrame:
    d = d[pd.Timestamp(lo) - pd.Timedelta(days=6):pd.Timestamp(hi) + pd.Timedelta(days=12)]
    sm = d.interpolate(limit=6).rolling(7, center=True, min_periods=3).median().to_numpy()
    ev = []
    for t in range(120, len(sm) - 72):
        if not np.isfinite(sm[t]):
            continue
        win = sm[t - 72:t + 73]
        if np.nanargmax(win) != 72:
            continue
        pre = sm[t - 120:t + 1]
        j = int(np.nanargmin(pre))
        if sm[t] - pre[j] >= rise_min:
            half = pre[j] + 0.5 * (sm[t] - pre[j])
            k = t - 120 + j + int(np.argmax(sm[t - 120 + j:t + 1] >= half))
            ev.append({'t_on': d.index[t - 120 + j], 't_half': d.index[k], 't_pk': d.index[t],
                       'd_on': float(pre[j]), 'd_pk': float(sm[t]), 'rise': float(sm[t] - pre[j])})
    e = pd.DataFrame(ev)
    return e[(e.t_pk >= pd.Timestamp(lo)) & (e.t_pk < pd.Timestamp(hi))].reset_index(drop=True) if len(e) else e


def score_rises(t0s: pd.DatetimeIndex, pred: np.ndarray, rises: pd.DataFrame, d: pd.Series) -> pd.DataFrame:
    rows = []
    for k, r in rises.iterrows():
        for i, t0 in enumerate(t0s):
            lead = (r.t_pk - t0) / pd.Timedelta(hours=1)
            if not (6 <= lead <= 168):
                continue
            hrs = t0 + pd.to_timedelta(np.arange(1, HORIZON + 1), unit='h')
            w = np.asarray((hrs >= r.t_pk - pd.Timedelta(hours=72)) & (hrs <= r.t_pk + pd.Timedelta(hours=72)))
            if w.sum() < 48:
                continue
            p = pred[i]
            j = int(np.argmax(np.where(w, p, -9)))
            thr = r.d_on + 0.5 * r.rise
            up = np.where(w & (p >= thr))[0]
            d0 = float(d.get(t0, np.nan))
            rows.append({'rise_id': k, 't0': t0, 'lead_to_peak': lead, 'rise': r.rise, 'before_onset': t0 < r.t_on,
                         'height_err': float(p[j] - r.d_pk),
                         'share_of_rise': float((p[j] - d0) / (r.d_pk - d0)) if r.d_pk - d0 > 0.05 else np.nan,
                         'time_err': float((hrs[j] - r.t_pk) / pd.Timedelta(hours=1)),
                         'onset_err': float((hrs[up[0]] - r.t_half) / pd.Timedelta(hours=1)) if len(up) and t0 < r.t_half else np.nan,
                         'missed_half': bool(t0 < r.t_half and not len(up))})
    return pd.DataFrame(rows)


def summarise(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s['lead'] = pd.cut(s.lead_to_peak, [0, 24, 48, 96, 168], labels=['6-24 h', '1-2 d', '2-4 d', '4-7 d'])
    return s.groupby('lead', observed=True).agg(
        n=('height_err', 'size'), height_err_median=('height_err', 'median'), height_err_mae=('height_err', lambda x: x.abs().mean()),
        time_err_median=('time_err', 'median'), time_err_mae=('time_err', lambda x: x.abs().mean()),
        onset_err_median=('onset_err', 'median'), onset_err_mae=('onset_err', lambda x: x.abs().mean()),
        missed_half=('missed_half', 'mean'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--locations', default='isis,godstow,wallingford')
    ap.add_argument('--models', default='deployed=' + '+'.join(HYBRID), help='label=name+name,...')
    a = ap.parse_args()
    models = dict(m.split('=') for m in a.models.split(','))
    torch.set_num_threads(2)
    ifs = load_ifs()
    out = {}
    for loc in a.locations.split(','):
        data = load_location_data(loc, ROOT)
        from flag_predictor.evaluation import load_merged, rain_station_columns
        stations = rain_station_columns(load_merged(loc, ROOT))
        t_idx = daily_starts(data)
        t0s = data.index[t_idx].tz_convert('UTC').tz_localize(None)
        d = observed_series(data)
        rises = pd.concat([find_rises(d, *PERIODS[p]) for p in PERIODS], ignore_index=True)
        obs = np.stack([data.diff[t + 1:t + 1 + HORIZON] for t in t_idx])
        sources = {'observed': None}
        if ifs is not None:
            sources['ifs'] = ifs_rain(data, t_idx, ifs, stations)
        for name, rain in sources.items():
            ok = np.ones(len(t_idx), bool) if rain is None else np.isfinite(rain).all(1)
            preds = {}
            for label, spec in models.items():           # 'names' or 'names@inputs' (extra inputs, train_extra_models.py)
                names, _, inputs = spec.partition('@')
                dd = data
                if inputs:
                    from train_extra_models import data_with_extra
                    dd = data_with_extra(loc, inputs)
                preds[label] = run_hybrid(loc, dd, t_idx[ok], None if rain is None else rain[ok], names.split('+'))
            same = np.all([np.isfinite(p).all(1) for p in preds.values()], axis=0)   # starts every model could run
            idx_ok = np.where(ok)[0][same]
            per = np.where(t0s[idx_ok] >= pd.Timestamp('2024-06-01'), 'test', '2023')
            for label, pred in preds.items():
                pred = pred[same]
                s = score_rises(t0s[idx_ok], pred, rises, d)
                s['period'] = np.where(s.t0 >= pd.Timestamp('2024-06-01'), 'test', '2023')
                err = np.abs(pred - obs[idx_ok])
                for pname in np.unique(per):
                    m = per == pname
                    print(f'\n== {loc}, {label}, {name} rain, {pname}: {m.sum()} starts, MAE {np.nanmean(err[m]):.4f} m '
                          f'(1-24 h {np.nanmean(err[m, :24]):.4f}, 2-3 d {np.nanmean(err[m, 24:72]):.4f}, '
                          f'4-7 d {np.nanmean(err[m, 72:168]):.4f})')
                    with pd.option_context('display.width', 200):
                        print(summarise(s[s.period == pname]).round(3).to_string())
                out[(loc, label, name)] = s
                out[(loc, label, name, 'mae')] = pd.DataFrame({'t0': t0s[idx_ok], 'mae': np.nanmean(err, 1)})
    pd.to_pickle(out, ROOT / 'figures' / 'eval' / 'rise_backtest.pkl')


if __name__ == '__main__':
    main()
