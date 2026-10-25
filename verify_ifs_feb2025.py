"""
Verification only: Feb 2025 with the rain that was actually forecast, not the rain that fell.

For forecasts issued every 6 hours from 20 Jan to 28 Feb 2025, the deployed hybrid (5 copies) is run three ways:

  rain that fell   the usual backtest (a ceiling: perfect rain forecast)
  IFS forecast     ECMWF IFS as issued: for a valid hour d days after the start, the run from d days before
                   that hour (Open-Meteo Previous Runs, fetch_ifs_previous_runs.py), 1-7 days ahead only;
                   one deterministic forecast, so its likely range is the model-error library alone (no rain
                   spread), as on the site
  no change        persistence

Scored against what the river read, by lead (6, 24, 48, 72, 120, 168 h): typical error, right flag band, and
how often the IFS likely range (10-90%) held the river. Writes figures/eval/ifs_feb2025.json.

    python verify_ifs_feb2025.py
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
import torch  # noqa: E402

from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, WindowSampler, load_location_data  # noqa: E402
from flag_predictor.prediction import official_flag as of  # noqa: E402
from flag_predictor.prediction.hybrid_live import ERROR_LIBRARY, HYBRID_MODELS, MODEL_TAG, TRAINING_GAUGES  # noqa: E402
from flag_predictor.prediction.model_error import ErrorLibrary, start_features, total_uncertainty  # noqa: E402

RAIN = ROOT / 'data' / 'verification' / 'ifs_previous_runs_2025.csv'
OUT = ROOT / 'figures' / 'eval' / 'ifs_feb2025.json'
LEADS = (6, 24, 48, 72, 120, 168)
MAX_LEAD = 168                      # Previous Runs go back 7 days
START, END = pd.Timestamp('2025-01-20', tz='UTC'), pd.Timestamp('2025-02-28', tz='UTC')


def ifs_rain(ifs: pd.DataFrame, gauges: list, t0: pd.Timestamp) -> np.ndarray:
    """(HORIZON,) mean-gauge rain (mm/h) as forecast at t0: valid hour t0+L from the run ceil(L/24) days earlier."""
    out = np.zeros(HORIZON, dtype=np.float32)
    sub = ifs[ifs.station.isin(gauges)]
    for L in range(1, MAX_LEAD + 1):
        t = t0 + pd.Timedelta(hours=L)
        d = int(np.ceil(L / 24))
        v = sub.loc[sub.time == t, f'day{d}']
        out[L - 1] = float(v.mean()) if v.notna().any() else 0.0
    return out


@torch.no_grad()
def run(names, loc, data, pos, rains=None) -> np.ndarray:
    from flag_predictor.models.physics_train import load_candidate
    preds = []
    for n in names:
        model, scaler, _ = load_candidate(n, loc, ROOT / 'models')
        b = WindowSampler(data, scaler, torch.device('cpu')).batch(pos)
        if rains is not None:
            b['rain'] = torch.from_numpy(rains)
        preds.append(model(b['x'], b['rain'], b['season'], b['rain24'], b['rain168'], b['d0']).numpy())
    return np.mean(preds, axis=0)


def main() -> None:
    ifs = pd.read_csv(RAIN, parse_dates=['time'])
    ifs['time'] = ifs.time.dt.tz_localize('UTC')
    torch.set_num_threads(2)
    report = {'leads': list(LEADS), 'period': '20 Jan - 28 Feb 2025, forecasts every 6 h', 'locations': {}}
    for loc in ('isis', 'godstow', 'wallingford'):
        data = load_location_data(loc, ROOT)
        idx = data.index.tz_convert('UTC')
        d = np.asarray(data.diff, float)
        enc_ok = np.isfinite(data.enc).all(axis=1)
        ok = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(enc_ok)] == SEQ_LEN
        pos = np.array([i for i in np.flatnonzero((idx >= START) & (idx < END) & ok & np.isfinite(d))
                        if idx[i].hour % 6 == 0])
        gauges = TRAINING_GAUGES[loc]
        rains = np.stack([ifs_rain(ifs, gauges, idx[i]) for i in pos])
        fell = run(HYBRID_MODELS, loc, data, pos)
        fc = run(HYBRID_MODELS, loc, data, pos, rains)
        lib = ErrorLibrary.load(ROOT / 'models' / ERROR_LIBRARY.format(tag=MODEL_TAG, location=loc))
        lines = list(of.lines(loc)[1:]) if loc != 'wallingford' else []
        rows = []
        for j, i in enumerate(pos):
            t0 = idx[i]
            d24 = d[i - 24] if np.isfinite(d[i - 24]) else np.nan
            feats = start_features(d[i], d24, rains[j], float(np.nansum(data.rain[i - 167:i + 1])), t0)[None, :]
            traj = np.r_[d[i], fc[j]][None, :]
            samples = total_uncertainty(traj, feats, lib, seed=j)
            p10, p90 = np.percentile(samples, 10, axis=0), np.percentile(samples, 90, axis=0)
            for L in LEADS:
                if i + L >= len(d) or not np.isfinite(d[i + L]):
                    continue
                o = d[i + L]
                rows.append({'lead': L, 'valid': (t0 + pd.Timedelta(hours=L)).strftime('%Y-%m-%dT%H:%M'), 'obs': o, 'fell': fell[j, L - 1], 'ifs': fc[j, L - 1], 'pers': d[i],
                             'inside': p10[L] <= o <= p90[L], 'rain_ifs': float(rains[j, :L].sum()),
                             'rain_fell': float(data.rain[i + 1:i + L + 1].sum())})
        df = pd.DataFrame(rows)
        res = {'n_forecasts': int(len(pos)), 'by_lead': {}}
        for L, g in df.groupby('lead'):
            cell = {k: round(float((g[k] - g.obs).abs().mean()), 3) for k in ('ifs', 'fell', 'pers')}
            cell['inside_ifs'] = round(float(g.inside.mean()), 2)
            cell['rain_ifs_mm'] = round(float(g.rain_ifs.mean()), 1)
            cell['rain_fell_mm'] = round(float(g.rain_fell.mean()), 1)
            if lines:
                band = lambda x: np.digitize(x, lines)  # noqa: E731
                for k in ('ifs', 'fell', 'pers'):
                    cell[f'band_{k}'] = round(float((band(g[k]) == band(g.obs)).mean()), 3)
            res['by_lead'][int(L)] = cell
        # the 24 h and 72 h traces for a plot
        for L in (24, 72):
            g = df[df.lead == L].sort_values('valid')
            res[f'series{L}'] = {'valid': g.valid.tolist(), 'obs': g.obs.round(3).tolist(),
                                 'ifs': g.ifs.round(3).tolist(), 'fell': g.fell.round(3).tolist()}
        report['locations'][loc] = res
        print(f'\n=== {loc}: {len(pos)} forecasts')
        print(pd.DataFrame(res['by_lead']).T.to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f'\nwritten {OUT}')


if __name__ == '__main__':
    main()
