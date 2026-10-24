"""
When it rains more than forecast: how quickly does the forecast catch up?

At held-out starts (Nov 2024 - Jan 2026) with 10 mm+ (catchment mean) in the next 24 h, the forecast
is run four ways, all scored on the same hours (24-72 h after the start) against what the river did:

  from the start, 6 h later and 12 h later, each twice:
  perfect   - with the rain that actually fell (the best the model can do from there)
  surprised - the rain of the first day (from the start) halved: the forecast was too dry. 6 and
              12 h in, the rain so far is measured (in the history); the rest of that day is still
              halved
The extra error of 'surprised' over 'perfect' at each start time is what the dry forecast still costs.

    python test_rain_surprise.py
"""
from __future__ import annotations

import sys
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
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS, TRAINING_GAUGES  # noqa: E402

N_STARTS = 60
FACTOR = 0.5


def main() -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    for loc in ('isis', 'godstow', 'wallingford'):
        F = load_merged(loc, ROOT)
        F.index = F.index.tz_localize('UTC') if F.index.tz is None else F.index.tz_convert('UTC')
        d = F['differential']
        stations = [c for c in F.columns if c not in ('differential', 'flow_m3s_Farmoor')]
        gauges = [g for g in TRAINING_GAUGES[loc] if g in F.columns]
        cm = F[gauges].clip(lower=0).mean(axis=1).fillna(0.0)
        predict = ensemble_predictor(HYBRID_MODELS, loc)
        nxt24 = cm[::-1].rolling(24, min_periods=24).sum()[::-1].shift(-1)
        cand = d.index[(d.index >= '2024-11-01') & (d.index < '2026-01-01') & (d.index.hour % 3 == 0)
                       & d.notna() & (nxt24 >= 10)]
        starts = rng.choice(cand, min(N_STARTS, len(cand)), replace=False)
        lags = (0, 6, 12)
        res = {(lag, s): [] for lag in lags for s in (False, True)}
        rain_obs = []
        for t0 in starts:
            t0 = pd.Timestamp(t0)
            score = pd.date_range(t0 + pd.Timedelta(hours=24), t0 + pd.Timedelta(hours=72), freq='1h')
            truth = d.reindex(score).to_numpy()
            if np.isnan(truth).mean() > 0.2:
                continue
            out = {}
            for lag in lags:
                ts = t0 + pd.Timedelta(hours=lag)
                hist = F.loc[ts - pd.Timedelta(days=30):ts]
                if not np.isfinite(hist['differential'].iloc[-1]):
                    break
                fut = pd.date_range(ts + pd.Timedelta(hours=1), periods=240, freq='1h')
                for scale in (False, True):
                    rain = F[stations].reindex(fut).fillna(0.0)
                    if scale:
                        first = fut <= t0 + pd.Timedelta(hours=24)
                        rain.loc[first] = rain.loc[first] * FACTOR
                    p = pd.Series(np.asarray(predict(hist, rain), float)[1:], fut)
                    out[(lag, scale)] = np.nanmean(np.abs(p.reindex(score).to_numpy() - truth))
            if len(out) == len(res) and all(np.isfinite(v) for v in out.values()):
                for k, v in out.items():
                    res[k].append(v)
                rain_obs.append(float(nxt24.loc[t0]))
        n = len(rain_obs)
        print(f'\n{loc}: {n} starts, {np.mean(rain_obs):.1f} mm in the next 24 h on average (the dry forecast had half). '
              f'Mean error 24-72 h after the first start (m):')
        for lag in lags:
            a, b = np.mean(res[(lag, False)]), np.mean(res[(lag, True)])
            print(f'  forecast made {lag:2d} h in: right rain {a:.3f}, dry forecast {b:.3f}, extra error {b - a:+.3f}')

if __name__ == '__main__':
    main()
