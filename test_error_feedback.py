"""
Does the hybrid keep making the same error, so that recent forecasts' errors could correct the next one?

At each start t (every 6 h, 2023 validation year and the Nov 2024 - Jan 2026 test window, rain
that fell), the forecasts issued 24/48/72 h earlier are compared with the river at t:
    e_k = obs(t) - forecast_from(t-k)(t)
(each forecast starts from the observed level, so this is the drift error over its first k hours).
A fading correction  corrected(h) = forecast(h) + beta_h * e  is fitted per lead time on 2023 and
tested on the test window, by stretch and for heavy rain ahead.

    python test_error_feedback.py
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
import torch  # noqa: E402

from build_error_library import predict_hybrid, start_positions  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402

LAGS = (24, 48, 72)
LEADS = {'1-24h': (0, 24), '25-72h': (24, 72), '73-240h': (72, 240), 'all': (0, 240)}


def main() -> None:
    torch.set_num_threads(2)
    report = {}
    for loc in ('isis', 'godstow', 'wallingford'):
        data = load_location_data(loc, ROOT)
        st = start_positions(data, stride_h=6)
        t_all = np.concatenate([st['val2023'], st['test']])
        is_val = np.isin(t_all, st['val2023'])
        # forecasts at every start and at the starts 24/48/72 h before it
        need = np.unique(np.concatenate([t_all] + [t_all - k for k in LAGS]))
        pred = dict(zip(need, predict_hybrid(HYBRID_MODELS, loc, data, need)))
        obs = np.stack([data.diff[t + 1:t + HORIZON + 1] for t in t_all])
        fc = np.stack([pred[t] for t in t_all])
        r = obs - fc
        E = np.full((len(t_all), len(LAGS)), np.nan)
        for j, k in enumerate(LAGS):
            for i, t in enumerate(t_all):
                if (t - k) in pred and np.isfinite(data.diff[t]):
                    E[i, j] = data.diff[t] - pred[t - k][k - 1]
        e = np.nanmean(E, axis=1)              # mean drift error of the last three days' forecasts
        ok = np.isfinite(e)
        rain3 = np.array([data.rain[t + 1:t + 73].sum() for t in t_all])
        heavy = rain3 >= 15
        # persistence: correlation of the past error with this forecast's error at each lead
        rep = {'n_val': int((ok & is_val).sum()), 'n_test': int((ok & ~is_val).sum())}
        corr = {}
        for h in (6, 24, 48, 72, 120, 240):
            m = ok & np.isfinite(r[:, h - 1])
            corr[h] = round(float(np.corrcoef(e[m], r[m, h - 1])[0, 1]), 3)
        rep['corr_past_error_vs_error_at_lead'] = corr
        # beta per lead, fitted on 2023 (least squares through 0), smoothed
        beta = np.zeros(HORIZON)
        for h in range(HORIZON):
            m = ok & is_val & np.isfinite(r[:, h])
            beta[h] = np.sum(e[m] * r[m, h]) / np.sum(e[m] ** 2) if m.sum() > 50 else 0.0
        beta = np.convolve(np.pad(beta, 6, mode='edge'), np.ones(13) / 13, 'valid')
        rep['beta'] = {h: round(float(beta[h - 1]), 3) for h in (6, 24, 48, 72, 120, 240)}
        fc2 = fc + np.where(ok[:, None], e[:, None] * beta[None, :], 0.0)
        rows = {}
        for sname, m in {'test window': ~is_val & ok, 'test, heavy rain ahead': ~is_val & ok & heavy,
                         'test, dry ahead (<5 mm in 3 d)': ~is_val & ok & (rain3 < 5)}.items():
            row = {'n': int(m.sum())}
            for lname, (a, b) in LEADS.items():
                row[lname] = [round(float(np.nanmean(np.abs(r[m, a:b]))), 4),
                              round(float(np.nanmean(np.abs(obs[m, a:b] - fc2[m, a:b]))), 4)]
            rows[sname] = row
        rep['mae_before_after'] = rows
        report[loc] = rep
        print(f'\n=== {loc}: corr(past error, error at lead h) {corr}')
        print(f'  beta {rep["beta"]}')
        for sname, row in rows.items():
            cells = '  '.join(f'{k} {v[0]:.4f}->{v[1]:.4f}' for k, v in row.items() if k != 'n')
            print(f'  {sname:32s} n={row["n"]:5d}  {cells}', flush=True)
    out = ROOT / 'figures' / 'eval' / 'error_feedback.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
