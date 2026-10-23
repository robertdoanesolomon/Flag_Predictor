"""
Is the Thames at Hannington Bridge (EA 0569TH, upper Thames, level) a useful early guide?

1. Lead time: how far ahead of each stretch's differential does Hannington move (cross-correlation of
   24 h changes), against Farmoor flow, which the model already sees.
2. Added value: on the deployed hybrid's held-out backtest (2023 + Nov 2024 - Jan 2026, rain that
   fell), does Hannington's recent rise explain the forecast's errors? If the model already captures
   what Hannington knows, its rise should not correlate with the errors.

    python test_hannington.py <hannington.csv>
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

from build_error_library import predict_hybrid, start_positions  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402


def main(path: str) -> None:
    torch.set_num_threads(2)
    han = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    han.index = pd.to_datetime(han.index, utc=True)
    han = han.interpolate(limit=6)
    for loc in ('isis', 'godstow', 'wallingford'):
        data = load_location_data(loc, ROOT)
        idx = data.index.tz_convert('UTC')
        d = pd.Series(np.asarray(data.diff, float), idx)
        h = han.reindex(idx)
        fl = pd.Series(np.asarray(data.enc[:, list(data.enc_cols).index('flow_m3s_Farmoor')], float), idx) \
            if 'flow_m3s_Farmoor' in data.enc_cols else None
        dd = d.diff(24)
        print(f'\n=== {loc}: correlation of the 24 h change at Hannington (and Farmoor flow) with the '
              f'24 h change of the differential k hours later')
        row_h, row_f = [], []
        for k in (0, 12, 24, 36, 48, 72, 96):
            row_h.append(f'{k}h {h.diff(24).corr(dd.shift(-k)):+.2f}')
            if fl is not None:
                row_f.append(f'{k}h {fl.diff(24).corr(dd.shift(-k)):+.2f}')
        print('  Hannington:', '  '.join(row_h))
        if fl is not None:
            print('  Farmoor   :', '  '.join(row_f))
        # added value on the backtest
        st = start_positions(data, stride_h=6)
        t_all = np.concatenate([st['val2023'], st['test']])
        pred = predict_hybrid(HYBRID_MODELS, loc, data, t_all)
        obs = np.stack([data.diff[t + 1:t + HORIZON + 1] for t in t_all])
        r = obs - pred
        hv = h.to_numpy()
        base = pd.Series(hv, idx).rolling('14D', min_periods=24).min().to_numpy()
        rise = np.array([hv[t] - base[t] for t in t_all])
        d24 = np.array([hv[t] - hv[t - 24] for t in t_all])
        rain3 = np.array([data.rain[t + 1:t + 73].sum() for t in t_all])
        print('  backtest: correlation of the forecast error at lead h with Hannington at the start')
        for name, x in (('rise above 14-day low', rise), ('24 h change', d24)):
            cells = []
            for lead in (12, 24, 48, 72, 120):
                m = np.isfinite(x) & np.isfinite(r[:, lead - 1])
                cells.append(f'{lead}h {np.corrcoef(x[m], r[m, lead - 1])[0, 1]:+.2f}')
            print(f'    {name:22s}', '  '.join(cells), f'(n={int(np.isfinite(x).sum())})')
        # rising at Hannington but little rain ahead: the case the user means
        m = np.isfinite(d24) & (d24 > np.nanpercentile(d24, 90)) & (rain3 < 10)
        if m.sum() > 10:
            print(f'    Hannington rising fast (top 10%), <10 mm rain ahead: n={int(m.sum())}, mean error '
                  f'24h {np.nanmean(r[m, 23]):+.3f}  48h {np.nanmean(r[m, 47]):+.3f}  72h {np.nanmean(r[m, 71]):+.3f} m '
                  f'(all starts: {np.nanmean(r[:, 23]):+.3f} {np.nanmean(r[:, 47]):+.3f} {np.nanmean(r[:, 71]):+.3f})')


if __name__ == '__main__':
    main(sys.argv[1])
