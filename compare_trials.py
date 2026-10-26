"""Main model against trial input sets on the held-out starts (2023, Nov 2024 - Jan 2026), rain that fell.
Error change vs main (positive = better), 90% range from resampling whole weeks.  python compare_trials.py"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, torch  # noqa: E402
from build_error_library import predict_hybrid, start_positions  # noqa: E402
from train_extra_models import data_with_extra  # noqa: E402
from flag_predictor.models.physics_data import load_location_data, HORIZON  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402
SETS = {'full trial (soil+9 gauges)': ('hybrid_srh', 'soil_rise_han', 5), 'Hannington only': ('hybrid_han', 'han', 3),
        '9 gauges, no soil': ('hybrid_rh', 'rise_han', 3)}
torch.set_num_threads(2); rng = np.random.default_rng(0)
for loc in ('isis', 'godstow', 'wallingford'):
    data = load_location_data(loc, ROOT); st = start_positions(data); t = np.concatenate([st['val2023'], st['test']])
    obs = np.stack([data.diff[i + 1:i + HORIZON + 1] for i in t])
    main = predict_hybrid(HYBRID_MODELS, loc, data, t)
    week = data.index[t].tz_convert('UTC').to_period('W').astype(str)
    print(f'\n=== {loc}')
    for name, (tag, inputs, n) in SETS.items():
        p = predict_hybrid([tag] + [f'{tag}_s{i}' for i in range(1, n)], loc, data_with_extra(loc, inputs), t)
        cells = []
        for L in (24, 72, 168):
            ea, eb = np.abs(main[:, L - 1] - obs[:, L - 1]), np.abs(p[:, L - 1] - obs[:, L - 1])
            ok = np.isfinite(ea) & np.isfinite(eb)
            g = pd.DataFrame({'w': week[ok], 'd': (eb - ea)[ok], 'a': ea[ok]}).groupby('w').sum()
            b = [1 - (s.d.sum() + s.a.sum()) / s.a.sum() for s in (g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _ in range(1000))]
            cells.append(f'{L}h {100 * (-g.d.sum() / g.a.sum()):+4.1f}% [{100 * np.percentile(b, 5):+.0f},{100 * np.percentile(b, 95):+.0f}]')
        print(f'  {name:28s} ' + '  '.join(cells))
