"""
Model-error libraries for the shared catchment model (models/error_library_hybrid_shared_<loc>.npz).

Same starts and situation descriptors as the deployed model's libraries (build_error_library.py:
every 6 h over 2023 and Nov 2024 - Jan 2026, the rain that fell); only the error trajectories
(observed - forecast) are the shared model's. The per-lead spread inflation is taken from the
deployed library (fitted there by cross-validation between the two periods).

    python build_shared_error_library.py hybrid_shared hybrid_shared_s1 ...
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

from flag_predictor.models.shared_catchment import (STRETCHES, JointSampler, batch_forecast,  # noqa: E402
                                                    build_joint_data, load)
from flag_predictor.prediction.model_error import ErrorLibrary  # noqa: E402


def main(names):
    data = build_joint_data(ROOT)
    pos = pd.Series(np.arange(len(data.index)), index=data.index)
    for j, loc in enumerate(STRETCHES):
        models = [load(n, ROOT / 'models', loc) for n in names]
        dep = ErrorLibrary.load(ROOT / 'models' / f'error_library_hybrid_c1_ps_{loc}.npz')
        t0 = pd.to_datetime(dep.t0.astype('datetime64[s]').astype('int64'), unit='s', utc=True)
        idx = pos.reindex(t0).to_numpy()
        keep = np.isfinite(idx)
        idx = idx[keep].astype(int)
        keep_ok = data.stretch_ok[idx, j]
        idx = idx[keep_ok]
        sel = np.flatnonzero(keep)[keep_ok]
        preds = np.mean([batch_forecast(m, JointSampler(data, s, 'cpu'), idx)[:, :, j] for m, s, _ in models], axis=0)
        y = data.diff[idx[:, None] + np.arange(1, 241)[None, :], j]
        lib = ErrorLibrary.build(loc, 'hybrid_shared', dep.t0[sel], dep.features[sel], y - preds)
        lib.inflation = dep.inflation_or_ones()
        lib.k = dep.k
        lib.save(ROOT / 'models' / f'error_library_hybrid_shared_{loc}.npz')
        print(f'{loc}: {len(sel)} starts, mean |error| {np.nanmean(np.abs(y - preds)):.4f} m', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['hybrid_shared'])
