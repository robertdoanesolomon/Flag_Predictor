"""
After a lock-keeper step (gates / bucks changed: the differential jumps within the hour), does the
forecast keep the new level or drift back?

The model reads a jump as a change in flow: it pins its channel store to the new level, and that store
drains over the next hours unless rain and the other stores keep it up. A gate change at steady flow
is really a step that stays. Held-out starts (2023, Nov 2024 - Jan 2026), rain that fell:

  * starts in the hour after a jump of 3 cm+ (Wallingford 4 cm+) that held for the next 2 hours;
  * against all other starts.

For each: mean error (observed - forecast, signed in the jump's direction: positive = the river kept
more of the jump than the forecast did) and mean absolute error at 1-6 h, 6-24 h, 24-48 h; and the
same for persistence (no change from the start).

    python test_jumps.py
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

from build_error_library import PERIODS, predict_hybrid  # noqa: E402
from flag_predictor.models.physics_data import HORIZON, SEQ_LEN, load_location_data  # noqa: E402
from flag_predictor.prediction.hybrid_live import HYBRID_MODELS  # noqa: E402

THRESH = {'isis': 0.03, 'godstow': 0.03, 'wallingford': 0.04}
LEADS = {'1-6 h': (0, 6), '6-24 h': (6, 24), '24-48 h': (24, 48)}


def main() -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    for loc in ('isis', 'godstow', 'wallingford'):
        data = load_location_data(loc, ROOT)
        d = np.asarray(data.diff, float)
        idx = data.index
        enc_ok = np.isfinite(data.enc).all(axis=1)
        ok_count = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[:len(enc_ok)]
        fut_ok = np.array([np.isfinite(d[t + 1:t + 49]).mean() > 0.8 if t + 49 < len(d) else False for t in range(len(d))])
        held = np.zeros(len(d), bool)
        for lo, hi in PERIODS.values():
            held |= (idx >= pd.Timestamp(lo, tz=idx.tz)) & (idx < pd.Timestamp(hi, tz=idx.tz))
        valid = held & (ok_count == SEQ_LEN) & np.isfinite(d) & fut_ok
        valid[:SEQ_LEN + 2] = False
        valid[len(d) - HORIZON - 2:] = False
        jump = np.r_[np.nan, np.diff(d)]
        stays = np.r_[np.full(1, np.nan), d[3:] - d[:-3], np.full(2, np.nan)]        # d[t+2] - d[t-1]
        is_jump = valid & (np.abs(jump) >= THRESH[loc]) & (np.sign(stays) == np.sign(jump)) \
            & (np.abs(stays) >= 0.7 * np.abs(jump))
        jumps = np.flatnonzero(is_jump)
        others = rng.choice(np.flatnonzero(valid & ~is_jump), min(3000, int((valid & ~is_jump).sum())), replace=False)
        print(f'\n=== {loc}: {len(jumps)} jumps (median size {np.nanmedian(np.abs(jump[jumps])):.3f} m), '
              f'{len(others)} other starts')
        for name, t_idx in (('after a jump', jumps), ('other starts', others)):
            pred = predict_hybrid(HYBRID_MODELS, loc, data, t_idx)
            obs = np.stack([d[t + 1:t + HORIZON + 1] for t in t_idx])
            sign = np.sign(jump[t_idx]) if name == 'after a jump' else np.ones(len(t_idx))
            err = (obs - pred) * sign[:, None]
            pers = (obs - d[t_idx][:, None]) * sign[:, None]
            cells = []
            for lname, (a, b) in LEADS.items():
                cells.append(f'{lname}: bias {np.nanmean(err[:, a:b]):+.3f} MAE {np.nanmean(np.abs(err[:, a:b])):.3f} '
                             f'(persistence {np.nanmean(np.abs(pers[:, a:b])):.3f})')
            print(f'  {name:13s} ' + ' | '.join(cells))
            if name == 'after a jump':
                kept_obs = np.nanmean((obs[:, 5] - d[t_idx]) * sign)
                kept_fc = np.nanmean((pred[:, 5] - d[t_idx]) * sign)
                print(f'    6 h after: the river moved {kept_obs:+.3f} m further in the jump\'s direction; '
                      f'the forecast {kept_fc:+.3f} m')


if __name__ == '__main__':
    main()
