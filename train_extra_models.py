"""
Train the deployed hybrid set-up with extra encoder inputs: measured soil moisture (COSMOS-UK,
Chimney Meadows) and upstream EA river levels (models/extra_inputs.py).

    python train_extra_models.py --name hybrid_su --seeds 0,1,2,3,4 --locations all

Seed 0 saves as models/redesign_<name>_<location>.*, seed k as <name>_s<k>. Same settings as
hybrid_c1_ps (per-sample parameters, smooth rating curve); only the encoder inputs differ.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flag_predictor.models.extra_inputs import INPUT_SETS, history_extra  # noqa: E402
from flag_predictor.models.physics_data import load_location_data  # noqa: E402
from flag_predictor.models.physics_train import save_candidate, train_candidate  # noqa: E402


def data_with_extra(location: str, inputs: str = 'all'):
    data = load_location_data(location, ROOT)
    cols = INPUT_SETS[inputs]
    extra = history_extra(data.index, ROOT)[cols].to_numpy(dtype=np.float32)
    data.enc = np.concatenate([data.enc, extra], axis=1)
    data.enc_cols = list(data.enc_cols) + cols
    if cols:
        cover = np.isfinite(extra[data.train_t]).mean(0)
        print(f"  extra inputs at training starts: {cover.min():.0%}-{cover.max():.0%} present", flush=True)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='hybrid_su')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--locations', default='all')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--inputs', default='all', choices=['none', 'all', 'soil', 'soil_rise'])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    locations = ['isis', 'godstow', 'wallingford'] if args.locations == 'all' else args.locations.split(',')
    for seed in [int(s) for s in args.seeds.split(',')]:
        for loc in locations:
            name = args.name if seed == 0 else f'{args.name}_s{seed}'
            if (ROOT / 'models' / f'redesign_{name}_{loc}.pt').exists():
                print(f'{name} {loc}: already trained', flush=True)
                continue
            t = time.time()
            print(f'\n=== {name} {loc} ===', flush=True)
            data = data_with_extra(loc, args.inputs)
            cfg = {'per_sample_params': True, 'smooth_rating': True, 'seed': seed}
            model, scaler, cfg_used, history = train_candidate('hybrid', data, cfg)
            save_candidate(model, scaler, cfg_used, history, 'hybrid', name, loc, data, ROOT / 'models')
            best = min(history, key=lambda r: r['val_mae'])
            print(f"saved {name}_{loc}: best val MAE {best['val_mae']:.4f} (epoch {best['epoch']}), "
                  f"{(time.time() - t) / 60:.0f} min", flush=True)


if __name__ == '__main__':
    main()
