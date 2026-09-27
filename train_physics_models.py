"""
Train physics-redesign candidates.

Usage:
    python train_physics_models.py hybrid isis --name hybrid_v1
    python train_physics_models.py lstm_v2 isis,godstow --name lstm_v2 --cfg '{"epochs": 10}'

Weights go to models/redesign_{name}_{location}.pt; live models are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
warnings.filterwarnings('ignore')

import torch  # noqa: E402

from flag_predictor.models.physics_data import load_location_data  # noqa: E402
from flag_predictor.models.physics_train import save_candidate, train_candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices=['hybrid', 'lstm_v2'])
    parser.add_argument('locations')
    parser.add_argument('--name', required=True)
    parser.add_argument('--cfg', default='{}')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    torch.set_num_threads(8)
    locations = ['isis', 'godstow', 'wallingford'] if args.locations == 'all' else args.locations.split(',')
    for location in locations:
        t = time.time()
        data = load_location_data(location, PROJECT_ROOT)
        print(f"\n=== {args.family} {args.name} {location}: train {len(data.train_t)} / "
              f"val {len(data.val_t)} windows, {data.enc.shape[1]} features ===", flush=True)
        model, scaler, cfg, history = train_candidate(
            args.family, data, json.loads(args.cfg), device=torch.device(args.device)
        )
        path = save_candidate(model, scaler, cfg, history, args.family, args.name,
                              location, data, PROJECT_ROOT / 'models')
        best = min(history, key=lambda r: r['val_mae'])
        print(f"saved {path.name}  best val_mae={best['val_mae']:.4f} (epoch {best['epoch']}) "
              f"in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
