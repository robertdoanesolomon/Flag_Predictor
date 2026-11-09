"""
Train the shared catchment model (models/shared_catchment.py): one model for all three stretches.

    python train_shared_catchment.py --name hybrid_shared            # seed 0
    python train_shared_catchment.py --name hybrid_shared_s1 --cfg '{"seed": 1}'

Weights: models/shared_<name>.pt (+ .pkl). Live models are untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')

import torch  # noqa: E402

from flag_predictor.models.shared_catchment import build_joint_data, save, train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cfg', default='{}')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    t = time.time()
    data = build_joint_data(ROOT)
    print(f'=== {a.name}: train {len(data.train_t)} / val {len(data.val_t)} windows, '
          f'{data.enc.shape[1]} features', flush=True)
    model, scaler, cfg, history = train(data, json.loads(a.cfg))
    path = save(model, scaler, cfg, history, data, a.name, ROOT / 'models')
    best = min(history, key=lambda r: r['val_mae'])
    print(f"saved {path.name}: best val {best['val_mae']:.4f} (G {best['val_mae_godstow']:.4f} "
          f"I {best['val_mae_isis']:.4f} W {best['val_mae_wallingford']:.4f}, epoch {best['epoch']}) "
          f"in {(time.time() - t) / 60:.1f} min", flush=True)
    print({k: round(v, 3) for k, v in model.describe().items()}, flush=True)


if __name__ == '__main__':
    main()
