#!/usr/bin/env python
"""
Continuous-state hybrid (hybrid_cs): pre-train the hydrology on the long
Farmoor flow record, then train the forecast model on the differential.

    # 1. hydrology only, on Farmoor flow 2000-2024 (needs data/hourly/*.npz)
    python train_continuous_models.py pretrain --core cs_cos --et cos
    python train_continuous_models.py pretrain --core cs_pet --et pet     # needs data/era5/et0_hourly.csv.gz

    # 2. forecast model per location, starting from a pre-trained core
    python train_continuous_models.py forecast --core cs_pet --name hybrid_cs_pet --locations all --seed 0

data/hourly/*.npz are hourly per-gauge rain and Farmoor flow made from the EA
qualified 15-min CSVs (data/rainfall_training_data/hourly/to_hourly.py).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from flag_predictor.models.continuous_hybrid import (  # noqa: E402
    ReservoirCore, build_long_record, flow_scores, loss_mask, pretrain_core, run_full,
)
from flag_predictor.data.download_et import load_et0  # noqa: E402

ALL_GAUGES = ['Aylesbury', 'Worsham', 'Osney', 'Bourton', 'Abingdon', 'St', 'Swindon', 'Eynsham',
              'Chipping', 'Stanford', 'Rapsgate', 'Wheatley', 'Stowell', 'Cleeve', 'Shorncote',
              'Benson', 'Grimsbury']
HOURLY_DIR = ROOT / 'data' / 'hourly'
ET0_PATH = ROOT / 'data' / 'era5' / 'et0_hourly.csv.gz'
MODELS = ROOT / 'models'


def cmd_pretrain(args) -> None:
    torch.set_num_threads(args.threads)
    et0 = load_et0(ET0_PATH) if args.et == 'pet' else None
    if args.et == 'pet' and et0 is None:
        raise SystemExit(f'{ET0_PATH} not found')
    rec = build_long_record(HOURLY_DIR, ALL_GAUGES, et0=et0)
    core = ReservoirCore(et_mode=args.et, direct=not args.no_direct)
    t = time.time()
    hist = pretrain_core(core, rec, epochs=args.epochs, chunk=args.chunk, lr=args.lr, seed=args.seed)
    q, _, _ = run_full(core, rec)
    q = q[0].numpy()
    masks = loss_mask(rec.index, rec.flow_mmh)
    scores = {k: flow_scores(q, rec.flow_mmh, m) for k, m in masks.items()}
    print(json.dumps(scores, indent=1))
    MODELS.mkdir(exist_ok=True)
    torch.save(core.state_dict(), MODELS / f'core_{args.core}.pt')
    with open(MODELS / f'core_{args.core}.pkl', 'wb') as f:
        pickle.dump({'et_mode': args.et, 'direct': not args.no_direct, 'physics': core.describe(),
                     'scores': scores, 'history': hist, 'secs': time.time() - t,
                     'gauges': ALL_GAUGES}, f)
    sim = pd.DataFrame({'q_sim_m3s': q * 447.0, 'q_obs_m3s': rec.flow_mmh * 447.0}, index=rec.index)
    sim.to_csv(ROOT / 'figures' / f'core_{args.core}_flow.csv.gz', float_format='%.3f')
    print(f'saved core_{args.core} ({time.time() - t:.0f}s)')


def cmd_forecast(args) -> None:
    from flag_predictor.models.continuous_train import train_location, save_forecast_model
    torch.set_num_threads(args.threads)
    locations = ['isis', 'godstow', 'wallingford'] if args.locations == 'all' else args.locations.split(',')
    cfg = json.loads(args.cfg) if args.cfg else {}
    cfg['seed'] = args.seed
    for loc in locations:
        print(f'\n=== {loc}: {args.name} (core {args.core}, seed {args.seed}) ===', flush=True)
        model, scaler, cfg_used, history, extras = train_location(loc, args.core, ROOT, cfg)
        save_forecast_model(model, scaler, cfg_used, history, args.name, args.core, loc, extras, MODELS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pretrain')
    p.add_argument('--core', required=True)
    p.add_argument('--et', choices=['cos', 'pet'], default='cos')
    p.add_argument('--no-direct', action='store_true')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--chunk', type=int, default=2016)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.set_defaults(func=cmd_pretrain)
    f = sub.add_parser('forecast')
    f.add_argument('--core', required=True)
    f.add_argument('--name', required=True)
    f.add_argument('--locations', default='all')
    f.add_argument('--seed', type=int, default=0)
    f.add_argument('--cfg', default=None)
    f.add_argument('--threads', type=int, default=2)
    f.set_defaults(func=cmd_forecast)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
