"""
Batch stage-1 (Farmoor flow) forecasts for training windows.

The September stage-2 model trains on the Farmoor flow that actually
happened but forecasts with stage 1's prediction. To train an LSTM on the
same kind of flow it will see at forecast time, run the September Farmoor
model (without the hold-at-last-value blend) over every training window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..processing.features import build_september_encoder_features, rain_station_columns
from .physics_data import HORIZON, LocationData
from .training import load_model

FLOW_MODEL = 'experiment_2026_09_farmoor'


def load_flow_model(models_dir: Path):
    return load_model(
        model_path=models_dir / f'multihorizon_model_{FLOW_MODEL}.pth',
        scaler_path=models_dir / f'scaler_{FLOW_MODEL}.pkl',
        config_path=models_dir / f'config_{FLOW_MODEL}.pkl',
    )


@torch.no_grad()
def batch_flow_forecast(merged: pd.DataFrame, positions: np.ndarray, models_dir: Path,
                        batch_size: int = 256) -> np.ndarray:
    """(N, 240) predicted Farmoor flow (m³/s) for forecasts starting at positions."""
    model, scaler, cfg = load_flow_model(models_dir)
    model = model.cpu().eval()
    flow_col = cfg['target_column']
    frame = merged.drop(columns=['differential'])
    enc = build_september_encoder_features(frame, target_col=flow_col)
    enc = enc.reindex(columns=cfg['feature_columns']).ffill().fillna(0.0)
    enc = scaler.transform(enc.to_numpy()).astype(np.float32)
    rain = merged[rain_station_columns(frame)].fillna(0).sum(axis=1).to_numpy(dtype=float)
    rain = cfg['decoder_scaler'].transform(rain.reshape(-1, 1)).astype(np.float32)[:, 0]
    flow_now = merged[flow_col].ffill().fillna(0).to_numpy(dtype=float)

    seq = cfg['sequence_length']
    out = np.zeros((len(positions), HORIZON), dtype=np.float32)
    past_off = np.arange(-seq + 1, 1)
    fut_off = np.arange(1, HORIZON + 1)
    for i in range(0, len(positions), batch_size):
        pos = positions[i:i + batch_size]
        x = torch.from_numpy(enc[pos[:, None] + past_off[None, :]])
        c = torch.from_numpy(rain[pos[:, None] + fut_off[None, :]])[..., None]
        delta = model(x, c).numpy()
        log_now = np.log1p(np.clip(flow_now[pos], 0, None))[:, None]
        out[i:i + batch_size] = np.expm1(log_now + delta).clip(min=0.0)
    return out


def attach_flow_forecasts(data: LocationData, merged: pd.DataFrame, project_root: Path) -> None:
    """Add data.pred_flow (N, 240) and data.pred_flow_row (T,) lookup, cached on disk."""
    cache = project_root / 'data' / f'stage1_flow_{data.location}.npz'
    positions = np.unique(np.concatenate([data.train_t, data.val_t]))
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z['positions'], positions):
            flows = z['flows']
        else:
            flows = None
    else:
        flows = None
    if flows is None:
        flows = batch_flow_forecast(merged, positions, project_root / 'models')
        np.savez_compressed(cache, positions=positions, flows=flows)
    lookup = np.full(len(data.index), -1, dtype=np.int64)
    lookup[positions] = np.arange(len(positions))
    data.pred_flow = flows
    data.pred_flow_row = lookup
