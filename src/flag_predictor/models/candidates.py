"""
Predictors for physics-redesign candidates, in the evaluation's interface:
``predict(history, future_rain) -> np.ndarray`` of 241 values (t0 first).

evaluate_candidates.py resolves ``redesign:<name>`` to redesign_predictor(name, location), which loads
models/redesign_<name>_<location>.pt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..evaluation import mean_station_rain
from .physics_data import HORIZON, SEQ_LEN, doy_phase, encoder_frame
from .physics_train import load_candidate

MODELS_DIR = Path(__file__).resolve().parents[3] / 'models'
# Enough history for the 720h rolling features to be exact over the encoder window.
HISTORY_H = 720 + SEQ_LEN + 200


def redesign_predictor(name: str, location: str, models_dir: Path = MODELS_DIR):
    model, scaler, meta = load_candidate(name, location, models_dir)
    enc_cols = meta['enc_cols']

    @torch.no_grad()
    def predict(history: pd.DataFrame, future_rain: pd.DataFrame) -> np.ndarray:
        hist = history.iloc[-HISTORY_H:]
        enc = encoder_frame(hist).reindex(columns=enc_cols)
        x = scaler.transform(enc.to_numpy(dtype=np.float32)[-SEQ_LEN:])
        rain_hist = mean_station_rain(hist).to_numpy(dtype=np.float32)
        rain_fut = future_rain.clip(lower=0).mean(axis=1).fillna(0.0).to_numpy(dtype=np.float32)
        t0 = hist.index[-1]
        fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
        d0 = float(hist['differential'].iloc[-1])
        out = model(
            torch.from_numpy(x)[None],
            torch.from_numpy(rain_fut)[None],
            torch.from_numpy(doy_phase(fut_idx))[None],
            torch.tensor([rain_hist[-24:].sum()]),
            torch.tensor([rain_hist[-168:].sum()]),
            torch.tensor([d0], dtype=torch.float32),
        )
        return np.concatenate([[d0], out[0].numpy().astype(float)])

    return predict
