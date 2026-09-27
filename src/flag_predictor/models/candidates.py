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
from .stage1_flow import batch_flow_forecast

MODELS_DIR = Path(__file__).resolve().parents[3] / 'models'
# Enough history for the 720h rolling features to be exact over the encoder window.
HISTORY_H = 720 + SEQ_LEN + 200

_PAST_CACHE: dict = {}


def _past_inputs(history: pd.DataFrame):
    """Encoder features and past rain for a forecast start, shared across calls.

    These depend only on the history, so every ensemble member, seed and rain
    variant for the same t0 reuses one computation.
    """
    hist = history.iloc[-HISTORY_H:]
    key = (hist.index[-1], len(hist), tuple(hist.columns),
           float(np.nansum(hist['differential'].to_numpy())), float(np.nansum(hist.iloc[:, 1:].to_numpy())))
    if key not in _PAST_CACHE:
        if len(_PAST_CACHE) > 8:
            _PAST_CACHE.clear()
        enc = encoder_frame(hist)
        rain_hist = mean_station_rain(hist).to_numpy(dtype=np.float32)
        _PAST_CACHE[key] = (hist, enc, rain_hist)
    return _PAST_CACHE[key]


def redesign_predictor(name: str, location: str, models_dir: Path = MODELS_DIR):
    model, scaler, meta = load_candidate(name, location, models_dir)
    enc_cols = meta['enc_cols']

    @torch.no_grad()
    def predict(history: pd.DataFrame, future_rain: pd.DataFrame) -> np.ndarray:
        hist, enc_all, rain_hist = _past_inputs(history)
        enc = enc_all.reindex(columns=enc_cols)
        x = scaler.transform(enc.to_numpy(dtype=np.float32)[-SEQ_LEN:])
        rain_fut = future_rain.clip(lower=0).mean(axis=1).fillna(0.0).to_numpy(dtype=np.float32)
        t0 = hist.index[-1]
        fut_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq='1h')
        d0 = float(hist['differential'].iloc[-1])
        extra = {}
        if getattr(model, 'n_stations', 0):
            st = future_rain.reindex(columns=meta['station_names']).clip(lower=0).to_numpy(dtype=np.float32)
            extra['rain_st'] = torch.from_numpy(np.nan_to_num(st, nan=0.0))[None]
            extra['rain_st_ok'] = torch.from_numpy(np.isfinite(st))[None]
        if getattr(model, 'use_flow', False):
            future = future_rain.reindex(columns=[c for c in hist.columns if c in future_rain.columns])
            frame = pd.concat([hist, future.set_axis(fut_idx)])
            flow = batch_flow_forecast(frame, np.array([len(hist) - 1]), models_dir)
            extra['pred_flow_log'] = torch.from_numpy(np.log1p(flow))
        out = model(
            torch.from_numpy(x)[None],
            torch.from_numpy(rain_fut)[None],
            torch.from_numpy(doy_phase(fut_idx))[None],
            torch.tensor([rain_hist[-24:].sum()]),
            torch.tensor([rain_hist[-168:].sum()]),
            torch.tensor([d0], dtype=torch.float32),
            **extra,
        )
        return np.concatenate([[d0], out[0].numpy().astype(float)])

    return predict


def ensemble_predictor(names, location: str, models_dir: Path = MODELS_DIR):
    """Mean of several redesign models (e.g. seeds).

    The structural guarantees survive averaging: a mean of trajectories that
    start at d0, never climb without water and rise with rain has the same
    properties.
    """
    members = [redesign_predictor(n, location, models_dir) for n in names]

    def predict(history: pd.DataFrame, future_rain: pd.DataFrame) -> np.ndarray:
        return np.mean([m(history, future_rain) for m in members], axis=0)

    return predict
