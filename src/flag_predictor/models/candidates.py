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


def redesign_predictor(name: str, location: str, models_dir: Path = MODELS_DIR,
                       extra_features: pd.DataFrame = None):
    """``extra_features``: hourly encoder columns not derived from the history frame
    (e.g. continuous-simulation states for hybrid_simfeat), joined on time."""
    model, scaler, meta = load_candidate(name, location, models_dir)
    enc_cols = meta['enc_cols']

    @torch.no_grad()
    def predict(history: pd.DataFrame, future_rain: pd.DataFrame, with_states: bool = False):
        hist, enc_all, rain_hist = _past_inputs(history)
        if extra_features is not None:
            enc_all = enc_all.join(extra_features.reindex(enc_all.index))
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
        args = (
            torch.from_numpy(x)[None],
            torch.from_numpy(rain_fut)[None],
            torch.from_numpy(doy_phase(fut_idx))[None],
            torch.tensor([rain_hist[-24:].sum()]),
            torch.tensor([rain_hist[-168:].sum()]),
            torch.tensor([d0], dtype=torch.float32),
        )
        if with_states:
            # (241,) differential, (240,) channel outflow mm/h, (241, 5) states: soil wetness
            # (0-1) and the quick, medium, slow and channel stores (mm), t0 first.
            d, q, st = model(*args, return_states=True, **extra)
            return (np.concatenate([[d0], d[0].numpy().astype(float)]), q[0].numpy().astype(float),
                    st[0].numpy().astype(float))
        out = model(*args, **extra)
        return np.concatenate([[d0], out[0].numpy().astype(float)])

    return predict


def ensemble_predictor(names, location: str, models_dir: Path = MODELS_DIR, extra_features: pd.DataFrame = None):
    """Mean of several redesign models (e.g. seeds).

    The structural guarantees survive averaging: a mean of trajectories that
    start at d0, never climb without water and rise with rain has the same
    properties.
    """
    members = [redesign_predictor(n, location, models_dir, extra_features) for n in names]

    def predict(history: pd.DataFrame, future_rain: pd.DataFrame) -> np.ndarray:
        return np.mean([m(history, future_rain) for m in members], axis=0)

    def predict_with_states(history: pd.DataFrame, future_rain: pd.DataFrame):
        """(differential, outflow, states) as predict, each the mean over the copies."""
        outs = [m(history, future_rain, with_states=True) for m in members]
        return tuple(np.mean([o[i] for o in outs], axis=0) for i in range(3))

    predict.with_states = predict_with_states
    return predict
