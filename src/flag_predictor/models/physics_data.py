"""
Windowed training data for the physics-redesign candidates.

Unlike the September trainer, everything lives on a regular hourly grid, so a
window never splices across a data gap; missing target hours are masked in the
loss instead. Windows are gathered on the fly from (T, F) arrays rather than
materialised, which keeps memory at O(T·F).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..evaluation import mean_station_rain
from ..processing.features import build_september_encoder_features

SEQ_LEN = 100
HORIZON = 240
# Training windows must finish before the frozen test window opens (2024-11-01);
# 2023 is the validation year. Windows whose 10-day target runs into 2023 from
# December 2022 are dropped from training so validation stays unseen.
TRAIN_END = '2024-10-21'
VAL_YEAR = 2023
ENC_FFILL_LIMIT = 24


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate NaN runs of at most max_gap hours; longer runs stay NaN.

    Every forecast starts at an observed hour, so a filled gap always lies
    wholly before t0 and never leaks future values into the encoder.
    """
    na = series.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform('size')
    filled = series.interpolate(limit_area='inside')
    return filled.where(~na | (run_len <= max_gap))


def encoder_frame(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged['differential'] = fill_short_gaps(merged['differential'], ENC_FFILL_LIMIT)
    merged['flow_m3s_Farmoor'] = fill_short_gaps(merged['flow_m3s_Farmoor'], ENC_FFILL_LIMIT)
    enc = build_september_encoder_features(merged, target_col='differential')
    enc = enc.replace([np.inf, -np.inf], np.nan).ffill(limit=ENC_FFILL_LIMIT)
    return enc.astype(np.float32)


def recent_rain_sums(rain: np.ndarray, t: int) -> np.ndarray:
    """Mean-station rain (mm) over the 24h and 168h ending at index t."""
    lo24, lo168 = max(0, t - 23), max(0, t - 167)
    return np.array([rain[lo24:t + 1].sum(), rain[lo168:t + 1].sum()], dtype=np.float32)


def doy_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """cos of day-of-year phase, peaking at the summer solstice (for ET)."""
    doy = index.dayofyear.to_numpy() + index.hour.to_numpy() / 24.0
    return np.cos(2 * np.pi * (doy - 172) / 365.25).astype(np.float32)


@dataclass
class LocationData:
    location: str
    index: pd.DatetimeIndex
    enc: np.ndarray          # (T, F) unscaled encoder features
    enc_cols: list
    rain: np.ndarray         # (T,) mean-station mm/h
    flow: np.ndarray         # (T,) Farmoor m3/s (NaN where missing)
    diff: np.ndarray         # (T,) differential (NaN where missing)
    season: np.ndarray       # (T,) cos day-of-year phase
    rain24: np.ndarray       # (T,) trailing 24h mean-station mm
    rain168: np.ndarray      # (T,) trailing 168h mean-station mm
    train_t: np.ndarray      # t0 positions
    val_t: np.ndarray


def build_location_data(location: str, merged: pd.DataFrame, stride: int = 4) -> LocationData:
    enc_df = encoder_frame(merged)
    rain = mean_station_rain(merged).to_numpy(dtype=np.float32)
    diff = merged['differential'].to_numpy(dtype=np.float32)
    flow = merged['flow_m3s_Farmoor'].to_numpy(dtype=np.float32)
    idx = merged.index
    enc = enc_df.to_numpy(dtype=np.float32)

    csum = np.concatenate([[0.0], np.cumsum(rain, dtype=np.float64)])
    pos = np.arange(len(rain))
    rain24 = (csum[pos + 1] - csum[np.maximum(0, pos - 23)]).astype(np.float32)
    rain168 = (csum[pos + 1] - csum[np.maximum(0, pos - 167)]).astype(np.float32)

    enc_ok = np.isfinite(enc).all(axis=1)
    # Encoder window fully finite: rolling count of finite rows.
    ok_count = np.convolve(enc_ok.astype(np.int32), np.ones(SEQ_LEN, dtype=np.int32), 'full')[: len(enc_ok)]
    fut_count = np.convolve(np.isfinite(diff).astype(np.int32), np.ones(HORIZON, dtype=np.int32), 'full')
    # fut_count[t + HORIZON] = finite targets in (t, t + HORIZON]
    cand = np.arange(SEQ_LEN, len(idx) - HORIZON - 1, stride)
    good = (
        (ok_count[cand] == SEQ_LEN)
        & np.isfinite(diff[cand])
        & (fut_count[cand + HORIZON] >= 200)
    )
    cand = cand[good]
    t0 = idx[cand]
    train_end = pd.Timestamp(TRAIN_END, tz=idx.tz)
    val_lo = pd.Timestamp(f'{VAL_YEAR}-01-01', tz=idx.tz)
    val_hi = pd.Timestamp(f'{VAL_YEAR + 1}-01-01', tz=idx.tz)
    near_val = (t0 >= val_lo - pd.Timedelta(hours=HORIZON)) & (t0 < val_hi)
    is_val = (t0 >= val_lo) & (t0 < val_hi - pd.Timedelta(hours=HORIZON))
    is_train = (t0 < train_end) & ~near_val
    return LocationData(
        location=location, index=idx, enc=enc, enc_cols=list(enc_df.columns),
        rain=rain, flow=flow, diff=diff, season=doy_phase(idx),
        rain24=rain24, rain168=rain168,
        train_t=cand[np.asarray(is_train)], val_t=cand[np.asarray(is_val)],
    )


class Standardiser:
    """Per-column z-score fitted on training rows, clipped to ±8 sd."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> 'Standardiser':
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.clip(np.nan_to_num(z, nan=0.0), -8, 8).astype(np.float32)

    def state(self) -> Dict:
        return {'mean': self.mean, 'std': self.std}


class WindowSampler:
    """Gathers batches of (encoder, future rain, targets, ...) on a device."""

    def __init__(self, data: LocationData, scaler: Standardiser, device: torch.device):
        self.device = device
        self.enc = torch.from_numpy(scaler.transform(data.enc)).to(device)
        self.rain = torch.from_numpy(data.rain).to(device)
        self.diff = torch.from_numpy(np.nan_to_num(data.diff, nan=0.0)).to(device)
        self.diff_ok = torch.from_numpy(np.isfinite(data.diff)).to(device)
        self.season = torch.from_numpy(data.season).to(device)
        self.rain24 = torch.from_numpy(data.rain24).to(device)
        self.rain168 = torch.from_numpy(data.rain168).to(device)
        self.flow = torch.from_numpy(np.nan_to_num(data.flow, nan=0.0)).to(device)
        self.flow_ok = torch.from_numpy(np.isfinite(data.flow)).to(device)
        months = data.index.month.to_numpy()
        self.winter = torch.from_numpy(np.isin(months, (11, 12, 1, 2, 3))).to(device)
        self.past_off = torch.arange(-SEQ_LEN + 1, 1, device=device)
        self.fut_off = torch.arange(1, HORIZON + 1, device=device)
        self.rain_past_off = torch.arange(-71, 1, device=device)
        self.rain_st = None
        if getattr(data, 'rain_st', None) is not None:
            self.rain_st = torch.from_numpy(np.nan_to_num(data.rain_st, nan=0.0)).to(device)
            self.rain_st_ok = torch.from_numpy(np.isfinite(data.rain_st)).to(device)
        self.pred_flow = None
        if getattr(data, 'pred_flow', None) is not None:
            self.pred_flow = torch.from_numpy(np.log1p(data.pred_flow)).to(device)
            self.pred_flow_row = torch.from_numpy(data.pred_flow_row).to(device)

    def batch(self, t0: np.ndarray) -> Dict[str, torch.Tensor]:
        t = torch.as_tensor(t0, device=self.device, dtype=torch.long)
        past = t[:, None] + self.past_off[None, :]
        fut = t[:, None] + self.fut_off[None, :]
        return {
            'x': self.enc[past],                           # (B, SEQ, F)
            'rain': self.rain[fut],                        # (B, H) mm/h
            'rain_past72': self.rain[t[:, None] + self.rain_past_off[None, :]],
            'season': self.season[fut],                    # (B, H)
            'rain24': self.rain24[t],
            'rain168': self.rain168[t],
            'd0': self.diff[t],
            'y': self.diff[fut],
            'y_ok': self.diff_ok[fut],
            'flow0': self.flow[t],
            'flow': self.flow[fut],
            'flow_ok': self.flow_ok[fut],
            'winter': self.winter[t],
            **({'pred_flow_log': self.pred_flow[self.pred_flow_row[t]]} if self.pred_flow is not None else {}),
            **({'rain_st': self.rain_st[fut], 'rain_st_ok': self.rain_st_ok[fut]} if self.rain_st is not None else {}),
        }


def cache_path(project_root: Path, location: str) -> Path:
    return project_root / 'data' / f'physics_data_{location}.pkl'


def load_location_data(location: str, project_root: Path, merged: Optional[pd.DataFrame] = None) -> LocationData:
    path = cache_path(project_root, location)
    if path.exists():
        return pd.read_pickle(path)
    from ..evaluation import load_merged
    merged = merged if merged is not None else load_merged(location, project_root)
    data = build_location_data(location, merged)
    pd.to_pickle(data, path)
    return data


def attach_station_rain(data: LocationData, merged: pd.DataFrame) -> None:
    """Add data.rain_st (T, S) per-station mm/h (NaN where a gauge is not reporting)."""
    from ..evaluation import rain_station_columns
    cols = rain_station_columns(merged)
    data.station_names = list(cols)
    data.rain_st = merged[cols].clip(lower=0).to_numpy(dtype=np.float32)
