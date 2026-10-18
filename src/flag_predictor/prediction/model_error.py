"""
Model-error spread from the model's own past mistakes.

The 50 forecast lines only carry rain-forecast uncertainty. Even with perfect
rain the model is wrong, and early in the forecast (when the rain members
barely disagree) that is most of the real uncertainty. This module adds it.

How: forecasts are re-run every 6 hours over periods the model never trained
on (the 2023 validation year and the Nov 2024 – Jan 2026 test window) with the
rain that actually fell ("clairvoyant" rain). Each run's error trajectory
(observed − forecast, hour by hour for 10 days) goes into a library, along
with a few numbers describing the situation it started from:

    level         differential at t0
    trend         change over the previous 24 h
    rain_72h      rain in the first 3 forecast days
    rain_240h     rain over the whole 10 days
    wetness       rain over the previous 7 days
    season        cosine of the day of year (winter / summer)

For a new forecast, each rain member finds the past starts most like its own
situation (k nearest neighbours) and borrows several of their error
trajectories. Adding them to the member's line gives a cloud of possible
outcomes that carries both rain and model uncertainty; the shaded band and
the flag probabilities come from that cloud. Whole trajectories are borrowed,
so errors keep their real persistence over time (a forecast that is too low
at hour 30 tends to still be too low at hour 40), and the spread grows with
lead time exactly as fast as the model's errors did.

Errors are only ever measured out of sample, so the spread is not flattered
by data the model was fitted to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

FEATURES = ('level', 'trend', 'rain_72h', 'rain_240h', 'wetness', 'season')
FEATURE_WEIGHTS = np.array([2.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)
K_NEIGHBOURS = 40
SAMPLES_PER_MEMBER = 10
QUANTILES = {'p10': 10, 'p25': 25, 'p50': 50, 'p75': 75, 'p90': 90}


def start_features(d0: float, d_24h_ago: float, rain_future: np.ndarray, rain_past_168h: float,
                   t0: pd.Timestamp) -> np.ndarray:
    """Situation descriptor for one forecast (rain in mm/h, mean over gauges)."""
    rain_future = np.nan_to_num(np.asarray(rain_future, dtype=float), nan=0.0).clip(min=0)
    trend = d0 - d_24h_ago if np.isfinite(d_24h_ago) else 0.0
    doy = pd.Timestamp(t0).dayofyear
    return np.array([
        d0,
        trend,
        np.log1p(rain_future[:72].sum()),
        np.log1p(rain_future[:240].sum()),
        np.log1p(max(rain_past_168h, 0.0)),
        np.cos(2 * np.pi * (doy - 15) / 365.25),   # +1 mid-January, -1 mid-July
    ], dtype=np.float32)


def fill_trajectory(r: np.ndarray) -> np.ndarray:
    """Interpolate gaps in an error trajectory; hold the last value past the end."""
    r = np.asarray(r, dtype=float).copy()
    ok = np.isfinite(r)
    if not ok.any():
        return np.zeros_like(r)
    x = np.arange(len(r))
    r[~ok] = np.interp(x[~ok], x[ok], r[ok])
    return r


@dataclass
class ErrorLibrary:
    location: str
    model: str
    t0: np.ndarray            # (N,) datetime64 forecast starts
    features: np.ndarray      # (N, F)
    residuals: np.ndarray     # (N, H) observed - forecast, gaps filled
    scale: np.ndarray         # (F,) feature scale used for distances
    inflation: Optional[np.ndarray] = None   # (H,) spread calibration from cross-validation
    k: int = K_NEIGHBOURS

    @classmethod
    def build(cls, location: str, model: str, t0, features, residuals) -> 'ErrorLibrary':
        features = np.asarray(features, dtype=np.float32)
        res = np.stack([fill_trajectory(r) for r in residuals]).astype(np.float32)
        scale = np.nanstd(features, axis=0).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        return cls(location, model, np.asarray(t0, dtype='datetime64[s]'), features, res, scale)

    def inflation_or_ones(self) -> np.ndarray:
        H = self.residuals.shape[1]
        return self.inflation if self.inflation is not None else np.ones(H, np.float32)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, location=self.location, model=self.model, t0=self.t0.astype('int64'),
                            features=self.features, residuals=self.residuals.astype(np.float16),
                            scale=self.scale, feature_names=np.array(FEATURES),
                            inflation=self.inflation_or_ones().astype(np.float32), k=np.array(self.k),
                            feature_weights=FEATURE_WEIGHTS)

    @classmethod
    def load(cls, path: Path) -> 'ErrorLibrary':
        z = np.load(path, allow_pickle=False)
        return cls(str(z['location']), str(z['model']), z['t0'].astype('datetime64[s]'),
                   z['features'].astype(np.float32), z['residuals'].astype(np.float32),
                   z['scale'].astype(np.float32),
                   z['inflation'].astype(np.float32) if 'inflation' in z.files else None,
                   int(z['k']) if 'k' in z.files else K_NEIGHBOURS)

    def subset(self, mask: np.ndarray) -> 'ErrorLibrary':
        return ErrorLibrary(self.location, self.model, self.t0[mask], self.features[mask],
                            self.residuals[mask], self.scale, self.inflation, self.k)

    def neighbours(self, query: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """(M, k) indices of the most similar past starts for each query row."""
        k = k or self.k
        q = np.atleast_2d(query).astype(np.float32)
        w = FEATURE_WEIGHTS / self.scale
        d = (((q[:, None, :] - self.features[None, :, :]) * w) ** 2).sum(-1)
        k = min(k, len(self.features))
        return np.argpartition(d, k - 1, axis=1)[:, :k]

    def sample(self, query: np.ndarray, n: int = SAMPLES_PER_MEMBER, k: Optional[int] = None,
               rng: Optional[np.random.Generator] = None, inflate: bool = True) -> np.ndarray:
        """(M, n, H) error trajectories borrowed from each query's neighbours."""
        rng = rng or np.random.default_rng(0)
        nb = self.neighbours(query, k)
        pick = nb[np.arange(len(nb))[:, None], rng.integers(0, nb.shape[1], size=(len(nb), n))]
        res = self.residuals[pick]
        return res * self.inflation_or_ones()[None, None, :] if inflate else res


def total_uncertainty(members: np.ndarray, member_features: np.ndarray, library: ErrorLibrary,
                      n: int = SAMPLES_PER_MEMBER, floor: Optional[float] = None,
                      seed: int = 0) -> np.ndarray:
    """
    members: (M, H+1) rain-member trajectories, index 0 = t0.
    Returns (M * n, H+1) outcomes = member + borrowed model error.
    """
    rng = np.random.default_rng(seed)
    H = members.shape[1] - 1
    err = library.sample(member_features, n=n, rng=rng)[:, :, :H]      # (M, n, H)
    if err.shape[2] < H:
        err = np.concatenate([err, np.repeat(err[:, :, -1:], H - err.shape[2], axis=2)], axis=2)
    out = members[:, None, :].repeat(n, axis=1)
    out[:, :, 1:] += err
    out = out.reshape(-1, H + 1)
    if floor is not None:
        out = np.maximum(out, floor)
    return out


def quantile_band(samples: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.nanpercentile(samples, q, axis=0) for k, q in QUANTILES.items()}


def flag_probabilities(samples: np.ndarray, thresholds: Dict[str, Sequence[float]],
                       order: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    for key in order:
        lo, hi = thresholds[key]
        out[key] = ((samples >= lo) & (samples < hi)).mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def crps_samples(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS per lead from an ensemble (S, H) and observations (H,) (energy form)."""
    s = np.sort(samples, axis=0)
    S = s.shape[0]
    term1 = np.nanmean(np.abs(s - obs[None, :]), axis=0)
    # E|X - X'| from sorted samples: 2/S^2 * sum_i (2i - S - 1) x_(i)
    i = np.arange(1, S + 1)[:, None]
    term2 = (2.0 / (S * S)) * np.sum((2 * i - S - 1) * s, axis=0)
    return term1 - 0.5 * term2


def coverage(samples: np.ndarray, obs: np.ndarray, lo_q: float = 10, hi_q: float = 90) -> np.ndarray:
    lo = np.nanpercentile(samples, lo_q, axis=0)
    hi = np.nanpercentile(samples, hi_q, axis=0)
    ok = np.isfinite(obs)
    out = np.full(obs.shape, np.nan)
    out[ok] = ((obs[ok] >= lo[ok]) & (obs[ok] <= hi[ok])).astype(float)
    return out
