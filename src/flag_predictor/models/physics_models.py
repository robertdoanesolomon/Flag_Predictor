"""
Physically structured differential forecaster (physics-redesign candidate B).

An LSTM encoder reads the recent past and sets the *initial state* of a small
differentiable rainfall-runoff model; future rain then drives that model hour
by hour, and a learned monotonic rating curve maps its channel outflow to the
differential.

Structure (all stores non-negative, units mm over the catchment, hourly):

    rain P ─► soil (wetness s) ─► runoff R = P·s^β
                                   ├─ quick store   (1 linear reservoir)
                                   ├─ medium stores (3-reservoir cascade)
                                   └─ groundwater   (1 slow reservoir)
                                                 ▼
                                      channel store C ─► Q = C/k_c ─► D = g(log Q)

What this guarantees by construction, with no clamps:

* The forecast starts exactly at the observed differential: C₀ is chosen so
  that g(log Q₀) = D(t0).
* No rise without water. Groundwater starts at ρ·Q₀ outflow with ρ ≤ 1, and
  the quick/medium stores can hold at most the rain of the last 24h/168h. With
  no recent and no future rain the channel inflow never exceeds its outflow,
  so Q, and therefore D, can only hold or fall.
* More rain never lowers the forecast: every path from rain to Q is
  non-decreasing and g is increasing.
* Smooth trajectories: Q is the output of a reservoir cascade.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bounded(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def _unbound(value: float, lo: float, hi: float) -> float:
    p = (value - lo) / (hi - lo)
    return math.log(p / (1 - p))


class MonotoneRatingCurve(nn.Module):
    """Increasing piecewise-linear D = g(z), z = log Q, with an exact inverse."""

    def __init__(self, d_lo: float, d_hi: float, z_min: float, z_max: float, n_knots: int = 24):
        super().__init__()
        self.d_lo = d_lo
        self.z_min, self.z_max = z_min, z_max
        self.register_buffer('z_knots', torch.linspace(z_min, z_max, n_knots + 1))
        self.dz = (z_max - z_min) / n_knots
        slope0 = (d_hi - d_lo) / (z_max - z_min)
        self.slope_raw = nn.Parameter(torch.full((n_knots,), math.log(math.expm1(slope0))))

    @torch.no_grad()
    def init_from_data(self, z_obs, d_obs) -> None:
        """Start from the empirical rating: quantile-match observed log-flow to D."""
        import numpy as np
        z_obs = np.asarray(z_obs, dtype=float)
        d_obs = np.asarray(d_obs, dtype=float)
        knots = self.z_knots.cpu().numpy()
        cdf = np.searchsorted(np.sort(z_obs), knots) / len(z_obs)
        d_k = np.quantile(d_obs, np.clip(cdf, 0, 1))
        d_k[0] = self.d_lo  # the curve is anchored at d_lo at z_min
        slopes = np.maximum(np.diff(d_k) / self.dz, 2e-3)
        self.slope_raw.copy_(torch.as_tensor(np.log(np.expm1(slopes - 1e-3 + 1e-6)), dtype=torch.float32))

    def slopes(self) -> torch.Tensor:
        return F.softplus(self.slope_raw) + 1e-3

    def d_knots(self) -> torch.Tensor:
        s = self.slopes()
        return torch.cat([s.new_tensor([self.d_lo]), self.d_lo + torch.cumsum(s * self.dz, 0)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.slopes()
        dk = self.d_knots()
        seg = torch.clamp(((z - self.z_min) / self.dz).floor().long(), 0, len(s) - 1)
        return dk[seg] + s[seg] * (z - self.z_knots[seg])

    def inverse(self, d: torch.Tensor) -> torch.Tensor:
        s = self.slopes()
        dk = self.d_knots()
        seg = torch.clamp(torch.searchsorted(dk, d.contiguous()) - 1, 0, len(s) - 1)
        return self.z_knots[seg] + (d - dk[seg]) / s[seg]


# Farmoor catchment ~1609 km²: 1 mm/h of runoff over it is ~447 m³/s.
M3S_PER_MM_H = 447.0


class ReservoirHybrid(nn.Module):
    architecture = 'reservoir_hybrid'
    N_MEDIUM = 3
    # Physical ranges (hours / mm / mm per hour) for the global parameters.
    RANGES = {
        'k_quick': (1.0, 36.0, 6.0),
        'k_medium': (3.0, 96.0, 16.0),
        'k_slow': (48.0, 4000.0, 600.0),
        'k_channel': (1.0, 72.0, 8.0),
        'beta': (0.3, 6.0, 2.0),
        'smax': (10.0, 400.0, 100.0),
        'et_base': (0.0, 0.1, 0.01),
        'et_summer': (0.0, 0.3, 0.08),
    }

    def __init__(
        self,
        n_features: int,
        d_lo: float,
        d_hi: float,
        hidden: int = 64,
        dropout: float = 0.2,
        z_min: float = math.log(1e-4),
        z_max: float = math.log(3.0),
        per_sample_params: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.per_sample_params = per_sample_params
        self.q_floor = math.exp(z_min)
        self.encoder = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        # s0, rho, quick fill, medium fill, 3 medium-distribution logits
        n_state = 4 + self.N_MEDIUM
        n_mod = 4 if per_sample_params else 0   # k_quick, k_medium, beta, split shift
        self.head = nn.Linear(hidden, n_state + n_mod)
        self.params_raw = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(_unbound(init, lo, hi)))
            for name, (lo, hi, init) in self.RANGES.items()
        })
        # Runoff split over quick / medium / slow paths.
        self.split_logits = nn.Parameter(torch.tensor([0.0, 0.5, 0.0]))
        self.rating = MonotoneRatingCurve(d_lo, d_hi, z_min, z_max)

    def param(self, name: str, shift: torch.Tensor = None) -> torch.Tensor:
        lo, hi, _ = self.RANGES[name]
        raw = self.params_raw[name]
        if shift is not None:
            raw = raw + shift
        return _bounded(raw, lo, hi)

    def initial_state(self, x: torch.Tensor):
        h, _ = self.encoder(x)
        return self.head(self.drop(h[:, -1]))

    def forward(
        self,
        x: torch.Tensor,
        rain: torch.Tensor,
        season: torch.Tensor,
        rain24: torch.Tensor,
        rain168: torch.Tensor,
        d0: torch.Tensor,
        return_states: bool = False,
        return_q: bool = False,
    ):
        """
        x: (B, SEQ, F) scaled encoder features; rain: (B, H) mm/h mean-station;
        season: (B, H) cos day-of-year phase; rain24/rain168: (B,) mm;
        d0: (B,) observed differential at t0.  Returns D: (B, H).
        """
        out = self.initial_state(x)
        s = torch.sigmoid(out[:, 0])
        rho = torch.sigmoid(out[:, 1])
        quick = torch.sigmoid(out[:, 2]) * rain24
        medium_total = torch.sigmoid(out[:, 3]) * rain168
        medium_w = torch.softmax(out[:, 4:4 + self.N_MEDIUM], dim=1)
        medium = [medium_total * medium_w[:, i] for i in range(self.N_MEDIUM)]

        mod = out[:, 4 + self.N_MEDIUM:] if self.per_sample_params else None
        k_q = self.param('k_quick', mod[:, 0] if mod is not None else None)
        k_m = self.param('k_medium', mod[:, 1] if mod is not None else None)
        beta = self.param('beta', mod[:, 2] if mod is not None else None)
        k_s = self.param('k_slow')
        k_c = self.param('k_channel')
        smax = self.param('smax')
        et_base = self.param('et_base')
        et_summer = self.param('et_summer')
        split = torch.softmax(self.split_logits, 0)
        if mod is not None:
            # Wetter-soil shift toward the quick path, still a valid split.
            logits = self.split_logits[None, :] + torch.stack(
                [mod[:, 3], torch.zeros_like(mod[:, 3]), -mod[:, 3]], dim=1
            )
            split = torch.softmax(logits, 1)
            f_q, f_m, f_s = split[:, 0], split[:, 1], split[:, 2]
        else:
            f_q, f_m, f_s = split[0], split[1], split[2]

        a_q = 1 - torch.exp(-1 / k_q)
        a_m = 1 - torch.exp(-1 / k_m)
        a_s = 1 - torch.exp(-1 / k_s)
        a_c = 1 - torch.exp(-1 / k_c)

        # Pin the channel to the observation: g(log(Q0 + floor)) = d0.
        # Stores hold their post-outflow content, so with inflow I the next
        # outflow is (1 - a)·Q0 + a·I: steady when I = Q0, falling when I < Q0.
        z0 = self.rating.inverse(d0)
        q0 = torch.clamp(torch.exp(z0) - self.q_floor, min=1e-7)
        channel = q0 * (1 - a_c) / a_c
        slow = rho * q0 * (1 - a_s) / a_s

        H = rain.shape[1]
        et = et_base + et_summer * torch.clamp(season, min=0)
        qs = []
        states = []
        for t in range(H):
            p = torch.clamp(rain[:, t], min=0)
            runoff = p * s.clamp(min=1e-6) ** beta
            s = torch.clamp(s + (p - runoff - et[:, t] * s) / smax, 0.0, 1.0)
            quick = quick + f_q * runoff
            o_q = a_q * quick
            quick = quick - o_q
            inflow_m = f_m * runoff
            for i in range(self.N_MEDIUM):
                medium[i] = medium[i] + inflow_m
                inflow_m = a_m * medium[i]
                medium[i] = medium[i] - inflow_m
            slow = slow + f_s * runoff
            o_s = a_s * slow
            slow = slow - o_s
            channel = channel + o_q + inflow_m + o_s
            q = a_c * channel
            channel = channel - q
            qs.append(q)
            if return_states:
                states.append(torch.stack([s, quick, sum(medium), slow, channel], 1))
        q = torch.stack(qs, 1)
        d = self.rating(torch.log(q + self.q_floor))
        if return_states:
            return d, q, torch.stack(states, 1)
        if return_q:
            return d, q
        return d

    def describe(self) -> Dict[str, float]:
        out = {name: float(self.param(name)) for name in self.RANGES}
        split = torch.softmax(self.split_logits, 0).tolist()
        out.update({'split_quick': split[0], 'split_medium': split[1], 'split_slow': split[2]})
        return out
