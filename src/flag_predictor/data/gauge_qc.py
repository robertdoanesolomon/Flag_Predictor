"""
Faulty rain gauge check, and filling gauge gaps from neighbours.

The common fault is a blocked funnel: the gauge keeps reporting but catches little or
nothing while the gauges around it fill up. The model's encoder sums rain over its
gauges with gaps counted as zero, so a blocked or late gauge reads as a dry patch.

Check (per gauge, at the end of each day): walk back over the most recent days its
neighbours call wet (neighbour rain >= WET_MM) until they add up to WINDOW_MM. The
gauge is faulty if it caught under LOW_RATIO of the neighbours' rain over those days
and at least MIN_AGREE of its NEIGHBOURS nearest gauges each had over 3x its total
(so a single neighbour's downpour can't cause a flag). The neighbour rain is the
median of the NEIGHBOURS nearest gauges, scaled by the gauge's long-run ratio to
them, and is only counted over hours the gauge itself reported, so a gauge that is
just late isn't called faulty.

Backtest (backtest_gauge_qc.py, EA qualified daily rain 2015-2026, 19 gauges): flags
~1.5% of gauge-days, mostly runs of weeks with the gauge near zero that the EA later
marked Suspect/Missing/Estimated (e.g. Benson Mar-Jun 2024, Stanford Oct-Nov 2024).

Days in a flagged window where the gauge was low (or dry) are blanked, as are the wet
days just before it that the gauge also under-caught (the fault's start), and every
blank hour (faulty, late or missing) is filled with the mean of its nearest reporting
gauges, scaled by the long-run ratio. Hours no neighbour reported stay blank.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Gauge positions (short names as in the live frames).
COORDS: Dict[str, Tuple[float, float]] = {
    'Osney': (51.750, -1.272), 'Eynsham': (51.789, -1.402), 'St': (51.7, -1.5),
    'Shorncote': (51.666, -1.916), 'Rapsgate': (51.815, -1.975), 'Stowell': (51.833, -1.821),
    'Bourton': (51.884, -1.758), 'Chipping': (51.942, -1.547), 'Grimsbury': (52.065, -1.326),
    'Bicester': (51.899, -1.155), 'Byfield': (52.179, -1.274), 'Swindon': (51.556, -1.779),
    'Worsham': (51.817, -1.498), 'Stanford': (51.650, -1.500), 'Abingdon': (51.670, -1.283),
    'Wheatley': (51.747, -1.140), 'Benson': (51.620, -1.110), 'Aylesbury': (51.817, -0.813),
    'Cleeve': (51.680, -1.200),
}
# Long-run rain at each gauge / mean of its NEIGHBOURS nearest gauges (EA qualified daily, 2015-2026).
RATIO: Dict[str, float] = {
    'Bicester': 0.96, 'Bourton': 1.03, 'Byfield': 0.95, 'Chipping': 1.06, 'Eynsham': 0.99,
    'Grimsbury': 0.98, 'Osney': 1.03, 'Rapsgate': 1.16, 'Shorncote': 0.92, 'St': 1.01,
    'Stowell': 1.01, 'Swindon': 1.07, 'Worsham': 1.08, 'Stanford': 0.93, 'Abingdon': 0.96,
    'Wheatley': 1.04, 'Benson': 0.96, 'Aylesbury': 1.01, 'Cleeve': 1.13,
}
NEIGHBOURS = 4        # nearest gauges compared with
WINDOW_MM = 15.0      # neighbour rain the check looks back over
WET_MM = 1.0          # a day counts if the neighbours had at least this
MAX_DAYS = 30         # never look back further than this
LOW_RATIO = 0.3       # faulty if the gauge caught less than this share
MIN_AGREE = 3         # ... and this many neighbours each had > 3x the gauge
BLANK_BELOW = 0.5     # within a flagged window, blank wet days where the gauge had < this share


def _km(a: str, b: str) -> float:
    (la, lo), (lb, lob) = COORDS[a], COORDS[b]
    return float(np.hypot((la - lb) * 111.0, (lo - lob) * 111.0 * np.cos(np.radians(51.8))))


def neighbours(gauge: str, available: List[str], k: int = NEIGHBOURS) -> List[str]:
    others = [g for g in available if g != gauge and g in COORDS]
    return sorted(others, key=lambda g: _km(gauge, g))[:k]


def _check_gauge(g: np.ndarray, nb: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Daily walk-back check for one gauge.

    g: (D,) gauge daily totals (NaN = no readings); nb: (D, K) neighbour daily totals over the
    hours the gauge reported. Returns (ratio caught at each day's end or NaN, faulty flag at each
    day's end, the [start, end] day span of each flagged window).
    """
    n = len(g)
    nbs = nb * ratio
    med = np.nanmedian(np.where(np.isfinite(nbs), nbs, np.nan), axis=1) if nbs.shape[1] else np.full(n, np.nan)
    caught = np.full(n, np.nan)
    flag = np.zeros(n, bool)
    spans: List[Tuple[int, int]] = []
    for t in range(n):
        sg = sn = 0.0
        snb = np.zeros(nbs.shape[1])
        first = None
        for s in range(t, max(-1, t - MAX_DAYS), -1):
            if not np.isfinite(med[s]) or med[s] < WET_MM:
                continue
            if not np.isfinite(g[s]):
                continue                    # no readings that day: nothing to compare
            sg += g[s]
            sn += med[s]
            snb += np.nan_to_num(nbs[s])
            first = s
            if sn >= WINDOW_MM:
                break
        if sn >= WINDOW_MM:
            caught[t] = sg / sn
            if caught[t] < LOW_RATIO and int((snb > 3 * sg).sum()) >= MIN_AGREE:
                flag[t] = True
                spans.append((first, t))
    return caught, flag, spans


def check_gauges(hourly: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Blank faulty days. ``hourly``: hourly rain (mm), one column per gauge, NaN where no reading.

    Returns (the frame with faulty days blanked, a report: gauge -> {'caught', 'since', 'faulty_now'}
    for every gauge flagged at some point in the frame).
    """
    gauges = [c for c in hourly.columns if c in COORDS]
    out = hourly.copy()
    days = hourly.index.floor('D')
    report: Dict[str, dict] = {}
    for gauge in gauges:
        nbs = neighbours(gauge, gauges)
        if len(nbs) < MIN_AGREE:
            continue
        has = hourly[gauge].notna()
        g_day = hourly[gauge].groupby(days).sum(min_count=1)
        nb_day = pd.concat([hourly[n].where(has).groupby(days).sum(min_count=1) for n in nbs], axis=1)
        caught, flag, spans = _check_gauge(g_day.to_numpy(float), nb_day.to_numpy(float), RATIO.get(gauge, 1.0))
        if not flag.any():
            continue
        med = nb_day.median(axis=1).to_numpy(float) * RATIO.get(gauge, 1.0)
        bad = np.zeros(len(g_day), bool)
        low = lambda s: np.isfinite(med[s]) and med[s] >= WET_MM and (
            not np.isfinite(g_day.iloc[s]) or g_day.iloc[s] < BLANK_BELOW * med[s])
        for i, (a, b) in enumerate(spans):
            # The fault usually began before the window that tripped the check: walk back over
            # wet days the gauge also under-caught (dry days in between go with them).
            s, first = a - 1, a
            while s >= max(0, a - MAX_DAYS):
                if low(s):
                    first = s
                elif np.isfinite(med[s]) and med[s] >= WET_MM:
                    break
                s -= 1
            spans[i] = (first, b)
        for a, b in spans:
            for s in range(a, b + 1):
                gs = g_day.iloc[s]
                if not np.isfinite(med[s]) or med[s] < WET_MM or not np.isfinite(gs) or gs < BLANK_BELOW * med[s]:
                    bad[s] = True
        bad_days = g_day.index[bad]
        out.loc[days.isin(bad_days), gauge] = np.nan
        last = int(np.where(np.isfinite(caught))[0][-1]) if np.isfinite(caught).any() else None
        report[gauge] = {
            'faulty_now': bool(last is not None and flag[last]),
            'caught': None if last is None else round(float(caught[last]), 2),
            'since': str(bad_days.min().date()) if len(bad_days) else None,
        }
    return out, report


def fill_from_neighbours(hourly: pd.DataFrame, k: int = NEIGHBOURS) -> Tuple[pd.DataFrame, pd.Series]:
    """Fill each blank hour with the mean of the gauge's nearest gauges that reported it, scaled
    by its long-run ratio. Returns (filled frame, hours filled per gauge)."""
    gauges = [c for c in hourly.columns if c in COORDS]
    out = hourly.copy()
    filled = {}
    for gauge in gauges:
        miss = hourly[gauge].isna()
        if not miss.any():
            filled[gauge] = 0
            continue
        nbs = neighbours(gauge, gauges, k)
        est = hourly[nbs].mean(axis=1) * RATIO.get(gauge, 1.0)  # NaN only if none reported
        out.loc[miss, gauge] = est[miss]
        filled[gauge] = int((miss & est.notna()).sum())
    return out, pd.Series(filled, dtype=int)


def clean_rain(hourly: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """check_gauges then fill_from_neighbours. Never raises: on any error returns the input."""
    try:
        checked, report = check_gauges(hourly)
        filled, n = fill_from_neighbours(checked)
    except Exception as exc:  # the check is a safeguard; never lose the forecast over it
        if verbose:
            print(f"  [gauge check] failed ({type(exc).__name__}: {exc}); rain left as reported")
        return hourly, {}
    if verbose:
        for g, r in report.items():
            state = 'faulty now' if r['faulty_now'] else 'was faulty'
            print(f"  [gauge check] {g}: {state} (caught {r['caught']} of its neighbours' rain); "
                  f"blanked from {r['since']} and filled from neighbours")
        late = {g: v for g, v in n.items() if v and g not in report}
        if late:
            print(f"  [gauge check] hours filled from neighbours (no reading): {late}")
    for g in report:
        report[g]['hours_filled'] = int(n.get(g, 0))
    return filled, report
