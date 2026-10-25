"""
Forecast archive: what the site forecast, and what the river then did.

Written by generate_all_location_figures.py on every CI run:

* ``docs/archive/observed.json`` – hourly observed differential per location,
  accumulated run by run (the EA API only keeps ~4 weeks, so this is the only
  long record of the live differential after the 2026-01-20 archive ends).
* ``docs/archive/issues/<slot>.json`` – one forecast per 6-hour slot (00, 06,
  12, 18 UTC; the first successful run in a slot is kept): per location the
  forecast start, the rain-ensemble mean and 10/50/90 % quantiles, and the
  total-uncertainty quantiles (rain + model error) when the run had them.
* ``docs/archive/index.json`` – the list of slots, for the history slider.
* ``docs/archive/now.json`` – for the latest observation at each location,
  what every archived forecast from the last 10 days said it would be.
* ``data/forecast_archive/<slot>.npz`` – the raw ensemble inputs (mean-station
  hourly rain per member, per location), so the model can later be
  backtested and recalibrated on real forecast rain rather than observed rain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

SLOT_HOURS = 6
NOW_LOOKBACK_DAYS = 10
ARCHIVE_VERSION = 1


def slot_id(t: pd.Timestamp) -> str:
    t = pd.Timestamp(t)
    t = t.tz_convert('UTC').tz_localize(None) if t.tzinfo else t
    return t.floor(f'{SLOT_HOURS}h').strftime('%Y-%m-%dT%H')


def _naive_utc(times: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(times)))
    return idx.tz_convert('UTC').tz_localize(None) if idx.tz is not None else idx


def _round(values, nd: int = 3) -> list:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(round(f, nd) if np.isfinite(f) else None)
    return out


def _load_json(path: Path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))


# ---------------------------------------------------------------------------
# Observed differential
# ---------------------------------------------------------------------------

def merge_observed(archive_dir: Path, payloads: Dict[str, dict]) -> Dict[str, pd.Series]:
    """Fold each payload's recent observations into observed.json (hourly)."""
    path = archive_dir / 'observed.json'
    stored = _load_json(path, {})
    out: Dict[str, pd.Series] = {}
    for loc, payload in payloads.items():
        series = observed_series(stored.get(loc))
        hist = payload.get('historical') or {}
        if hist.get('times'):
            new = pd.Series(pd.to_numeric(pd.Series(hist['differential']), errors='coerce').to_numpy(),
                            index=_naive_utc(hist['times'])).dropna()
            new = new[~new.index.duplicated(keep='last')]
            new = new.resample('1h').mean().dropna()
            series = new if series is None else new.combine_first(series)
        if series is not None and len(series):
            out[loc] = series.sort_index()
    for loc, s in stored.items():   # keep locations missing from this run
        if loc not in out and observed_series(s) is not None:
            out[loc] = observed_series(s)
    _dump(path, {loc: encode_series(s) for loc, s in out.items()})
    return out


def payload_estimates(payload: dict) -> Optional[pd.Series]:
    """Hourly estimated differential while a gauge is offline: the stand-in gauge (e.g. Botanic Gardens
    for King's Mill) and the neighbouring-stretch estimate. Never used where a real reading exists."""
    parts = []
    h15 = payload.get('historical_15min') or {}
    if h15.get('times') and h15.get('stand_in_from'):
        s = pd.Series(pd.to_numeric(pd.Series(h15['differential']), errors='coerce').to_numpy(), index=_naive_utc(h15['times']))
        parts.append(s[s.index >= _naive_utc([h15['stand_in_from']])[0]])
    for key in ('gauge_stand_in', 'estimated'):
        part = payload.get(key) or {}
        if part.get('times') and part.get('values'):
            parts.append(pd.Series(pd.to_numeric(pd.Series(part['values']), errors='coerce').to_numpy(),
                                   index=_naive_utc(part['times'])))
    if not parts:
        return None
    s = pd.concat(parts).dropna()
    s = s[~s.index.duplicated(keep='first')].sort_index()          # 15-min stand-in readings before hourly estimates
    return s.resample('15min').mean().dropna() if len(s) else None


def merge_estimated(archive_dir: Path, payloads: Dict[str, dict], observed: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """Fold each payload's estimated differential into estimated.json (15-minute), minus times with a real reading."""
    path = archive_dir / 'estimated.json'
    stored = _load_json(path, {})
    out: Dict[str, pd.Series] = {}
    for loc in set(stored) | set(payloads):
        series = observed_series(stored.get(loc))
        new = payload_estimates(payloads[loc]) if loc in payloads else None
        if new is not None and len(new):
            series = new if series is None else new.combine_first(series)
        if series is None or not len(series):
            continue
        real = observed.get(loc)
        if real is not None and len(real):
            series = series[~series.index.floor('1h').isin(real.index)]
        if len(series):
            out[loc] = series.sort_index()
    _dump(path, {loc: encode_series(s, step_min=15) for loc, s in out.items()})
    return out


def encode_series(s: pd.Series, step_min: int = 60) -> dict:
    """Compact regular series: a start time and one value per step (null for gaps)."""
    s = s.sort_index()
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq=f'{step_min}min'))
    return {'start': full.index[0].strftime('%Y-%m-%dT%H:%M:%S'), 'step_h': step_min / 60,
            'values': _round(full.to_numpy(), 4)}


def observed_series(enc: Optional[dict]) -> Optional[pd.Series]:
    if not enc or not enc.get('values'):
        return None
    idx = pd.date_range(pd.Timestamp(enc['start']), periods=len(enc['values']),
                        freq=f"{int(round(60 * enc.get('step_h', 1)))}min")
    return pd.Series(pd.to_numeric(pd.Series(enc['values']), errors='coerce').to_numpy(), index=idx).dropna()


# ---------------------------------------------------------------------------
# Forecast issues
# ---------------------------------------------------------------------------

def summarise_payload(payload: dict) -> dict:
    """The parts of one location's forecast the history views need."""
    ens = payload['ensemble']
    members = np.array([[np.nan if v is None else v for v in m] for m in ens['members']], dtype=float)
    out = {
        't0': pd.Timestamp(payload['forecast_start']).strftime('%Y-%m-%dT%H:%M:%S'),
        'd0': payload.get('current', {}).get('differential'),
        'n_members': int(members.shape[0]),
        'model': payload.get('model', 'september'),
        'mean': _round(np.nanmean(members, axis=0)),
        'p10': _round(np.nanpercentile(members, 10, axis=0)),
        'p50': _round(np.nanpercentile(members, 50, axis=0)),
        'p90': _round(np.nanpercentile(members, 90, axis=0)),
    }
    total = ens.get('total')
    if total:
        for k in ('p10', 'p25', 'p50', 'p75', 'p90'):
            if k in total:
                out[f'total_{k}'] = _round(total[k])
    return out


def write_issue(archive_dir: Path, slot: str, payloads: Dict[str, dict], model: str,
                issued_at: pd.Timestamp, overwrite: bool = False) -> bool:
    path = archive_dir / 'issues' / f'{slot}.json'
    if path.exists() and not overwrite:
        return False
    issue = {
        'version': ARCHIVE_VERSION,
        'slot': slot,
        'issued_at': pd.Timestamp(issued_at).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model': model,
        'locations': {loc: summarise_payload(p) for loc, p in payloads.items() if p.get('ensemble')},
    }
    if not issue['locations']:
        return False
    _dump(path, issue)
    return True


def rebuild_index(archive_dir: Path) -> list:
    entries = []
    for path in sorted((archive_dir / 'issues').glob('*.json')):
        issue = _load_json(path, None)
        if not issue:
            continue
        entries.append({
            'slot': issue['slot'], 'issued_at': issue['issued_at'], 'model': issue.get('model'),
            'has_total': any('total_p10' in v for v in issue['locations'].values()),
            't0': {loc: v['t0'] for loc, v in issue['locations'].items()},
        })
    _dump(archive_dir / 'index.json', {'version': ARCHIVE_VERSION, 'slot_hours': SLOT_HOURS, 'issues': entries})
    return entries


def build_now(archive_dir: Path, observed: Dict[str, pd.Series], lookback_days: int = NOW_LOOKBACK_DAYS,
              estimated: Optional[Dict[str, pd.Series]] = None) -> dict:
    """For the latest observation per location (or the latest estimate, if the gauge has been offline since):
    what each recent issue forecast for it."""
    out = {'version': ARCHIVE_VERSION, 'locations': {}}
    issues = sorted((archive_dir / 'issues').glob('*.json'))
    for loc in set(observed) | set(estimated or {}):
        obs = observed.get(loc)
        est = (estimated or {}).get(loc)
        is_est = False
        if est is not None and len(est) and (obs is None or not len(obs) or est.index[-1] > obs.index[-1]):
            obs, is_est = est, True
        if obs is None or not len(obs):
            continue
        target = obs.index[-1]
        rows = []
        for path in issues:
            slot_time = pd.Timestamp(path.stem.replace('T', ' ') + ':00')
            if slot_time < target - pd.Timedelta(days=lookback_days + 1) or slot_time > target:
                continue
            issue = _load_json(path, None)
            v = (issue or {}).get('locations', {}).get(loc)
            if not v:
                continue
            lead = int(round((target - pd.Timestamp(v['t0'])) / pd.Timedelta(hours=1)))
            if lead < 1 or lead >= len(v['mean']):
                continue
            row = {'slot': issue['slot'], 't0': v['t0'], 'lead_h': lead,
                   'mean': v['mean'][lead], 'p10': v['p10'][lead], 'p90': v['p90'][lead]}
            for k in ('total_p10', 'total_p90', 'total_p25', 'total_p75'):
                if k in v:
                    row[k] = v[k][lead]
            rows.append(row)
        out['locations'][loc] = {'target': target.strftime('%Y-%m-%dT%H:%M:%S'),
                                 'observed': round(float(obs.iloc[-1]), 4), 'estimated': is_est, 'forecasts': rows}
    _dump(archive_dir / 'now.json', out)
    return out


def write_raw_inputs(raw_dir: Path, slot: str, member_rain: Dict[str, pd.DataFrame],
                     d0: Dict[str, float]) -> Optional[Path]:
    """Mean-station hourly rain per member and location, in 0.01 mm (uint16)."""
    if not member_rain:
        return None
    path = raw_dir / f'{slot}.npz'
    if path.exists():
        return None
    arrays = {}
    for loc, df in member_rain.items():
        arr = np.clip(np.nan_to_num(df.to_numpy(dtype=float).T, nan=0.0) * 100, 0, 65535).round().astype(np.uint16)
        arrays[f'{loc}_rain_cmm'] = arr                     # (members, hours)
        arrays[f'{loc}_start'] = np.array(str(pd.Timestamp(df.index[0])))
        arrays[f'{loc}_d0'] = np.array(d0.get(loc, np.nan))
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def fill_gaps_from_git(docs_dir: Path, repo_root: Path, now: Optional[pd.Timestamp] = None,
                       max_days: int = 30, default_model: str = 'hybrid_c1_ps') -> list:
    """Archive slots missed since the last archived one, from the git history of forecast_data.json.

    Every CI run commits docs/forecast_data.json, so after an outage of the
    archive (or before this code was merged) the forecasts are still in git.
    In a shallow clone (CI) it first fetches the history since the last archived slot.
    """
    import subprocess
    archive_dir = Path(docs_dir) / 'archive'
    index = _load_json(archive_dir / 'index.json', {}).get('issues', [])
    now = pd.Timestamp.utcnow() if now is None else pd.Timestamp(now)
    now = now.tz_convert('UTC').tz_localize(None) if now.tzinfo else now
    if index:
        since = pd.Timestamp(index[-1]['slot'].replace('T', ' ') + ':00') + pd.Timedelta(hours=SLOT_HOURS)
    else:
        since = now - pd.Timedelta(days=max_days)
    since = max(since, now - pd.Timedelta(days=max_days))
    if slot_id(since) >= slot_id(now):
        return []
    # CI checks out only the latest commit: fetch just the history since the last archived slot.
    if (Path(repo_root) / '.git' / 'shallow').exists():
        subprocess.run(['git', 'fetch', '--quiet', f'--shallow-since={since.isoformat()}Z', 'origin', 'main'],
                       cwd=repo_root, capture_output=True, text=True, timeout=300)
    log = subprocess.run(['git', 'log', '--format=%H %cI', f'--since={since.isoformat()}Z', '--reverse',
                          '--', 'docs/forecast_data.json'], cwd=repo_root, capture_output=True, text=True, check=True)
    first_in_slot = {}
    for line in log.stdout.split('\n'):
        if not line.strip():
            continue
        sha, date = line.split()
        first_in_slot.setdefault(slot_id(pd.Timestamp(date)), sha)
    written = []
    for slot, sha in sorted(first_in_slot.items()):
        if slot >= slot_id(now) or (archive_dir / 'issues' / f'{slot}.json').exists():
            continue
        try:
            raw = subprocess.run(['git', 'show', f'{sha}:docs/forecast_data.json'], cwd=repo_root,
                                 capture_output=True, text=True, check=True).stdout
            bundle = json.loads(raw)
        except Exception:
            continue
        payloads = bundle.get('locations', {})
        for p in payloads.values():
            p.setdefault('model', default_model)
        issued = pd.Timestamp(bundle.get('generated_at', slot))
        merge_observed(archive_dir, payloads)
        if write_issue(archive_dir, slot, payloads, f'{default_model} (from git history)', issued):
            written.append(slot)
    if written:
        rebuild_index(archive_dir)
    return written


def update_archive(docs_dir: Path, payloads: Dict[str, dict], model: str,
                   member_rain: Optional[Dict[str, pd.DataFrame]] = None,
                   raw_dir: Optional[Path] = None, now: Optional[pd.Timestamp] = None) -> dict:
    """Everything a CI run adds to the archive. Never raises on bad input."""
    archive_dir = Path(docs_dir) / 'archive'
    now = pd.Timestamp.utcnow() if now is None else pd.Timestamp(now)
    observed = merge_observed(archive_dir, payloads)
    estimated = merge_estimated(archive_dir, payloads, observed)
    slot = slot_id(now)
    wrote = write_issue(archive_dir, slot, payloads, model, now)
    raw = None
    if wrote and member_rain and raw_dir is not None:
        raw = write_raw_inputs(Path(raw_dir), slot, member_rain,
                               {loc: p.get('current', {}).get('differential') for loc, p in payloads.items()})
    rebuild_index(archive_dir)
    build_now(archive_dir, observed, estimated=estimated)
    return {'slot': slot, 'wrote_issue': wrote, 'raw': str(raw) if raw else None}
