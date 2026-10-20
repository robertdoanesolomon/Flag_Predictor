"""
Extra encoder inputs: measured soil moisture and upstream river levels (branch soil-upstream).

The hybrid's encoder sets the starting state of its stores from the last few weeks of rain,
river level and Farmoor flow. Two things it cannot see:

* how wet the ground actually is: COSMOS-UK cosmic-ray soil moisture at Chimney Meadows
  (UKCEH, daily, published about a day late; EXPERIMENTS.md "Model soil wetness vs measured
  soil moisture");
* water already on its way down the tributaries: EA level gauges upstream (Windrush, Evenlode,
  Coln, upper Thames, Cherwell, Ray, Thame, Ock), 15-min, in the flood-monitoring API live
  and the hydrology API (qualified) for training.

Features (hourly, unscaled; the model's Standardiser maps missing values to the mean):
  up_<ref>_rise   level above its 14-day minimum (m): how far above baseflow the river is
  up_<ref>_d6     6 h change (m)
  up_<ref>_d24    24 h change (m)
  soil_vwc        measured volumetric water content (%), each day's value from the next noon
  soil_vwc_d7     its change over the previous week (%)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# EA station reference -> (river, place). Level gauges on natural reaches (not lock cuts).
UPSTREAM: Dict[str, str] = {
    '1080TH': 'Windrush at Worsham',
    '1238TH': 'Evenlode at Shipton',
    '0680TH': 'Coln at Fairford',
    '0190TH': 'Thames at Cricklade',
    '1453TH': 'Cherwell at Heyford Bridge',
    '1489TH': 'Ray at Islip',
    '1980TH': 'Thame at Wheatley',
    '1790TH': 'Ock at Abingdon',
}
# EA hydrology API measure (15-min level, qualified; it keeps up within the hour) per station.
HYDROLOGY_LEVEL = {
    '1080TH': 'bed0ef93-6faa-4f98-af56-bfb3fed3ed61', '1238TH': '22eb53c1-ea9b-4094-a813-6d8f9328b8aa',
    '0680TH': 'acf6c077-f35a-4af5-9191-f8aec1831c80', '0190TH': 'bf85754c-4e6e-481b-9f58-a4fe2bf249cd',
    '1453TH': 'e60c6eae-7b77-4d67-986d-66065bae341c', '1489TH': '34e23121-3c10-454d-80dc-45d824b9e2cf',
    '1980TH': 'b86817d4-c71b-4bd4-ab64-4982bf812b12', '1790TH': 'c257f8de-5cf1-4c1f-9b46-deea2006d745',
}
HYDROLOGY_READINGS = 'https://environment.data.gov.uk/hydrology/id/measures/{guid}-level-i-900-m-qualified/readings.csv'
COSMOS_SITE = 'CHIMN'   # Chimney Meadows
SOIL_DELAY_H = 36       # a day's value (stamped 00:00) is published the next day: use it from noon

EXTRA_COLS = [f'up_{r}_{k}' for r in UPSTREAM for k in ('rise', 'd6', 'd24')] + ['soil_vwc', 'soil_vwc_d7']


def _utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    return index.tz_localize('UTC') if index.tz is None else index.tz_convert('UTC')


def level_features(levels: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """levels: hourly level (m) per station reference, UTC index."""
    idx = _utc(index)
    out = {}
    for ref in UPSTREAM:
        if ref in levels.columns:
            lv = levels[ref].astype(float)
            lv = lv.reindex(lv.index.union(idx)).interpolate(method='time', limit=6, limit_area='inside')
            base = lv.rolling('14D', min_periods=24).min()
            out[f'up_{ref}_rise'] = (lv - base).reindex(idx)
            out[f'up_{ref}_d6'] = (lv - lv.shift(freq='6h').reindex(lv.index)).reindex(idx)
            out[f'up_{ref}_d24'] = (lv - lv.shift(freq='24h').reindex(lv.index)).reindex(idx)
        else:
            for k in ('rise', 'd6', 'd24'):
                out[f'up_{ref}_{k}'] = pd.Series(np.nan, index=idx)
    return pd.DataFrame(out, index=idx)


def soil_features(vwc_daily: Optional[pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """vwc_daily: daily COSMOS volumetric water content (%), stamped at 00:00 UTC of its day."""
    idx = _utc(index)
    if vwc_daily is None or vwc_daily.dropna().empty:
        return pd.DataFrame({'soil_vwc': np.nan, 'soil_vwc_d7': np.nan}, index=idx)
    v = vwc_daily.astype(float).copy()
    v.index = _utc(v.index)
    v = v.sort_index()
    d7 = v - v.shift(freq='7D').reindex(v.index)
    avail = v.index + pd.Timedelta(hours=SOIL_DELAY_H)
    frame = pd.DataFrame({'soil_vwc': v.to_numpy(), 'soil_vwc_d7': d7.to_numpy()}, index=avail)
    frame = frame[~frame.index.duplicated(keep='last')]
    frame = frame.reindex(frame.index.union(idx)).ffill(limit=24 * 4).reindex(idx)
    return frame


def extra_feature_frame(index: pd.DatetimeIndex, levels: pd.DataFrame,
                        vwc_daily: Optional[pd.Series]) -> pd.DataFrame:
    out = pd.concat([level_features(levels, index), soil_features(vwc_daily, index)], axis=1)
    return out[EXTRA_COLS].astype(np.float32)


# ---- history (training) ----

def load_upstream_history(project_root: Path) -> pd.DataFrame:
    cols = {}
    for ref in UPSTREAM:
        p = Path(project_root) / 'data' / 'upstream' / f'{ref}.csv.gz'
        if p.exists():
            s = pd.read_csv(p, index_col=0, parse_dates=True)['level']
            s.index = _utc(s.index)
            cols[ref] = s
    return pd.DataFrame(cols)


def load_soil_history(project_root: Path) -> Optional[pd.Series]:
    p = Path(project_root) / 'data' / 'cosmos' / f'{COSMOS_SITE}_1D_data.csv'
    if not p.exists():
        return None
    return parse_cosmos_csv(p.read_text())


def parse_cosmos_csv(text: str) -> pd.Series:
    """COSMOS-UK API CSV (metadata lines, then timestamp,cosmos_vwc,...) -> daily vwc."""
    rows = []
    for line in text.splitlines():
        parts = line.split(',')
        if len(parts) >= 2 and parts[0][:2] == '20' and 'T' in parts[0]:
            try:
                rows.append((pd.Timestamp(parts[0]), float(parts[1]) if parts[1] not in ('', 'NA') else np.nan))
            except ValueError:
                continue
    s = pd.Series(dict(rows), dtype=float).sort_index()
    s.index = _utc(s.index)
    return s


def history_extra(index: pd.DatetimeIndex, project_root: Path) -> pd.DataFrame:
    return extra_feature_frame(index, load_upstream_history(project_root), load_soil_history(project_root))


# ---- live ----

COSMOS_DAILY = ('https://cosmos-api.ceh.ac.uk/collections/1D/locations/{site}'
                '?datetime={start}/{end}&parameter-name=cosmos_vwc&f=CSV')


def fetch_live_levels(days: int = 28) -> pd.DataFrame:
    """Recent 15-min levels from the EA hydrology API (the training series), hourly means."""
    import io
    import requests
    since = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    cols = {}
    for ref, guid in HYDROLOGY_LEVEL.items():
        try:
            r = requests.get(HYDROLOGY_READINGS.format(guid=guid), params={'mineq-date': since, '_limit': 20000},
                             timeout=90)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                continue
            s = pd.Series(pd.to_numeric(df['value'], errors='coerce').to_numpy(),
                          index=pd.to_datetime(df['dateTime'], utc=True)).sort_index()
            cols[ref] = s.resample('1h').mean()
        except Exception as exc:
            print(f"  [extra inputs] no live level for {ref} ({type(exc).__name__}: {exc})")
    return pd.DataFrame(cols)


def fetch_live_soil(days: int = 40) -> Optional[pd.Series]:
    import io
    import zipfile
    import requests
    end = pd.Timestamp.now(tz='UTC').ceil('D')
    start = end - pd.Timedelta(days=days)
    url = COSMOS_DAILY.format(site=COSMOS_SITE, start=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                              end=end.strftime('%Y-%m-%dT%H:%M:%SZ'))
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = r.content
        if raw[:2] == b'PK':   # the API answers CSV requests with a one-file zip
            z = zipfile.ZipFile(io.BytesIO(raw))
            raw = z.read(z.namelist()[0])
        return parse_cosmos_csv(raw.decode('utf-8', 'replace'))
    except Exception as exc:
        print(f"  [extra inputs] no COSMOS soil moisture ({type(exc).__name__}: {exc})")
        return None


def live_extra(index: pd.DatetimeIndex, project_root: Path) -> pd.DataFrame:
    """Extra inputs over a live history frame's hours: live API data, the stored history behind it."""
    levels = fetch_live_levels()
    hist = load_upstream_history(project_root)
    if len(hist):
        levels = levels.combine_first(hist) if len(levels) else hist
    soil = fetch_live_soil()
    soil_hist = load_soil_history(project_root)
    if soil_hist is not None:
        soil = soil.combine_first(soil_hist) if soil is not None else soil_hist
    out = extra_feature_frame(index, levels, soil)
    last = out.iloc[-1]
    print(f"  [extra inputs] {int(last.notna().sum())}/{len(EXTRA_COLS)} inputs at t0; soil {last['soil_vwc']:.1f}%")
    return out
