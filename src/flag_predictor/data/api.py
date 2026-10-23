"""
API data fetching for river levels and rainfall.

Functions for fetching real-time data from:
- Environment Agency Flood Monitoring API (river levels, rainfall)
- Open-Meteo API (rainfall forecasts)
"""

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from urllib3.util.retry import Retry

from ..config import (
    API_URLS,
    FLOW_API_BASE,
    RAINFALL_API_URLS,
    RAINFALL_STATION_NAMES,
    RAINFALL_STATION_COORDINATES,
    WALLINGFORD_RAINFALL_API_URLS,
    WALLINGFORD_RAINFALL_STATION_NAMES,
    WALLINGFORD_RAINFALL_STATION_COORDINATES,
)

_DEFAULT_HTTP_TIMEOUT = (10, 180)  # (connect_timeout_s, read_timeout_s)
# EA readings endpoints hard-cap at 10000 items. The flood-monitoring API only
# retains ~28 days of readings, so asking for more just strains the backend.
_EA_READINGS_LIMIT = 10000
_EA_LOOKBACK_DAYS = 28
# When the backend 503s on a query, progressively smaller windows often still
# succeed; a couple of days of fresh readings beats failing the whole publish.
_EA_FALLBACK_LOOKBACK_DAYS = (7, 2)
_EA_HTTP_TIMEOUT = (10, 30)
# A gauge's last reading is carried forward this long when building a differential. Kept to one
# missed hour: carrying one gauge forward while the other moves makes a false trend (Osney
# downstream arriving hours late on 23 Oct 2026 pulled the Isis down 3 cm as Iffley rose). Longer
# gaps are estimated (hybrid_live.outage_nowcast) and drawn dotted.
GAUGE_HOLD_H = 1


def _requests_session_with_retries() -> requests.Session:
    """
    Build a requests Session with retries/backoff.

    GitHub Actions runners can intermittently time out when calling external APIs.
    This makes Open-Meteo / EA calls much more reliable without changing call sites.
    """
    # Kept short: a run fetches ~40 EA series, and when the EA answers 503s (as on 21 Oct 2026)
    # five retries with long backoff each pushed runs past their 25-minute limit.
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _normalize_readings_url(url: str) -> str:
    """Force https and a .json readings path; strip any old query string."""
    base = url.split("?", 1)[0].replace("http://", "https://")
    if base.endswith("/readings"):
        base = base + ".json"
    elif not base.endswith(".json") and "/readings" not in base:
        # Allow bare measure URLs for safety
        base = base.rstrip("/") + "/readings.json"
    return base


def _ea_readings_params(lookback_days: int = _EA_LOOKBACK_DAYS) -> dict:
    """Toby-style EA params: recent window only, within the 10k hard cap."""
    since = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "since": since,
        "_limit": _EA_READINGS_LIMIT,
        "_sorted": "",
    }


_EA_CACHE: Dict[str, dict] = {}
_EA_CACHE_LOCK = __import__('threading').Lock()


def _fetch_many(urls_names, verbose: bool = True):
    """{name: EA readings JSON or the exception}, fetched 8 at a time (each thread its own session)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(pair):
        url, name = pair
        try:
            return name, _fetch_ea_readings_json(url, session=_requests_session_with_retries())
        except Exception as exc:
            return name, exc
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, list(urls_names)))


def _fetch_ea_readings_json(url: str, session: Optional[requests.Session] = None) -> dict:
    """
    GET an EA readings endpoint with retries and sane limits.

    Each window already gets HTTP-level retries from the session; if the full
    window still fails (EA backends intermittently 503 heavy queries), fall
    back to smaller lookback windows before giving up.
    """
    normalized = _normalize_readings_url(url)
    with _EA_CACHE_LOCK:
        if normalized in _EA_CACHE:      # one run asks for the same series for several stretches
            return _EA_CACHE[normalized]
    sess = session or _requests_session_with_retries()
    last_error: Optional[Exception] = None
    for lookback_days in (_EA_LOOKBACK_DAYS, *_EA_FALLBACK_LOOKBACK_DAYS):
        try:
            response = sess.get(
                normalized,
                params=_ea_readings_params(lookback_days),
                timeout=_EA_HTTP_TIMEOUT,
                headers={"User-Agent": "Flag_Predictor/1.0 (+GitHubActions)"},
            )
            response.raise_for_status()
            data = response.json()
            with _EA_CACHE_LOCK:
                _EA_CACHE[normalized] = data
            return data
        except Exception as exc:
            last_error = exc
            print(f"    EA readings fetch failed ({lookback_days}d window): {exc}")
    assert last_error is not None
    raise last_error


def _require_level_series(river_levels: Dict[str, pd.DataFrame], name: str) -> pd.Series:
    """Return a level series or raise a clear error if the EA fetch was empty."""
    df = river_levels.get(name)
    if df is None or df.empty or "level" not in df.columns:
        raise RuntimeError(
            f"Missing EA river-level data for '{name}'. "
            "Flood-monitoring fetch returned no usable readings."
        )
    return df["level"]


def _coerce_reading_values(series: pd.Series) -> pd.Series:
    """
    Coerce EA reading values to floats.

    The EA API occasionally returns `value` as a JSON list (duplicate readings
    for one timestamp, e.g. [0.107, 0.107]); take the first element. Anything
    non-numeric becomes NaN and is dropped by callers.
    """
    cleaned = series.map(lambda v: v[0] if isinstance(v, list) and v else v)
    return pd.to_numeric(cleaned, errors="coerce")


def _process_level_api_response(data: dict) -> pd.DataFrame:
    """
    Process API response from flood monitoring service for water levels.
    
    Args:
        data: JSON response from API
        
    Returns:
        DataFrame with timestamp index and 'level' column
    """
    if 'items' not in data or not data['items']:
        return pd.DataFrame()
    
    temp_df = pd.DataFrame(data['items'])
    if 'dateTime' not in temp_df.columns or 'value' not in temp_df.columns:
        return pd.DataFrame()
    
    temp_df = temp_df[['dateTime', 'value']]
    temp_df.rename(columns={'dateTime': 'timestamp', 'value': 'level'}, inplace=True)
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df['level'] = _coerce_reading_values(temp_df['level'])
    temp_df = temp_df.dropna(subset=['level'])
    df = temp_df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')]
    return df


def _process_rainfall_api_response(data: dict) -> pd.DataFrame:
    """
    Process API response from flood monitoring service for rainfall.
    
    Args:
        data: JSON response from API
        
    Returns:
        DataFrame with timestamp index and 'rainfall' column
    """
    if 'items' not in data or not data['items']:
        return pd.DataFrame()
    
    temp_df = pd.DataFrame(data['items'])
    if 'dateTime' not in temp_df.columns or 'value' not in temp_df.columns:
        return pd.DataFrame()
    
    temp_df = temp_df[['dateTime', 'value']]
    temp_df.rename(columns={'dateTime': 'timestamp', 'value': 'rainfall'}, inplace=True)
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df['rainfall'] = _coerce_reading_values(temp_df['rainfall'])
    temp_df = temp_df.dropna(subset=['rainfall'])
    df = temp_df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')]
    return df


def _process_flow_api_response(data: dict, column_name: str = 'flow_m3s_Farmoor') -> pd.DataFrame:
    """
    Process API response from Hydrology API for flow (same structure as rainfall).
    
    Args:
        data: JSON response with 'items' array containing dateTime, value
        column_name: Column name for the flow values
        
    Returns:
        DataFrame with timestamp index and flow column
    """
    if 'items' not in data or not data['items']:
        return pd.DataFrame()
    
    temp_df = pd.DataFrame(data['items'])
    if 'dateTime' not in temp_df.columns or 'value' not in temp_df.columns:
        return pd.DataFrame()
    
    temp_df = temp_df[['dateTime', 'value']]
    temp_df.rename(columns={'dateTime': 'timestamp', 'value': column_name}, inplace=True)
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df[column_name] = _coerce_reading_values(temp_df[column_name])
    temp_df = temp_df.dropna(subset=[column_name])
    df = temp_df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')]
    return df


def fetch_river_level_data(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch all river level data from Environment Agency API.
    
    Args:
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping station names to DataFrames
    """
    if verbose:
        print("Fetching river level data...")
    
    fetched = _fetch_many([(url, name) for name, url in API_URLS.items()])
    results = {}
    for name, url in API_URLS.items():
        try:
            got = fetched[name]
            if isinstance(got, Exception):
                raise got
            df = _process_level_api_response(got)
            results[name] = df
            if verbose:
                if df.empty:
                    print(f"  ✗ {name}: empty response")
                else:
                    print(
                        f"  ✓ {name}: {len(df)} records "
                        f"({df.index.min()} → {df.index.max()})"
                    )
        except Exception as e:
            if verbose:
                print(f"  ✗ {name}: {e}")
            results[name] = pd.DataFrame()
    
    return results


def fetch_rainfall_data(location: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch rainfall data from all stations via Environment Agency API.
    
    Args:
        location: Location name ('wallingford' uses additional stations, others use default)
        verbose: Whether to print progress
        
    Returns:
        DataFrame with all stations as columns, timestamp index
    """
    if verbose:
        print("Fetching rainfall data...")
    
    # Get base stations
    api_urls = list(RAINFALL_API_URLS)
    station_names = list(RAINFALL_STATION_NAMES)
    
    # Filter out stations for specific locations
    if location:
        location_lower = location.lower()
        
        # Exclude Bicester and Grimsbury for Godstow
        if location_lower == 'godstow':
            godstow_excluded = {'Bicester', 'Grimsbury'}
            filtered_pairs = [(url, name) for url, name in zip(api_urls, station_names) 
                            if name not in godstow_excluded]
            api_urls, station_names = zip(*filtered_pairs) if filtered_pairs else ([], [])
            api_urls = list(api_urls)
            station_names = list(station_names)
            if verbose:
                print(f"  Excluding Bicester and Grimsbury for Godstow")
        
        # Add Wallingford-specific stations if location is wallingford
        if location_lower == 'wallingford':
            api_urls.extend(WALLINGFORD_RAINFALL_API_URLS)
            station_names.extend(WALLINGFORD_RAINFALL_STATION_NAMES)
            if verbose:
                print(f"  Including {len(WALLINGFORD_RAINFALL_STATION_NAMES)} Wallingford-specific stations")
    
    fetched = _fetch_many(list(zip(api_urls, station_names)))
    rainfall_dfs = {}
    for url, name in zip(api_urls, station_names):
        try:
            got = fetched[name]
            if isinstance(got, Exception):
                raise got
            df = _process_rainfall_api_response(got)
            if not df.empty:
                df.rename(columns={'rainfall': name}, inplace=True)
                rainfall_dfs[name] = df
                if verbose:
                    print(f"  ✓ {name}: {len(df)} records")
            elif verbose:
                print(f"  ✗ {name}: empty response")
        except Exception as e:
            if verbose:
                print(f"  ✗ {name}: {e}")
    
    if not rainfall_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(rainfall_dfs.values(), axis=1)
    if verbose:
        print(f"\n✓ Combined rainfall API data: {combined.shape}")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


LIVE_FLOW_URL = ('https://environment.data.gov.uk/flood-monitoring/id/measures/'
                 '1100TH-flow--Mean-15_min-m3_s/readings.json')


def _fetch_live_farmoor_flow(verbose: bool = True, days: int = 28) -> pd.DataFrame:
    """Farmoor flow from the EA flood-monitoring API (raw, near real time), last ``days`` days."""
    try:
        since = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
        r = requests.get(LIVE_FLOW_URL, params={'since': since, '_sorted': '', '_limit': 10000}, timeout=60)
        r.raise_for_status()
        items = r.json().get('items', [])
        s = pd.Series({pd.Timestamp(i['dateTime']): i.get('value') for i in items if 'dateTime' in i}, dtype=object)
        s = pd.to_numeric(s, errors='coerce').dropna().sort_index()
        if s.empty:
            return pd.DataFrame()
        s.index = s.index.tz_convert('UTC') if s.index.tz is not None else s.index.tz_localize('UTC')
        if verbose:
            print(f"  ✓ Farmoor flow (live feed): {len(s)} readings to {s.index.max()}")
        return s.to_frame('flow_m3s_Farmoor')
    except Exception as e:
        if verbose:
            print(f"  ✗ Farmoor flow (live feed): {e}")
        return pd.DataFrame()


def fetch_flow_data(verbose: bool = True, min_days_back: int = 90) -> pd.DataFrame:
    """
    Fetch recent Farmoor flow data from Hydrology API.
    
    Same pattern as rainfall: returns DataFrame with flow_m3s_Farmoor column,
    timestamp index. Used for training and forecasting to get recent flow
    (API data overrides historical CSV for overlapping dates).
    
    Hydrology API returns oldest-first by default, so we use min-date to fetch
    only recent data (last min_days_back days).
    
    Args:
        verbose: Whether to print progress
        min_days_back: Fetch data from this many days ago to present (default 90)
        
    Returns:
        DataFrame with flow_m3s_Farmoor column, or empty DataFrame on error
    """
    if verbose:
        print("Fetching flow data...")
    
    try:
        min_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=min_days_back)).strftime('%Y-%m-%d')
        url = f"{FLOW_API_BASE}?_limit=20000&min-date={min_date}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = _process_flow_api_response(response.json(), column_name='flow_m3s_Farmoor')
        # The qualified (checked) series is published days late; the flood-monitoring feed has the
        # same gauge's raw 15-min flow within the hour (identical where both exist). Fill the
        # recent hours from it.
        live = _fetch_live_farmoor_flow(verbose=verbose)
        if not live.empty:
            if not df.empty and df.index.tz is None:   # the qualified series comes back naive (UTC)
                live.index = live.index.tz_localize(None)
            df = df.combine_first(live) if not df.empty else live
        if not df.empty and verbose:
            print(f"  ✓ Farmoor flow: {len(df)} records")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        if verbose:
            print(f"  ✗ Farmoor flow: {e}")
        return pd.DataFrame()


def fetch_all_api_data(
    location: Optional[str] = None, verbose: bool = True
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Fetch all API data (river levels, rainfall, and flow).
    
    Args:
        location: Location name (for location-specific rainfall stations)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (river_levels_dict, rainfall_df, flow_df)
    """
    river_levels = fetch_river_level_data(verbose=verbose)
    rainfall = fetch_rainfall_data(location=location, verbose=verbose)
    flow = fetch_flow_data(verbose=verbose)
    return river_levels, rainfall, flow


def _utc_series(level: pd.Series) -> pd.Series:
    s = level.sort_index().copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    return s


def _attach_reading_times(out: pd.DataFrame, gauges: Dict[str, pd.Series], n_hours: int = 48) -> pd.DataFrame:
    """Record, for each recent hourly point, when its oldest contributing raw reading was taken.

    Hourly points are labelled by the start of the hour but built from the last reading in
    that hour, and gauges are forward-filled up to 6h, so the true age of a point is the
    oldest of its gauges' latest readings. Stored in ``out.attrs["reading_time"]`` as
    {naive UTC hour: naive UTC reading time} for the last ``n_hours`` points.
    """
    times = {}
    for name, level in gauges.items():
        s = _utc_series(level).dropna()
        stamps = pd.Series(s.index, index=s.index)
        times[name] = stamps.resample("1h").last()
    as_of = pd.concat(times, axis=1).sort_index().ffill(limit=GAUGE_HOLD_H).min(axis=1)
    as_of = as_of.reindex(out.index).dropna().iloc[-n_hours:]
    out.attrs["reading_time"] = {
        idx.tz_localize(None): pd.Timestamp(t).tz_convert("UTC").tz_localize(None)
        for idx, t in as_of.items()
    }
    return out


def _level_series_hourly_last(level: pd.Series) -> pd.Series:
    """Resample irregular EA readings to hourly using the last value in each hour (UTC).

    Raw flood-monitoring points are unevenly spaced; subtracting aligned Series without
    this step can silently drop trailing hours wherever indices do not coincide. Three-gauge
    blends (Isis) are especially sensitive versus two-gauge stretches.
    """
    return _utc_series(level).resample("1h").last()


def calculate_isis_differential(river_levels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate ISIS differential from river level data.
    
    Formula:
        isis_diff = 0.71 * (osney_downstream - iffley_upstream - 2.14) +
                    0.29 * (kings_mill_downstream - iffley_upstream - 0.73)
    
    Args:
        river_levels: Dictionary of river level DataFrames
        
    Returns:
        DataFrame with 'differential' column
    """
    gauges = {
        name: _require_level_series(river_levels, name)
        for name in ("osney_downstream", "iffley_upstream")
    }
    km = river_levels.get("kings_mill_downstream")
    gauges["kings_mill_downstream"] = km["level"] if km is not None and not km.empty and "level" in km.columns \
        else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    o = _level_series_hourly_last(gauges["osney_downstream"])
    i = _level_series_hourly_last(gauges["iffley_upstream"])
    k = _level_series_hourly_last(gauges["kings_mill_downstream"])
    k, stand_in = _kings_mill_stand_in(k, o, i, river_levels.get("cherwell_botanic"))
    reading_gauges = dict(gauges)
    if stand_in is not None:
        reading_gauges["kings_mill_downstream"] = pd.concat(
            [_utc_series(gauges["kings_mill_downstream"]).dropna(), stand_in.pop("_raw")])
        # the time of the last King's Mill reading itself (k is hourly: its hour, not the reading)
        km_raw = _utc_series(gauges["kings_mill_downstream"]).dropna()
        if len(km_raw):
            stand_in["since"] = km_raw.index[-1].tz_convert("UTC").tz_localize(None).isoformat() + "Z"
    merged = pd.concat({"o": o, "i": i, "k": k}, axis=1).sort_index()
    # Allow small gaps—EA stations often miss occasional hours; avoids a frozen tail/hourly NaNs.
    merged = merged.ffill(limit=GAUGE_HOLD_H).dropna(how="any")
    if merged.empty:
        raise RuntimeError("ISIS differential is empty after aligning EA level series.")
    differential = (
        0.71 * (merged["o"] - merged["i"] - 2.14)
        + 0.29 * (merged["k"] - merged["i"] - 0.73)
    )
    out = _attach_reading_times(pd.DataFrame({"differential": differential}), reading_gauges)
    if stand_in is not None:
        out.attrs["gauge_stand_in"] = stand_in
    return out


# King's Mill downstream (1491TH) tracks the Cherwell at the Botanic Gardens (1490TH, same reach, a few
# hundred metres down) to about 2 mm at low water: K = C - 3.944 (Sep-Oct 2026). While King's Mill is
# offline its level is taken from the Botanic Gardens gauge, the offset re-anchored on the last 48 h both
# reported. The drop between the two grows with Cherwell flow and is not published for high water, so
# the stand-in is used only while the Cherwell is within KM_STAND_IN_MAX_RISE of its level at the last
# King's Mill reading; beyond that the Isis is treated as offline (estimated from Godstow).
KM_BOTANIC_OFFSET = -3.944
KM_STAND_IN_AFTER_H = 4
KM_STAND_IN_MAX_RISE = 0.30


def _kings_mill_stand_in(k: pd.Series, o: pd.Series, i: pd.Series, botanic):
    """(King's Mill hourly level, extended from the Botanic Gardens gauge if it is offline; info or None)."""
    if botanic is None or botanic.empty or "level" not in botanic.columns:
        return k, None
    c_raw = _utc_series(botanic["level"]).dropna()
    c = _level_series_hourly_last(c_raw)
    others_end = min(o.last_valid_index() or pd.Timestamp.min.tz_localize("UTC"),
                     i.last_valid_index() or pd.Timestamp.min.tz_localize("UTC"))
    k_end = k.last_valid_index()
    if k_end is not None and others_end - k_end <= pd.Timedelta(hours=KM_STAND_IN_AFTER_H):
        return k, None                                   # King's Mill is reporting
    if c.last_valid_index() is None or others_end - c.last_valid_index() > pd.Timedelta(hours=KM_STAND_IN_AFTER_H):
        return k, None                                   # Botanic Gardens offline too
    if k_end is not None:
        both = (k - c).loc[k_end - pd.Timedelta(hours=48):k_end].dropna()
        offset = float(both.median()) if len(both) >= 6 else KM_BOTANIC_OFFSET
        c_then = c.loc[:k_end].dropna()
        c_then = float(c_then.iloc[-1]) if len(c_then) else float("nan")
    else:
        offset, c_then = KM_BOTANIC_OFFSET, float("nan")
    rise = float(c.dropna().iloc[-1] - c_then) if np.isfinite(c_then) else float("nan")
    if not np.isfinite(rise) or abs(rise) > KM_STAND_IN_MAX_RISE:
        print(f"  [isis] King's Mill offline since {k_end}; Cherwell moved {rise:+.2f} m since, "
              "too far to stand in from the Botanic Gardens gauge")
        return k, None
    after = c.loc[c.index > k_end] if k_end is not None else c
    filled = k.combine_first(after + offset)
    print(f"  [isis] King's Mill offline since {k_end}: its level from the Botanic Gardens gauge "
          f"(offset {offset:+.3f} m, Cherwell {rise:+.2f} m since)")
    raw = c_raw.loc[c_raw.index > k_end] if k_end is not None else c_raw
    return filled, {"gauge": "King's Mill", "stand_in": "Botanic Gardens (Cherwell)",
                    "since": (pd.Timestamp(k_end).tz_convert("UTC").tz_localize(None).isoformat() + "Z")
                    if k_end is not None else None,
                    "offset_m": round(offset, 4), "cherwell_change_m": round(rise, 3),
                    "_raw": raw}


def calculate_godstow_differential(river_levels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate Godstow differential from river level data.
    
    Formula:
        godstow_diff = godstow_downstream - osney_upstream - 1.63
    
    Args:
        river_levels: Dictionary of river level DataFrames
        
    Returns:
        DataFrame with 'differential' column
    """
    gauges = {
        name: _require_level_series(river_levels, name)
        for name in ("godstow_downstream", "osney_upstream")
    }
    g = _level_series_hourly_last(gauges["godstow_downstream"])
    u = _level_series_hourly_last(gauges["osney_upstream"])
    merged = pd.concat({"godstow": g, "osney_us": u}, axis=1).sort_index().ffill(limit=GAUGE_HOLD_H).dropna(
        how="any"
    )
    if merged.empty:
        raise RuntimeError("Godstow differential is empty after aligning EA level series.")
    differential = merged["godstow"] - merged["osney_us"] - 1.63

    return _attach_reading_times(pd.DataFrame({'differential': differential}), gauges)

def calculate_wallingford_differential(river_levels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate Wallingford differential from river level data.
    
    Formula:
        wallingford_diff = benson_downstream - cleeve_upstream - 2.13
    
    Args:
        river_levels: Dictionary of river level DataFrames
        
    Returns:
        DataFrame with 'differential' column
    """
    gauges = {
        name: _require_level_series(river_levels, name)
        for name in ("benson_downstream", "cleeve_upstream")
    }
    b = _level_series_hourly_last(gauges["benson_downstream"])
    c = _level_series_hourly_last(gauges["cleeve_upstream"])
    merged = pd.concat({"benson": b, "cleeve": c}, axis=1).sort_index().ffill(limit=GAUGE_HOLD_H).dropna(how="any")
    if merged.empty:
        raise RuntimeError("Wallingford differential is empty after aligning EA level series.")
    differential = merged["benson"] - merged["cleeve"] - 2.13

    return _attach_reading_times(pd.DataFrame({'differential': differential}), gauges)

def get_rainfall_forecast(
    locations: Optional[Dict] = None,
    location: Optional[str] = None,
    forecast_days: int = 10
) -> pd.DataFrame:
    """
    Fetch rainfall forecast from Open-Meteo API.
    
    Args:
        locations: Dictionary of location names and coordinates
                  (defaults to RAINFALL_STATION_COORDINATES or Wallingford-specific)
        location: Location name ('wallingford' uses additional stations)
        forecast_days: Number of days to forecast
        
    Returns:
        DataFrame with hourly precipitation forecasts
    """
    if locations is None:
        if location and location.lower() == 'wallingford':
            # Combine default and Wallingford-specific stations
            locations = {**RAINFALL_STATION_COORDINATES, **WALLINGFORD_RAINFALL_STATION_COORDINATES}
        elif location and location.lower() == 'godstow':
            # Exclude Bicester and Grimsbury for Godstow
            locations = {k: v for k, v in RAINFALL_STATION_COORDINATES.items() 
                        if k not in {'Bicester', 'Grimsbury'}}
        else:
            locations = RAINFALL_STATION_COORDINATES
    
    location_names = list(locations.keys())
    latitudes = [loc['latitude'] for loc in locations.values()]
    longitudes = [loc['longitude'] for loc in locations.values()]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "precipitation",
        "forecast_days": forecast_days
    }

    try:
        session = _requests_session_with_retries()
        response = session.get(
            url,
            params=params,
            timeout=_DEFAULT_HTTP_TIMEOUT,
            headers={"User-Agent": "Flag_Predictor/1.0 (+GitHubActions)"},
        )
        response.raise_for_status()
        data = response.json()

        forecast_dfs = []
        for i, location_data in enumerate(data):
            df = pd.DataFrame(location_data['hourly'])
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.set_index('timestamp')
            df = df[['precipitation']]
            df = df.rename(columns={'precipitation': location_names[i]})
            forecast_dfs.append(df)
        
        combined_df = pd.concat(forecast_dfs, axis=1)
        print("✓ Successfully fetched rainfall forecast")
        return combined_df

    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return pd.DataFrame()
    except (KeyError, TypeError) as e:
        print(f"✗ Failed to parse API response: {e}")
        return pd.DataFrame()


def get_rainfall_forecast_ensemble(
    locations: Optional[Dict] = None,
    location: Optional[str] = None,
    ensemble_model: str = 'ecmwf_aifs025',
    n_members: Optional[int] = None,
    forecast_days: int = 10
) -> pd.DataFrame:
    """
    Fetch ensemble rainfall forecast from Open-Meteo Ensemble API.
    
    Returns ALL ensemble members for probabilistic forecasting.
    
    Args:
        locations: Dictionary of location names and coordinates
        location: Location name ('wallingford' uses additional stations)
        ensemble_model: Model to use ('ecmwf_ifs', 'ecmwf_aifs025',)
        n_members: Number of members to return (None = all)
        forecast_days: Number of days to forecast (default 10)
        
    Returns:
        DataFrame with columns like 'Osney_member_0', 'Osney_member_1', etc.
    """
    if locations is None:
        if location and location.lower() == 'wallingford':
            # Combine default and Wallingford-specific stations
            locations = {**RAINFALL_STATION_COORDINATES, **WALLINGFORD_RAINFALL_STATION_COORDINATES}
        elif location and location.lower() == 'godstow':
            # Exclude Bicester and Grimsbury for Godstow
            locations = {k: v for k, v in RAINFALL_STATION_COORDINATES.items() 
                        if k not in {'Bicester', 'Grimsbury'}}
        else:
            locations = RAINFALL_STATION_COORDINATES
    
    location_names = list(locations.keys())
    latitudes = [loc['latitude'] for loc in locations.values()]
    longitudes = [loc['longitude'] for loc in locations.values()]

    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "precipitation",
        "forecast_days": forecast_days,
        "models": ensemble_model
    }

    try:
        session = _requests_session_with_retries()
        response = session.get(
            url,
            params=params,
            timeout=_DEFAULT_HTTP_TIMEOUT,
            headers={"User-Agent": "Flag_Predictor/1.0 (+GitHubActions)"},
        )
        response.raise_for_status()
        data = response.json()

        all_dfs = []
        for i, location_data in enumerate(data):
            df = pd.DataFrame(location_data['hourly'])
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.set_index('timestamp')
            
            # Get precipitation columns (ensemble members)
            precip_cols = [col for col in df.columns if col.startswith('precipitation')]
            
            if n_members is not None:
                precip_cols = precip_cols[:n_members]
            
            # Rename columns to station_member_N format
            df_members = df[precip_cols].copy()
            df_members.columns = [
                f"{location_names[i]}_member_{j}" 
                for j in range(len(precip_cols))
            ]
            all_dfs.append(df_members)
        
        combined_df = pd.concat(all_dfs, axis=1)
        
        n_actual_members = len([c for c in combined_df.columns if '_member_0' in c or c.endswith('_member_0')])
        print(f"✓ Successfully fetched ensemble forecast: {len(location_names)} stations × {len(precip_cols)} members")
        
        return combined_df

    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return pd.DataFrame()
    except (KeyError, TypeError) as e:
        print(f"✗ Failed to parse API response: {e}")
        return pd.DataFrame()
