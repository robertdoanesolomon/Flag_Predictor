"""
API data fetching for river levels and rainfall.

Functions for fetching real-time data from:
- Environment Agency Flood Monitoring API (river levels, rainfall)
- Open-Meteo API (rainfall forecasts)
"""

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
# EA readings endpoints hard-cap at 10000 items. 90d of 15-min data is ~8640.
_EA_READINGS_LIMIT = 10000
_EA_LOOKBACK_DAYS = 90
_EA_HTTP_TIMEOUT = (10, 60)


def _requests_session_with_retries() -> requests.Session:
    """
    Build a requests Session with retries/backoff.

    GitHub Actions runners can intermittently time out when calling external APIs.
    This makes Open-Meteo / EA calls much more reliable without changing call sites.
    """
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
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


def _fetch_ea_readings_json(url: str, session: Optional[requests.Session] = None) -> dict:
    """GET an EA readings endpoint with retries and sane limits."""
    sess = session or _requests_session_with_retries()
    response = sess.get(
        _normalize_readings_url(url),
        params=_ea_readings_params(),
        timeout=_EA_HTTP_TIMEOUT,
        headers={"User-Agent": "Flag_Predictor/1.0 (+GitHubActions)"},
    )
    response.raise_for_status()
    return response.json()


def _require_level_series(river_levels: Dict[str, pd.DataFrame], name: str) -> pd.Series:
    """Return a level series or raise a clear error if the EA fetch was empty."""
    df = river_levels.get(name)
    if df is None or df.empty or "level" not in df.columns:
        raise RuntimeError(
            f"Missing EA river-level data for '{name}'. "
            "Flood-monitoring fetch returned no usable readings."
        )
    return df["level"]


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
    df = temp_df.set_index('timestamp')
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
    df = temp_df.set_index('timestamp')
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
    temp_df[column_name] = pd.to_numeric(temp_df[column_name], errors='coerce')
    df = temp_df.set_index('timestamp')
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
    
    session = _requests_session_with_retries()
    results = {}
    for name, url in API_URLS.items():
        try:
            df = _process_level_api_response(_fetch_ea_readings_json(url, session=session))
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
    
    session = _requests_session_with_retries()
    rainfall_dfs = {}
    for url, name in zip(api_urls, station_names):
        try:
            df = _process_rainfall_api_response(_fetch_ea_readings_json(url, session=session))
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


def _level_series_hourly_last(level: pd.Series) -> pd.Series:
    """Resample irregular EA readings to hourly using the last value in each hour (UTC).

    Raw flood-monitoring points are unevenly spaced; subtracting aligned Series without
    this step can silently drop trailing hours wherever indices do not coincide. Three-gauge
    blends (Isis) are especially sensitive versus two-gauge stretches.
    """
    s = level.sort_index().copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    return s.resample("1h").last()


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
    o = _level_series_hourly_last(_require_level_series(river_levels, "osney_downstream"))
    i = _level_series_hourly_last(_require_level_series(river_levels, "iffley_upstream"))
    k = _level_series_hourly_last(_require_level_series(river_levels, "kings_mill_downstream"))
    merged = pd.concat({"o": o, "i": i, "k": k}, axis=1).sort_index()
    # Allow small gaps—EA stations often miss occasional hours; avoids a frozen tail/hourly NaNs.
    merged = merged.ffill(limit=6).dropna(how="any")
    if merged.empty:
        raise RuntimeError("ISIS differential is empty after aligning EA level series.")
    differential = (
        0.71 * (merged["o"] - merged["i"] - 2.14)
        + 0.29 * (merged["k"] - merged["i"] - 0.73)
    )
    return pd.DataFrame({"differential": differential})


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
    g = _level_series_hourly_last(_require_level_series(river_levels, "godstow_downstream"))
    u = _level_series_hourly_last(_require_level_series(river_levels, "osney_upstream"))
    merged = pd.concat({"godstow": g, "osney_us": u}, axis=1).sort_index().ffill(limit=6).dropna(
        how="any"
    )
    if merged.empty:
        raise RuntimeError("Godstow differential is empty after aligning EA level series.")
    differential = merged["godstow"] - merged["osney_us"] - 1.63

    return pd.DataFrame({'differential': differential})

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
    b = _level_series_hourly_last(_require_level_series(river_levels, "benson_downstream"))
    c = _level_series_hourly_last(_require_level_series(river_levels, "cleeve_upstream"))
    merged = pd.concat({"benson": b, "cleeve": c}, axis=1).sort_index().ffill(limit=6).dropna(how="any")
    if merged.empty:
        raise RuntimeError("Wallingford differential is empty after aligning EA level series.")
    differential = merged["benson"] - merged["cleeve"] - 2.13

    return pd.DataFrame({'differential': differential})

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
