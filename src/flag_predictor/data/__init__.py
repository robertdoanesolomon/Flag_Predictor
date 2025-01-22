"""Data loading and API access modules."""

from .api import (
    fetch_river_level_data,
    fetch_rainfall_data,
    fetch_all_api_data,
    get_rainfall_forecast,
    get_rainfall_forecast_ensemble,
)
from .loader import (
    load_historical_differential,
    load_historical_rainfall,
    load_all_historical_data,
)
