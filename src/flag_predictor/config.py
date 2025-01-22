"""
Configuration for different river locations (ISIS, Godstow).

This module defines location-specific settings including:
- API URLs for water level data
- Differential calculation formulas
- Historical data file paths
- Model hyperparameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class LocationConfig:
    """Configuration for a river monitoring location."""
    
    name: str
    display_name: str
    
    # Data file paths (relative to project root)
    historical_data_file: str
    differential_column: str
    
    # Differential calculation parameters
    differential_offset: float
    
    # API URLs for real-time data
    api_urls: Dict[str, str] = field(default_factory=dict)
    
    # Differential value bounds for data cleaning
    min_differential: float = -0.1
    max_differential: float = 1.5
    
    # Default prediction horizons (hours)
    horizons: List[int] = field(default_factory=lambda: [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,  # 0-24h: every 2h
        30, 36, 42, 48,                               # 24-48h: every 6h
        72, 96, 120, 144, 168, 192, 216, 240          # 48-240h: every 24h (daily)
    ])


# Station coordinates for rainfall forecast API
RAINFALL_STATION_COORDINATES = {
    'Osney': {'latitude': 51.750, 'longitude': -1.272},
    'Eynsham': {'latitude': 51.789, 'longitude': -1.402},
    'St': {'latitude': 51.7, 'longitude': -1.5},
    'Shorncote': {'latitude': 51.666, 'longitude': -1.916},
    'Rapsgate': {'latitude': 51.815, 'longitude': -1.975},
    'Stowell': {'latitude': 51.833, 'longitude': -1.821},
    'Bourton': {'latitude': 51.884, 'longitude': -1.758},
    'Chipping': {'latitude': 51.942, 'longitude': -1.547},
    'Grimsbury': {'latitude': 52.065, 'longitude': -1.326},
    'Bicester': {'latitude': 51.899, 'longitude': -1.155},
    'Byfield': {'latitude': 52.179, 'longitude': -1.274},
    'Swindon': {'latitude': 51.556, 'longitude': -1.779},
    'Worsham': {'latitude': 51.817, 'longitude': -1.498}
}

RAINFALL_STATION_NAMES = list(RAINFALL_STATION_COORDINATES.keys())

# Wallingford-specific rainfall stations (only used for Wallingford location)
WALLINGFORD_RAINFALL_STATION_COORDINATES = {
    'Stanford': {'latitude': 51.650, 'longitude': -1.500},  # Stanford-in-the-Vale
    'Abingdon': {'latitude': 51.670, 'longitude': -1.283},  # Abingdon-on-Thames
    'Wheatley': {'latitude': 51.747, 'longitude': -1.140},  # Wheatley, Oxfordshire
    'Benson': {'latitude': 51.620, 'longitude': -1.110},  # Benson, Oxfordshire
    'Aylesbury': {'latitude': 51.817, 'longitude': -0.813},  # Aylesbury, Buckinghamshire
    'Cleeve': {'latitude': 51.680, 'longitude': -1.200},  # Cleeve, Oxfordshire
}

# Wallingford-specific rainfall API URLs (measure IDs from user)
WALLINGFORD_RAINFALL_API_URLS = [
    'http://environment.data.gov.uk/flood-monitoring/id/measures/260221TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Stanford
    'http://environment.data.gov.uk/flood-monitoring/id/measures/261021TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Abingdon
    'http://environment.data.gov.uk/flood-monitoring/id/measures/263541TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Wheatley
    'http://environment.data.gov.uk/flood-monitoring/id/measures/264254TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Benson
    'http://environment.data.gov.uk/flood-monitoring/id/measures/261923TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Aylesbury
    'http://environment.data.gov.uk/flood-monitoring/id/measures/264845TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',  # Cleeve
]

WALLINGFORD_RAINFALL_STATION_NAMES = list(WALLINGFORD_RAINFALL_STATION_COORDINATES.keys())


# River level API URLs
API_URLS = {
    'kings_mill_downstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/1491TH-level-downstage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'godstow_downstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/1302TH-level-downstage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'osney_upstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/1303TH-level-stage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'osney_downstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/1303TH-level-downstage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'iffley_upstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/1501TH-level-stage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'benson_downstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/2001TH-level-downstage-i-15_min-mASD/readings?_sorted&_limit=90000',
    'cleeve_upstream': 'http://environment.data.gov.uk/flood-monitoring/id/measures/2002TH-level-stage-i-15_min-mASD/readings?_sorted&_limit=90000',

}

RAINFALL_API_URLS = [
    'http://environment.data.gov.uk/flood-monitoring/id/measures/256230TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/254336TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/251530TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/248332TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/248965TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/251556TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/253340TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/254829TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/257039TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/259110TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/256345TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/249744TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000',
    'http://environment.data.gov.uk/flood-monitoring/id/measures/253861TP-rainfall-tipping_bucket_raingauge-t-15_min-mm/readings?_sorted&_limit=90000'
]


# Pre-defined location configurations
ISIS_CONFIG = LocationConfig(
    name='isis',
    display_name='Isis Stretch',
    historical_data_file='data/isis_flag_model_data.csv',
    differential_column='jameson_isis_differential',
    differential_offset=0.0,  # Calculated from multiple components
    api_urls={
        'osney_downstream': API_URLS['osney_downstream'],
        'iffley_upstream': API_URLS['iffley_upstream'],
        'kings_mill_downstream': API_URLS['kings_mill_downstream'],
    },
)

GODSTOW_CONFIG = LocationConfig(
    name='godstow',
    display_name='Godstow Stretch',
    historical_data_file='data/godstow_flag_model_data.csv',
    differential_column='jameson_godstow_differential',
    differential_offset=1.63,
    api_urls={
        'godstow_downstream': API_URLS['godstow_downstream'],
        'osney_upstream': API_URLS['osney_upstream'],
    },
)

WALLINGFORD_CONFIG = LocationConfig(
    name='wallingford',
    display_name='Wallingford Stretch',
    historical_data_file='data/wallingford_flag_model_data.csv',
    differential_column='jameson_wallingford_differential',
    differential_offset=2.13,  # Calculated from multiple components
    api_urls={
        'benson_downstream': API_URLS['benson_downstream'],
        'cleeve_upstream': API_URLS['cleeve_upstream'],
    },
    min_differential=-0.1,
    max_differential=3.0,  # Wallingford can go higher than other locations
)

# Location registry
LOCATIONS = {
    'isis': ISIS_CONFIG,
    'godstow': GODSTOW_CONFIG,
    'wallingford': WALLINGFORD_CONFIG,
}


def get_location_config(location: str) -> LocationConfig:
    """
    Get configuration for a specific location.
    
    Args:
        location: Location name ('isis' or 'godstow')
        
    Returns:
        LocationConfig object
        
    Raises:
        ValueError: If location is not recognized
    """
    location = location.lower()
    if location not in LOCATIONS:
        raise ValueError(f"Unknown location: {location}. Available: {list(LOCATIONS.keys())}")
    return LOCATIONS[location]


# Model hyperparameters
MODEL_CONFIG = {
    'sequence_length': 100,      
    'hidden_sizes': [192, 128, 64],
    'dropout_rate': 0.3,
    'catchment_lag': 18,         # hours - rain at time T affects river at T+18h
}

TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'patience': 7,
    'validation_split': 0.2,
    'max_grad_norm': 1.0,
}

"""
Flag thresholds for mapping differentials to flag colours.

We support location-dependent thresholds while keeping backwards
compatibility with the original global thresholds used for ISIS.
"""

# ISIS thresholds - same as original notebook
FIXED_THRESHOLDS = {
    'green': (-float('inf'), 0.215),
    'light_blue': (0.215, 0.33),
    'dark_blue': (0.33, 0.44),
    'amber': (0.44, 0.535),
    'red': (0.535, float('inf')),
}

# Historical thresholds (unused by default but kept for reference)
HISTORICAL_THRESHOLDS = {
    'green': (-float('inf'), 0.1366),
    'light_blue': (0.1366, 0.2582),
    'dark_blue': (0.2582, 0.387),
    'amber': (0.387, 0.6047),
    'red': (0.6047, float('inf')),
}

# Godstow thresholds (requested):
#   green  < 0.45
#   amber  0.45–0.75
#   red    > 0.75
# We keep light_blue / dark_blue as zero-width bands so code expecting
# these keys continues to work, but effectively only green/amber/red are used.
GODSTOW_THRESHOLDS = {
    'green': (-float('inf'), 0.45),
    'light_blue': (0.45, 0.45),   # effectively unused
    'dark_blue': (0.45, 0.45),    # effectively unused
    'amber': (0.45, 0.75),
    'red': (0.75, float('inf')),
}

# Wallingford thresholds: Not used (white flags - no flags)
# Thresholds are kept as placeholders for code compatibility but are never used.
WALLINGFORD_THRESHOLDS = {
    'green': (-float('inf'), 0.0),
    'light_blue': (0.0, 0.0),
    'dark_blue': (0.0, 0.0),
    'amber': (0.0, 0.0),
    'red': (0.0, float('inf')),
}

# Location-dependent registry
LOCATION_FLAG_THRESHOLDS = {
    'isis': FIXED_THRESHOLDS,
    'godstow': GODSTOW_THRESHOLDS,
    'wallingford': WALLINGFORD_THRESHOLDS,
}


def get_flag_thresholds(location: str) -> dict:
    """
    Get flag thresholds for a specific location.

    Args:
        location: 'isis', 'godstow', or 'wallingford'

    Returns:
        Dict mapping flag names to (lower, upper) differential bounds.
    """
    location = location.lower()
    return LOCATION_FLAG_THRESHOLDS.get(location, FIXED_THRESHOLDS)


# Backwards-compatible global thresholds (ISIS defaults).
# NOTE: New code should prefer get_flag_thresholds(location).
FLAG_THRESHOLDS = FIXED_THRESHOLDS


FLAG_COLORS = {
    'green': '#008001',
    'light_blue': '#02bfff',
    'dark_blue': '#000080',
    'amber': '#ffa503',
    'red': '#ff0000',
}
