"""
Flag Predictor - River Differential Prediction System

A multi-horizon LSTM model that predicts river differential levels
for Oxford's ISIS and Godstow locks using rainfall forecasts.
"""

__version__ = "1.0.0"
__author__ = "Flag Predictor Team"

from .config import LocationConfig, get_location_config
from .models.lstm import MultiHorizonLSTMModel
