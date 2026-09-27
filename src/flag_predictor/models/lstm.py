"""
Multi-Horizon LSTM Model for river differential prediction.

This module defines the PyTorch LSTM model architecture that predicts
multiple future time horizons simultaneously.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def get_device() -> torch.device:
    """
    Get the best available device (MPS/CUDA/CPU).
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: cuda (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: cpu")
    return device


class MultiHorizonLSTMModel(nn.Module):
    """
    LSTM model that predicts multiple future time horizons simultaneously.
    
    Architecture:
    - Stacked LSTM layers with dropout
    - Separate output head for each prediction horizon
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes (e.g., [192, 128, 64])
        dropout_rate: Dropout rate between layers
        n_horizons: Number of future horizons to predict
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2,
        n_horizons: int = 10
    ):
        super(MultiHorizonLSTMModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.n_horizons = n_horizons
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Multiple output heads for different time horizons
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], 1) for _ in range(n_horizons)
        ])
    
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            debug: Whether to print debug information
            
        Returns:
            Predictions tensor of shape (batch_size, n_horizons)
        """
        if debug:
            print(f"[FORWARD] Input shape: {x.shape}")
        
        # Pass through LSTM layers
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
            if debug:
                print(f"[FORWARD] LSTM layer {i+1} output shape: {x.shape}")
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        if debug:
            print(f"[FORWARD] After last timestep: {x.shape}")
        
        # Predict multiple horizons through separate heads
        predictions = []
        for fc in self.fc_layers:
            predictions.append(fc(x))
        
        # Stack predictions: (batch_size, n_horizons)
        predictions = torch.cat(predictions, dim=1)
        
        if debug:
            print(f"[FORWARD] Final output shape: {predictions.shape}")
        
        return predictions
    
    def get_config(self) -> dict:
        """
        Get model configuration for serialization.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'hidden_sizes': self.hidden_sizes,
            'num_layers': self.num_layers,
            'n_horizons': self.n_horizons,
        }


class HourlyDecoderModel(nn.Module):
    """
    Encoder-decoder LSTM that emits an hourly trajectory.

    The encoder reads `sequence_length` hours of past features. The decoder
    then steps across `horizon` future hours, seeing known future covariates
    at each step (rainfall, and for location models predicted Farmoor flow).
    Outputs are deltas from the value at t0.
    """

    architecture = 'hourly_decoder'

    def __init__(
        self,
        input_size: int,
        decoder_input_size: int,
        hidden_sizes: List[int] = [192, 128, 64],
        dropout_rate: float = 0.3,
        horizon: int = 240,
    ):
        super().__init__()
        self.input_size = input_size
        self.decoder_input_size = decoder_input_size
        self.hidden_sizes = list(hidden_sizes)
        self.num_layers = len(self.hidden_sizes)
        self.dropout_rate = dropout_rate
        self.horizon = horizon

        self.encoder_layers = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        for i, hidden_size in enumerate(self.hidden_sizes):
            in_dim = input_size if i == 0 else self.hidden_sizes[i - 1]
            self.encoder_layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
            self.encoder_dropouts.append(nn.Dropout(dropout_rate))

        dec_hidden = self.hidden_sizes[-1]
        self.decoder = nn.LSTM(decoder_input_size, dec_hidden, batch_first=True)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(dec_hidden, 1)

    def forward(
        self,
        x_past: torch.Tensor,
        future_cov: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_past: (batch, sequence_length, input_size)
            future_cov: (batch, horizon, decoder_input_size)

        Returns:
            (batch, horizon) predicted deltas
        """
        x = x_past
        h_n = c_n = None
        for lstm, dropout in zip(self.encoder_layers, self.encoder_dropouts):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
        if debug:
            print(f"[HOURLY] encoder last {x.shape} h {h_n.shape}")

        dec_out, _ = self.decoder(future_cov, (h_n, c_n))
        dec_out = self.decoder_dropout(dec_out)
        pred = self.head(dec_out).squeeze(-1)
        if debug:
            print(f"[HOURLY] pred {pred.shape}")
        return pred

    def get_config(self) -> dict:
        return {
            'architecture': self.architecture,
            'hidden_sizes': self.hidden_sizes,
            'num_layers': self.num_layers,
            'decoder_input_size': self.decoder_input_size,
            'horizon': self.horizon,
        }
