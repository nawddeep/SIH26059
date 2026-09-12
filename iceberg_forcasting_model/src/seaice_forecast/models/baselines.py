"""
Baseline models for sea-ice concentration forecasting.

Implements:
1. Persistence: SIC_hat(t+1) = SIC(t)
2. Climatology: Day-of-year expected SIC from training period
"""

import numpy as np
import torch
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistenceModel:
    """
    Persistence baseline: assumes tomorrow's SIC equals today's SIC.
    
    This is the simplest forecast model and serves as the minimum
    performance bar any ML model should beat.
    """
    
    def __init__(self, name: str = "persistence"):
        """
        Initialize persistence model.
        
        Args:
            name: Model name
        """
        self.name = name
    
    def predict(
        self, 
        inputs: np.ndarray,
        return_torch: bool = False
    ) -> np.ndarray:
        """
        Make persistence forecast: use the last day's SIC as the forecast.
        
        Args:
            inputs: Input SIC sequence [N, T, H, W] or [T, H, W]
            return_torch: Return torch tensor instead of numpy
            
        Returns:
            Forecast [N, 1, H, W] or [1, H, W]
        """
        # Handle both batched and unbatched inputs
        if inputs.ndim == 4:
            # Batched: [N, T, H, W]
            # Take the last timestep from each sequence
            forecast = inputs[:, -1:, :, :]  # [N, 1, H, W]
        elif inputs.ndim == 3:
            # Unbatched: [T, H, W]
            forecast = inputs[-1:, :, :]  # [1, H, W]
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {inputs.shape}")
        
        if return_torch and not isinstance(forecast, torch.Tensor):
            forecast = torch.from_numpy(forecast).float()
        
        return forecast
    
    def __repr__(self):
        return f"PersistenceModel(name='{self.name}')"


class ClimatologyModel:
    """
    Climatology baseline: uses day-of-year expected SIC from training period.
    
    For each day of the year (1-365/366), computes the average SIC across
    all training years for that day, with optional smoothing window.
    """
    
    def __init__(
        self, 
        name: str = "climatology",
        smoothing_window: int = 15
    ):
        """
        Initialize climatology model.
        
        Args:
            name: Model name
            smoothing_window: Days to smooth climatology (centered moving average)
        """
        self.name = name
        self.smoothing_window = smoothing_window
        self.climatology = None  # Will be [366, H, W] array
        self.is_fitted = False
    
    def fit(
        self,
        sic_data: np.ndarray,
        dates: List[datetime]
    ):
        """
        Fit climatology from training data.
        
        Args:
            sic_data: SIC data [N, H, W] from training set
            dates: Corresponding dates for each sample
        """
        if len(sic_data) != len(dates):
            raise ValueError("Number of samples must match number of dates")
        
        logger.info(f"Fitting climatology from {len(dates)} training samples...")
        
        # Get spatial dimensions
        H, W = sic_data.shape[1], sic_data.shape[2]
        
        # Initialize climatology array for each day of year (1-366)
        climatology = np.zeros((366, H, W))
        counts = np.zeros(366)
        
        # Accumulate SIC for each day of year
        for sic, date in zip(sic_data, dates):
            doy = date.timetuple().tm_yday  # Day of year (1-366)
            climatology[doy - 1] += sic
            counts[doy - 1] += 1
        
        # Compute mean for each day
        for doy in range(366):
            if counts[doy] > 0:
                climatology[doy] /= counts[doy]
            else:
                # If no data for this day, use average of neighbors
                if doy > 0 and doy < 365:
                    climatology[doy] = (climatology[doy - 1] + climatology[doy + 1]) / 2
        
        # Apply smoothing if requested
        if self.smoothing_window > 1:
            climatology = self._smooth_climatology(climatology, self.smoothing_window)
        
        self.climatology = climatology
        self.is_fitted = True
        
        # Log statistics
        days_with_data = np.sum(counts > 0)
        logger.info(f"Climatology fitted: {days_with_data}/366 days have data")
        logger.info(f"Applied {self.smoothing_window}-day smoothing window")
    
    def _smooth_climatology(
        self, 
        climatology: np.ndarray, 
        window: int
    ) -> np.ndarray:
        """
        Apply centered moving average smoothing to climatology.
        
        Args:
            climatology: Raw climatology [366, H, W]
            window: Smoothing window size (days)
            
        Returns:
            Smoothed climatology [366, H, W]
        """
        half_window = window // 2
        smoothed = np.zeros_like(climatology)
        
        for doy in range(366):
            # Get window indices (wrap around at year boundaries)
            indices = []
            for offset in range(-half_window, half_window + 1):
                idx = (doy + offset) % 366
                indices.append(idx)
            
            # Average over window
            smoothed[doy] = np.mean(climatology[indices], axis=0)
        
        return smoothed
    
    def predict(
        self,
        inputs: np.ndarray,
        dates: List[datetime],
        return_torch: bool = False
    ) -> np.ndarray:
        """
        Make climatology forecast: return expected SIC for the target date.
        
        Args:
            inputs: Input SIC sequence [N, T, H, W] (not actually used, included for API consistency)
            dates: Target dates for forecasts (one per sample)
            return_torch: Return torch tensor instead of numpy
            
        Returns:
            Forecast [N, 1, H, W]
        """
        if not self.is_fitted:
            raise RuntimeError("Climatology must be fitted before prediction")
        
        if inputs.ndim == 4:
            N = inputs.shape[0]
        else:
            N = 1
            dates = [dates]
        
        if len(dates) != N:
            raise ValueError(f"Number of dates ({len(dates)}) must match batch size ({N})")
        
        forecasts = []
        
        for date in dates:
            doy = date.timetuple().tm_yday  # Day of year (1-366)
            forecast = self.climatology[doy - 1]  # [H, W]
            forecasts.append(forecast)
        
        # Stack forecasts
        forecasts = np.stack(forecasts, axis=0)  # [N, H, W]
        forecasts = forecasts[:, np.newaxis, :, :]  # [N, 1, H, W]
        
        if return_torch:
            forecasts = torch.from_numpy(forecasts).float()
        
        return forecasts
    
    def save(self, save_path: str):
        """
        Save fitted climatology to disk.
        
        Args:
            save_path: Path to save file (.npz)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted climatology")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            save_path,
            climatology=self.climatology,
            smoothing_window=self.smoothing_window
        )
        
        # Save metadata
        metadata = {
            'name': self.name,
            'smoothing_window': self.smoothing_window,
            'shape': list(self.climatology.shape),
            'is_fitted': self.is_fitted
        }
        
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Climatology saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load fitted climatology from disk.
        
        Args:
            load_path: Path to load file (.npz)
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Climatology file not found: {load_path}")
        
        data = np.load(load_path)
        self.climatology = data['climatology']
        self.smoothing_window = int(data['smoothing_window'])
        self.is_fitted = True
        
        logger.info(f"Climatology loaded from {load_path}")
        logger.info(f"Shape: {self.climatology.shape}, smoothing: {self.smoothing_window} days")
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ClimatologyModel(name='{self.name}', smoothing={self.smoothing_window}, {status})"


def fit_climatology_from_dataset(
    data_file: str,
    metadata_file: str,
    smoothing_window: int = 15,
    save_path: Optional[str] = None
) -> ClimatologyModel:
    """
    Fit climatology model from processed dataset.
    
    Args:
        data_file: Path to processed data .npz file
        metadata_file: Path to metadata JSON file with dates
        smoothing_window: Smoothing window for climatology
        save_path: Optional path to save fitted model
        
    Returns:
        Fitted ClimatologyModel
    """
    # Load data
    data = np.load(data_file)
    inputs = data['inputs']  # [N, T, H, W]
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get target dates (end of input window = start of forecast)
    date_pairs = metadata['date_pairs']
    target_dates = [
        datetime.strptime(date_pair[1], "%Y-%m-%d")
        for date_pair in date_pairs
    ]
    
    # Use last timestep from each input sequence for training climatology
    # Shape: [N, H, W]
    sic_data = inputs[:, -1, :, :]
    
    # Create and fit model
    model = ClimatologyModel(smoothing_window=smoothing_window)
    model.fit(sic_data, target_dates)
    
    # Save if requested
    if save_path:
        model.save(save_path)
    
    return model


def main():
    """Test baseline models."""
    # Create dummy data
    N, T, H, W = 10, 7, 50, 50
    inputs = np.random.rand(N, T, H, W).astype(np.float32)
    
    # Test persistence
    print("Testing Persistence Model:")
    persistence = PersistenceModel()
    forecast = persistence.predict(inputs)
    print(f"  Input shape: {inputs.shape}")
    print(f"  Forecast shape: {forecast.shape}")
    print(f"  Forecast equals last timestep: {np.allclose(forecast, inputs[:, -1:, :, :])}")
    
    # Test climatology
    print("\nTesting Climatology Model:")
    
    # Create training data with dates
    train_dates = [datetime(2010, 1, 1) + np.timedelta64(i, 'D') for i in range(365)]
    train_data = np.random.rand(365, H, W).astype(np.float32)
    
    climatology = ClimatologyModel(smoothing_window=15)
    climatology.fit(train_data, train_dates)
    
    # Test prediction
    test_dates = [datetime(2020, 6, 15) for _ in range(N)]
    forecast = climatology.predict(inputs, test_dates)
    print(f"  Fitted: {climatology.is_fitted}")
    print(f"  Climatology shape: {climatology.climatology.shape}")
    print(f"  Forecast shape: {forecast.shape}")
    
    print(f"\n{persistence}")
    print(climatology)


if __name__ == "__main__":
    main()
