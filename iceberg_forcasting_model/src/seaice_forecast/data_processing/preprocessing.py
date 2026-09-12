"""
Preprocess NSIDC SIC data for model training.

This module handles:
- Loading raw NetCDF files
- Creating land-ocean mask
- Normalizing SIC values
- Handling missing data
- Creating sliding window sequences
- Train/val/test splitting
"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SICPreprocessor:
    """Preprocess sea-ice concentration data."""
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.raw_data_dir = Path(config['data']['paths']['raw'])
        self.processed_data_dir = Path(config['data']['paths']['processed'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing settings
        self.sic_scale = config['preprocessing']['sic']['scale']
        self.fill_missing = config['preprocessing']['sic']['fill_missing']
        
        # Window settings
        self.input_window = config['forecast']['input_window']
        self.forecast_horizon = config['forecast']['forecast_horizon']
        
        # Mask will be created/loaded
        self.land_ocean_mask = None
    
    def load_sic_file(self, file_path: Path) -> xr.Dataset:
        """
        Load a single SIC NetCDF file.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            xarray Dataset
        """
        ds = xr.open_dataset(file_path)
        return ds
    
    def create_land_ocean_mask(
        self, 
        sample_files: List[Path],
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create land-ocean mask from multiple SIC files.
        
        Strategy: A pixel is land if it's consistently land (SIC missing or flagged)
        across multiple days. Ocean pixels have valid SIC values most of the time.
        
        Args:
            sample_files: List of sample NetCDF files to analyze
            output_path: Path to save mask (optional)
            
        Returns:
            Binary mask: 1 = ocean, 0 = land
        """
        logger.info(f"Creating land-ocean mask from {len(sample_files)} sample files...")
        
        valid_counts = None
        total = 0
        
        for file_path in sample_files:
            ds = self.load_sic_file(file_path)
            
            # Get SIC variable (variable name may vary)
            if 'cdr_seaice_conc' in ds.variables:
                sic = ds['cdr_seaice_conc'].values
            elif 'seaice_conc_cdr' in ds.variables:
                sic = ds['seaice_conc_cdr'].values
            else:
                # Try to find SIC variable
                for var in ds.variables:
                    if 'conc' in var.lower():
                        sic = ds[var].values
                        break
                else:
                    raise ValueError(f"Cannot find SIC variable in {file_path}")
            
            # Squeeze any singleton dimensions
            sic = np.squeeze(sic)
            
            # Check for valid data (not NaN, not fill value, within reasonable range)
            valid = np.logical_and(
                np.isfinite(sic),
                np.logical_and(sic >= 0, sic <= 100)
            )
            
            if valid_counts is None:
                valid_counts = valid.astype(int)
            else:
                valid_counts += valid.astype(int)
            
            total += 1
            ds.close()
        
        # A pixel is ocean if it has valid data in at least 50% of samples
        threshold = total * 0.5
        mask = (valid_counts >= threshold).astype(np.uint8)
        
        ocean_pixels = np.sum(mask)
        land_pixels = np.sum(1 - mask)
        
        logger.info(f"Mask created: {ocean_pixels} ocean pixels, {land_pixels} land pixels")
        logger.info(f"Ocean coverage: {ocean_pixels / (ocean_pixels + land_pixels):.1%}")
        
        # Save mask
        if output_path is None:
            output_path = self.processed_data_dir / "land_ocean_mask.npy"
        
        np.save(output_path, mask)
        logger.info(f"Mask saved to {output_path}")
        
        # Also save as NetCDF for visualization
        mask_ds = xr.Dataset({
            'mask': (['y', 'x'], mask)
        })
        mask_ds.to_netcdf(str(output_path).replace('.npy', '.nc'))
        
        self.land_ocean_mask = mask
        return mask
    
    def load_mask(self, mask_path: Optional[Path] = None) -> np.ndarray:
        """
        Load existing land-ocean mask.
        
        Args:
            mask_path: Path to mask file (optional)
            
        Returns:
            Binary mask array
        """
        if mask_path is None:
            mask_path = self.processed_data_dir / "land_ocean_mask.npy"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = np.load(mask_path)
        logger.info(f"Loaded mask from {mask_path}")
        
        self.land_ocean_mask = mask
        return mask
    
    def normalize_sic(self, sic: np.ndarray) -> np.ndarray:
        """
        Normalize SIC values from 0-100 to 0-1.
        
        Args:
            sic: Raw SIC array (0-100)
            
        Returns:
            Normalized SIC (0-1)
        """
        # Handle invalid values
        sic = np.where(np.isnan(sic), self.fill_missing, sic)
        sic = np.where(sic < 0, self.fill_missing, sic)
        sic = np.where(sic > 100, 100, sic)
        
        # Scale to [0, 1]
        return sic * self.sic_scale
    
    def load_sic_sequence(
        self, 
        file_paths: List[Path]
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Load a sequence of SIC files.
        
        Args:
            file_paths: List of paths to NetCDF files
            
        Returns:
            Tuple of (SIC array [T, H, W], list of dates)
        """
        sic_arrays = []
        dates = []
        
        for file_path in file_paths:
            ds = self.load_sic_file(file_path)
            
            # Extract SIC variable
            if 'cdr_seaice_conc' in ds.variables:
                sic = ds['cdr_seaice_conc'].values
            elif 'seaice_conc_cdr' in ds.variables:
                sic = ds['seaice_conc_cdr'].values
            else:
                for var in ds.variables:
                    if 'conc' in var.lower():
                        sic = ds[var].values
                        break
            
            sic = np.squeeze(sic)
            sic = self.normalize_sic(sic)
            sic_arrays.append(sic)
            
            # Extract date from filename
            filename = file_path.name
            date_str = filename.split('_')[4]  # seaice_conc_daily_sh_YYYYMMDD_...
            date = datetime.strptime(date_str, "%Y%m%d")
            dates.append(date)
            
            ds.close()
        
        sic_sequence = np.stack(sic_arrays, axis=0)
        return sic_sequence, dates
    
    def create_windows(
        self,
        file_paths: List[Path],
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[datetime, datetime]]]:
        """
        Create sliding window input-target pairs.
        
        Args:
            file_paths: Sorted list of SIC file paths
            stride: Stride for sliding window
            
        Returns:
            Tuple of (inputs [N, input_window, H, W], 
                     targets [N, 1, H, W],
                     list of (input_end_date, target_date) tuples)
        """
        window_size = self.input_window + self.forecast_horizon
        
        # Load all SIC data
        logger.info(f"Loading {len(file_paths)} SIC files...")
        sic_sequence, dates = self.load_sic_sequence(file_paths)
        
        # Create windows
        inputs = []
        targets = []
        date_pairs = []
        
        for i in range(0, len(sic_sequence) - window_size + 1, stride):
            input_seq = sic_sequence[i:i + self.input_window]
            target = sic_sequence[i + self.input_window:i + window_size]
            
            inputs.append(input_seq)
            targets.append(target)
            
            input_end_date = dates[i + self.input_window - 1]
            target_date = dates[i + window_size - 1]
            date_pairs.append((input_end_date, target_date))
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        logger.info(f"Created {len(inputs)} windows")
        logger.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        
        return inputs, targets, date_pairs
    
    def split_data(
        self,
        file_paths: List[Path]
    ) -> Dict[str, List[Path]]:
        """
        Split data chronologically into train/val/test sets.
        
        Args:
            file_paths: All available SIC file paths
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to file lists
        """
        # Sort files by date
        file_paths = sorted(file_paths)
        
        # Extract dates from config
        train_start = datetime.strptime(
            self.config['data']['date_ranges']['train_start'], "%Y-%m-%d"
        )
        train_end = datetime.strptime(
            self.config['data']['date_ranges']['train_end'], "%Y-%m-%d"
        )
        val_start = datetime.strptime(
            self.config['data']['date_ranges']['val_start'], "%Y-%m-%d"
        )
        val_end = datetime.strptime(
            self.config['data']['date_ranges']['val_end'], "%Y-%m-%d"
        )
        test_start = datetime.strptime(
            self.config['data']['date_ranges']['test_start'], "%Y-%m-%d"
        )
        test_end = datetime.strptime(
            self.config['data']['date_ranges']['test_end'], "%Y-%m-%d"
        )
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for file_path in file_paths:
            # Extract date from filename
            filename = file_path.name
            try:
                date_str = filename.split('_')[4]
                date = datetime.strptime(date_str, "%Y%m%d")
            except (IndexError, ValueError):
                logger.warning(f"Cannot parse date from {filename}, skipping")
                continue
            
            # Assign to split
            if train_start <= date <= train_end:
                splits['train'].append(file_path)
            elif val_start <= date <= val_end:
                splits['val'].append(file_path)
            elif test_start <= date <= test_end:
                splits['test'].append(file_path)
        
        logger.info(f"Split data: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def save_processed_data(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        date_pairs: List[Tuple[datetime, datetime]],
        split_name: str
    ):
        """
        Save processed data to disk.
        
        Args:
            inputs: Input arrays
            targets: Target arrays
            date_pairs: List of date tuples
            split_name: 'train', 'val', or 'test'
        """
        output_file = self.processed_data_dir / f"{split_name}_data.npz"
        
        # Convert dates to strings for JSON serialization
        date_strings = [
            (d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d"))
            for d1, d2 in date_pairs
        ]
        
        # Save arrays
        np.savez_compressed(
            output_file,
            inputs=inputs,
            targets=targets
        )
        
        # Save metadata
        metadata = {
            'split': split_name,
            'n_samples': len(inputs),
            'input_shape': list(inputs.shape),
            'target_shape': list(targets.shape),
            'date_pairs': date_strings,
            'input_window': self.input_window,
            'forecast_horizon': self.forecast_horizon
        }
        
        metadata_file = self.processed_data_dir / f"{split_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {split_name} data to {output_file}")
        logger.info(f"Saved {split_name} metadata to {metadata_file}")
    
    def compute_statistics(self, train_inputs: np.ndarray) -> Dict[str, float]:
        """
        Compute normalization statistics from training data.
        
        Args:
            train_inputs: Training input array [N, T, H, W]
            
        Returns:
            Dictionary with mean and std
        """
        # Only compute over ocean pixels if mask exists
        if self.land_ocean_mask is not None:
            ocean_mask = self.land_ocean_mask == 1
            ocean_data = train_inputs[:, :, ocean_mask]
            mean = float(np.mean(ocean_data))
            std = float(np.std(ocean_data))
        else:
            mean = float(np.mean(train_inputs))
            std = float(np.std(train_inputs))
        
        stats = {'mean': mean, 'std': std}
        
        # Save statistics
        stats_file = self.processed_data_dir / "normalization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Computed statistics: mean={mean:.4f}, std={std:.4f}")
        logger.info(f"Saved statistics to {stats_file}")
        
        return stats


def main():
    """Example usage."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    from seaice_forecast.config import load_config, resolve_paths
    
    config = load_config()
    config = resolve_paths(config)
    
    preprocessor = SICPreprocessor(config)
    
    # Get all raw data files
    raw_files = sorted(Path(config['data']['paths']['raw']).glob("*.nc"))
    
    if len(raw_files) == 0:
        print("No data files found. Please run download script first.")
        return
    
    print(f"Found {len(raw_files)} data files")
    
    # Create/load mask
    mask_path = Path(config['data']['paths']['mask']).with_suffix('.npy')
    if not mask_path.exists():
        print("Creating land-ocean mask...")
        preprocessor.create_land_ocean_mask(raw_files[:100], mask_path)
    else:
        print("Loading existing mask...")
        preprocessor.load_mask(mask_path)
    
    # Split data
    print("Splitting data...")
    splits = preprocessor.split_data(raw_files)
    
    # Process each split
    for split_name, files in splits.items():
        if len(files) == 0:
            print(f"No files for {split_name}, skipping")
            continue
        
        print(f"\nProcessing {split_name} split ({len(files)} files)...")
        inputs, targets, date_pairs = preprocessor.create_windows(files)
        preprocessor.save_processed_data(inputs, targets, date_pairs, split_name)
    
    # Compute statistics from training data
    if len(splits['train']) > 0:
        train_file = preprocessor.processed_data_dir / "train_data.npz"
        if train_file.exists():
            print("\nComputing normalization statistics...")
            data = np.load(train_file)
            preprocessor.compute_statistics(data['inputs'])
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
