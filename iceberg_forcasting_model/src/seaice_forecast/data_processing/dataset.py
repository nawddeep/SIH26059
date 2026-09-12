"""
PyTorch Dataset and DataLoader for SIC forecasting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SICDataset(Dataset):
    """PyTorch Dataset for Sea-Ice Concentration forecasting."""
    
    def __init__(
        self,
        data_file: str,
        mask: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to .npz file containing inputs and targets
            mask: Land-ocean mask (optional)
            transform: Optional transform to apply
        """
        self.data_file = Path(data_file)
        self.mask = mask
        self.transform = transform
        
        # Load data
        data = np.load(self.data_file)
        self.inputs = data['inputs']  # [N, T, H, W]
        self.targets = data['targets']  # [N, 1, H, W]
        
        # Load metadata
        metadata_file = self.data_file.parent / self.data_file.name.replace(
            '_data.npz', '_metadata.json'
        )
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        logger.info(f"Loaded dataset from {self.data_file}")
        logger.info(f"  Samples: {len(self.inputs)}")
        logger.info(f"  Input shape: {self.inputs.shape}")
        logger.info(f"  Target shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input, target, mask)
        """
        # Get input and target
        x = self.inputs[idx]  # [T, H, W]
        y = self.targets[idx]  # [1, H, W]
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        # Add mask
        if self.mask is not None:
            mask = torch.from_numpy(self.mask).float()
        else:
            mask = torch.ones_like(y)
        
        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y, mask
    
    def get_date_pair(self, idx: int) -> Tuple[str, str]:
        """
        Get the date pair for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_end_date, target_date)
        """
        if 'date_pairs' in self.metadata:
            return tuple(self.metadata['date_pairs'][idx])
        else:
            return ("unknown", "unknown")


def create_dataloaders(
    config: dict,
    mask: Optional[np.ndarray] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        config: Configuration dictionary
        mask: Land-ocean mask (optional)
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    processed_dir = Path(config['data']['paths']['processed'])
    batch_size = config['training']['batch_size']
    num_workers = config['compute']['num_workers']
    pin_memory = config['compute']['pin_memory']
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        data_file = processed_dir / f"{split}_data.npz"
        
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            continue
        
        dataset = SICDataset(
            data_file=str(data_file),
            mask=mask
        )
        
        # Shuffle only for training
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')  # Drop incomplete batches in training
        )
        
        dataloaders[split] = dataloader
        logger.info(f"Created {split} DataLoader: {len(dataset)} samples, "
                   f"{len(dataloader)} batches")
    
    return dataloaders


class SICNormalizer:
    """Normalize SIC data using pre-computed statistics."""
    
    def __init__(self, stats_file: str):
        """
        Initialize normalizer.
        
        Args:
            stats_file: Path to JSON file with mean and std
        """
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        self.mean = stats['mean']
        self.std = stats['std']
        
        logger.info(f"Loaded normalization stats: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize data."""
        return (x - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize data."""
        return x * self.std + self.mean


def collate_with_mask(batch):
    """
    Custom collate function that handles mask.
    
    Args:
        batch: List of (input, target, mask) tuples
        
    Returns:
        Batched tensors
    """
    inputs, targets, masks = zip(*batch)
    
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return inputs, targets, masks


def main():
    """Test dataset loading."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    from seaice_forecast.config import load_config, resolve_paths
    
    config = load_config()
    config = resolve_paths(config)
    
    # Load mask
    mask_file = Path(config['data']['paths']['processed']) / "land_ocean_mask.npy"
    if mask_file.exists():
        mask = np.load(mask_file)
        print(f"Loaded mask: {mask.shape}")
    else:
        mask = None
        print("No mask found")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config, mask)
    
    # Test loading a batch
    for split, dataloader in dataloaders.items():
        print(f"\n{split.upper()} Split:")
        for inputs, targets, masks in dataloader:
            print(f"  Input batch: {inputs.shape}")
            print(f"  Target batch: {targets.shape}")
            print(f"  Mask batch: {masks.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            break  # Just show first batch


if __name__ == "__main__":
    main()
