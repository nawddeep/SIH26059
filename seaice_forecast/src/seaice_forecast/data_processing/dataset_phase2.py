"""
Phase 2 Dataset: Multi-variable environmental forcing for SIC forecasting.

Extends Phase 1 dataset to handle 49-channel input:
- 7 days × 7 variables = 49 channels
- Variables: SIC, wind_u, wind_v, air_temp, sst, current_u, current_v

Maintains compatibility with Phase 1 evaluation framework.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2Dataset(Dataset):
    """
    PyTorch Dataset for Phase 2 multi-variable SIC forecasting.

    Input: [49, H, W] = 7 days × 7 variables
    Target: [1, H, W] = next-day SIC
    """

    def __init__(
        self,
        data_file: str,
        mask: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        validate_channels: bool = True
    ):
        """
        Initialize Phase 2 dataset.

        Args:
            data_file: Path to .npz file with multi-variable inputs
            mask: Land-ocean mask (optional)
            transform: Optional transform
            validate_channels: Verify input has 49 channels
        """
        self.data_file = Path(data_file)
        self.mask = mask
        self.transform = transform

        # Load data
        data = np.load(self.data_file, allow_pickle=True)
        self.inputs = data['inputs']  # [N, 49, H, W]
        self.targets = data['targets']  # [N, 1, H, W]

        # Validate input shape
        if validate_channels:
            expected_channels = 49  # 7 days × 7 variables
            actual_channels = self.inputs.shape[1]

            if actual_channels != expected_channels:
                logger.warning(
                    f"Expected {expected_channels} input channels, "
                    f"got {actual_channels}. Setting validate_channels=False to proceed."
                )

        # Load metadata
        try:
            metadata_dict = data.get('metadata', None)
            if metadata_dict is not None:
                if isinstance(metadata_dict, np.ndarray):
                    self.metadata = metadata_dict.item()
                else:
                    self.metadata = metadata_dict
            else:
                self.metadata = {}
        except:
            self.metadata = {}

        # Variable info
        self.variable_order = self.metadata.get(
            'variable_order',
            ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v']
        )
        self.channels_per_variable = self.metadata.get('channels_per_variable', 7)

        logger.info(f"Loaded Phase 2 dataset from {self.data_file.name}")
        logger.info(f"  Samples: {len(self.inputs)}")
        logger.info(f"  Input shape: {self.inputs.shape} (N, 49, H, W)")
        logger.info(f"  Target shape: {self.targets.shape} (N, 1, H, W)")
        logger.info(f"  Variables: {self.variable_order}")
        logger.info(f"  Channels per variable: {self.channels_per_variable}")

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
            - input: [49, H, W]
            - target: [1, H, W]
            - mask: [1, H, W] or [H, W]
        """
        # Get input and target
        x = self.inputs[idx]  # [49, H, W]
        y = self.targets[idx]  # [1, H, W]

        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Add mask
        if self.mask is not None:
            mask = torch.from_numpy(self.mask).float()
            # Ensure mask has batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # [1, H, W]
        else:
            mask = torch.ones_like(y)

        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y, mask

    def get_variable_slice(self, idx: int, variable_name: str) -> torch.Tensor:
        """
        Extract a specific variable's time series from a sample.

        Args:
            idx: Sample index
            variable_name: Variable name (e.g., 'wind_u')

        Returns:
            Tensor of shape [7, H, W] for the variable
        """
        if variable_name not in self.variable_order:
            raise ValueError(f"Unknown variable: {variable_name}")

        var_idx = self.variable_order.index(variable_name)
        start_channel = var_idx * self.channels_per_variable
        end_channel = start_channel + self.channels_per_variable

        x = self.inputs[idx]
        var_data = x[start_channel:end_channel]  # [7, H, W]

        return torch.from_numpy(var_data).float()

    def get_channel_info(self) -> Dict:
        """
        Get information about channel organization.

        Returns:
            Dictionary describing channel layout
        """
        channel_info = {}

        for i, var_name in enumerate(self.variable_order):
            start_ch = i * self.channels_per_variable
            end_ch = start_ch + self.channels_per_variable

            channel_info[var_name] = {
                'channels': list(range(start_ch, end_ch)),
                'start': start_ch,
                'end': end_ch,
                'n_channels': self.channels_per_variable
            }

        return channel_info


class AblationDataset(Dataset):
    """
    Dataset for ablation studies - allows selecting subsets of variables.

    Useful for Phase 2 feature importance analysis.
    """

    def __init__(
        self,
        base_dataset: Phase2Dataset,
        enabled_variables: List[str],
        fill_disabled: float = 0.0
    ):
        """
        Create ablation dataset from base Phase 2 dataset.

        Args:
            base_dataset: Phase2Dataset instance
            enabled_variables: List of variables to keep (others set to fill_disabled)
            fill_disabled: Value to fill disabled variables
        """
        self.base_dataset = base_dataset
        self.enabled_variables = enabled_variables
        self.fill_disabled = fill_disabled

        # Validate enabled variables
        for var in enabled_variables:
            if var not in self.base_dataset.variable_order:
                raise ValueError(f"Unknown variable: {var}")

        logger.info(f"Created ablation dataset with variables: {enabled_variables}")
        logger.info(f"Disabled variables filled with: {fill_disabled}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample with ablation applied."""
        x, y, mask = self.base_dataset[idx]

        # Create ablated input
        x_ablated = x.clone()

        # Zero out disabled variables
        channel_info = self.base_dataset.get_channel_info()

        for var_name in self.base_dataset.variable_order:
            if var_name not in self.enabled_variables:
                ch_start = channel_info[var_name]['start']
                ch_end = channel_info[var_name]['end']
                x_ablated[ch_start:ch_end] = self.fill_disabled

        return x_ablated, y, mask


def create_phase2_dataloaders(
    config: dict,
    mask: Optional[np.ndarray] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for Phase 2 multi-variable training.

    Args:
        config: Configuration dictionary
        mask: Land-ocean mask
        data_dir: Directory containing Phase 2 data files
                 (defaults to data/processed/phase2/)

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    if data_dir is None:
        project_root = Path(config.get('project_root', '.'))
        data_dir = project_root / 'data' / 'processed' / 'phase2'

    batch_size = config['training']['batch_size']
    num_workers = config['compute'].get('num_workers', 4)
    pin_memory = config['compute'].get('pin_memory', True)

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        data_file = data_dir / f"{split}_data_phase2.npz"

        if not data_file.exists():
            logger.warning(f"Phase 2 data file not found: {data_file}")
            logger.info(f"Run prepare_environmental_data.py first")
            continue

        dataset = Phase2Dataset(
            data_file=str(data_file),
            mask=mask,
            validate_channels=True
        )

        # Shuffle only for training
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')
        )

        dataloaders[split] = dataloader
        logger.info(f"Created Phase 2 {split} DataLoader: {len(dataset)} samples, "
                   f"{len(dataloader)} batches")

    return dataloaders


def create_ablation_dataloaders(
    base_dataloaders: Dict[str, DataLoader],
    enabled_variables: List[str],
    config: dict
) -> Dict[str, DataLoader]:
    """
    Create ablation dataloaders from base Phase 2 dataloaders.

    Args:
        base_dataloaders: Base Phase 2 dataloaders
        enabled_variables: Variables to enable
        config: Configuration dictionary

    Returns:
        Dictionary of ablation dataloaders
    """
    batch_size = config['training']['batch_size']
    num_workers = config['compute'].get('num_workers', 4)
    pin_memory = config['compute'].get('pin_memory', True)

    ablation_loaders = {}

    for split, base_loader in base_dataloaders.items():
        base_dataset = base_loader.dataset

        ablation_dataset = AblationDataset(
            base_dataset,
            enabled_variables,
            fill_disabled=0.0
        )

        shuffle = (split == 'train')

        ablation_loader = DataLoader(
            ablation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')
        )

        ablation_loaders[split] = ablation_loader

    logger.info(f"Created ablation dataloaders for variables: {enabled_variables}")

    return ablation_loaders


def test_phase2_dataset():
    """Test Phase 2 dataset loading."""
    from seaice_forecast.config import load_config

    config = load_config()

    # Load mask
    project_root = Path(config.get('project_root', '.'))
    mask_file = project_root / 'data' / 'processed' / 'land_ocean_mask.npy'

    if mask_file.exists():
        mask = np.load(mask_file)
        print(f"Loaded mask: {mask.shape}")
    else:
        mask = None
        print("No mask found")

    # Create dataloaders
    dataloaders = create_phase2_dataloaders(config, mask)

    if not dataloaders:
        print("No dataloaders created - run prepare_environmental_data.py first")
        return

    # Test loading
    for split, dataloader in dataloaders.items():
        print(f"\n{split.upper()} Split:")

        for inputs, targets, masks in dataloader:
            print(f"  Input batch: {inputs.shape}")
            print(f"  Target batch: {targets.shape}")
            print(f"  Mask batch: {masks.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")

            # Show channel info
            dataset = dataloader.dataset
            if isinstance(dataset, Phase2Dataset):
                channel_info = dataset.get_channel_info()
                print(f"  Channel organization:")
                for var, info in channel_info.items():
                    print(f"    {var}: channels {info['start']}-{info['end']-1}")

            break


if __name__ == "__main__":
    test_phase2_dataset()
