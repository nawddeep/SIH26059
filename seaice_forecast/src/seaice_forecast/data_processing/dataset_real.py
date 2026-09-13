"""
On-the-Fly Sliding-Window PyTorch Dataset for Real Polar Data.

Constructs training/validation/testing samples directly from daily regridded arrays
on disk without pre-materializing multi-gigabyte overlapping sequence copies.

Input representation:
  - Default: [49, H=332, W=316] (7 days x 7 variables concatenated channels)
  - Or explicit temporal: [T=7, C=7, H=332, W=316]
Target representation:
  - Next-day SIC: [1, H=332, W=316] (bounded [0, 1])
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Union
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RealSeaIceDataset(Dataset):
    """
    On-the-fly sliding window dataset over daily regridded real-data arrays.
    """

    def __init__(
        self,
        daily_data_dir: Union[str, Path],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        input_window: int = 7,
        forecast_horizon: int = 1,
        channel_concat: bool = True,
        phase: str = "phase2",
        variables: Optional[List[str]] = None,
        normalization_stats: Optional[Dict] = None,
        mask_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            daily_data_dir: Directory containing daily regridded files (YYYY/YYYYMMDD.npz)
            start_date: Inclusive start date (string 'YYYY-MM-DD' or datetime)
            end_date: Inclusive end date (string 'YYYY-MM-DD' or datetime)
            input_window: Number of history days (default: 7)
            forecast_horizon: Lead time in days (default: 1 for next-day forecast)
            channel_concat: If True, returns concatenated channels [C_in, H, W]
            phase: 'phase1' (SIC only, C_in=7) or 'phase2' (all variables, C_in=49)
            variables: Optional specific variable subset
            normalization_stats: Optional dictionary with mean/std per variable
            mask_path: Optional path to land_ocean_mask_ps25.npy
        """
        self.daily_dir = Path(daily_data_dir)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.channel_concat = channel_concat
        self.phase = phase.lower()
        self.variables = variables or (["sic"] if self.phase == "phase1" else [
            "sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"
        ])
        self.all_variables = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]
        self.var_indices = [self.all_variables.index(v) for v in self.variables]
        self.norm_stats = normalization_stats

        # Parse date bounds
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.start_date = start_date
        self.end_date = end_date

        # Load land/ocean mask if available
        self.mask: Optional[np.ndarray] = None
        if mask_path:
            mp = Path(mask_path)
            if mp.exists():
                self.mask = np.load(mp)
        else:
            default_mask = self.daily_dir.parent.parent / "land_ocean_mask_ps25.npy"
            if default_mask.exists():
                self.mask = np.load(default_mask)

        # Build index of valid sliding windows
        self.samples: List[Tuple[List[Path], Path, str]] = []
        self._build_index()

    def _build_index(self):
        """
        Scan daily files and construct valid sequence windows.
        A valid sample requires input_window consecutive days followed by
        forecast_horizon day target with NO temporal gaps.
        """
        all_npz = sorted(self.daily_dir.glob("*/*.npz"))
        if not all_npz:
            all_npz = sorted(self.daily_dir.glob("*.npz"))

        # Map date -> file path
        date_to_file: Dict[datetime.date, Path] = {}
        for f in all_npz:
            stem = f.stem
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue

            if self.start_date and d < self.start_date.date():
                continue
            if self.end_date and d > self.end_date.date():
                continue

            date_to_file[d] = f

        sorted_dates = sorted(date_to_file.keys())
        total_days = len(sorted_dates)
        logger.info(f"Dataset indexed {total_days} available daily files.")

        if total_days < self.input_window + self.forecast_horizon:
            logger.warning(
                f"Not enough daily files ({total_days}) to build a single "
                f"{self.input_window}-day window with horizon {self.forecast_horizon}."
            )
            return

        date_set = set(sorted_dates)
        required_len = self.input_window + self.forecast_horizon

        # Find every valid sequence
        for start_d in sorted_dates:
            window_dates = [start_d + timedelta(days=i) for i in range(required_len)]
            # Check if all dates in window exist continuously
            if all(wd in date_set for wd in window_dates):
                input_files = [date_to_file[wd] for wd in window_dates[:self.input_window]]
                target_file = date_to_file[window_dates[-1]]
                target_date_str = window_dates[-1].isoformat()
                self.samples.append((input_files, target_file, target_date_str))

        logger.info(
            f"Built {len(self.samples)} valid sliding-window samples "
            f"from {total_days} daily arrays (stride=1)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_daily_array(self, path: Path) -> np.ndarray:
        """Load single daily array: [7, H, W]."""
        with np.load(path) as data:
            arr = data["data"].astype(np.float32)
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and assemble the sample on the fly.

        Returns:
            input_tensor: [49, H, W] if channel_concat=True else [7, 7, H, W]
            target_tensor: [1, H, W] (next-day SIC)
        """
        input_files, target_file, _ = self.samples[idx]

        # Load input arrays: list of (7, H, W) -> stack to (T=7, C=7, H, W)
        input_stack = np.stack([self._load_daily_array(f) for f in input_files], axis=0)

        # Apply z-score normalization if provided
        if self.norm_stats:
            for c_idx, var_name in enumerate(["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]):
                if var_name in self.norm_stats and var_name != "sic":
                    mean = self.norm_stats[var_name]["mean"]
                    std = self.norm_stats[var_name]["std"]
                    if std > 1e-6:
                        input_stack[:, c_idx] = (input_stack[:, c_idx] - mean) / std

        # Target: SIC only (channel index 0) from the forecast horizon day
        target_bundle = self._load_daily_array(target_file)
        target_sic = target_bundle[0:1]  # [1, H, W]

        # Select variables
        input_stack = input_stack[:, self.var_indices]  # [T=7, C_selected, H, W]

        # Reshape inputs
        if self.channel_concat:
            # Flatten [T, C, H, W] -> [T * C, H, W]
            T, C, H, W = input_stack.shape
            input_tensor = input_stack.reshape(T * C, H, W)
        else:
            input_tensor = input_stack

        # Convert to PyTorch tensors
        x = torch.from_numpy(input_tensor).float()
        y = torch.from_numpy(target_sic).float()

        return x, y

    def compute_normalization_stats(self) -> Dict:
        """
        Compute mean and standard deviation for each environmental variable
        across all indexed daily arrays (training split only).
        """
        logger.info(f"Computing normalization statistics across {len(self.samples)} sample windows...")
        var_names = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]

        # Collect sample files
        sampled_files = set()
        for input_files, _, _ in self.samples[::max(1, len(self.samples) // 100)]:
            sampled_files.update(input_files)

        all_arrays = [self._load_daily_array(f) for f in sampled_files]
        big_stack = np.stack(all_arrays, axis=0)  # [N, 7, H, W]

        stats = {}
        for c_idx, var_name in enumerate(var_names):
            vals = big_stack[:, c_idx]
            if var_name == "sic":
                stats[var_name] = {"mean": 0.0, "std": 1.0, "scaling": "bounded [0, 1]"}
            else:
                if self.mask is not None:
                    ocean_vals = vals[:, self.mask == 1]
                    stats[var_name] = {
                        "mean": float(np.mean(ocean_vals)),
                        "std": float(np.std(ocean_vals)),
                    }
                else:
                    stats[var_name] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }

        logger.info(f"Computed normalization stats: {json.dumps(stats, indent=2)}")
        return stats


class MultiHorizonSeaIceDataset(Dataset):
    """
    On-the-fly sliding window dataset that yields multi-horizon targets {1: y1, 3: y3, 5: y5, 7: y7}.
    """

    def __init__(
        self,
        daily_data_dir: Union[str, Path],
        horizons: List[int] = [1, 3, 5, 7],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        input_window: int = 7,
        channel_concat: bool = False,
        phase: str = "phase2",
        variables: Optional[List[str]] = None,
        normalization_stats: Optional[Dict] = None,
        mask_path: Optional[Union[str, Path]] = None,
    ):
        self.daily_dir = Path(daily_data_dir)
        self.horizons = sorted(horizons)
        self.max_horizon = max(self.horizons)
        self.input_window = input_window
        self.channel_concat = channel_concat
        self.phase = phase.lower()
        self.variables = variables or (["sic"] if self.phase == "phase1" else [
            "sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"
        ])
        self.all_variables = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]
        self.var_indices = [self.all_variables.index(v) for v in self.variables]
        self.norm_stats = normalization_stats

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.start_date = start_date
        self.end_date = end_date

        self.mask: Optional[np.ndarray] = None
        if mask_path:
            mp = Path(mask_path)
            if mp.exists():
                self.mask = np.load(mp)
        else:
            default_mask = self.daily_dir.parent.parent / "land_ocean_mask_ps25.npy"
            if default_mask.exists():
                self.mask = np.load(default_mask)

        # Index sequences
        self.samples: List[Tuple[List[Path], Dict[int, Path], str]] = []
        self._build_index()

    def _build_index(self):
        all_npz = sorted(self.daily_dir.glob("*/*.npz"))
        if not all_npz:
            all_npz = sorted(self.daily_dir.glob("*.npz"))

        date_to_file: Dict[datetime.date, Path] = {}
        for f in all_npz:
            stem = f.stem
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue
            if self.start_date and d < self.start_date.date():
                continue
            if self.end_date and d > self.end_date.date():
                continue
            date_to_file[d] = f

        sorted_dates = sorted(date_to_file.keys())
        total_days = len(sorted_dates)
        logger.info(f"MultiHorizonDataset indexed {total_days} available daily files.")

        required_len = self.input_window + self.max_horizon
        if total_days < required_len:
            logger.warning(
                f"Available days ({total_days}) < required window ({required_len}) for horizons {self.horizons}."
            )
            return

        date_set = set(sorted_dates)
        for start_d in sorted_dates:
            window_dates = [start_d + timedelta(days=i) for i in range(required_len)]
            if all(wd in date_set for wd in window_dates):
                input_files = [date_to_file[wd] for wd in window_dates[:self.input_window]]
                target_files = {
                    h: date_to_file[start_d + timedelta(days=self.input_window + h - 1)]
                    for h in self.horizons
                }
                self.samples.append((input_files, target_files, str(start_d)))

        logger.info(f"Built {len(self.samples)} valid multi-horizon samples (horizons={self.horizons}).")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_daily_array(self, path: Path) -> np.ndarray:
        with np.load(path) as data:
            arr = data["data"].astype(np.float32)
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        input_files, target_files, _ = self.samples[idx]

        input_stack = np.stack([self._load_daily_array(f) for f in input_files], axis=0)

        # Apply z-score normalization
        if self.norm_stats:
            for c_idx, var_name in enumerate(self.all_variables):
                if var_name in self.norm_stats and var_name != "sic":
                    mean = self.norm_stats[var_name]["mean"]
                    std = self.norm_stats[var_name]["std"]
                    if std > 1e-6:
                        input_stack[:, c_idx] = (input_stack[:, c_idx] - mean) / std

        # Variable selection
        input_stack = input_stack[:, self.var_indices]

        # Reshape inputs
        if self.channel_concat:
            T, C, H, W = input_stack.shape
            input_tensor = input_stack.reshape(T * C, H, W)
        else:
            input_tensor = input_stack

        x = torch.from_numpy(input_tensor).float()

        # Targets for all horizons
        targets_dict = {}
        for h, t_file in target_files.items():
            t_bundle = self._load_daily_array(t_file)
            target_sic = torch.from_numpy(t_bundle[0:1]).float()  # [1, H, W]
            targets_dict[h] = target_sic

        return x, targets_dict
