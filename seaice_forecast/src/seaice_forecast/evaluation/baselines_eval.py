"""
Evaluate baseline models on validation/test sets.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import logging

from .metrics import compute_all_metrics, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluate baseline models (persistence and climatology)."""

    def __init__(
        self,
        config: dict,
        mask: Optional[np.ndarray] = None
    ):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            mask: Land-ocean mask
        """
        self.config = config
        self.mask = mask
        self.ice_edge_threshold = config['evaluation']['ice_edge']['threshold']

    def evaluate_persistence(
        self,
        data_file: str,
        split_name: str = "val"
    ) -> Dict[str, float]:
        """
        Evaluate persistence baseline on a dataset.

        Args:
            data_file: Path to .npz data file
            split_name: Name of split being evaluated

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating persistence baseline on {split_name} set...")

        # Load data
        data = np.load(data_file)
        inputs = data['inputs']  # [N, T, H, W]
        targets = data['targets']  # [N, 1, H, W]

        # Persistence forecast: last timestep
        predictions = inputs[:, -1:, :, :]  # [N, 1, H, W]

        # Compute metrics
        metrics = compute_all_metrics(
            predictions,
            targets,
            self.mask,
            self.ice_edge_threshold
        )

        logger.info(f"Persistence {split_name} - MAE: {metrics['mae']:.6f}, "
                   f"RMSE: {metrics['rmse']:.6f}")

        return metrics

    def evaluate_climatology(
        self,
        climatology_model,
        data_file: str,
        metadata_file: str,
        split_name: str = "val"
    ) -> Dict[str, float]:
        """
        Evaluate climatology baseline on a dataset.

        Args:
            climatology_model: Fitted climatology model
            data_file: Path to .npz data file
            metadata_file: Path to metadata JSON with dates
            split_name: Name of split being evaluated

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating climatology baseline on {split_name} set...")

        # Load data
        data = np.load(data_file)
        inputs = data['inputs']  # [N, T, H, W]
        targets = data['targets']  # [N, 1, H, W]

        # Load dates
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        date_pairs = metadata['date_pairs']
        target_dates = [
            datetime.strptime(date_pair[1], "%Y-%m-%d")
            for date_pair in date_pairs
        ]

        # Make predictions
        predictions = climatology_model.predict(inputs, target_dates)

        # Compute metrics
        metrics = compute_all_metrics(
            predictions,
            targets,
            self.mask,
            self.ice_edge_threshold
        )

        logger.info(f"Climatology {split_name} - MAE: {metrics['mae']:.6f}, "
                   f"RMSE: {metrics['rmse']:.6f}")

        return metrics

    def compare_baselines(
        self,
        persistence_metrics: Dict[str, float],
        climatology_metrics: Dict[str, float],
        split_name: str = "val"
    ):
        """
        Compare baseline performance.

        Args:
            persistence_metrics: Persistence metrics
            climatology_metrics: Climatology metrics
            split_name: Name of split
        """
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON - {split_name.upper()} SET")
        print(f"{'='*80}")

        # Metric display info
        metric_names = {
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Squared Error',
            'spatial_correlation': 'Spatial Correlation',
            'ice_edge_displacement': 'Ice Edge Displacement (px)'
        }

        print(f"\n{'Metric':<35s} {'Persistence':>15s} {'Climatology':>15s} {'Better':>12s}")
        print(f"{'-'*80}")

        for key in ['mae', 'rmse', 'spatial_correlation', 'ice_edge_displacement']:
            if key not in persistence_metrics or key not in climatology_metrics:
                continue

            pers_val = persistence_metrics[key]
            clim_val = climatology_metrics[key]

            # Determine which is better (lower is better except for correlation)
            if key == 'spatial_correlation':
                better = 'Clim' if clim_val > pers_val else 'Pers'
            else:
                better = 'Clim' if clim_val < pers_val else 'Pers'

            if not np.isnan(pers_val) and not np.isnan(clim_val):
                if key == 'spatial_correlation':
                    print(f"{metric_names[key]:<35s} {pers_val:>15.4f} {clim_val:>15.4f} {better:>12s}")
                elif key == 'ice_edge_displacement':
                    print(f"{metric_names[key]:<35s} {pers_val:>15.2f} {clim_val:>15.2f} {better:>12s}")
                else:
                    print(f"{metric_names[key]:<35s} {pers_val:>15.6f} {clim_val:>15.6f} {better:>12s}")

        print(f"{'='*80}\n")

    def save_results(
        self,
        persistence_metrics: Dict[str, float],
        climatology_metrics: Dict[str, float],
        output_file: str,
        split_name: str = "val"
    ):
        """
        Save baseline evaluation results to JSON.

        Args:
            persistence_metrics: Persistence metrics
            climatology_metrics: Climatology metrics
            output_file: Path to output JSON file
            split_name: Name of split
        """
        results = {
            'split': split_name,
            'timestamp': datetime.now().isoformat(),
            'baselines': {
                'persistence': persistence_metrics,
                'climatology': climatology_metrics
            },
            'config': {
                'ice_edge_threshold': self.ice_edge_threshold,
                'mask_used': self.mask is not None
            }
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """Test evaluator."""
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

    evaluator = BaselineEvaluator(config, mask)

    # Test with dummy data
    processed_dir = Path(config['data']['paths']['processed'])
    val_data = processed_dir / "val_data.npz"

    if val_data.exists():
        metrics = evaluator.evaluate_persistence(str(val_data), "val")
        print_metrics(metrics, "Persistence Validation Metrics")


if __name__ == "__main__":
    main()
