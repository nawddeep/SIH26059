"""
Comprehensive evaluation for trained U-Net model.

Evaluates on test set and compares against baselines.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime
import logging

from .metrics import compute_all_metrics, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained U-Net model comprehensively."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        mask: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained U-Net model
            config: Configuration dictionary
            mask: Land-ocean mask
            device: Device for evaluation
        """
        self.model = model
        self.config = config
        self.mask = mask
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.ice_edge_threshold = config['evaluation']['ice_edge']['threshold']

    def predict_batch(
        self,
        inputs: torch.Tensor
    ) -> np.ndarray:
        """
        Make predictions for a batch.

        Args:
            inputs: Input tensor [B, T, H, W]

        Returns:
            Predictions as numpy array [B, 1, H, W]
        """
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs.cpu().numpy()

    def evaluate_dataset(
        self,
        dataloader,
        split_name: str = "test"
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of split being evaluated

        Returns:
            Tuple of (metrics dict, all predictions, all targets)
        """
        logger.info(f"Evaluating model on {split_name} set...")

        all_predictions = []
        all_targets = []

        # Collect all predictions and targets
        for inputs, targets, masks in dataloader:
            predictions = self.predict_batch(inputs)

            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        metrics = compute_all_metrics(
            all_predictions,
            all_targets,
            self.mask,
            self.ice_edge_threshold
        )

        logger.info(f"Model {split_name} - MAE: {metrics['mae']:.6f}, "
                   f"RMSE: {metrics['rmse']:.6f}, "
                   f"Correlation: {metrics['spatial_correlation']:.4f}")

        return metrics, all_predictions, all_targets

    def compare_with_baselines(
        self,
        model_metrics: Dict[str, float],
        baseline_results_file: str,
        split_name: str = "test"
    ):
        """
        Compare model metrics with baseline results.

        Args:
            model_metrics: Model evaluation metrics
            baseline_results_file: Path to baseline results JSON
            split_name: Name of split
        """
        # Load baseline results
        with open(baseline_results_file, 'r') as f:
            baseline_results = json.load(f)

        persistence_metrics = baseline_results['baselines']['persistence']
        climatology_metrics = baseline_results['baselines']['climatology']

        print(f"\n{'='*90}")
        print(f"MODEL VS BASELINES COMPARISON - {split_name.upper()} SET")
        print(f"{'='*90}")

        metric_names = {
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Squared Error',
            'spatial_correlation': 'Spatial Correlation',
            'ice_edge_displacement': 'Ice Edge Displacement (px)'
        }

        print(f"\n{'Metric':<35s} {'Persistence':>15s} {'Climatology':>15s} {'U-Net':>15s} {'Best':>12s}")
        print(f"{'-'*90}")

        for key in ['mae', 'rmse', 'spatial_correlation', 'ice_edge_displacement']:
            if key not in model_metrics:
                continue

            pers_val = persistence_metrics[key]
            clim_val = climatology_metrics[key]
            model_val = model_metrics[key]

            # Determine best (lower is better except for correlation)
            if key == 'spatial_correlation':
                best_val = max(pers_val, clim_val, model_val)
                if model_val == best_val:
                    best = 'U-Net ✓'
                elif clim_val == best_val:
                    best = 'Clim'
                else:
                    best = 'Pers'
            else:
                best_val = min(pers_val, clim_val, model_val)
                if model_val == best_val:
                    best = 'U-Net ✓'
                elif clim_val == best_val:
                    best = 'Clim'
                else:
                    best = 'Pers'

            if not np.isnan(pers_val) and not np.isnan(clim_val) and not np.isnan(model_val):
                if key == 'spatial_correlation':
                    print(f"{metric_names[key]:<35s} {pers_val:>15.4f} {clim_val:>15.4f} {model_val:>15.4f} {best:>12s}")
                elif key == 'ice_edge_displacement':
                    print(f"{metric_names[key]:<35s} {pers_val:>15.2f} {clim_val:>15.2f} {model_val:>15.2f} {best:>12s}")
                else:
                    print(f"{metric_names[key]:<35s} {pers_val:>15.6f} {clim_val:>15.6f} {model_val:>15.6f} {best:>12s}")

        print(f"{'='*90}")

        # Compute improvement percentages
        print(f"\nIMPROVEMENT OVER BASELINES:")
        print(f"{'-'*90}")

        for key in ['mae', 'rmse']:
            if key in model_metrics:
                pers_val = persistence_metrics[key]
                clim_val = climatology_metrics[key]
                model_val = model_metrics[key]

                # Best baseline to beat
                best_baseline = min(pers_val, clim_val)
                improvement = ((best_baseline - model_val) / best_baseline) * 100

                if improvement > 0:
                    print(f"{metric_names[key]:<35s}: {improvement:>6.2f}% better than best baseline")
                else:
                    print(f"{metric_names[key]:<35s}: {abs(improvement):>6.2f}% worse than best baseline ⚠")

        # Check correlation improvement
        if 'spatial_correlation' in model_metrics:
            pers_val = persistence_metrics['spatial_correlation']
            clim_val = climatology_metrics['spatial_correlation']
            model_val = model_metrics['spatial_correlation']

            best_baseline = max(pers_val, clim_val)
            improvement = ((model_val - best_baseline) / abs(best_baseline)) * 100 if best_baseline != 0 else 0

            if improvement > 0:
                print(f"{'Spatial Correlation':<35s}: {improvement:>6.2f}% better than best baseline")
            else:
                print(f"{'Spatial Correlation':<35s}: {abs(improvement):>6.2f}% worse than best baseline ⚠")

        print(f"{'='*90}\n")

    def compute_per_sample_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each sample individually.

        Args:
            predictions: All predictions [N, 1, H, W]
            targets: All targets [N, 1, H, W]

        Returns:
            List of metric dictionaries
        """
        sample_metrics = []

        for i in range(len(predictions)):
            metrics = compute_all_metrics(
                predictions[i:i+1],
                targets[i:i+1],
                self.mask,
                self.ice_edge_threshold
            )
            sample_metrics.append(metrics)

        return sample_metrics

    def find_best_worst_samples(
        self,
        sample_metrics: List[Dict[str, float]],
        metric_key: str = 'mae',
        n_samples: int = 3
    ) -> Tuple[List[int], List[int]]:
        """
        Find indices of best and worst predictions.

        Args:
            sample_metrics: List of metric dictionaries
            metric_key: Metric to use for ranking
            n_samples: Number of best/worst to return

        Returns:
            Tuple of (best_indices, worst_indices)
        """
        values = [m[metric_key] for m in sample_metrics]

        # Sort by metric (ascending for MAE/RMSE, descending for correlation)
        if metric_key == 'spatial_correlation':
            sorted_indices = np.argsort(values)[::-1]  # Descending
        else:
            sorted_indices = np.argsort(values)  # Ascending

        best_indices = sorted_indices[:n_samples].tolist()
        worst_indices = sorted_indices[-n_samples:][::-1].tolist()

        return best_indices, worst_indices

    def save_results(
        self,
        metrics: Dict[str, float],
        output_file: str,
        split_name: str = "test",
        checkpoint_info: Optional[Dict] = None
    ):
        """
        Save evaluation results to JSON.

        Args:
            metrics: Evaluation metrics
            output_file: Path to output JSON file
            split_name: Name of split
            checkpoint_info: Optional checkpoint metadata
        """
        results = {
            'model': self.config['model']['name'],
            'split': split_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': {
                'ice_edge_threshold': self.ice_edge_threshold,
                'mask_used': self.mask is not None,
                'device': self.device
            }
        }

        if checkpoint_info:
            clean_checkpoint = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in checkpoint_info.items()
                if k not in ['model_state_dict', 'optimizer_state_dict']
            }
            results['checkpoint'] = clean_checkpoint

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def load_model_for_evaluation(
    checkpoint_path: str,
    config: dict,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    from ..models.unet import create_unet_from_config

    # Create model
    model = create_unet_from_config(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")

    return model, checkpoint


def main():
    """Test evaluator."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

    from seaice_forecast.config import load_config, resolve_paths
    from seaice_forecast.models.unet import create_unet_from_config

    config = load_config()
    config = resolve_paths(config)

    # Create dummy model
    model = create_unet_from_config(config)

    # Load mask
    mask_file = Path(config['data']['paths']['processed']) / "land_ocean_mask.npy"
    if mask_file.exists():
        mask = np.load(mask_file)
        print(f"Loaded mask: {mask.shape}")
    else:
        mask = None

    evaluator = ModelEvaluator(model, config, mask, device='cpu')
    print("Evaluator initialized successfully")


if __name__ == "__main__":
    main()
