#!/usr/bin/env python
"""
Evaluate trained U-Net model on test set.

This script loads a trained model checkpoint, evaluates it on the test set,
compares with baselines, and generates visualization plots.

Usage:
    python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt
    python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt --split val
    python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt --visualize 5
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths
from seaice_forecast.data_processing.dataset_phase1 import create_dataloaders
from seaice_forecast.evaluation.model_eval import load_model_for_evaluation, ModelEvaluator
from seaice_forecast.evaluation.metrics import print_metrics
from seaice_forecast.utils.visualization import (
    plot_prediction_comparison,
    plot_error_map,
    plot_ice_edge_comparison,
    plot_metric_comparison
)
import argparse
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net model"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: use default config)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: cuda or cpu (default: from config)"
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=3,
        help="Number of best/worst samples to visualize (default: 3)"
    )
    parser.add_argument(
        "--no-baseline-comparison",
        action="store_true",
        help="Skip baseline comparison"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    config = resolve_paths(config)

    # Set device
    device = args.device or config['compute']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    logger.info("="*80)
    logger.info("SEA-ICE CONCENTRATION FORECASTING - MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Evaluating on: {args.split} set")
    logger.info(f"Device: {device}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load processed data
    processed_dir = Path(config['data']['paths']['processed'])
    eval_data = processed_dir / f"{args.split}_data.npz"

    if not eval_data.exists():
        logger.error(f"Evaluation data not found: {eval_data}")
        logger.error("Please run: python scripts/prepare_data.py")
        return

    # Load land-ocean mask
    mask_file = processed_dir / "land_ocean_mask.npy"
    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"Loaded mask: {mask.shape}, ocean pixels: {np.sum(mask)}")
    else:
        logger.warning("No mask found, evaluating on all pixels")
        mask = None

    # Load model
    logger.info("\nLoading model...")
    model, checkpoint_info = load_model_for_evaluation(
        str(checkpoint_path),
        config,
        device
    )

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    dataloaders = create_dataloaders(config, mask)

    if args.split not in dataloaders:
        logger.error(f"Failed to create {args.split} dataloader")
        return

    eval_loader = dataloaders[args.split]
    logger.info(f"{args.split.capitalize()}: {len(eval_loader.dataset)} samples, "
               f"{len(eval_loader)} batches")

    # Create evaluator
    logger.info("\nInitializing evaluator...")
    evaluator = ModelEvaluator(
        model=model,
        config=config,
        mask=mask,
        device=device
    )

    # Evaluate model
    logger.info("\n" + "="*80)
    logger.info("EVALUATING MODEL")
    logger.info("="*80)

    metrics, predictions, targets = evaluator.evaluate_dataset(
        eval_loader,
        args.split
    )

    print_metrics(metrics, f"U-Net - {args.split.upper()} Set")

    # Save results
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"model_results_{args.split}.json"
    evaluator.save_results(
        metrics,
        str(results_file),
        args.split,
        checkpoint_info
    )
    logger.info(f"Results saved to {results_file}")

    # Compare with baselines
    if not args.no_baseline_comparison:
        baseline_file = output_dir / f"baseline_results_{args.split}.json"

        if baseline_file.exists():
            logger.info("\n" + "="*80)
            logger.info("COMPARING WITH BASELINES")
            logger.info("="*80)

            evaluator.compare_with_baselines(
                metrics,
                str(baseline_file),
                args.split
            )

            # Create comparison plot
            plots_dir = Path(config['output']['plots_dir'])
            plots_dir.mkdir(parents=True, exist_ok=True)

            comparison_plot = plots_dir / f"metric_comparison_{args.split}.png"
            try:
                plot_metric_comparison(
                    str(baseline_file),
                    str(results_file),
                    str(comparison_plot)
                )
                logger.info(f"Comparison plot saved to {comparison_plot}")
            except Exception as e:
                logger.warning(f"Failed to create comparison plot: {e}")
        else:
            logger.warning(f"Baseline results not found: {baseline_file}")
            logger.warning("Run: python scripts/evaluate_baselines.py")

    # Visualizations
    if args.visualize > 0:
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)

        plots_dir = Path(config['output']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata for dates
        metadata_file = processed_dir / f"{args.split}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            date_pairs = metadata['date_pairs']
        else:
            date_pairs = None

        # Compute per-sample metrics
        logger.info("Computing per-sample metrics...")
        sample_metrics = evaluator.compute_per_sample_metrics(predictions, targets)

        # Find best and worst samples
        best_indices, worst_indices = evaluator.find_best_worst_samples(
            sample_metrics,
            metric_key='mae',
            n_samples=args.visualize
        )

        # Load inputs for last timestep
        data = np.load(eval_data)
        inputs = data['inputs']

        # Visualize best samples
        logger.info(f"\nVisualizing {args.visualize} best predictions...")
        for i, idx in enumerate(best_indices):
            input_last = inputs[idx, -1, :, :]
            pred = predictions[idx]
            target = targets[idx]

            date_str = f"Date: {date_pairs[idx][1]}" if date_pairs else None
            mae = sample_metrics[idx]['mae']

            save_path = plots_dir / f"best_prediction_{i+1}_mae{mae:.4f}.png"
            plot_prediction_comparison(
                input_last, pred, target, mask,
                save_path=str(save_path),
                title=f"Best Prediction #{i+1} (MAE={mae:.4f})",
                date_str=date_str
            )
            logger.info(f"  Saved: {save_path.name}")

        # Visualize worst samples
        logger.info(f"\nVisualizing {args.visualize} worst predictions...")
        for i, idx in enumerate(worst_indices):
            input_last = inputs[idx, -1, :, :]
            pred = predictions[idx]
            target = targets[idx]

            date_str = f"Date: {date_pairs[idx][1]}" if date_pairs else None
            mae = sample_metrics[idx]['mae']

            save_path = plots_dir / f"worst_prediction_{i+1}_mae{mae:.4f}.png"
            plot_prediction_comparison(
                input_last, pred, target, mask,
                save_path=str(save_path),
                title=f"Worst Prediction #{i+1} (MAE={mae:.4f})",
                date_str=date_str
            )
            logger.info(f"  Saved: {save_path.name}")

        # Create error map
        logger.info("\nCreating spatial error map...")
        error_map_path = plots_dir / f"error_map_{args.split}.png"
        plot_error_map(
            predictions, targets, mask,
            save_path=str(error_map_path),
            title=f'Mean Absolute Error Map - {args.split.upper()} Set'
        )
        logger.info(f"  Saved: {error_map_path.name}")

        # Create ice edge comparison for a sample
        logger.info("\nCreating ice edge comparison...")
        sample_idx = best_indices[0]
        ice_edge_path = plots_dir / f"ice_edge_comparison_{args.split}.png"
        plot_ice_edge_comparison(
            predictions[sample_idx], targets[sample_idx],
            threshold=config['evaluation']['ice_edge']['threshold'],
            mask=mask,
            save_path=str(ice_edge_path),
            title=f"Ice Edge Comparison - {args.split.upper()} Set"
        )
        logger.info(f"  Saved: {ice_edge_path.name}")

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {plots_dir}")
    print("\n" + "-"*80)
    print("KEY METRICS:")
    print("-"*80)
    print(f"MAE:                     {metrics['mae']:.6f}")
    print(f"RMSE:                    {metrics['rmse']:.6f}")
    print(f"Spatial Correlation:     {metrics['spatial_correlation']:.4f}")
    print(f"Ice Edge Displacement:   {metrics['ice_edge_displacement']:.2f} pixels")
    print("="*80)


if __name__ == "__main__":
    main()
