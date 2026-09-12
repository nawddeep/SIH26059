#!/usr/bin/env python
"""
Evaluate baseline models (persistence and climatology).

This script MUST be run before training the U-Net to establish
the performance bar that the ML model needs to beat.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --split test
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths
from seaice_forecast.models.baselines import (
    PersistenceModel,
    ClimatologyModel,
    fit_climatology_from_dataset
)
from seaice_forecast.evaluation.baselines_eval import BaselineEvaluator
from seaice_forecast.evaluation.metrics import print_metrics
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: use default config)"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate (default: val)"
    )
    parser.add_argument(
        "--save-climatology",
        action="store_true",
        help="Save fitted climatology model"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    config = resolve_paths(config)

    processed_dir = Path(config['data']['paths']['processed'])
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Evaluating on: {args.split} set")

    # Load land-ocean mask
    mask_file = processed_dir / "land_ocean_mask.npy"
    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"Loaded mask: {mask.shape}, ocean pixels: {np.sum(mask)}")
    else:
        logger.warning("No mask found, evaluating on all pixels")
        mask = None

    # Check if data exists
    train_data = processed_dir / "train_data.npz"
    train_meta = processed_dir / "train_metadata.json"
    eval_data = processed_dir / f"{args.split}_data.npz"
    eval_meta = processed_dir / f"{args.split}_metadata.json"

    if not train_data.exists():
        logger.error(f"Training data not found: {train_data}")
        logger.error("Please run: python scripts/prepare_data.py")
        return

    if not eval_data.exists():
        logger.error(f"Evaluation data not found: {eval_data}")
        logger.error("Please run: python scripts/prepare_data.py")
        return

    # Create evaluator
    evaluator = BaselineEvaluator(config, mask)

    # ========================================
    # 1. Evaluate Persistence
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("EVALUATING PERSISTENCE BASELINE")
    logger.info("="*80)

    persistence_metrics = evaluator.evaluate_persistence(
        str(eval_data),
        args.split
    )

    print_metrics(persistence_metrics, f"Persistence - {args.split.upper()} Set")

    # ========================================
    # 2. Fit and Evaluate Climatology
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("FITTING CLIMATOLOGY FROM TRAINING DATA")
    logger.info("="*80)

    # Fit climatology
    smoothing_window = config['baselines']['climatology']['smoothing_window']

    climatology_save_path = None
    if args.save_climatology:
        climatology_save_path = output_dir / "climatology_model.npz"

    climatology = fit_climatology_from_dataset(
        str(train_data),
        str(train_meta),
        smoothing_window=smoothing_window,
        save_path=str(climatology_save_path) if climatology_save_path else None
    )

    logger.info(f"Climatology fitted with {smoothing_window}-day smoothing")

    logger.info("\n" + "="*80)
    logger.info("EVALUATING CLIMATOLOGY BASELINE")
    logger.info("="*80)

    climatology_metrics = evaluator.evaluate_climatology(
        climatology,
        str(eval_data),
        str(eval_meta),
        args.split
    )

    print_metrics(climatology_metrics, f"Climatology - {args.split.upper()} Set")

    # ========================================
    # 3. Compare Baselines
    # ========================================
    evaluator.compare_baselines(
        persistence_metrics,
        climatology_metrics,
        args.split
    )

    # ========================================
    # 4. Save Results
    # ========================================
    results_file = output_dir / f"baseline_results_{args.split}.json"
    evaluator.save_results(
        persistence_metrics,
        climatology_metrics,
        str(results_file),
        args.split
    )

    # ========================================
    # 5. Summary and Next Steps
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline evaluation complete on {args.split} set")
    print(f"\nResults saved to: {results_file}")

    if args.save_climatology and climatology_save_path:
        print(f"Climatology model saved to: {climatology_save_path}")

    print("\n" + "-"*80)
    print("PERFORMANCE BAR FOR U-NET:")
    print("-"*80)
    print(f"The U-Net model MUST beat both baselines to be considered successful.")
    print(f"\nTarget metrics to beat:")
    print(f"  MAE:  < {min(persistence_metrics['mae'], climatology_metrics['mae']):.6f}")
    print(f"  RMSE: < {min(persistence_metrics['rmse'], climatology_metrics['rmse']):.6f}")
    print(f"  Correlation: > {max(persistence_metrics['spatial_correlation'], climatology_metrics['spatial_correlation']):.4f}")

    print("\n" + "-"*80)
    print("NEXT STEP:")
    print("-"*80)
    print("Train the U-Net model:")
    print("  python scripts/train.py")
    print("="*80)


if __name__ == "__main__":
    main()
