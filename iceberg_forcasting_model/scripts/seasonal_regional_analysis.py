#!/usr/bin/env python
"""
Phase 2: Seasonal and Regional Performance Breakdown

Analyzes where environmental forcing helps most:
- By austral season (summer, autumn, winter, spring)
- By Antarctic sector (Weddell, Ross, Amundsen-Bellingshausen, Indian, Pacific)

Identifies if Phase 2 improves performance uniformly or only in specific
conditions, informing when environmental forcing is most valuable.

Usage:
    python scripts/seasonal_regional_analysis.py \
        --phase1-checkpoint models/sic_unet_v001_best.pt \
        --phase2-checkpoint models/sic_unet_env_v001_best.pt
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import get_project_root
from seaice_forecast.models.unet import UNet
from seaice_forecast.data_processing.dataset import SICDataset
from seaice_forecast.data_processing.dataset_phase2 import Phase2Dataset
from seaice_forecast.evaluation.metrics import masked_mae, masked_rmse
from torch.utils.data import DataLoader
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def get_austral_season(date_str: str) -> str:
    """
    Get austral season from date string.

    Southern Hemisphere seasons:
    - Summer: Dec, Jan, Feb
    - Autumn: Mar, Apr, May
    - Winter: Jun, Jul, Aug
    - Spring: Sep, Oct, Nov
    """
    date = datetime.fromisoformat(date_str)
    month = date.month

    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'


def define_antarctic_sectors() -> Dict[str, Tuple[slice, slice]]:
    """
    Define Antarctic sectors as grid indices.

    This is a simplified regional definition based on grid position.
    For a real implementation, would use longitude ranges and proper
    polar stereographic coordinate mapping.

    Returns:
        Dictionary mapping sector names to (y_slice, x_slice) tuples
    """
    H, W = 316, 332

    # Simplified sector definitions (adjust based on actual grid)
    sectors = {
        'Weddell': (slice(0, H//2), slice(0, W//3)),  # Top-left
        'Indian': (slice(0, H//2), slice(W//3, 2*W//3)),  # Top-middle
        'Pacific': (slice(0, H//2), slice(2*W//3, W)),  # Top-right
        'Ross': (slice(H//2, H), slice(W//3, 2*W//3)),  # Bottom-middle
        'Amundsen': (slice(H//2, H), slice(0, W//3)),  # Bottom-left
    }

    return sectors


def evaluate_by_season(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: List[str],
    mask: np.ndarray
) -> Dict[str, Dict]:
    """
    Evaluate performance broken down by season.

    Args:
        predictions: Model predictions [N, 1, H, W]
        targets: Ground truth [N, 1, H, W]
        dates: List of date strings
        mask: Land/ocean mask

    Returns:
        Dictionary of metrics per season
    """
    logger.info("\nComputing seasonal breakdown...")

    # Assign seasons to each sample
    seasons = [get_austral_season(date) for date in dates]
    unique_seasons = ['Summer', 'Autumn', 'Winter', 'Spring']

    results = {}

    for season in unique_seasons:
        # Get indices for this season
        indices = [i for i, s in enumerate(seasons) if s == season]

        if len(indices) == 0:
            logger.warning(f"No samples for {season}")
            continue

        # Select samples for this season
        season_preds = predictions[indices]
        season_targets = targets[indices]

        # Compute metrics
        mae = masked_mae(season_preds, season_targets, mask)
        rmse = masked_rmse(season_preds, season_targets, mask)

        results[season] = {
            'season': season,
            'n_samples': len(indices),
            'mae': float(mae),
            'rmse': float(rmse)
        }

        logger.info(f"  {season:8s}: N={len(indices):4d}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return results


def evaluate_by_region(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    sectors: Dict[str, Tuple[slice, slice]]
) -> Dict[str, Dict]:
    """
    Evaluate performance broken down by Antarctic sector.

    Args:
        predictions: Model predictions [N, 1, H, W]
        targets: Ground truth [N, 1, H, W]
        mask: Land/ocean mask
        sectors: Sector definitions

    Returns:
        Dictionary of metrics per sector
    """
    logger.info("\nComputing regional breakdown...")

    results = {}

    for sector_name, (y_slice, x_slice) in sectors.items():
        # Create sector mask
        sector_mask = np.zeros_like(mask)
        sector_mask[y_slice, x_slice] = mask[y_slice, x_slice]

        # Check if sector has ocean pixels
        if sector_mask.sum() == 0:
            logger.warning(f"Sector {sector_name} has no ocean pixels")
            continue

        # Compute metrics for this sector
        mae = masked_mae(predictions, targets, sector_mask)
        rmse = masked_rmse(predictions, targets, sector_mask)

        results[sector_name] = {
            'sector': sector_name,
            'n_ocean_pixels': int(sector_mask.sum()),
            'mae': float(mae),
            'rmse': float(rmse)
        }

        logger.info(f"  {sector_name:12s}: Pixels={int(sector_mask.sum()):5d}, "
                   f"MAE={mae:.6f}, RMSE={rmse:.6f}")

    return results


def plot_seasonal_comparison(
    phase1_results: Dict,
    phase2_results: Dict,
    output_path: Path
):
    """Plot seasonal performance comparison."""
    seasons = ['Summer', 'Autumn', 'Winter', 'Spring']

    phase1_mae = [phase1_results[s]['mae'] for s in seasons if s in phase1_results]
    phase2_mae = [phase2_results[s]['mae'] for s in seasons if s in phase2_results]
    valid_seasons = [s for s in seasons if s in phase1_results and s in phase2_results]

    x = np.arange(len(valid_seasons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, phase1_mae, width, label='Phase 1 (SIC-only)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, phase2_mae, width, label='Phase 2 (Environmental)',
                   color='coral', alpha=0.8)

    ax.set_ylabel('MAE', fontsize=12)
    ax.set_xlabel('Austral Season', fontsize=12)
    ax.set_title('Seasonal Performance Comparison: Phase 1 vs Phase 2',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_seasons)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.5f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Seasonal comparison plot saved to {output_path}")
    plt.close()


def plot_regional_comparison(
    phase1_results: Dict,
    phase2_results: Dict,
    output_path: Path
):
    """Plot regional performance comparison."""
    sectors = list(phase1_results.keys())

    phase1_mae = [phase1_results[s]['mae'] for s in sectors]
    phase2_mae = [phase2_results[s]['mae'] for s in sectors]

    x = np.arange(len(sectors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, phase1_mae, width, label='Phase 1 (SIC-only)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, phase2_mae, width, label='Phase 2 (Environmental)',
                   color='coral', alpha=0.8)

    ax.set_ylabel('MAE', fontsize=12)
    ax.set_xlabel('Antarctic Sector', fontsize=12)
    ax.set_title('Regional Performance Comparison: Phase 1 vs Phase 2',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.5f}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Regional comparison plot saved to {output_path}")
    plt.close()


def generate_analysis_report(
    phase1_seasonal: Dict,
    phase2_seasonal: Dict,
    phase1_regional: Dict,
    phase2_regional: Dict,
    output_path: Path
):
    """Generate text report of seasonal/regional analysis."""
    report = []
    report.append("="*80)
    report.append("PHASE 2: SEASONAL AND REGIONAL PERFORMANCE ANALYSIS")
    report.append("="*80)
    report.append("")

    # Seasonal analysis
    report.append("SEASONAL BREAKDOWN")
    report.append("-"*80)
    report.append("")

    seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
    report.append(f"{'Season':<12} {'Phase 1 MAE':<15} {'Phase 2 MAE':<15} {'Improvement':<15}")
    report.append("-"*60)

    for season in seasons:
        if season in phase1_seasonal and season in phase2_seasonal:
            p1_mae = phase1_seasonal[season]['mae']
            p2_mae = phase2_seasonal[season]['mae']
            improvement = ((p1_mae - p2_mae) / p1_mae) * 100
            report.append(f"{season:<12} {p1_mae:<15.6f} {p2_mae:<15.6f} {improvement:>+.2f}%")

    report.append("")
    report.append("Key finding (seasonal):")

    # Find season with biggest improvement
    improvements = {}
    for season in seasons:
        if season in phase1_seasonal and season in phase2_seasonal:
            p1_mae = phase1_seasonal[season]['mae']
            p2_mae = phase2_seasonal[season]['mae']
            improvements[season] = ((p1_mae - p2_mae) / p1_mae) * 100

    if improvements:
        best_season = max(improvements, key=improvements.get)
        worst_season = min(improvements, key=improvements.get)

        report.append(f"  - Environmental forcing most helpful in {best_season} "
                     f"({improvements[best_season]:+.1f}% improvement)")
        report.append(f"  - Environmental forcing least helpful in {worst_season} "
                     f"({improvements[worst_season]:+.1f}% improvement)")

    report.append("")
    report.append("="*80)

    # Regional analysis
    report.append("REGIONAL BREAKDOWN")
    report.append("-"*80)
    report.append("")

    report.append(f"{'Sector':<18} {'Phase 1 MAE':<15} {'Phase 2 MAE':<15} {'Improvement':<15}")
    report.append("-"*65)

    for sector in phase1_regional.keys():
        if sector in phase2_regional:
            p1_mae = phase1_regional[sector]['mae']
            p2_mae = phase2_regional[sector]['mae']
            improvement = ((p1_mae - p2_mae) / p1_mae) * 100
            report.append(f"{sector:<18} {p1_mae:<15.6f} {p2_mae:<15.6f} {improvement:>+.2f}%")

    report.append("")
    report.append("Key finding (regional):")

    # Find region with biggest improvement
    regional_improvements = {}
    for sector in phase1_regional.keys():
        if sector in phase2_regional:
            p1_mae = phase1_regional[sector]['mae']
            p2_mae = phase2_regional[sector]['mae']
            regional_improvements[sector] = ((p1_mae - p2_mae) / p1_mae) * 100

    if regional_improvements:
        best_region = max(regional_improvements, key=regional_improvements.get)
        worst_region = min(regional_improvements, key=regional_improvements.get)

        report.append(f"  - Environmental forcing most helpful in {best_region} "
                     f"({regional_improvements[best_region]:+.1f}% improvement)")
        report.append(f"  - Environmental forcing least helpful in {worst_region} "
                     f"({regional_improvements[worst_region]:+.1f}% improvement)")

    report.append("")
    report.append("="*80)

    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Seasonal and Regional Performance Analysis"
    )
    parser.add_argument(
        "--phase1-checkpoint",
        default="models/sic_unet_v001_best.pt",
        help="Path to Phase 1 model checkpoint"
    )
    parser.add_argument(
        "--phase2-checkpoint",
        default="models/sic_unet_env_v001_best.pt",
        help="Path to Phase 2 model checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SEASONAL AND REGIONAL PERFORMANCE ANALYSIS")
    logger.info("="*80)

    project_root = get_project_root()
    output_dir = project_root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = project_root / 'data' / 'processed'

    # Load mask
    mask = np.load(processed_dir / 'land_ocean_mask.npy')

    # Define sectors
    sectors = define_antarctic_sectors()
    logger.info(f"Defined {len(sectors)} Antarctic sectors: {list(sectors.keys())}")

    # Load test metadata for dates
    test_meta_file = processed_dir / 'phase2' / 'test_metadata.json'
    if not test_meta_file.exists():
        test_meta_file = processed_dir / 'test_metadata.json'

    with open(test_meta_file, 'r') as f:
        test_meta = json.load(f)

    date_pairs = test_meta.get('date_pairs', [])
    target_dates = [pair[1] for pair in date_pairs]

    # Load Phase 1 test dataset and model
    phase1_test_file = processed_dir / 'test_data.npz'
    phase1_dataset = SICDataset(str(phase1_test_file), mask=mask)
    phase1_loader = DataLoader(phase1_dataset, batch_size=8, shuffle=False)

    phase1_model = UNet(
        input_channels=7,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )
    ckpt1 = torch.load(args.phase1_checkpoint, map_location=args.device)
    phase1_model.load_state_dict(ckpt1['model_state_dict'])
    phase1_model.to(args.device)
    phase1_model.eval()

    # Load Phase 2 test dataset and model
    phase2_test_file = processed_dir / 'phase2' / 'test_data_phase2.npz'
    phase2_dataset = Phase2Dataset(str(phase2_test_file), mask=mask, validate_channels=False)
    phase2_loader = DataLoader(phase2_dataset, batch_size=8, shuffle=False)

    phase2_model = UNet(
        input_channels=49,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )
    ckpt2 = torch.load(args.phase2_checkpoint, map_location=args.device)
    phase2_model.load_state_dict(ckpt2['model_state_dict'])
    phase2_model.to(args.device)
    phase2_model.eval()

    # Generate predictions
    logger.info("\nGenerating predictions for Phase 1...")
    p1_preds, p1_targets = [], []
    with torch.no_grad():
        for x, y, _ in phase1_loader:
            x, y = x.to(args.device), y.to(args.device)
            p1_preds.append(phase1_model(x).cpu().numpy())
            p1_targets.append(y.cpu().numpy())
    p1_preds = np.concatenate(p1_preds, axis=0)
    p1_targets = np.concatenate(p1_targets, axis=0)

    logger.info("Generating predictions for Phase 2...")
    p2_preds, p2_targets = [], []
    with torch.no_grad():
        for x, y, _ in phase2_loader:
            x, y = x.to(args.device), y.to(args.device)
            p2_preds.append(phase2_model(x).cpu().numpy())
            p2_targets.append(y.cpu().numpy())
    p2_preds = np.concatenate(p2_preds, axis=0)
    p2_targets = np.concatenate(p2_targets, axis=0)

    # 1. Seasonal breakdown
    p1_seasonal = evaluate_by_season(p1_preds, p1_targets, target_dates, mask)
    p2_seasonal = evaluate_by_season(p2_preds, p2_targets, target_dates, mask)
    plot_seasonal_comparison(p1_seasonal, p2_seasonal, plots_dir / 'seasonal_comparison.png')

    # 2. Regional breakdown
    p1_regional = evaluate_by_region(p1_preds, p1_targets, mask, sectors)
    p2_regional = evaluate_by_region(p2_preds, p2_targets, mask, sectors)
    plot_regional_comparison(p1_regional, p2_regional, plots_dir / 'regional_comparison.png')

    # Save combined seasonal/regional results
    results = {
        'seasonal': {'phase1': p1_seasonal, 'phase2': p2_seasonal},
        'regional': {'phase1': p1_regional, 'phase2': p2_regional}
    }
    with open(output_dir / 'seasonal_regional_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate written report
    generate_analysis_report(p1_seasonal, p2_seasonal, p1_regional, p2_regional, output_dir / 'seasonal_regional_report.txt')

    logger.info("\n" + "="*80)
    logger.info("SEASONAL AND REGIONAL ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
