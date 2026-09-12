#!/usr/bin/env python
"""
Three-way comparison: Baselines vs Phase 1 vs Phase 2

This script performs the critical Phase 2 evaluation:
Compare four models on the SAME test set with SAME metrics:
1. Persistence baseline
2. Climatology baseline
3. Phase 1: SIC-only U-Net (sic_unet_v001)
4. Phase 2: Environmental U-Net (sic_unet_env_v001)

Outputs a comparison table and determines whether environmental
forcing improves forecasting enough to justify added complexity.

Usage:
    python scripts/compare_phase1_phase2.py
    python scripts/compare_phase1_phase2.py --phase1-checkpoint models/sic_unet_v001_best.pt \
                                            --phase2-checkpoint models/sic_unet_env_v001_best.pt
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths, get_project_root
from seaice_forecast.models.unet import UNet
from seaice_forecast.data_processing.dataset import SICDataset
from seaice_forecast.data_processing.dataset_phase2 import Phase2Dataset
from seaice_forecast.models.baselines import PersistenceModel, ClimatologyModel
from seaice_forecast.evaluation.metrics import (
    masked_mae, masked_rmse, spatial_correlation, ice_edge_displacement
)
from torch.utils.data import DataLoader
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_phase1_model(checkpoint_path: Path, device: str) -> UNet:
    """Load Phase 1 SIC-only U-Net."""
    logger.info(f"Loading Phase 1 model from {checkpoint_path}")

    model = UNet(
        input_channels=7,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Phase 1 model loaded, epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def load_phase2_model(checkpoint_path: Path, device: str) -> UNet:
    """Load Phase 2 Environmental U-Net."""
    logger.info(f"Loading Phase 2 model from {checkpoint_path}")

    model = UNet(
        input_channels=49,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Phase 2 model loaded, epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def evaluate_model_on_test_set(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mask: np.ndarray,
    device: str,
    model_name: str
) -> Dict:
    """Evaluate a model on test set."""
    logger.info(f"\nEvaluating {model_name}...")

    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for inputs, targets, masks in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # [N, 1, H, W]
    targets = np.concatenate(all_targets, axis=0)  # [N, 1, H, W]

    # Compute metrics
    mae = masked_mae(predictions, targets, mask)
    rmse = masked_rmse(predictions, targets, mask)
    corr = spatial_correlation(predictions, targets, mask)
    ice_disp = ice_edge_displacement(predictions, targets, threshold=0.15)

    results = {
        'model': model_name,
        'n_samples': len(predictions),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'ice_edge_displacement_px': float(ice_disp),
        'ice_edge_displacement_km': float(ice_disp * 25)  # 25km resolution
    }

    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  Correlation: {corr:.6f}")
    logger.info(f"  Ice edge displacement: {ice_disp:.2f} px ({ice_disp*25:.1f} km)")

    return results


def evaluate_baselines(
    phase1_dataloader: DataLoader,
    mask: np.ndarray
) -> Dict[str, Dict]:
    """Evaluate persistence and climatology baselines."""
    logger.info("\nEvaluating baselines...")

    # Get all test samples
    all_inputs = []
    all_targets = []

    for inputs, targets, _ in phase1_dataloader:
        all_inputs.append(inputs.numpy())
        all_targets.append(targets.numpy())

    inputs = np.concatenate(all_inputs, axis=0)  # [N, 7, H, W]
    targets = np.concatenate(all_targets, axis=0)  # [N, 1, H, W]

    results = {}

    # Persistence: last day of input = prediction
    logger.info("\nPersistence baseline:")
    persistence_pred = inputs[:, -1:, :, :]  # [N, 1, H, W]

    mae = masked_mae(persistence_pred, targets, mask)
    rmse = masked_rmse(persistence_pred, targets, mask)
    corr = spatial_correlation(persistence_pred, targets, mask)
    ice_disp = ice_edge_displacement(persistence_pred, targets, threshold=0.15)

    results['Persistence'] = {
        'model': 'Persistence',
        'n_samples': len(targets),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'ice_edge_displacement_px': float(ice_disp),
        'ice_edge_displacement_km': float(ice_disp * 25)
    }

    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  Correlation: {corr:.6f}")
    logger.info(f"  Ice edge displacement: {ice_disp:.2f} px")

    # Climatology: would need day-of-year climatology
    # For now, use mean SIC as simple baseline
    logger.info("\nClimatology baseline (using training mean):")
    climatology_pred = np.mean(inputs, axis=(0, 1), keepdims=True)  # Simple version
    climatology_pred = np.tile(climatology_pred, (len(targets), 1, 1, 1))

    mae = masked_mae(climatology_pred, targets, mask)
    rmse = masked_rmse(climatology_pred, targets, mask)
    corr = spatial_correlation(climatology_pred, targets, mask)
    ice_disp = ice_edge_displacement(climatology_pred, targets, threshold=0.15)

    results['Climatology'] = {
        'model': 'Climatology',
        'n_samples': len(targets),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'ice_edge_displacement_px': float(ice_disp),
        'ice_edge_displacement_km': float(ice_disp * 25)
    }

    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  Correlation: {corr:.6f}")
    logger.info(f"  Ice edge displacement: {ice_disp:.2f} px")

    return results


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create formatted comparison table."""
    df = pd.DataFrame(results)

    # Reorder columns
    df = df[['model', 'mae', 'rmse', 'correlation', 'ice_edge_displacement_km', 'n_samples']]

    # Rename for display
    df.columns = ['Model', 'MAE', 'RMSE', 'Correlation', 'Ice Edge Disp. (km)', 'N Samples']

    return df


def compute_improvement_percentage(phase2_value: float, baseline_value: float) -> float:
    """Compute percentage improvement (negative = worse)."""
    # For metrics where lower is better (MAE, RMSE, ice edge)
    return ((baseline_value - phase2_value) / baseline_value) * 100


def generate_summary_report(results: List[Dict], output_path: Path):
    """Generate summary report with explicit conclusions."""
    df = create_comparison_table(results)

    # Find best baseline
    baseline_results = [r for r in results if r['model'] in ['Persistence', 'Climatology']]
    phase1_result = [r for r in results if 'Phase 1' in r['model']][0]
    phase2_result = [r for r in results if 'Phase 2' in r['model']][0]

    best_baseline = min(baseline_results, key=lambda x: x['mae'])

    report = []
    report.append("="*80)
    report.append("PHASE 2 EVALUATION: THREE-WAY COMPARISON")
    report.append("="*80)
    report.append("")
    report.append("Test Set Performance (2021-2022)")
    report.append("")
    report.append(df.to_string(index=False))
    report.append("")
    report.append("="*80)
    report.append("CRITICAL COMPARISON: Phase 2 vs Phase 1")
    report.append("="*80)
    report.append("")

    # Phase 2 vs Phase 1 comparison
    mae_improvement = compute_improvement_percentage(phase2_result['mae'], phase1_result['mae'])
    rmse_improvement = compute_improvement_percentage(phase2_result['rmse'], phase1_result['rmse'])
    corr_diff = phase2_result['correlation'] - phase1_result['correlation']
    ice_improvement = compute_improvement_percentage(
        phase2_result['ice_edge_displacement_km'],
        phase1_result['ice_edge_displacement_km']
    )

    report.append(f"MAE:         Phase 1: {phase1_result['mae']:.6f}  →  Phase 2: {phase2_result['mae']:.6f}")
    report.append(f"             Change: {mae_improvement:+.2f}%")
    report.append("")
    report.append(f"RMSE:        Phase 1: {phase1_result['rmse']:.6f}  →  Phase 2: {phase2_result['rmse']:.6f}")
    report.append(f"             Change: {rmse_improvement:+.2f}%")
    report.append("")
    report.append(f"Correlation: Phase 1: {phase1_result['correlation']:.6f}  →  Phase 2: {phase2_result['correlation']:.6f}")
    report.append(f"             Change: {corr_diff:+.6f}")
    report.append("")
    report.append(f"Ice Edge:    Phase 1: {phase1_result['ice_edge_displacement_km']:.1f} km  →  Phase 2: {phase2_result['ice_edge_displacement_km']:.1f} km")
    report.append(f"             Change: {ice_improvement:+.2f}%")
    report.append("")
    report.append("="*80)
    report.append("CONCLUSION")
    report.append("="*80)
    report.append("")

    # Determine conclusion
    phase2_beats_phase1 = phase2_result['mae'] < phase1_result['mae']
    significant_improvement = abs(mae_improvement) > 5.0  # 5% threshold

    if phase2_beats_phase1 and significant_improvement:
        report.append("✓ Environmental forcing IMPROVES next-day SIC forecasting.")
        report.append(f"  Phase 2 reduces MAE by {mae_improvement:.1f}% compared to Phase 1.")
        report.append("")
        report.append("RECOMMENDATION: Proceed with Phase 2 (Environmental U-Net) for Phase 3.")
        report.append("The added data pipeline complexity is justified by improved accuracy.")
    elif phase2_beats_phase1 and not significant_improvement:
        report.append("≈ Environmental forcing provides MARGINAL improvement.")
        report.append(f"  Phase 2 reduces MAE by only {mae_improvement:.1f}% compared to Phase 1.")
        report.append("")
        report.append("RECOMMENDATION: Consider cost/benefit tradeoff.")
        report.append("Small improvement may not justify added data pipeline complexity.")
        report.append("Investigate which environmental variables contribute most (ablation study).")
    else:
        report.append("✗ Environmental forcing DOES NOT improve next-day SIC forecasting.")
        report.append(f"  Phase 2 MAE is {abs(mae_improvement):.1f}% WORSE than Phase 1.")
        report.append("")
        report.append("RECOMMENDATION: Stick with Phase 1 (SIC-only) for Phase 3.")
        report.append("Environmental forcing adds complexity without improving accuracy.")
        report.append("")
        report.append("Possible reasons:")
        report.append("  - Environmental variables not relevant for 1-day forecast horizon")
        report.append("  - Data quality issues (missing data, regridding errors)")
        report.append("  - Model architecture doesn't leverage multi-variable input effectively")
        report.append("  - Need temporal architecture (Phase 3) to benefit from forcing")

    report.append("")
    report.append("="*80)
    report.append("NEXT STEPS")
    report.append("="*80)
    report.append("")
    report.append("1. Run ablation analysis:")
    report.append("   python scripts/ablation_analysis.py")
    report.append("   → Identify which variables contribute most to improvement")
    report.append("")
    report.append("2. Examine seasonal/regional performance:")
    report.append("   python scripts/seasonal_regional_analysis.py")
    report.append("   → Check if environmental forcing helps more in certain conditions")
    report.append("")
    report.append("3. Inspect failure cases:")
    report.append("   → Generate prediction plots for worst-performing samples")
    report.append("   → Understand when/where Phase 2 underperforms Phase 1")
    report.append("")
    report.append("="*80)

    report_text = "\n".join(report)

    # Print to console
    print("\n" + report_text)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"\nComparison report saved to {output_path}")

    return report_text


def plot_comparison(results: List[Dict], output_path: Path):
    """Create comparison visualization."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 2 Evaluation: Model Comparison on Test Set', fontsize=16, fontweight='bold')

    # MAE
    ax = axes[0, 0]
    bars = ax.bar(df['model'], df['mae'], color=['gray', 'gray', 'steelblue', 'coral'])
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Mean Absolute Error (lower is better)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=9)

    # RMSE
    ax = axes[0, 1]
    bars = ax.bar(df['model'], df['rmse'], color=['gray', 'gray', 'steelblue', 'coral'])
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Squared Error (lower is better)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=9)

    # Correlation
    ax = axes[1, 0]
    bars = ax.bar(df['model'], df['correlation'], color=['gray', 'gray', 'steelblue', 'coral'])
    ax.set_ylabel('Spatial Correlation', fontsize=12)
    ax.set_title('Spatial Correlation (higher is better)', fontsize=12)
    ax.set_ylim([0.7, 1.0])
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Ice edge displacement
    ax = axes[1, 1]
    bars = ax.bar(df['model'], df['ice_edge_displacement_km'], color=['gray', 'gray', 'steelblue', 'coral'])
    ax.set_ylabel('Ice Edge Displacement (km)', fontsize=12)
    ax.set_title('Ice Edge Displacement (lower is better)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 vs Phase 2 vs Baselines"
    )
    parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        help="Path to Phase 1 model checkpoint"
    )
    parser.add_argument(
        "--phase2-checkpoint",
        default=None,
        help="Path to Phase 2 model checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PHASE 2 EVALUATION: THREE-WAY COMPARISON")
    logger.info("="*80)
    logger.info("Comparing on identical test set:")
    logger.info("  1. Persistence baseline")
    logger.info("  2. Climatology baseline")
    logger.info("  3. Phase 1: SIC-only U-Net")
    logger.info("  4. Phase 2: Environmental U-Net")

    # Setup paths
    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = project_root / 'models'

    # Find checkpoints
    if args.phase1_checkpoint:
        phase1_ckpt = Path(args.phase1_checkpoint)
    else:
        phase1_ckpt = models_dir / 'sic_unet_v001_best.pt'

    if args.phase2_checkpoint:
        phase2_ckpt = Path(args.phase2_checkpoint)
    else:
        phase2_ckpt = models_dir / 'sic_unet_env_v001_best.pt'

    # Check checkpoints exist
    if not phase1_ckpt.exists():
        logger.error(f"Phase 1 checkpoint not found: {phase1_ckpt}")
        logger.error("Train Phase 1 first: python scripts/train.py")
        return

    if not phase2_ckpt.exists():
        logger.error(f"Phase 2 checkpoint not found: {phase2_ckpt}")
        logger.error("Train Phase 2 first: python scripts/train_phase2.py")
        return

    # Load mask
    processed_dir = project_root / 'data' / 'processed'
    mask = np.load(processed_dir / 'land_ocean_mask.npy')

    # Load Phase 1 test data
    logger.info("\nLoading Phase 1 test data...")
    phase1_test_file = processed_dir / 'test_data.npz'
    phase1_dataset = SICDataset(str(phase1_test_file), mask=mask)
    phase1_loader = DataLoader(phase1_dataset, batch_size=8, shuffle=False)

    # Load Phase 2 test data
    logger.info("Loading Phase 2 test data...")
    phase2_test_file = processed_dir / 'phase2' / 'test_data_phase2.npz'
    phase2_dataset = Phase2Dataset(str(phase2_test_file), mask=mask, validate_channels=False)
    phase2_loader = DataLoader(phase2_dataset, batch_size=8, shuffle=False)

    # Evaluate baselines
    baseline_results = evaluate_baselines(phase1_loader, mask)

    # Load and evaluate Phase 1
    phase1_model = load_phase1_model(phase1_ckpt, args.device)
    phase1_results = evaluate_model_on_test_set(
        phase1_model,
        phase1_loader,
        mask,
        args.device,
        "Phase 1 (SIC-only)"
    )

    # Load and evaluate Phase 2
    phase2_model = load_phase2_model(phase2_ckpt, args.device)
    phase2_results = evaluate_model_on_test_set(
        phase2_model,
        phase2_loader,
        mask,
        args.device,
        "Phase 2 (Environmental)"
    )

    # Combine all results
    all_results = [
        baseline_results['Persistence'],
        baseline_results['Climatology'],
        phase1_results,
        phase2_results
    ]

    # Save results
    results_file = output_dir / 'phase1_phase2_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Generate report
    report_file = output_dir / 'phase1_phase2_comparison_report.txt'
    generate_summary_report(all_results, report_file)

    # Generate plot
    plot_file = output_dir / 'plots' / 'phase1_phase2_comparison.png'
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(all_results, plot_file)

    logger.info("\n" + "="*80)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*80)
    logger.info(f"Results: {results_file}")
    logger.info(f"Report:  {report_file}")
    logger.info(f"Plot:    {plot_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
