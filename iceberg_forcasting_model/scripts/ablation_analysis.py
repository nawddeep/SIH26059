#!/usr/bin/env python
"""
Phase 2 Ablation Analysis: Feature Importance for Environmental Variables

Determines which environmental variables contribute most to forecasting accuracy.

Ablation strategies:
1. SIC-only (Phase 1 equivalent)
2. SIC + Wind (U/V)
3. SIC + Temperature
4. SIC + SST
5. SIC + Currents (U/V)
6. SIC + All (Phase 2 full)

Evaluates each configuration on the test set and reports which variables
provide the most value, informing whether a subset of variables would
be sufficient.

Usage:
    python scripts/ablation_analysis.py --checkpoint models/sic_unet_env_v001_best.pt
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

from seaice_forecast.config import load_config, get_project_root
from seaice_forecast.models.unet import UNet
from seaice_forecast.data_processing.dataset_phase2 import Phase2Dataset, AblationDataset
from seaice_forecast.evaluation.metrics import masked_mae, masked_rmse, spatial_correlation
from torch.utils.data import DataLoader
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class AblationStudy:
    """Conduct ablation study on Phase 2 model."""

    def __init__(
        self,
        model: torch.nn.Module,
        base_dataset: Phase2Dataset,
        mask: np.ndarray,
        device: str,
        batch_size: int = 8
    ):
        """
        Initialize ablation study.

        Args:
            model: Trained Phase 2 model
            base_dataset: Phase 2 test dataset
            mask: Land/ocean mask
            device: Device to use
            batch_size: Batch size for evaluation
        """
        self.model = model
        self.base_dataset = base_dataset
        self.mask = mask
        self.device = device
        self.batch_size = batch_size

        # Define ablation configurations
        self.configurations = {
            'SIC only': ['sic'],
            'SIC + Wind': ['sic', 'wind_u', 'wind_v'],
            'SIC + Air Temp': ['sic', 'air_temp'],
            'SIC + SST': ['sic', 'sst'],
            'SIC + Currents': ['sic', 'current_u', 'current_v'],
            'SIC + Wind + Temp': ['sic', 'wind_u', 'wind_v', 'air_temp'],
            'SIC + Wind + SST': ['sic', 'wind_u', 'wind_v', 'sst'],
            'SIC + Wind + Currents': ['sic', 'wind_u', 'wind_v', 'current_u', 'current_v'],
            'All variables': ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v']
        }

    def evaluate_configuration(
        self,
        config_name: str,
        enabled_variables: List[str]
    ) -> Dict:
        """
        Evaluate a specific variable configuration.

        Args:
            config_name: Name of configuration
            enabled_variables: List of enabled variables

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating: {config_name}")
        logger.info(f"  Variables: {enabled_variables}")

        # Create ablation dataset
        ablation_dataset = AblationDataset(
            self.base_dataset,
            enabled_variables,
            fill_disabled=0.0
        )

        # Create dataloader
        dataloader = DataLoader(
            ablation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Evaluate
        all_predictions = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for inputs, targets, masks in dataloader:
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        mae = masked_mae(predictions, targets, self.mask)
        rmse = masked_rmse(predictions, targets, self.mask)
        corr = spatial_correlation(predictions, targets, self.mask)

        results = {
            'configuration': config_name,
            'n_variables': len(enabled_variables),
            'variables': enabled_variables,
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(corr)
        }

        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Correlation: {corr:.6f}")

        return results

    def run_full_ablation(self) -> List[Dict]:
        """
        Run complete ablation study across all configurations.

        Returns:
            List of results for each configuration
        """
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY: ENVIRONMENTAL VARIABLE IMPORTANCE")
        logger.info("="*80)

        results = []

        for config_name, enabled_vars in self.configurations.items():
            result = self.evaluate_configuration(config_name, enabled_vars)
            results.append(result)

        return results

    def run_leave_one_out(self) -> List[Dict]:
        """
        Leave-one-out ablation: remove each variable individually.

        Returns:
            List of results with each variable removed
        """
        logger.info("\n" + "="*80)
        logger.info("LEAVE-ONE-OUT ABLATION")
        logger.info("="*80)

        all_vars = ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v']

        results = []

        for var_to_remove in all_vars:
            if var_to_remove == 'sic':
                continue  # Can't remove SIC

            enabled_vars = [v for v in all_vars if v != var_to_remove]
            config_name = f"All except {var_to_remove}"

            result = self.evaluate_configuration(config_name, enabled_vars)
            result['removed_variable'] = var_to_remove
            results.append(result)

        return results


def analyze_results(results: List[Dict], output_dir: Path):
    """Analyze and visualize ablation results."""
    df = pd.DataFrame(results)

    # Sort by MAE
    df = df.sort_values('mae')

    logger.info("\n" + "="*80)
    logger.info("ABLATION RESULTS (sorted by MAE)")
    logger.info("="*80)
    print(df[['configuration', 'n_variables', 'mae', 'rmse', 'correlation']].to_string(index=False))

    # Find best and worst
    best = df.iloc[0]
    worst = df.iloc[-1]
    sic_only = df[df['configuration'] == 'SIC only'].iloc[0]
    all_vars = df[df['configuration'] == 'All variables'].iloc[0]

    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS")
    logger.info("="*80)
    logger.info(f"\nBest configuration: {best['configuration']}")
    logger.info(f"  MAE: {best['mae']:.6f}")
    logger.info(f"  Variables: {best['n_variables']}")

    logger.info(f"\nWorst configuration: {worst['configuration']}")
    logger.info(f"  MAE: {worst['mae']:.6f}")

    # Compare to SIC-only
    improvement_vs_sic = ((sic_only['mae'] - best['mae']) / sic_only['mae']) * 100
    logger.info(f"\nBest vs SIC-only:")
    logger.info(f"  Improvement: {improvement_vs_sic:.2f}%")

    # Check if all variables is best
    if best['configuration'] == 'All variables':
        logger.info("\n✓ All environmental variables together provide best performance")
        logger.info("  Recommendation: Use full Phase 2 model")
    else:
        all_vs_best = ((best['mae'] - all_vars['mae']) / best['mae']) * 100
        logger.info(f"\n✗ Subset of variables performs better than all variables")
        logger.info(f"  '{best['configuration']}' is {abs(all_vs_best):.2f}% better than 'All variables'")
        logger.info(f"  Recommendation: Consider using {best['configuration']} for efficiency")

    # Variable importance ranking
    logger.info("\n" + "-"*80)
    logger.info("Variable Group Importance (by MAE when added to SIC):")
    logger.info("-"*80)

    single_var_configs = [
        ('Wind', 'SIC + Wind'),
        ('Air Temp', 'SIC + Air Temp'),
        ('SST', 'SIC + SST'),
        ('Currents', 'SIC + Currents')
    ]

    improvements = []
    for var_name, config_name in single_var_configs:
        config_result = df[df['configuration'] == config_name]
        if not config_result.empty:
            mae = config_result.iloc[0]['mae']
            improvement = ((sic_only['mae'] - mae) / sic_only['mae']) * 100
            improvements.append((var_name, improvement, mae))

    improvements.sort(key=lambda x: x[1], reverse=True)

    for i, (var_name, improvement, mae) in enumerate(improvements, 1):
        logger.info(f"{i}. {var_name:12s}: {improvement:+.2f}% (MAE: {mae:.6f})")

    return df


def plot_ablation_results(results: List[Dict], output_dir: Path):
    """Create visualizations of ablation results."""
    df = pd.DataFrame(results)
    df = df.sort_values('mae')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Phase 2 Ablation Study: Variable Importance', fontsize=16, fontweight='bold')

    # MAE comparison
    ax = axes[0]
    colors = ['coral' if 'All' in name else 'steelblue' if 'SIC only' in name else 'lightblue'
              for name in df['configuration']]
    bars = ax.barh(range(len(df)), df['mae'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['configuration'], fontsize=9)
    ax.set_xlabel('MAE (lower is better)', fontsize=12)
    ax.set_title('Mean Absolute Error by Configuration', fontsize=12)
    ax.invert_yaxis()

    # Add values
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['mae'], i, f" {row['mae']:.6f}", va='center', fontsize=8)

    # RMSE comparison
    ax = axes[1]
    bars = ax.barh(range(len(df)), df['rmse'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['configuration'], fontsize=9)
    ax.set_xlabel('RMSE (lower is better)', fontsize=12)
    ax.set_title('Root Mean Squared Error', fontsize=12)
    ax.invert_yaxis()

    # Correlation
    ax = axes[2]
    bars = ax.barh(range(len(df)), df['correlation'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['configuration'], fontsize=9)
    ax.set_xlabel('Correlation (higher is better)', fontsize=12)
    ax.set_title('Spatial Correlation', fontsize=12)
    ax.set_xlim([0.7, 1.0])
    ax.invert_yaxis()

    plt.tight_layout()

    plot_path = output_dir / 'plots' / 'ablation_analysis.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nAblation plot saved to {plot_path}")
    plt.close()

    # Create improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sic_only_mae = df[df['configuration'] == 'SIC only']['mae'].values[0]
    improvements = ((sic_only_mae - df['mae']) / sic_only_mae) * 100

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(range(len(df)), improvements, color=colors, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['configuration'], fontsize=9)
    ax.set_xlabel('Improvement vs SIC-only (%)', fontsize=12)
    ax.set_title('MAE Improvement Relative to SIC-only Baseline', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add values
    for i, imp in enumerate(improvements):
        ax.text(imp, i, f" {imp:+.1f}%", va='center', fontsize=8)

    plt.tight_layout()

    improvement_plot = output_dir / 'plots' / 'ablation_improvement.png'
    plt.savefig(improvement_plot, dpi=300, bbox_inches='tight')
    logger.info(f"Improvement plot saved to {improvement_plot}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Ablation Analysis"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained Phase 2 model checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--leave-one-out",
        action='store_true',
        help="Also run leave-one-out ablation"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PHASE 2 ABLATION ANALYSIS")
    logger.info("="*80)
    logger.info("Objective: Identify which environmental variables contribute most")
    logger.info("Method: Evaluate model with different variable subsets")

    # Setup paths
    project_root = get_project_root()
    output_dir = project_root / 'output'

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"\nLoading model from {checkpoint_path}")

    model = UNet(
        input_channels=49,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    logger.info(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

    # Load test dataset
    processed_dir = project_root / 'data' / 'processed'
    phase2_dir = processed_dir / 'phase2'
    test_file = phase2_dir / 'test_data_phase2.npz'

    if not test_file.exists():
        logger.error(f"Test data not found: {test_file}")
        logger.error("Run prepare_environmental_data.py first")
        return

    # Load mask
    mask = np.load(processed_dir / 'land_ocean_mask.npy')

    # Load dataset
    logger.info(f"\nLoading test dataset from {test_file}")
    test_dataset = Phase2Dataset(str(test_file), mask=mask, validate_channels=False)
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Create ablation study
    study = AblationStudy(
        model=model,
        base_dataset=test_dataset,
        mask=mask,
        device=args.device,
        batch_size=args.batch_size
    )

    # Run main ablation
    results = study.run_full_ablation()

    # Run leave-one-out if requested
    if args.leave_one_out:
        loo_results = study.run_leave_one_out()

        logger.info("\n" + "="*80)
        logger.info("LEAVE-ONE-OUT RESULTS")
        logger.info("="*80)
        logger.info("Impact of removing each variable (MAE change):\n")

        all_vars_mae = [r['mae'] for r in results if r['configuration'] == 'All variables'][0]

        for result in loo_results:
            mae_change = result['mae'] - all_vars_mae
            impact = "WORSE" if mae_change > 0 else "better"
            logger.info(f"Remove {result['removed_variable']:12s}: "
                       f"MAE {result['mae']:.6f} ({mae_change:+.6f}) - {impact}")

    # Analyze results
    df = analyze_results(results, output_dir)

    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Create visualizations
    plot_ablation_results(results, output_dir)

    # Save summary report
    report_file = output_dir / 'ablation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2 ABLATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Objective: Determine which environmental variables contribute most\n")
        f.write("to sea-ice concentration forecasting accuracy.\n\n")
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        f.write(df[['configuration', 'n_variables', 'mae', 'rmse', 'correlation']].to_string(index=False))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*80 + "\n\n")

        best = df.iloc[0]
        sic_only = df[df['configuration'] == 'SIC only'].iloc[0]
        improvement = ((sic_only['mae'] - best['mae']) / sic_only['mae']) * 100

        f.write(f"Best configuration: {best['configuration']}\n")
        f.write(f"Improvement over SIC-only: {improvement:.2f}%\n\n")

        if best['configuration'] == 'All variables':
            f.write("All environmental variables together provide the best performance.\n")
            f.write("Recommendation: Use the full Phase 2 model with all variables.\n")
        else:
            f.write(f"A subset of variables ({best['configuration']}) performs best.\n")
            f.write("Recommendation: Consider using this reduced configuration for efficiency.\n")

    logger.info(f"Report saved to {report_file}")

    logger.info("\n" + "="*80)
    logger.info("ABLATION ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
