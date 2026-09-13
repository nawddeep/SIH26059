#!/usr/bin/env python3
"""
Phase E Demonstration Script: Uncertainty Quantification & IMO POLARIS Risk Function.

Executes:
1. Empirical lead-time uncertainty quantification on synthetic multi-horizon errors.
2. Fits expanding-uncertainty curves (linear and exponential).
3. Generates 90% confidence intervals and saves diagnostic plots.
4. Evaluates IMO POLARIS risk function across Polar Classes (PC1, PC3, PC5, PC7, Unclassed).
5. Prints tabular risk comparisons across key SIC operational thresholds (0.0, 0.15, 0.5, 0.85, 1.0).
6. Plots risk vs. SIC across Polar Classes and saves to output/phase_e/.
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from seaice_forecast.uncertainty.quantification import (
    compute_horizon_error_stats,
    fit_uncertainty_curve,
    plot_uncertainty_curve,
    generate_uncertainty_report,
)
from seaice_forecast.risk.polaris import (
    PolarisRiskConfig,
    risk_ice,
    risk_ice_array,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("demo_phase_e")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase E Uncertainty and Risk Demonstration.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase_e.yaml",
        help="Path to Phase E YAML configuration file.",
    )
    return parser.parse_args()


def run_uncertainty_demo(output_dir: Path, cfg: dict):
    logger.info("=" * 75)
    logger.info("PHASE E [1/2]: LEAD-TIME UNCERTAINTY QUANTIFICATION DEMO")
    logger.info("=" * 75)

    horizons = cfg["uncertainty"].get("horizons", [1, 3, 5, 7])

    # Generate synthetic error distributions where error variance strictly expands with lead time
    np.random.seed(42)
    n_samples_per_horizon = 500
    synthetic_errors = {}

    # Synthetic standard deviation expands: h=1 -> 0.04, h=3 -> 0.08, h=5 -> 0.13, h=7 -> 0.18
    base_stds = {1: 0.04, 3: 0.08, 5: 0.13, 7: 0.18}

    for h in horizons:
        std_target = base_stds.get(h, 0.04 * np.sqrt(h))
        # Zero-mean forecast errors with expanding variance
        errs = np.random.normal(loc=0.0, scale=std_target, size=n_samples_per_horizon)
        synthetic_errors[h] = errs

    # Compute empirical horizon statistics
    stats = compute_horizon_error_stats(synthetic_errors)
    logger.info("Empirical Horizon Statistics:")
    for h in horizons:
        st = stats[h]
        logger.info(
            f"  +{h}d: Mean={st['mean']:+.4f} | Std={st['std']:.4f} | "
            f"5th %ile={st['q05']:+.4f} | 95th %ile={st['q95']:+.4f}"
        )

    # Fit expanding uncertainty curves
    stds = [stats[h]["std"] for h in horizons]
    linear_fit = fit_uncertainty_curve(horizons, stds, method="linear")
    exp_fit = fit_uncertainty_curve(horizons, stds, method="exponential")

    # Assert that fitted uncertainty is strictly expanding
    pred_stds = [linear_fit(h) for h in horizons]
    for i in range(len(pred_stds) - 1):
        assert pred_stds[i] < pred_stds[i + 1], (
            f"Uncertainty must expand monotonically with lead time: {pred_stds[i]} !< {pred_stds[i+1]}"
        )
    logger.info("✓ Verified: Fitted uncertainty curve expands strictly monotonically with lead time.")

    # Save diagnostic uncertainty plot
    demo_png = Path(cfg["output"].get("uncertainty_demo_png", output_dir / "phase_e_uncertainty_demo.png"))
    demo_pdf = Path(cfg["output"].get("uncertainty_demo_pdf", output_dir / "phase_e_uncertainty_demo.pdf"))

    plot_uncertainty_curve(
        horizons=horizons,
        stds=stds,
        fit_fn=linear_fit,
        output_path_png=demo_png,
        output_path_pdf=demo_pdf,
        stats_dict=stats,
        is_smoke_test=True,
        title_suffix="Empirical Error Variance Model",
    )
    logger.info(f"✓ Saved Uncertainty Demo PNG: {demo_png}")
    logger.info(f"✓ Saved Uncertainty Demo PDF: {demo_pdf}")

    # Generate full report JSON
    report = generate_uncertainty_report(
        error_pairs=synthetic_errors,
        output_dir=output_dir,
        method="linear",
        is_smoke_test=True,
    )
    logger.info(f"✓ Saved Uncertainty JSON Report to: {output_dir}")

    return stats, linear_fit


def run_risk_demo(output_dir: Path, cfg: dict):
    logger.info("\n" + "=" * 75)
    logger.info("PHASE E [2/2]: IMO POLARIS NAVIGATIONAL RISK FUNCTION DEMO")
    logger.info("=" * 75)

    polaris_cfg = PolarisRiskConfig(
        open_water_threshold=cfg["polaris_risk"].get("open_water_threshold", 0.15),
        open_water_risk=cfg["polaris_risk"].get("open_water_risk", 0.0),
        max_risk=cfg["polaris_risk"].get("max_risk", 1.0),
    )

    test_sics = [0.0, 0.15, 0.50, 0.85, 1.0]
    polar_classes = ["PC1", "PC3", "PC5", "PC7", "UNCLASSED"]

    table_records = []
    for sic in test_sics:
        row = {"SIC": sic}
        for pc in polar_classes:
            val = risk_ice(sic, polar_class=pc, config=polaris_cfg)
            row[pc] = round(val, 4)
        table_records.append(row)

    df = pd.DataFrame(table_records)
    logger.info("\nIMO POLARIS Navigational Risk R_ice = f(SIC, PolarClass):")
    print(df.to_string(index=False))

    # Assertions for physical sanity
    for pc in polar_classes:
        # At SIC=0.0, risk is near zero
        assert risk_ice(0.0, pc, polaris_cfg) < 0.01, f"{pc} risk at SIC=0.0 must be near zero"
        # At SIC=0.15, risk is low (< 0.05)
        assert risk_ice(0.15, pc, polaris_cfg) <= 0.05, f"{pc} risk at SIC=0.15 must be <= 0.05"
        # At SIC=1.0, risk approaches maximum (> 0.9)
        assert risk_ice(1.0, pc, polaris_cfg) >= 0.90, f"{pc} risk at SIC=1.0 must be >= 0.90"
        # Monotonicity test
        risks = [risk_ice(s, pc, polaris_cfg) for s in test_sics]
        assert all(risks[i] <= risks[i + 1] for i in range(len(risks) - 1)), f"{pc} risks must be monotonic"

    # Higher ice class (PC1) must have lower risk than lower ice class (PC7) for same pack ice
    assert risk_ice(0.5, "PC1", polaris_cfg) < risk_ice(0.5, "PC7", polaris_cfg)
    assert risk_ice(0.85, "PC1", polaris_cfg) < risk_ice(0.85, "PC7", polaris_cfg)
    logger.info("✓ Verified: Risk function satisfies all boundary, monotonicity, and class hierarchy criteria.")

    # Vectorized array check
    sic_grid = np.array([0.0, 0.15, 0.5, 0.85, 1.0])
    arr_risks = risk_ice_array(sic_grid, "PC4", polaris_cfg)
    elem_risks = np.array([risk_ice(s, "PC4", polaris_cfg) for s in sic_grid])
    np.testing.assert_allclose(arr_risks, elem_risks, rtol=1e-5)
    logger.info("✓ Verified: Vectorized risk_ice_array matches element-wise risk_ice exactly.")

    # Save risk table JSON
    risk_json_path = Path(cfg["output"].get("risk_table_json", output_dir / "phase_e_risk_table.json"))
    with open(risk_json_path, "w") as f:
        json.dump(table_records, f, indent=2)
    logger.info(f"✓ Saved Risk Table JSON: {risk_json_path}")

    # Plot Risk vs. SIC curves across Polar Classes
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    sic_dense = np.linspace(0.0, 1.0, 300)
    palette = {
        "PC1": ("#1f77b4", "-", "PC1 (Heavy Icebreaker / Year-round Polar Waters)"),
        "PC3": ("#2ca02c", "-", "PC3 (Dedicated Polar Research Vessel)"),
        "PC5": ("#ff7f0e", "-", "PC5 (Medium Icebreaker / First-Year Ice)"),
        "PC7": ("#d62728", "-", "PC7 (Thin First-Year Ice / Summer Operation)"),
        "UNCLASSED": ("#7f7f7f", "--", "Unclassed (Open Water Hull)"),
    }

    for pc, (color, linestyle, label) in palette.items():
        curve = risk_ice_array(sic_dense, polar_class=pc, config=polaris_cfg)
        ax.plot(sic_dense, curve, color=color, linestyle=linestyle, linewidth=2.5, label=label)

    # Highlight open water threshold
    ax.axvline(
        polaris_cfg.open_water_threshold,
        color="darkblue",
        linestyle=":",
        linewidth=1.8,
        label=f"Open Water Boundary (SIC = {polaris_cfg.open_water_threshold})",
    )

    ax.set_title(
        "Navigational Risk vs. Sea-Ice Concentration (IMO POLARIS Model)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Sea-Ice Concentration (SIC [0, 1])", fontsize=12)
    ax.set_ylabel("Navigation Risk / Impairment Score $R_{ice} \in [0, 1]$", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    risk_png = Path(cfg["output"].get("risk_demo_png", output_dir / "phase_e_risk_vs_sic.png"))
    risk_pdf = Path(cfg["output"].get("risk_demo_pdf", output_dir / "phase_e_risk_vs_sic.pdf"))

    fig.savefig(risk_png, dpi=300, bbox_inches="tight")
    fig.savefig(risk_pdf, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved Risk Plot PNG: {risk_png}")
    logger.info(f"✓ Saved Risk Plot PDF: {risk_pdf}")


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"].get("dir", "output/phase_e"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_uncertainty_demo(output_dir, cfg)
    run_risk_demo(output_dir, cfg)

    logger.info("\n" + "=" * 75)
    logger.info("PHASE E DEMONSTRATION EXECUTED SUCCESSFULLY (EXIT 0).")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
