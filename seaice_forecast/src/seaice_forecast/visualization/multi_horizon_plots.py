"""
Multi-Horizon Error vs. Lead-Time Plotting Module.

Generates clean 4-panel diagnostic curves:
1. Masked MAE vs. Lead Time (+1, +3, +5, +7 days)
2. Masked RMSE vs. Lead Time
3. Spatial Correlation vs. Lead Time
4. Ice-Edge Displacement (km) vs. Lead Time

Compares Model vs. Persistence baseline, saves PNG and PDF,
and prominently labels smoke-test figures with 'SMOKE TEST ONLY'.
"""

from typing import Union, Optional, Tuple
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_error_vs_lead_time(
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path],
    filename_stem: str = "error_vs_lead_time",
    is_smoke_test: bool = True,
    model_name: str = "ConvLSTM Multi-Horizon"
) -> Tuple[Path, Path]:
    """
    Generate 4-panel diagnostic figure comparing Model vs Persistence across lead times.

    Args:
        metrics_df: DataFrame containing per-horizon metrics (columns: horizon_days, model_mae, etc.)
        output_dir: Target directory to save figures
        filename_stem: Base filename for output PNG and PDF
        is_smoke_test: If True, adds 'SMOKE TEST ONLY' watermark/banner
        model_name: Name of the model for legend and title

    Returns:
        Tuple of (png_path, pdf_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = metrics_df.sort_values("horizon_days")
    horizons = df["horizon_days"].values

    # Setup styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)

    model_color = "#1f77b4"  # Blue
    persist_color = "#ff7f0e"  # Orange

    has_persist = "persist_mae" in df.columns

    # 1. Masked MAE
    ax = axes[0, 0]
    ax.plot(horizons, df["model_mae"], "o-", color=model_color, lw=2, ms=6, label=f"{model_name}")
    if has_persist:
        ax.plot(horizons, df["persist_mae"], "s--", color=persist_color, lw=2, ms=6, label="Persistence Baseline")
    ax.set_title("Masked MAE vs. Lead Time (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax.set_ylabel("MAE (Ocean SIC [0, 1])", fontsize=10)
    ax.set_xticks(horizons)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Masked RMSE
    ax = axes[0, 1]
    ax.plot(horizons, df["model_rmse"], "o-", color=model_color, lw=2, ms=6, label=f"{model_name}")
    if has_persist:
        ax.plot(horizons, df["persist_rmse"], "s--", color=persist_color, lw=2, ms=6, label="Persistence Baseline")
    ax.set_title("Masked RMSE vs. Lead Time (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax.set_ylabel("RMSE (Ocean SIC [0, 1])", fontsize=10)
    ax.set_xticks(horizons)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Spatial Correlation
    ax = axes[1, 0]
    ax.plot(horizons, df["model_corr"], "o-", color=model_color, lw=2, ms=6, label=f"{model_name}")
    if has_persist:
        ax.plot(horizons, df["persist_corr"], "s--", color=persist_color, lw=2, ms=6, label="Persistence Baseline")
    ax.set_title("Spatial Correlation vs. Lead Time (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax.set_ylabel("Pearson Correlation r", fontsize=10)
    ax.set_xticks(horizons)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Ice-Edge Displacement
    ax = axes[1, 1]
    ax.plot(horizons, df["model_ice_edge_km"], "o-", color=model_color, lw=2, ms=6, label=f"{model_name}")
    if has_persist:
        ax.plot(horizons, df["persist_ice_edge_km"], "s--", color=persist_color, lw=2, ms=6, label="Persistence Baseline")
    ax.set_title("Ice-Edge Displacement vs. Lead Time (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=10)
    ax.set_ylabel("Mean Edge Displacement (km)", fontsize=10)
    ax.set_xticks(horizons)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Super title & banner
    if is_smoke_test:
        fig.suptitle(
            "[SMOKE TEST ONLY — PENDING TRAINED MODEL]\nMulti-Horizon Sea-Ice Forecasting Performance (+1d to +7d)",
            fontsize=13,
            fontweight="bold",
            color="#d62728",
            y=0.98
        )
    else:
        fig.suptitle(
            f"Multi-Horizon Sea-Ice Forecasting Performance (+1d to +7d)\n{model_name} vs. Persistence",
            fontsize=13,
            fontweight="bold",
            y=0.98
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path
