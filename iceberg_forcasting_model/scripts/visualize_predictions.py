#!/usr/bin/env python
"""
Generate multi-panel prediction visualizations for Phase 2 Environmental U-Net.

Creates 3-panel plots for samples across different seasons:
1. Actual SIC field (with 0.15 ice-edge contour)
2. Predicted SIC field (with 0.15 ice-edge contour)
3. Difference Map (Predicted - Actual)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import get_project_root
from seaice_forecast.models.unet import UNet
from seaice_forecast.data_processing.dataset_phase2 import Phase2Dataset
from seaice_forecast.evaluation.metrics import masked_mae, masked_rmse


def get_season(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    m = dt.month
    if m in [12, 1, 2]:
        return 'Summer'
    elif m in [3, 4, 5]:
        return 'Autumn'
    elif m in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'


def main():
    project_root = get_project_root()
    output_dir = project_root / 'output' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load mask
    mask_file = project_root / 'data' / 'processed' / 'land_ocean_mask.npy'
    mask = np.load(mask_file)

    # Load Phase 2 test dataset
    test_file = project_root / 'data' / 'processed' / 'phase2' / 'test_data_phase2.npz'
    dataset = Phase2Dataset(str(test_file), mask=mask, validate_channels=False)

    # Load metadata
    meta_file = project_root / 'data' / 'processed' / 'phase2' / 'test_metadata.json'
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    date_pairs = meta.get('date_pairs', [])

    # Load model
    ckpt_path = project_root / 'models' / 'sic_unet_env_v001_best.pt'
    model = UNet(
        input_channels=49,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256],
        use_batch_norm=True,
        output_activation='sigmoid'
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Find representative samples across available seasons/dates
    dates = [pair[1] for pair in date_pairs]
    seasons = [get_season(d) for d in dates]

    selected_indices = []
    seen_seasons = set()
    for idx, s in enumerate(seasons):
        if s not in seen_seasons:
            selected_indices.append(idx)
            seen_seasons.add(s)

    # Also add early, middle, late samples if fewer seasons
    if len(selected_indices) < 3:
        n = len(dataset)
        selected_indices = [0, n // 2, n - 1]

    print(f"Plotting predictions for sample indices: {selected_indices}")

    for idx in selected_indices:
        target_date = dates[idx] if idx < len(dates) else f"sample_{idx}"
        season = seasons[idx] if idx < len(seasons) else "Unknown"

        x, y, m = dataset[idx]
        with torch.no_grad():
            x_in = x.unsqueeze(0).to(device)
            pred = model(x_in).squeeze(0).cpu().numpy()[0]

        target = y.numpy()[0]

        # Compute metrics for this sample
        sample_mae = masked_mae(pred, target, mask)
        sample_rmse = masked_rmse(pred, target, mask)

        # Mask out land for display
        pred_disp = np.ma.masked_where(mask == 0, pred)
        target_disp = np.ma.masked_where(mask == 0, target)
        diff = pred - target
        diff_disp = np.ma.masked_where(mask == 0, diff)

        # Plot 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(
            f"Phase 2 Environmental U-Net Forecast | Date: {target_date} ({season})\n"
            f"Sample MAE: {sample_mae:.5f}, RMSE: {sample_rmse:.5f}",
            fontsize=14, fontweight='bold'
        )

        # 1. Actual
        ax = axes[0]
        im0 = ax.imshow(target_disp, cmap='Blues_r', vmin=0, vmax=1)
        ax.contour(target >= 0.15, levels=[0.5], colors='red', linewidths=1.5)
        ax.set_title("Actual SIC (Red = 15% Edge)", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label='SIC')

        # 2. Predicted
        ax = axes[1]
        im1 = ax.imshow(pred_disp, cmap='Blues_r', vmin=0, vmax=1)
        ax.contour(pred >= 0.15, levels=[0.5], colors='red', linewidths=1.5)
        ax.set_title("Predicted SIC (Red = 15% Edge)", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='SIC')

        # 3. Difference
        ax = axes[2]
        vlim = max(0.05, float(np.percentile(np.abs(diff[mask == 1]), 98)))
        im2 = ax.imshow(diff_disp, cmap='coolwarm', vmin=-vlim, vmax=vlim)
        ax.set_title(f"Difference (Pred - Actual)\nvlim = +/-{vlim:.4f}", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Error')

        plt.tight_layout()
        save_file = output_dir / f"env_unet_forecast_{season.lower()}_{target_date}.png"
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {save_file.name}")

    print("All forecast visualizations generated successfully.")


if __name__ == "__main__":
    main()
