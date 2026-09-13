"""
Per-Horizon Evaluation Module for Multi-Horizon Sea-Ice Forecasting.

Computes metrics for each lead time (+1, +3, +5, +7 days):
- Masked MAE (Mean Absolute Error) over ocean cells
- Masked RMSE (Root Mean Squared Error) over ocean cells
- Spatial Correlation (Pearson's r) over ocean cells
- Ice-Edge Displacement in physical kilometers (and pixels)

Compares model forecasts against Persistence baseline and saves tabular summaries to CSV and JSON.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from seaice_forecast.evaluation.metrics import (
    masked_mae,
    masked_rmse,
    spatial_correlation,
    ice_edge_displacement
)


def evaluate_multi_horizon(
    predictions: Dict[int, Union[torch.Tensor, np.ndarray]],
    targets: Dict[int, Union[torch.Tensor, np.ndarray]],
    persistence_inputs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask: Optional[np.ndarray] = None,
    pixel_size_km: float = 25.0,
    horizons: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute comprehensive per-horizon evaluation metrics for model and persistence baseline.

    Args:
        predictions: Dict mapping horizon h -> predicted SIC [N, 1, H, W]
        targets: Dict mapping horizon h -> target SIC [N, 1, H, W]
        persistence_inputs: Optional input tensor [N, T, C, H, W] or [N, 1, H, W] to extract persistence
        mask: Optional land/ocean mask [H, W], 1=ocean, 0=land
        pixel_size_km: Grid resolution in km (default: 25.0)
        horizons: Optional list of horizons to evaluate (defaults to sorted keys in targets)

    Returns:
        Tuple of (metrics_df, metrics_dict)
    """
    if horizons is None:
        horizons = sorted(targets.keys())

    # Convert tensors to numpy if needed
    preds_np = {}
    targets_np = {}
    for h in horizons:
        p = predictions[h]
        t = targets[h]
        preds_np[h] = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        targets_np[h] = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

    # Extract persistence prediction if available (most recent observed SIC)
    persist_sic = None
    if persistence_inputs is not None:
        p_in = persistence_inputs.detach().cpu().numpy() if isinstance(persistence_inputs, torch.Tensor) else persistence_inputs
        if p_in.ndim == 5:  # [N, T, C, H, W]
            persist_sic = p_in[:, -1, 0:1]  # Latest timestep SIC
        elif p_in.ndim == 4:
            if p_in.shape[1] > 1:
                persist_sic = p_in[:, -1:, :, :]
            else:
                persist_sic = p_in

    rows = []
    summary_dict = {}

    for h in horizons:
        pred_h = preds_np[h]
        target_h = targets_np[h]

        # Model metrics
        mae = float(masked_mae(pred_h, target_h, mask))
        rmse = float(masked_rmse(pred_h, target_h, mask))
        corr = float(spatial_correlation(pred_h, target_h, mask))
        disp_km = float(ice_edge_displacement(pred_h, target_h, threshold=0.15, mask=mask, pixel_size_km=pixel_size_km))
        disp_px = disp_km / pixel_size_km if not np.isnan(disp_km) else np.nan

        row = {
            "horizon_days": h,
            "lead_time": f"+{h}d",
            "model_mae": mae,
            "model_rmse": rmse,
            "model_corr": corr,
            "model_ice_edge_km": disp_km,
            "model_ice_edge_px": disp_px
        }

        # Baseline metrics
        if persist_sic is not None:
            p_mae = float(masked_mae(persist_sic, target_h, mask))
            p_rmse = float(masked_rmse(persist_sic, target_h, mask))
            p_corr = float(spatial_correlation(persist_sic, target_h, mask))
            p_disp_km = float(ice_edge_displacement(persist_sic, target_h, threshold=0.15, mask=mask, pixel_size_km=pixel_size_km))
            p_disp_px = p_disp_km / pixel_size_km if not np.isnan(p_disp_km) else np.nan

            row["persist_mae"] = p_mae
            row["persist_rmse"] = p_rmse
            row["persist_corr"] = p_corr
            row["persist_ice_edge_km"] = p_disp_km
            row["persist_ice_edge_px"] = p_disp_px

            # Difference (% improvement over persistence)
            row["mae_improvement_pct"] = ((p_mae - mae) / max(1e-6, p_mae)) * 100.0

        rows.append(row)
        summary_dict[h] = row

    df = pd.DataFrame(rows)
    return df, summary_dict


def save_evaluation_results(
    df: pd.DataFrame,
    summary_dict: Dict,
    output_dir: Union[str, Path],
    prefix: str = "multi_horizon_metrics"
) -> Tuple[Path, Path]:
    """
    Save evaluation results to CSV and JSON formats.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"{prefix}.csv"
    json_path = out_path / f"{prefix}.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        # Convert non-serializable types
        clean_dict = json.loads(df.to_json(orient="records"))
        json.dump(clean_dict, f, indent=2)

    return csv_path, json_path
