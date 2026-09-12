"""
Evaluation metrics for sea-ice concentration forecasting.

All metrics respect the land-ocean mask (land pixels excluded).
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy import ndimage


def masked_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Mean Absolute Error, excluding land pixels.

    Args:
        predictions: Predicted SIC [N, H, W] or [N, 1, H, W]
        targets: Target SIC [N, H, W] or [N, 1, H, W]
        mask: Land-ocean mask [H, W], 1=ocean, 0=land (optional)

    Returns:
        MAE value
    """
    # Squeeze singleton dimensions
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)

    # Ensure same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

    # Apply mask
    if mask is not None:
        # Compute errors first
        errors = np.abs(predictions - targets)

        # Broadcast mask to match batch dimension if needed
        if predictions.ndim == 3:  # [N, H, W]
            ocean_mask = (mask == 1)[np.newaxis, :, :]
            ocean_mask = np.broadcast_to(ocean_mask, predictions.shape)
        else:  # [H, W]
            ocean_mask = mask == 1

        # Only compute error over ocean pixels
        masked_errors = errors[ocean_mask]
    else:
        masked_errors = np.abs(predictions - targets).flatten()

    return float(np.mean(masked_errors))


def masked_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Root Mean Squared Error, excluding land pixels.

    Args:
        predictions: Predicted SIC [N, H, W] or [N, 1, H, W]
        targets: Target SIC [N, H, W] or [N, 1, H, W]
        mask: Land-ocean mask [H, W], 1=ocean, 0=land (optional)

    Returns:
        RMSE value
    """
    # Squeeze singleton dimensions
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)

    # Apply mask
    if mask is not None:
        # Compute errors first
        errors = (predictions - targets) ** 2

        # Broadcast mask to match batch dimension if needed
        if predictions.ndim == 3:  # [N, H, W]
            ocean_mask = (mask == 1)[np.newaxis, :, :]
            ocean_mask = np.broadcast_to(ocean_mask, predictions.shape)
        else:  # [H, W]
            ocean_mask = mask == 1

        masked_errors = errors[ocean_mask]
    else:
        masked_errors = ((predictions - targets) ** 2).flatten()

    return float(np.sqrt(np.mean(masked_errors)))


def spatial_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute spatial correlation coefficient between predicted and actual fields.

    This measures how well the spatial pattern is captured, regardless of
    the absolute values.

    Args:
        predictions: Predicted SIC [N, H, W] or [H, W]
        targets: Target SIC [N, H, W] or [H, W]
        mask: Land-ocean mask [H, W], 1=ocean, 0=land (optional)

    Returns:
        Correlation coefficient (Pearson's r)
    """
    # Squeeze singleton dimensions
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)

    # If batched, compute mean correlation across batch
    if predictions.ndim == 3:  # [N, H, W]
        correlations = []
        for pred, tgt in zip(predictions, targets):
            if mask is not None:
                ocean_mask = mask == 1
                pred_flat = pred[ocean_mask]
                tgt_flat = tgt[ocean_mask]
            else:
                pred_flat = pred.flatten()
                tgt_flat = tgt.flatten()

            # Compute Pearson correlation
            corr = np.corrcoef(pred_flat, tgt_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    else:  # [H, W]
        if mask is not None:
            ocean_mask = mask == 1
            pred_flat = predictions[ocean_mask]
            tgt_flat = targets[ocean_mask]
        else:
            pred_flat = predictions.flatten()
            tgt_flat = targets.flatten()

        corr = np.corrcoef(pred_flat, tgt_flat)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0


def extract_ice_edge(
    sic: np.ndarray,
    threshold: float = 0.15,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract ice edge as a binary mask using SIC threshold.

    Ice edge is defined as the boundary between ice-covered (SIC >= threshold)
    and ice-free (SIC < threshold) regions.

    Args:
        sic: Sea-ice concentration [H, W]
        threshold: SIC threshold for ice edge (default: 15%)
        mask: Land-ocean mask [H, W] (optional)

    Returns:
        Binary edge mask [H, W]
    """
    # Create ice/no-ice mask
    ice_mask = sic >= threshold

    # Apply ocean mask if provided
    if mask is not None:
        ice_mask = ice_mask & (mask == 1)

    # Find edges using morphological gradient
    # Edge = dilation - erosion
    from scipy.ndimage import binary_dilation, binary_erosion

    dilated = binary_dilation(ice_mask)
    eroded = binary_erosion(ice_mask)
    edge = dilated ^ eroded  # XOR gives the boundary

    return edge.astype(np.uint8)


def ice_edge_displacement(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.15,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute average displacement between predicted and actual ice edges.

    This measures how far apart the predicted and actual ice edge locations are,
    in pixels. Lower is better.

    Args:
        predictions: Predicted SIC [N, H, W] or [H, W]
        targets: Target SIC [N, H, W] or [H, W]
        threshold: SIC threshold for ice edge
        mask: Land-ocean mask [H, W] (optional)

    Returns:
        Mean edge displacement in pixels
    """
    # Squeeze singleton dimensions
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)

    # If batched, compute mean displacement across batch
    if predictions.ndim == 3:  # [N, H, W]
        displacements = []
        for pred, tgt in zip(predictions, targets):
            disp = _compute_edge_displacement_single(pred, tgt, threshold, mask)
            if not np.isnan(disp):
                displacements.append(disp)

        return float(np.mean(displacements)) if displacements else np.nan

    else:  # [H, W]
        return _compute_edge_displacement_single(predictions, targets, threshold, mask)


def _compute_edge_displacement_single(
    pred: np.ndarray,
    tgt: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray]
) -> float:
    """Compute edge displacement for a single sample."""
    # Extract ice edges
    pred_edge = extract_ice_edge(pred, threshold, mask)
    tgt_edge = extract_ice_edge(tgt, threshold, mask)

    # If either edge is empty, return NaN
    if pred_edge.sum() == 0 or tgt_edge.sum() == 0:
        return np.nan

    # Compute distance transform for each edge
    # This gives the distance from each pixel to the nearest edge pixel
    pred_dist = ndimage.distance_transform_edt(1 - pred_edge)
    tgt_dist = ndimage.distance_transform_edt(1 - tgt_edge)

    # Average distance from predicted edge to actual edge
    displacement_pred_to_tgt = float(np.mean(pred_dist[tgt_edge == 1]))

    # Average distance from actual edge to predicted edge
    displacement_tgt_to_pred = float(np.mean(tgt_dist[pred_edge == 1]))

    # Symmetric displacement (average of both directions)
    displacement = (displacement_pred_to_tgt + displacement_tgt_to_pred) / 2

    return displacement


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ice_edge_threshold: float = 0.15
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: Predicted SIC
        targets: Target SIC
        mask: Land-ocean mask (optional)
        ice_edge_threshold: Threshold for ice edge detection

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'mae': masked_mae(predictions, targets, mask),
        'rmse': masked_rmse(predictions, targets, mask),
        'spatial_correlation': spatial_correlation(predictions, targets, mask),
        'ice_edge_displacement': ice_edge_displacement(
            predictions, targets, ice_edge_threshold, mask
        )
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metric names and values
        title: Title for the metrics display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    # Define display names and formats
    display_info = {
        'mae': ('Mean Absolute Error', '.6f'),
        'rmse': ('Root Mean Squared Error', '.6f'),
        'spatial_correlation': ('Spatial Correlation', '.4f'),
        'ice_edge_displacement': ('Ice Edge Displacement (px)', '.2f')
    }

    for key, value in metrics.items():
        if key in display_info:
            name, fmt = display_info[key]
            if not np.isnan(value):
                print(f"{name:30s}: {value:{fmt}}")
            else:
                print(f"{name:30s}: N/A")
        else:
            print(f"{key:30s}: {value}")

    print(f"{'='*60}\n")


def main():
    """Test metrics."""
    # Create dummy data
    H, W = 50, 50
    predictions = np.random.rand(10, H, W) * 0.8
    targets = predictions + np.random.randn(10, H, W) * 0.1
    targets = np.clip(targets, 0, 1)

    # Create mask
    mask = np.ones((H, W))
    mask[:10, :] = 0  # Land strip at top

    # Compute metrics
    metrics = compute_all_metrics(predictions, targets, mask)
    print_metrics(metrics, "Test Metrics")


if __name__ == "__main__":
    main()
