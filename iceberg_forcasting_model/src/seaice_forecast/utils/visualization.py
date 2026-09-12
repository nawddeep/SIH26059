"""
Visualization utilities for sea-ice forecasting results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, List, Tuple
import json


def create_ice_colormap():
    """
    Create colormap for sea-ice concentration.
    
    Returns:
        Matplotlib colormap
    """
    # Blue (no ice) -> White (full ice)
    colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff', '#ffffff']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('ice', colors, N=n_bins)
    return cmap


def plot_prediction_comparison(
    input_last: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    date_str: Optional[str] = None
):
    """
    Plot comparison of input (last day), prediction, target, and error.
    
    Args:
        input_last: Last day of input [H, W]
        prediction: Model prediction [H, W]
        target: Ground truth [H, W]
        mask: Land-ocean mask [H, W] (optional)
        save_path: Path to save figure (optional)
        title: Figure title (optional)
        date_str: Date string for subtitle (optional)
    """
    # Squeeze any extra dimensions
    input_last = np.squeeze(input_last)
    prediction = np.squeeze(prediction)
    target = np.squeeze(target)
    
    # Compute error
    error = prediction - target
    
    # Apply mask for visualization
    if mask is not None:
        input_last = np.ma.masked_where(mask == 0, input_last)
        prediction = np.ma.masked_where(mask == 0, prediction)
        target = np.ma.masked_where(mask == 0, target)
        error = np.ma.masked_where(mask == 0, error)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title or 'Sea-Ice Concentration Forecast', fontsize=16, fontweight='bold')
    
    if date_str:
        fig.text(0.5, 0.96, date_str, ha='center', fontsize=12)
    
    ice_cmap = create_ice_colormap()
    
    # Last input day
    im1 = axes[0, 0].imshow(input_last, cmap=ice_cmap, vmin=0, vmax=1)
    axes[0, 0].set_title('Last Input Day (t)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label='SIC')
    
    # Prediction
    im2 = axes[0, 1].imshow(prediction, cmap=ice_cmap, vmin=0, vmax=1)
    axes[0, 1].set_title('Prediction (t+1)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label='SIC')
    
    # Ground truth
    im3 = axes[1, 0].imshow(target, cmap=ice_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth (t+1)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label='SIC')
    
    # Error
    max_abs_error = max(abs(np.nanmin(error)), abs(np.nanmax(error)))
    im4 = axes[1, 1].imshow(error, cmap='RdBu_r', vmin=-max_abs_error, vmax=max_abs_error)
    axes[1, 1].set_title('Error (Pred - Truth)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Error')
    
    # Add metrics text
    mae = np.nanmean(np.abs(error))
    rmse = np.sqrt(np.nanmean(error**2))
    
    metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    axes[1, 1].text(0.02, 0.98, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history_file: str,
    save_path: Optional[str] = None
):
    """
    Plot training history (loss curves and learning rate).
    
    Args:
        history_file: Path to history JSON file
        save_path: Path to save figure (optional)
    """
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Masked MAE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])
    axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best (epoch {best_epoch})')
    axes[0].scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_map(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = 'Mean Absolute Error Map'
):
    """
    Plot spatial map of mean absolute error.
    
    Args:
        predictions: All predictions [N, 1, H, W]
        targets: All targets [N, 1, H, W]
        mask: Land-ocean mask [H, W] (optional)
        save_path: Path to save figure (optional)
        title: Figure title
    """
    # Squeeze and compute MAE per pixel
    predictions = np.squeeze(predictions)
    targets = np.squeeze(targets)
    
    # Compute absolute errors
    abs_errors = np.abs(predictions - targets)
    
    # Mean error per pixel across all samples
    mean_error = np.mean(abs_errors, axis=0)
    
    # Apply mask
    if mask is not None:
        mean_error = np.ma.masked_where(mask == 0, mean_error)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(mean_error, cmap='YlOrRd', vmin=0, vmax=np.nanpercentile(mean_error, 95))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Absolute Error', fontsize=12)
    
    # Add statistics
    stats_text = (f'Mean: {np.nanmean(mean_error):.4f}\n'
                 f'Median: {np.nanmedian(mean_error):.4f}\n'
                 f'Max: {np.nanmax(mean_error):.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ice_edge_comparison(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.15,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot ice edge comparison between prediction and ground truth.
    
    Args:
        prediction: Model prediction [H, W]
        target: Ground truth [H, W]
        threshold: SIC threshold for ice edge
        mask: Land-ocean mask [H, W] (optional)
        save_path: Path to save figure (optional)
        title: Figure title (optional)
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    # Squeeze
    prediction = np.squeeze(prediction)
    target = np.squeeze(target)
    
    # Create ice masks
    pred_ice = prediction >= threshold
    target_ice = target >= threshold
    
    # Apply ocean mask
    if mask is not None:
        ocean = mask == 1
        pred_ice = pred_ice & ocean
        target_ice = target_ice & ocean
    
    # Extract edges
    pred_edge = binary_dilation(pred_ice) ^ binary_erosion(pred_ice)
    target_edge = binary_dilation(target_ice) ^ binary_erosion(target_ice)
    
    # Create RGB image
    img = np.ones((*prediction.shape, 3))
    
    # Base SIC in grayscale
    if mask is not None:
        sic_display = np.ma.masked_where(mask == 0, target)
    else:
        sic_display = target
    
    img[:, :, 0] = sic_display
    img[:, :, 1] = sic_display
    img[:, :, 2] = sic_display
    
    # Overlay edges: red = predicted, blue = actual, purple = both
    img[pred_edge, :] = [1, 0, 0]  # Red
    img[target_edge, :] = [0, 0, 1]  # Blue
    img[pred_edge & target_edge, :] = [0.5, 0, 0.5]  # Purple
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(title or f'Ice Edge Comparison (threshold={threshold:.0%})', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Predicted Edge'),
        Patch(facecolor='blue', label='Actual Edge'),
        Patch(facecolor='purple', label='Both')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_comparison(
    baseline_file: str,
    model_file: str,
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        baseline_file: Path to baseline results JSON
        model_file: Path to model results JSON
        save_path: Path to save figure (optional)
    """
    # Load results
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)
    
    with open(model_file, 'r') as f:
        model_results = json.load(f)
    
    # Extract metrics
    pers_metrics = baseline_results['baselines']['persistence']
    clim_metrics = baseline_results['baselines']['climatology']
    model_metrics = model_results['metrics']
    
    metrics_to_plot = ['mae', 'rmse', 'spatial_correlation', 'ice_edge_displacement']
    metric_names = ['MAE', 'RMSE', 'Correlation', 'Ice Edge Disp. (px)']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(3)
    width = 0.6
    
    for i, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        values = [
            pers_metrics[metric_key],
            clim_metrics[metric_key],
            model_metrics[metric_key]
        ]
        
        # Color best in green
        if metric_key == 'spatial_correlation':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        bar_colors = [colors[0], colors[1], colors[2] if best_idx == 2 else 'gray']
        
        axes[i].bar(x_pos, values, width, color=bar_colors, alpha=0.8, edgecolor='black')
        axes[i].set_ylabel(metric_name, fontsize=11)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(['Persistence', 'Climatology', 'U-Net'], rotation=15, ha='right')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Mark best with star
        axes[i].text(best_idx, values[best_idx], '★', ha='center', va='bottom', 
                    fontsize=20, color='gold', weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Test visualization functions."""
    # Create dummy data
    H, W = 100, 100
    
    input_last = np.random.rand(H, W) * 0.8
    prediction = input_last + np.random.randn(H, W) * 0.1
    target = input_last + np.random.randn(H, W) * 0.05
    
    prediction = np.clip(prediction, 0, 1)
    target = np.clip(target, 0, 1)
    
    mask = np.ones((H, W))
    mask[:20, :] = 0  # Land strip
    
    print("Testing visualization functions...")
    
    plot_prediction_comparison(
        input_last, prediction, target, mask,
        title="Test Prediction",
        date_str="2020-06-15"
    )
    
    print("✓ Visualization test complete")


if __name__ == "__main__":
    main()
