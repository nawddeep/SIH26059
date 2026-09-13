#!/usr/bin/env python3
"""
Phase C: Train Spatial-Temporal ConvLSTM + U-Net for Antarctic Sea-Ice Forecasting.

Architecture:
- Input: [Batch, T=7, C=7, H=332, W=316] (explicit temporal sequence)
- Shared 2D spatial encoder: 4 downsampling stages (32, 64, 128, 256)
- Temporal Bottleneck: Recurrent ConvLSTM layer (512 channels)
- Spatial Decoder: Progressive upsampling with skip connections from latest timestep
- Output: [Batch, 1, H=332, W=316] (next-day SIC forecast, sigmoid bounded [0, 1])

Stability safeguards:
- Gradient norm clipping (torch.nn.utils.clip_grad_norm_) with telemetry
- Dynamic learning rate schedule (ReduceLROnPlateau)
- Masked MAE loss excluding land cells via reference mask
- Per-epoch atomic checkpointing with seamless resume
- Fixed ice-edge displacement metric (physically plausible km)
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.models.unet_convlstm import UNetConvLSTM
from seaice_forecast.data_processing.dataset_real import RealSeaIceDataset
from seaice_forecast.training.temporal_trainer import TemporalTrainer
from seaice_forecast.evaluation.metrics import (
    masked_mae,
    masked_rmse,
    spatial_correlation,
    ice_edge_displacement
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_phase3_convlstm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Phase C ConvLSTM + U-Net temporal sea-ice forecasting model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/regridded/daily",
        help="Path to daily regridded files directory."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2 for Mac mini MPS/CPU memory efficiency)."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization."
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (0 to disable)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Spatial dropout rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('mps', 'cuda', or 'cpu'). Auto-detected if omitted."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke-test (1-2 epochs) to verify architecture, gradients, and checkpointing."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/phase3",
        help="Directory to store model checkpoints."
    )
    return parser.parse_args()


def select_device(requested: str = None) -> str:
    """Select optimal compute device with preference for Apple Silicon MPS."""
    if requested:
        return requested

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS available. Using 'mps' device.")
        return "mps"
    else:
        logger.info("Using 'cpu' device.")
        return "cpu"


def main():
    args = parse_args()
    device = select_device(args.device)

    logger.info("=" * 70)
    logger.info("PHASE C: TEMPORAL ARCHITECTURE TRAINING (U-NET + CONVLSTM)")
    logger.info("=" * 70)
    logger.info(f"Target Device: {device}")
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Batch Size: {args.batch_size} | LR: {args.learning_rate} | Grad Clip: {args.grad_clip}")

    # 1. Load ground truth land/ocean mask
    mask_file = Path("data/processed/land_ocean_mask_ps25.npy")
    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"Loaded ground-truth land/ocean mask: shape {mask.shape} ({int(np.sum(mask == 1))} ocean cells)")
    else:
        logger.warning(f"Mask file {mask_file} not found. Running without explicit ocean mask.")
        mask = None

    # 2. Build dataset (explicit temporal shape [B, T=7, C=7, H, W])
    # In smoke-test mode or when available data on disk is short:
    # Adapt input_window to fit available days if necessary
    daily_files = list(Path(args.data_dir).glob("*/*.npz")) + list(Path(args.data_dir).glob("*.npz"))
    num_available_days = len(daily_files)
    logger.info(f"Found {num_available_days} daily regridded files on disk.")

    input_window = 7
    if args.smoke_test and num_available_days < 8:
        input_window = min(3, max(1, num_available_days - 1))
        logger.info(f"Smoke-test mode: adjusting input window to {input_window} days to fit {num_available_days} available daily files.")

    dataset = RealSeaIceDataset(
        daily_data_dir=args.data_dir,
        input_window=input_window,
        forecast_horizon=1,
        channel_concat=False,  # Explicit temporal [T, C, H, W]
        phase="phase2",        # All 7 environmental variables
        mask_path=mask_file if mask_file.exists() else None
    )

    total_samples = len(dataset)
    logger.info(f"Dataset constructed: {total_samples} total sliding-window samples (window={input_window}d).")

    if total_samples == 0:
        logger.error(
            "Zero samples available. Ingest real data via `run_data_pipeline.py` before running full training."
        )
        sys.exit(1)

    # Train / Val split
    if total_samples > 10:
        val_size = max(2, int(total_samples * 0.2))
        train_size = total_samples - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        # For small smoke tests, use same dataset for train and val
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # 3. Instantiate model
    model = UNetConvLSTM(
        in_channels=7,
        output_channels=1,
        seq_len=input_window,
        encoder_channels=[32, 64, 128, 256],
        convlstm_layers=1,
        dropout=args.dropout,
        output_activation="sigmoid"
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"UNetConvLSTM initialized with {total_params:,} parameters.")

    # 4. Instantiate trainer
    epochs = args.epochs if args.epochs != 50 else (2 if args.smoke_test else args.epochs)
    trainer = TemporalTrainer(
        model=model,
        mask=mask,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        lr_scheduler_type="plateau",
        patience=10,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name="sic_unet_convlstm_v001"
    )

    # 5. Execute training
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        resume_checkpoint=args.resume
    )

    # 6. Evaluation and metric computation on test / validation set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION & METRIC BENCHMARKING")
    logger.info("=" * 70)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            p_b = model(x_b)
            all_preds.append(p_b.cpu().numpy())
            all_targets.append(y_b.numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    mae_val = masked_mae(preds_np, targets_np, mask)
    rmse_val = masked_rmse(preds_np, targets_np, mask)
    corr_val = spatial_correlation(preds_np, targets_np, mask)
    disp_km = ice_edge_displacement(preds_np, targets_np, mask=mask, pixel_size_km=25.0)

    logger.info(f"Masked MAE:                {mae_val:.5f}")
    logger.info(f"Masked RMSE:               {rmse_val:.5f}")
    logger.info(f"Spatial Correlation:       {corr_val:.4f}")
    logger.info(f"Ice-Edge Displacement:     {disp_km:.2f} km")
    logger.info(f"Gradient Clip Triggers:    {trainer.clip_trigger_count} times")
    logger.info("=" * 70)

    results = {
        "model": "Phase 3: U-Net + ConvLSTM",
        "mode": "smoke_test" if args.smoke_test else "full_training",
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": trainer.best_val_loss,
        "metrics": {
            "mae": mae_val,
            "rmse": rmse_val,
            "correlation": corr_val,
            "ice_edge_displacement_km": disp_km
        },
        "telemetry": {
            "clip_trigger_count": trainer.clip_trigger_count,
            "device": device,
            "total_parameters": total_params
        }
    }

    results_file = Path(args.checkpoint_dir) / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()
