#!/usr/bin/env python3
"""
Phase D: Train Multi-Horizon (+1d, +3d, +5d, +7d) Sea-Ice Forecasting Model.

Supports:
- Direct multi-head forecasting (dedicated spatial decoder heads per horizon)
- Recursive autoregressive rollout from single-step ConvLSTM
- Per-horizon weighted loss on ocean cells
- Gradient clipping (1.0) and Cosine Annealing learning rate schedule
- Atomic checkpointing with resume capability
- Evaluation with persistence baseline comparison
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import platform
import os
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Inline Hardware Detection and Auto-Tuning (No external dependencies)
# =============================================================================

def detect_hardware():
    """Detect Mac hardware and print configuration."""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        chip = platform.processor() or "Unknown CPU"
    
    cores = os.cpu_count() or 1
    
    try:
        mem_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip())
        mem_gb = mem_bytes / (1024**3)
    except Exception:
        mem_gb = 8.0  # fallback
    
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    cuda_available = torch.cuda.is_available()
    
    print("=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)
    print(f"Chip:      {chip}")
    print(f"Cores:     {cores}")
    print(f"RAM:       {mem_gb:.1f} GB")
    print(f"MPS:       {mps_available}")
    print(f"CUDA:      {cuda_available}")
    print(f"PyTorch:   {torch.__version__}")
    print("=" * 70)
    
    return {
        "chip": chip,
        "cores": cores,
        "mem_gb": mem_gb,
        "mps": mps_available,
        "cuda": cuda_available
    }


def quick_batch_size_probe(model, sample_input_shape, device, candidate_sizes=[1, 2, 4, 8]):
    """
    Quick batch size calibration (~15-20 seconds).
    Tests a few candidates and picks the largest that runs cleanly.
    
    Args:
        model: The model to test
        sample_input_shape: Tuple of (T, C, H, W) for temporal input
        device: torch device
        candidate_sizes: List of batch sizes to test
        
    Returns:
        Best batch size (int)
    """
    print("\nQuick batch size probe (testing a few candidates)...")
    model.eval()
    
    best_size = candidate_sizes[0]
    best_throughput = 0.0
    
    for batch_size in candidate_sizes:
        try:
            # Create dummy batch
            dummy_input = torch.randn(batch_size, *sample_input_shape, device=device)
            
            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Time a few iterations
            torch.mps.synchronize() if device.type == "mps" else None
            start_time = time.time()
            n_iters = 3
            
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(dummy_input)
            
            torch.mps.synchronize() if device.type == "mps" else None
            elapsed = time.time() - start_time
            
            samples_per_sec = (batch_size * n_iters) / elapsed
            
            print(f"  Batch size {batch_size:2d}: {samples_per_sec:.1f} samples/sec - OK")
            
            if samples_per_sec > best_throughput:
                best_throughput = samples_per_sec
                best_size = batch_size
            
            # Clean up
            del dummy_input
            torch.mps.empty_cache() if device.type == "mps" else None
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print(f"  Batch size {batch_size:2d}: Memory allocation failed - stopping")
                break
            else:
                print(f"  Batch size {batch_size:2d}: Error ({str(e)[:50]}) - stopping")
                break
        except Exception as e:
            print(f"  Batch size {batch_size:2d}: Unexpected error - stopping")
            break
    
    print(f"\nSelected batch size: {best_size} ({best_throughput:.1f} samples/sec)")
    return best_size


def quick_worker_probe(dataset, batch_size, worker_candidates=[0, 2]):
    """
    Quick worker count check.
    Tests 0 vs 2 workers, picks faster. Falls back to 0 on errors.
    
    Args:
        dataset: Dataset to test
        batch_size: Batch size to use
        worker_candidates: Worker counts to test
        
    Returns:
        Best num_workers (int)
    """
    print("\nQuick worker count probe...")
    
    best_workers = 0
    best_time = float('inf')
    
    for num_workers in worker_candidates:
        try:
            # Test small number of batches
            n_batches = min(5, max(1, len(dataset) // batch_size))
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(num_workers > 0),
                persistent_workers=(num_workers > 0),
                drop_last=False,
                timeout=10 if num_workers > 0 else 0
            )
            
            start_time = time.time()
            batches_loaded = 0
            
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                batches_loaded += 1
            
            elapsed = time.time() - start_time
            
            if batches_loaded > 0:
                time_per_batch = elapsed / batches_loaded
                print(f"  Workers {num_workers}: {time_per_batch*1000:.1f} ms/batch - OK")
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_workers = num_workers
            
        except Exception as e:
            print(f"  Workers {num_workers}: Error ({str(e)[:50]}) - skipping")
            continue
    
    print(f"\nSelected num_workers: {best_workers}")
    return best_workers


def auto_select_device(hw_info):
    """Select best device based on hardware detection."""
    if hw_info["mps"]:
        device = torch.device("mps")
        print(f"\n[Device] Using: MPS (Apple Silicon GPU)")
    elif hw_info["cuda"]:
        device = torch.device("cuda")
        print(f"\n[Device] Using: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n[Device] Using: CPU (no GPU acceleration)")
    
    return device

from seaice_forecast.models.multi_horizon import (
    MultiHorizonForecaster,
    MultiHorizonLoss,
    MultiHorizonTrainer
)
from seaice_forecast.data_processing.dataset_real import MultiHorizonSeaIceDataset
from seaice_forecast.evaluation.multi_horizon_eval import evaluate_multi_horizon
from seaice_forecast.visualization.multi_horizon_plots import plot_multi_horizon_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_phase4_multi_horizon")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Phase D Multi-Horizon Sea-Ice Forecasting Model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase_d_full.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "recursive"],
        default=None,
        help="Override forecasting mode (direct or recursive)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('mps', 'cuda', or 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # =============================================================================
    # AUTO-DETECTION: Hardware, Device, Batch Size, Workers
    # =============================================================================
    
    print("\n")
    hw_info = detect_hardware()

    # Device selection (auto-detect or use command-line override)
    if args.device:
        device = torch.device(args.device)
        print(f"\n[Device] Using command-line override: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[Device] Using: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[Device] Using: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n[Device] Using: CPU")
    
    logger.info(f"Using device: {device}")

    mode = args.mode or cfg["model"].get("mode", "direct")
    epochs = args.epochs or cfg["training"].get("epochs", 50)
    horizons = cfg["data"].get("horizons", [1, 3, 5, 7])
    input_window = cfg["data"].get("input_window", 7)
    lr = float(cfg["training"].get("learning_rate", 1e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    loss_weights = {int(k): float(v) for k, v in cfg["training"].get("loss_weights", {}).items()}
    
    # Get batch size from config (will be overridden by auto-tuning if not specified by args)
    config_batch_size = cfg["training"].get("batch_size", 2)

    # Land mask
    mask_path = Path(cfg["data"]["mask_path"])
    mask = np.load(mask_path) if mask_path.exists() else None

    # Datasets
    daily_dir = cfg["data"]["daily_dir"]
    phase = cfg["data"].get("phase", "phase2")

    logger.info(f"Building MultiHorizon datasets (window={input_window}, horizons={horizons}, phase={phase})...")
    train_dataset = MultiHorizonSeaIceDataset(
        daily_dir=daily_dir,
        split="train",
        input_window=input_window,
        horizons=horizons,
        phase=phase,
        mask_path=str(mask_path) if mask_path.exists() else None
    )
    val_dataset = MultiHorizonSeaIceDataset(
        daily_dir=daily_dir,
        split="val",
        input_window=input_window,
        horizons=horizons,
        phase=phase,
        mask_path=str(mask_path) if mask_path.exists() else None
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize model
    logger.info(f"Initializing MultiHorizonForecaster (mode={mode})...")
    forecaster = MultiHorizonForecaster(
        mode=mode,
        horizons=horizons,
        in_channels=cfg["model"].get("in_channels", 7),
        seq_len=input_window,
        encoder_channels=cfg["model"].get("encoder_channels", [32, 64, 128, 256]),
        convlstm_layers=cfg["model"].get("convlstm_layers", 1),
        dropout=cfg["model"].get("dropout", 0.1),
        checkpoint_path=cfg["model"].get("checkpoint_path", None)
    )
    
    # =============================================================================
    # AUTO-TUNE: Batch size and num_workers
    # =============================================================================
    
    # Move model to device for batch size probing
    forecaster.model.to(device)
    
    # Phase 4: Multi-horizon with multiple decoder heads - most memory intensive
    sample_h, sample_w = 332, 316  # Antarctic grid size
    
    # Quick batch size probe
    print("\n[Phase 4 Note] Testing batch sizes with Multi-Horizon forecaster")
    print("Multi-horizon has multiple decoder heads - most memory intensive, expect smallest batches")
    optimized_batch_size = quick_batch_size_probe(
        model=forecaster.model,
        sample_input_shape=(input_window, cfg["model"].get("in_channels", 7), sample_h, sample_w),
        device=device,
        candidate_sizes=[1, 2, 4]  # Start very small due to multi-head architecture
    )
    
    # Create dataloaders with optimized batch size
    train_loader = DataLoader(train_dataset, batch_size=optimized_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=optimized_batch_size, shuffle=False) if len(val_dataset) > 0 else None
    
    # Quick worker count probe on training dataset
    optimized_num_workers = quick_worker_probe(
        dataset=train_dataset,
        batch_size=optimized_batch_size,
        worker_candidates=[0, 2]
    )
    
    # Recreate dataloaders with optimized worker count
    if optimized_num_workers != 0:
        logger.info(f"\nRecreating dataloaders with optimized num_workers={optimized_num_workers}...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=optimized_batch_size,
            shuffle=True,
            num_workers=optimized_num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=optimized_batch_size,
            shuffle=False,
            num_workers=optimized_num_workers,
            pin_memory=True,
            persistent_workers=True
        ) if len(val_dataset) > 0 else None
    
    # =============================================================================
    # Print final optimized settings
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("OPTIMIZED TRAINING SETTINGS (PHASE 4: MULTI-HORIZON)")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Batch Size:       {optimized_batch_size}")
    print(f"Num Workers:      {optimized_num_workers}")
    print(f"Precision:        fp32 (stable default)")
    print(f"Mode:             {mode}")
    print(f"Horizons:         {horizons}")
    print(f"LR:               {lr}")
    print(f"Grad Clip:        {grad_clip}")
    print("=" * 70)

    output_dir = Path(cfg["output"].get("dir", "output/phase_d"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MultiHorizonTrainer(
        forecaster=forecaster,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        horizons=horizons,
        horizon_weights=loss_weights,
        mask=mask,
        output_dir=output_dir / "checkpoints"
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(f"Starting Multi-Horizon training for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    # Final evaluation on val/test set
    eval_loader = val_loader or train_loader
    eval_results = evaluate_multi_horizon(
        forecaster=forecaster,
        dataloader=eval_loader,
        device=device,
        mask=mask,
        horizons=horizons,
        pixel_size_km=cfg["data"].get("pixel_size_km", 25.0),
        output_csv_path=cfg["output"].get("metrics_csv"),
        output_json_path=cfg["output"].get("metrics_json"),
        is_smoke_test=cfg["experiment"].get("is_smoke_test", False)
    )

    # Plot metrics vs lead time
    plots_dir = Path(cfg["output"].get("plots_dir", output_dir / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix = "SMOKE_TEST_ONLY_" if cfg["experiment"].get("is_smoke_test", False) else ""
    plot_multi_horizon_metrics(
        metrics_df=eval_results["metrics_df"],
        save_path_png=str(plots_dir / f"{plot_prefix}error_vs_lead_time.png"),
        save_path_pdf=str(plots_dir / f"{plot_prefix}error_vs_lead_time.pdf"),
        is_smoke_test=cfg["experiment"].get("is_smoke_test", False)
    )

    logger.info("Multi-horizon training and evaluation pipeline complete.")


if __name__ == "__main__":
    main()
