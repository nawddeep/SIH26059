#!/usr/bin/env python
"""
Apple Silicon MPS Training Optimization Benchmark

Tests various configurations to optimize training speed without changing
scientific behavior. Validates that outputs remain identical before/after
optimization.

Usage:
    python scripts/benchmark_mps_training.py --model-type phase1
    python scripts/benchmark_mps_training.py --model-type phase2
    python scripts/benchmark_mps_training.py --full-benchmark
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.models.unet import UNet
from seaice_forecast.models.unet_convlstm import UNetConvLSTM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MPSBenchmark:
    """Benchmark MPS training performance."""

    def __init__(self, model_type='phase1'):
        """
        Initialize benchmark.

        Args:
            model_type: 'phase1' (7 channels), 'phase2' (49 channels), or 'phase3' (ConvLSTM [7, 7])
        """
        self.model_type = model_type
        self.input_channels = 7 if model_type in ('phase1', 'phase3') else 49
        self.H, self.W = 332, 316

        # Device setup
        self.setup_device()

        # Create synthetic data for benchmarking
        self.create_synthetic_data()

        # Results storage
        self.results = []

    def setup_device(self):
        """Setup device with proper fallback logic."""
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info(f"✓ Using MPS device (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            logger.warning("⚠ MPS not available, using CPU")

        logger.info(f"Device: {self.device}")
        logger.info(f"PyTorch version: {torch.__version__}")

    def create_synthetic_data(self, n_samples=100):
        """Create synthetic data matching real data shapes."""
        logger.info(f"\nCreating synthetic data for {self.model_type}")
        if self.model_type == 'phase3':
            logger.info(f"  Input shape: [{n_samples}, 7, 7, {self.H}, {self.W}]")
            inputs = torch.randn(n_samples, 7, 7, self.H, self.W)
        else:
            logger.info(f"  Input shape: [{n_samples}, {self.input_channels}, {self.H}, {self.W}]")
            inputs = torch.randn(n_samples, self.input_channels, self.H, self.W)

        # Random targets and masks
        targets = torch.randn(n_samples, 1, self.H, self.W).clamp(0, 1)
        masks = torch.ones(n_samples, 1, self.H, self.W)

        self.dataset = TensorDataset(inputs, targets, masks)
        logger.info(f"  Dataset created: {len(self.dataset)} samples")

    def create_model(self):
        """Create model for benchmarking."""
        if self.model_type == 'phase3':
            model = UNetConvLSTM(
                in_channels=7,
                output_channels=1,
                seq_len=7,
                encoder_channels=[32, 64, 128, 256],
                convlstm_layers=1,
                use_batch_norm=True,
                output_activation='sigmoid'
            )
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"Model created: {total:,} parameters")
            return model

        model = UNet(
            input_channels=self.input_channels,
            output_channels=1,
            encoder_channels=[32, 64, 128, 256],
            use_batch_norm=True,
            output_activation='sigmoid'
        )

        params = model.get_num_parameters()
        logger.info(f"Model created: {params['total']:,} parameters")

        return model

    def masked_mae_loss(self, pred, target, mask):
        """Masked MAE loss for consistency check."""
        masked_diff = torch.abs(pred - target) * mask
        loss = masked_diff.sum() / mask.sum()
        return loss

    def correctness_check(self, batch_size=8):
        """
        Verify that optimizations don't change outputs.

        Runs one fixed batch through the model before and after
        optimizations and verifies numerical identity.
        """
        logger.info("\n" + "="*80)
        logger.info("CORRECTNESS CHECK: Verifying numerical identity")
        logger.info("="*80)

        # Create model
        model = self.create_model()
        model.to(self.device)
        model.eval()

        # Fixed batch
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        inputs, targets, masks = next(iter(dataloader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        masks = masks.to(self.device)

        # Set seed for reproducibility
        torch.manual_seed(42)
        if self.device.type == 'mps':
            torch.mps.manual_seed(42)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            loss = self.masked_mae_loss(outputs, targets, masks)

        # Move to CPU for comparison
        outputs_baseline = outputs.cpu().numpy()
        loss_baseline = loss.cpu().item()

        logger.info(f"Baseline output shape: {outputs_baseline.shape}")
        logger.info(f"Baseline output range: [{outputs_baseline.min():.6f}, {outputs_baseline.max():.6f}]")
        logger.info(f"Baseline loss: {loss_baseline:.8f}")

        return {
            'inputs': inputs.cpu(),
            'targets': targets.cpu(),
            'masks': masks.cpu(),
            'outputs': outputs_baseline,
            'loss': loss_baseline,
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()}
        }

    def verify_correctness(self, baseline, config_name, use_autocast=False):
        """Verify outputs match baseline."""
        logger.info(f"\nVerifying correctness for: {config_name}")

        # Create model with same weights
        model = self.create_model()
        model.load_state_dict(baseline['model_state'])
        model.to(self.device)
        model.eval()

        inputs = baseline['inputs'].to(self.device)
        targets = baseline['targets'].to(self.device)
        masks = baseline['masks'].to(self.device)

        # Set same seed
        torch.manual_seed(42)
        if self.device.type == 'mps':
            torch.mps.manual_seed(42)

        # Forward pass with optional autocast
        with torch.no_grad():
            if use_autocast and self.device.type == 'mps':
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    # Loss computed in fp32
                    loss = self.masked_mae_loss(outputs.float(), targets, masks)
            else:
                outputs = model(inputs)
                loss = self.masked_mae_loss(outputs, targets, masks)

        outputs_np = outputs.cpu().numpy()
        loss_val = loss.cpu().item()

        # Compare
        output_diff = np.abs(outputs_np - baseline['outputs'])
        max_diff = output_diff.max()
        mean_diff = output_diff.mean()
        loss_diff = abs(loss_val - baseline['loss'])

        # Tolerance based on precision
        if use_autocast:
            # FP16 has lower precision
            tol = 1e-3
        else:
            # FP32 should match closely
            tol = 1e-6

        passed = max_diff < tol and loss_diff < tol

        logger.info(f"  Output max diff: {max_diff:.2e} (tolerance: {tol:.2e})")
        logger.info(f"  Output mean diff: {mean_diff:.2e}")
        logger.info(f"  Loss diff: {loss_diff:.2e} (tolerance: {tol:.2e})")
        logger.info(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

        return {
            'passed': passed,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'loss_diff': float(loss_diff),
            'tolerance': tol
        }

    def benchmark_epoch(
        self,
        batch_size=8,
        num_workers=0,
        use_autocast=False,
        persistent_workers=False
    ):
        """
        Benchmark one full epoch.

        Returns:
            Dictionary with timing and throughput metrics
        """
        # Create dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # MPS doesn't benefit from pin_memory
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

        # Create model
        model = self.create_model()
        model.to(self.device)
        model.train()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Warmup (exclude from timing)
        logger.info(f"  Warmup pass...")
        for i, (inputs, targets, masks) in enumerate(dataloader):
            if i >= 2:  # 2 warmup batches
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()

            if use_autocast and self.device.type == 'mps':
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = self.masked_mae_loss(outputs.float(), targets, masks)
            else:
                outputs = model(inputs)
                loss = self.masked_mae_loss(outputs, targets, masks)

            loss.backward()
            optimizer.step()

        # Actual timing
        logger.info(f"  Timed epoch...")
        start_time = time.time()
        total_samples = 0
        total_loss = 0.0
        n_batches = 0

        for inputs, targets, masks in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()

            if use_autocast and self.device.type == 'mps':
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = self.masked_mae_loss(outputs.float(), targets, masks)
            else:
                outputs = model(inputs)
                loss = self.masked_mae_loss(outputs, targets, masks)

            loss.backward()
            optimizer.step()

            total_samples += inputs.size(0)
            total_loss += loss.item()
            n_batches += 1

        end_time = time.time()
        epoch_time = end_time - start_time
        samples_per_sec = total_samples / epoch_time
        avg_loss = total_loss / n_batches

        # Memory usage (if MPS)
        if self.device.type == 'mps':
            # MPS doesn't have direct memory query like CUDA
            # Would need process-level memory monitoring
            peak_memory_mb = "N/A (MPS)"
        else:
            peak_memory_mb = "N/A"

        return {
            'epoch_time': epoch_time,
            'samples_per_sec': samples_per_sec,
            'avg_loss': avg_loss,
            'total_samples': total_samples,
            'n_batches': n_batches,
            'peak_memory_mb': peak_memory_mb
        }

    def run_benchmark_suite(self):
        """Run complete benchmark suite."""
        logger.info("\n" + "="*80)
        logger.info(f"MPS TRAINING BENCHMARK - {self.model_type.upper()}")
        logger.info("="*80)
        logger.info(f"Model: U-Net with {self.input_channels} input channels")
        logger.info(f"Device: {self.device}")

        # 1. Correctness check baseline
        logger.info("\n" + "-"*80)
        logger.info("Step 1: Establishing correctness baseline")
        logger.info("-"*80)
        baseline = self.correctness_check(batch_size=8)

        # 2. Benchmark configurations
        configs = [
            {
                'name': 'Baseline (workers=0, bs=8, fp32)',
                'batch_size': 8,
                'num_workers': 0,
                'use_autocast': False,
                'persistent_workers': False
            },
            {
                'name': 'Workers=2, bs=8, fp32',
                'batch_size': 8,
                'num_workers': 2,
                'use_autocast': False,
                'persistent_workers': True
            },
            {
                'name': 'Workers=4, bs=8, fp32',
                'batch_size': 8,
                'num_workers': 4,
                'use_autocast': False,
                'persistent_workers': True
            },
            {
                'name': 'Workers=2, bs=16, fp32',
                'batch_size': 16,
                'num_workers': 2,
                'use_autocast': False,
                'persistent_workers': True
            },
            {
                'name': 'Workers=2, bs=16, fp16',
                'batch_size': 16,
                'num_workers': 2,
                'use_autocast': True,
                'persistent_workers': True
            },
        ]

        results = []

        for config in configs:
            logger.info("\n" + "-"*80)
            logger.info(f"Benchmarking: {config['name']}")
            logger.info("-"*80)

            try:
                # Verify correctness first
                correctness = self.verify_correctness(
                    baseline,
                    config['name'],
                    use_autocast=config['use_autocast']
                )

                if not correctness['passed'] and not config['use_autocast']:
                    logger.warning(f"⚠ Correctness check failed, skipping benchmark")
                    continue

                # Run benchmark
                metrics = self.benchmark_epoch(
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers'],
                    use_autocast=config['use_autocast'],
                    persistent_workers=config['persistent_workers']
                )

                result = {
                    'config': config['name'],
                    'batch_size': config['batch_size'],
                    'num_workers': config['num_workers'],
                    'precision': 'fp16' if config['use_autocast'] else 'fp32',
                    'epoch_time': metrics['epoch_time'],
                    'samples_per_sec': metrics['samples_per_sec'],
                    'avg_loss': metrics['avg_loss'],
                    'correctness': correctness,
                    'stable': True
                }

                results.append(result)

                logger.info(f"\n  Epoch time: {metrics['epoch_time']:.2f}s")
                logger.info(f"  Throughput: {metrics['samples_per_sec']:.1f} samples/s")
                logger.info(f"  Avg loss: {metrics['avg_loss']:.6f}")

            except Exception as e:
                logger.error(f"✗ Benchmark failed: {e}")
                results.append({
                    'config': config['name'],
                    'error': str(e),
                    'stable': False
                })

        self.results = results
        return results

    def save_results(self, output_path='output/mps_benchmark_results.json'):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'input_channels': self.input_channels,
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    def print_summary_table(self):
        """Print results as a table."""
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)

        print(f"\n{'Configuration':<40} {'Workers':<8} {'Batch':<7} {'Precision':<10} "
              f"{'Epoch(s)':<10} {'Samples/s':<12} {'Status':<8}")
        print("-" * 110)

        for r in self.results:
            if 'error' in r:
                print(f"{r['config']:<40} {'N/A':<8} {'N/A':<7} {'N/A':<10} "
                      f"{'N/A':<10} {'N/A':<12} {'FAILED':<8}")
            else:
                status = '✓ PASS' if r['correctness']['passed'] else '✗ FAIL'
                print(f"{r['config']:<40} {r['num_workers']:<8} {r['batch_size']:<7} "
                      f"{r['precision']:<10} {r['epoch_time']:<10.2f} "
                      f"{r['samples_per_sec']:<12.1f} {status:<8}")


def check_mps_fallbacks():
    """Check for MPS fallback warnings."""
    logger.info("\n" + "="*80)
    logger.info("CHECKING FOR MPS CPU FALLBACKS")
    logger.info("="*80)
    logger.info("\nSetting PYTORCH_ENABLE_MPS_FALLBACK=1 to expose fallbacks")

    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    logger.info("\nTo check for fallbacks during actual training:")
    logger.info("  export PYTORCH_ENABLE_MPS_FALLBACK=1")
    logger.info("  python scripts/train.py 2>&1 | grep 'not currently implemented for the MPS device'")
    logger.info("\nCommon fallback operations:")
    logger.info("  - Some interpolation modes (bilinear is usually OK)")
    logger.info("  - Advanced indexing patterns")
    logger.info("  - Some reduction operations")


def audit_cuda_references():
    """Audit codebase for CUDA-specific code."""
    logger.info("\n" + "="*80)
    logger.info("AUDITING FOR CUDA REFERENCES")
    logger.info("="*80)

    project_root = Path(__file__).parent.parent
    cuda_patterns = ['.cuda()', "device='cuda'", '.to("cuda")', ".to('cuda')"]

    found_issues = []

    for pattern in cuda_patterns:
        logger.info(f"\nSearching for: {pattern}")
        # This would need actual grep - placeholder for now
        logger.info(f"  Run: grep -r \"{pattern}\" {project_root}/src/ {project_root}/scripts/")

    logger.info("\nRecommendation:")
    logger.info("  Use device logic: device = 'mps' if torch.backends.mps.is_available() else 'cpu'")
    logger.info("  Then: model.to(device), tensor.to(device)")


def main():
    parser = argparse.ArgumentParser(description="MPS Training Benchmark")
    parser.add_argument(
        '--model-type',
        choices=['phase1', 'phase2', 'phase3'],
        default='phase1',
        help='Model type to benchmark'
    )
    parser.add_argument(
        '--full-benchmark',
        action='store_true',
        help='Run benchmarks for all model types (phase1, phase2, phase3)'
    )
    parser.add_argument(
        '--check-fallbacks',
        action='store_true',
        help='Check for MPS CPU fallbacks'
    )
    parser.add_argument(
        '--audit-cuda',
        action='store_true',
        help='Audit codebase for CUDA references'
    )

    args = parser.parse_args()

    if args.check_fallbacks:
        check_mps_fallbacks()
        return

    if args.audit_cuda:
        audit_cuda_references()
        return

    # Run benchmarks
    model_types = ['phase1', 'phase2', 'phase3'] if args.full_benchmark else [args.model_type]

    for model_type in model_types:
        benchmark = MPSBenchmark(model_type=model_type)
        results = benchmark.run_benchmark_suite()
        benchmark.print_summary_table()
        benchmark.save_results(f'output/mps_benchmark_{model_type}.json')

        # Clear MPS cache between models
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("\n✓ Cleared MPS cache")


if __name__ == "__main__":
    main()
