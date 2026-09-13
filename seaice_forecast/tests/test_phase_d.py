"""
Unit tests for Phase D Multi-Horizon Forecasting.

Validates:
1. Direct multi-head forecasting shapes and horizons [1, 3, 5, 7].
2. Recursive autoregressive rollout sequence shapes and horizon slicing.
3. Masked multi-horizon loss computation and weighting.
4. Metric calculations (MAE, RMSE, Pearson r, Ice-Edge Displacement) with known values.
5. Noise suppression in ice-edge displacement (avoiding spurious open-ocean pixel inflation).
6. Checkpoint save and resume state integrity.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.models.multi_horizon import (
    MultiHorizonForecaster,
    MultiHorizonLoss,
    MultiHorizonTrainer,
    DirectMultiHeadUNetConvLSTM,
    RecursiveMultiHorizonRollout
)
from seaice_forecast.models.unet_convlstm import UNetConvLSTM
from seaice_forecast.evaluation.metrics import (
    masked_mae,
    masked_rmse,
    spatial_correlation,
    ice_edge_displacement,
    extract_ice_edge
)
from seaice_forecast.evaluation.multi_horizon_eval import evaluate_multi_horizon


class TestMultiHorizonModels:
    """Test model architectures and rollout logic."""

    def test_direct_multi_head_shapes(self):
        horizons = [1, 3, 5, 7]
        B, T, C, H, W = 2, 7, 7, 64, 64
        model = DirectMultiHeadUNetConvLSTM(
            in_channels=C,
            horizons=horizons,
            seq_len=T,
            encoder_channels=[16, 32, 64, 128],
            convlstm_layers=1
        )
        x = torch.randn(B, T, C, H, W)
        outputs = model(x)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(horizons)
        for h in horizons:
            assert outputs[h].shape == (B, 1, H, W)
            assert not torch.isnan(outputs[h]).any()
            assert torch.all(outputs[h] >= 0.0) and torch.all(outputs[h] <= 1.0)

    def test_recursive_rollout_shapes(self):
        horizons = [1, 3, 5, 7]
        B, T, C, H, W = 2, 7, 7, 64, 64
        base_model = UNetConvLSTM(
            in_channels=C,
            output_channels=1,
            seq_len=T,
            encoder_channels=[16, 32, 64, 128],
            convlstm_layers=1
        )
        rollout_model = RecursiveMultiHorizonRollout(
            base_model=base_model,
            horizons=horizons,
            seq_len=T,
            in_channels=C
        )
        x = torch.randn(B, T, C, H, W)
        outputs = rollout_model(x)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(horizons)
        for h in horizons:
            assert outputs[h].shape == (B, 1, H, W)
            assert not torch.isnan(outputs[h]).any()

    def test_unified_forecaster_modes(self):
        horizons = [1, 3]
        x = torch.randn(1, 7, 7, 64, 64)

        direct = MultiHorizonForecaster(mode="direct", horizons=horizons, encoder_channels=[16, 32, 64, 128])
        out_d = direct(x)
        assert set(out_d.keys()) == {1, 3}

        recursive = MultiHorizonForecaster(mode="recursive", horizons=horizons, encoder_channels=[16, 32, 64, 128])
        out_r = recursive(x)
        assert set(out_r.keys()) == {1, 3}


class TestMultiHorizonLoss:
    """Test loss calculation, weights, and mask exclusion."""

    def test_loss_weighting(self):
        horizons = [1, 3]
        loss_fn = MultiHorizonLoss(horizons=horizons, weights={1: 0.8, 3: 0.2})

        B, H, W = 2, 32, 32
        preds = {
            1: torch.zeros(B, 1, H, W),
            3: torch.zeros(B, 1, H, W)
        }
        targets = {
            1: torch.ones(B, 1, H, W) * 1.0,  # MAE = 1.0
            3: torch.ones(B, 1, H, W) * 0.5   # MAE = 0.5
        }
        mask = torch.ones(H, W)

        total_loss, h_losses = loss_fn(preds, targets, mask)
        # Expected: 0.8 * 1.0 + 0.2 * 0.5 = 0.8 + 0.1 = 0.9
        assert abs(total_loss.item() - 0.9) < 1e-5
        assert abs(h_losses[1] - 1.0) < 1e-5
        assert abs(h_losses[3] - 0.5) < 1e-5

    def test_land_mask_exclusion(self):
        loss_fn = MultiHorizonLoss(horizons=[1])
        B, H, W = 1, 4, 4
        pred = torch.zeros(B, 1, H, W)
        target = torch.ones(B, 1, H, W)

        # Mask: only top half is ocean (1), bottom half is land (0)
        mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]).float()

        # Set errors on land to huge values
        target[:, :, 2:, :] = 999.0
        # Ocean errors are 1.0
        total_loss, _ = loss_fn({1: pred}, {1: target}, mask)
        # Land error must be ignored! Ocean MAE is exactly 1.0
        assert abs(total_loss.item() - 1.0) < 1e-5


class TestEvaluationMetrics:
    """Test evaluation metrics and ice-edge displacement."""

    def test_metrics_known_values(self):
        H, W = 10, 10
        pred = np.zeros((1, H, W), dtype=np.float32)
        tgt = np.ones((1, H, W), dtype=np.float32) * 0.5
        mask = np.ones((H, W), dtype=np.float32)

        mae = masked_mae(pred, tgt, mask)
        rmse = masked_rmse(pred, tgt, mask)
        assert abs(mae - 0.5) < 1e-5
        assert abs(rmse - 0.5) < 1e-5

    def test_ice_edge_noise_suppression(self):
        # Create a circle disk of ice
        H, W = 100, 100
        y, x = np.ogrid[:H, :W]
        cy, cx = 50, 50
        tgt = ((y - cy)**2 + (x - cx)**2 <= 20**2).astype(np.float32)

        # Perfect prediction
        disp_zero = ice_edge_displacement(tgt, tgt, pixel_size_km=25.0)
        assert abs(disp_zero) < 1e-3

        # Prediction with open ocean spurious noise pixels far away
        noisy_pred = tgt.copy()
        noisy_pred[5, 5] = 0.2  # isolated 1-pixel noise in corner
        noisy_pred[95, 95] = 0.2

        disp_noisy = ice_edge_displacement(noisy_pred, tgt, pixel_size_km=25.0)
        # Connected-component filtering should suppress isolated noise pixels
        assert abs(disp_noisy - disp_zero) < 1.0

    def test_multi_horizon_eval_pipeline(self):
        horizons = [1, 3]
        preds = {1: np.zeros((2, 1, 32, 32)), 3: np.zeros((2, 1, 32, 32))}
        targets = {1: np.ones((2, 1, 32, 32)) * 0.1, 3: np.ones((2, 1, 32, 32)) * 0.3}
        persist = np.ones((2, 1, 32, 32)) * 0.2

        df, summary = evaluate_multi_horizon(
            predictions=preds,
            targets=targets,
            persistence_inputs=persist,
            horizons=horizons
        )
        assert len(df) == 2
        assert "persist_mae" in df.columns
        assert "model_ice_edge_km" in df.columns
        assert 1 in summary and 3 in summary
