#!/usr/bin/env python
"""
Training script for iceberg drift prediction model.

Usage:
    python scripts/train.py --config config/settings.yaml --model-type xgboost
    python scripts/train.py --config config/settings.yaml --model-type pinn --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceberg_drift.config import load_config
from iceberg_drift.data_processing import (
    download_iceberg_positions,
    download_era5_wind,
    download_copernicus_currents,
    match_environmental_data,
    engineer_features,
    create_targets,
    split_trajectories,
    get_feature_columns,
    scale_features,
    DataQualityPipeline,
)
from iceberg_drift.models import (
    PhysicsDriftModel,
    PINNDriftModel,
    HybridDriftModel,
    create_model,
    MultiTargetGBM,
    PhysicsInformedGBM,
    create_gbm_config,
)
from iceberg_drift.evaluation import TrainingConfig, train_model, validate_model, ValidationConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/training.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train iceberg drift prediction model")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "pinn", "hybrid", "lstm", "gru", "transformer", "mlp", "physics_informed"])
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (NN only)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (NN only)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (NN only)")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of trees (GBM only)")
    parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth (GBM only)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="GBM learning rate")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for GBM")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # =============================================================================
    # Step 1: Download/Generate Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)

    # For development, generate synthetic data
    # In production, replace with actual downloads
    logger.info("Generating synthetic training data...")
    iceberg_df = download_iceberg_positions(
        output_dir=config.paths.raw_data,
        start_date="2015-01-01",
        end_date="2023-12-31",
        min_length_m=100,
        source="SYNTHETIC",
    )
    logger.info(f"Generated {len(iceberg_df)} iceberg position records")

    # Download environmental data (synthetic for dev)
    wind_ds = download_era5_wind(
        output_dir=config.paths.raw_data,
        start_date="2015-01-01",
        end_date="2023-12-31",
    )
    current_ds = download_copernicus_currents(
        output_dir=config.paths.raw_data,
        start_date="2015-01-01",
        end_date="2023-12-31",
    )

    # =============================================================================
    # Step 2: Data Quality Pipeline
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Data Quality Pipeline")
    logger.info("=" * 60)

    quality_pipeline = DataQualityPipeline(
        min_observations=10,
        max_gap_hours=48,
        max_speed_kmh=50.0,
        resample_freq="6H",
        coastal_depth_threshold=100.0,
        fill_coastal_nan=True,
    )

    iceberg_df, wind_ds, current_ds = quality_pipeline.run(
        iceberg_df, wind_ds, current_ds
    )

    quality_summary = quality_pipeline.get_summary()
    logger.info(f"Quality report: {quality_summary}")

    # =============================================================================
    # Step 3: Match Environmental Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Environmental Data Matching")
    logger.info("=" * 60)

    matched_df = match_environmental_data(
        iceberg_df,
        wind_ds,
        current_ds,
    )
    logger.info(f"Matched data shape: {matched_df.shape}")

    # =============================================================================
    # Step 4: Feature Engineering
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)

    features_df = engineer_features(
        matched_df,
        add_cyclical_time=True,
        add_physics_features=True,
        add_lag_features=True,
    )
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Feature columns: {list(features_df.columns)}")

    # =============================================================================
    # Step 5: Create Targets
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Target Creation")
    logger.info("=" * 60)

    prediction_horizon = config.model.ml_correction.prediction_horizon
    targets_df = create_targets(
        features_df,
        prediction_horizon_hours=prediction_horizon,
        target_type="velocity",
    )
    logger.info(f"Targets shape: {targets_df.shape}")

    # =============================================================================
    # Step 6: Train/Val/Test Split
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Data Splitting")
    logger.info("=" * 60)

    train_df, val_df, test_df = split_trajectories(
        targets_df,
        train_frac=config.training.train_split,
        val_frac=config.training.val_split,
        test_frac=config.training.test_split,
        random_seed=config.training.random_seed,
        strategy="trajectory",
    )

    # =============================================================================
    # Step 7: Feature Selection & Scaling
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Feature Selection & Scaling")
    logger.info("=" * 60)

    feature_cols = [c for c in train_df.columns if not c.startswith(('target_', 'iceberg_id', 'datetime', 'lat', 'lon'))]
    target_cols = [c for c in train_df.columns if c.startswith('target_')]

    logger.info(f"Selected {len(feature_cols)} features, {len(target_cols)} targets")
    logger.info(f"Target columns: {target_cols}")

    # Scale features
    train_df, val_df, test_df, scaler = scale_features(
        train_df, val_df, test_df, feature_cols, scaler_type="standard"
    )

    # Save scaler
    Path(config.paths.models).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, Path(config.paths.models) / "feature_scaler.pkl")

    # =============================================================================
    # Step 8: Prepare arrays for GBM or DataLoaders for NN
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Prepare Training Data")
    logger.info("=" * 60)

    # Convert to numpy arrays
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_cols].values.astype(np.float32)

    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # =============================================================================
    # Step 9: Model Training
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 9: Model Training")
    logger.info("=" * 60)

    output_dim = len(target_cols)

    if args.model_type in ["xgboost", "lightgbm"]:
        # Gradient Boosted Machines
        logger.info(f"Training {args.model_type.upper()} model...")

        gbm_config = create_gbm_config(
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            use_gpu=args.use_gpu,
        )

        if args.model_type == "xgboost":
            from iceberg_drift.models import MultiTargetGBM
            model = MultiTargetGBM(config=gbm_config, target_names=target_cols)
        else:
            from iceberg_drift.models import MultiTargetGBM
            model = MultiTargetGBM(config=gbm_config, target_names=target_cols)

        # Train GBM
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        # Save model
        model.save(Path(config.paths.models) / f"{args.model_type}_model")

        # Evaluate on test
        test_pred = model.predict(X_test)
        from iceberg_drift.evaluation.metrics import regression_metrics
        test_metrics = regression_metrics(test_pred, y_test)
        logger.info(f"Test Metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}")

        # Feature importance
        importance = model.get_feature_importance_summary()
        logger.info(f"Top 10 features:\n{importance.head(10)}")

        # Save feature importance
        importance.to_csv(Path(config.paths.output) / f"{args.model_type}_feature_importance.csv", index=False)

        # Physics-informed GBM
        if args.model_type == "xgboost":  # Also train physics-informed version
            logger.info("Training Physics-Informed GBM...")
            physics_model = PhysicsDriftModel()
            pi_gbm = PhysicsInformedGBM(
                gbm_config=gbm_config,
                physics_model=physics_model,
                target_names=target_cols,
            )
            pi_gbm.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            pi_gbm.save(Path(config.paths.models) / "physics_informed_gbm")

    elif args.model_type == "physics_informed":
        # Physics-Informed GBM (residual learning)
        logger.info("Training Physics-Informed GBM...")

        gbm_config = create_gbm_config(
            model_type="xgboost",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            use_gpu=args.use_gpu,
        )

        physics_model = PhysicsDriftModel()
        model = PhysicsInformedGBM(
            gbm_config=gbm_config,
            physics_model=physics_model,
            target_names=target_cols,
        )
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
        model.save(Path(config.paths.models) / "physics_informed_gbm")

    else:
        # Neural Networks (PyTorch)
        logger.info(f"Training {args.model_type.upper()} neural network...")

        # Create DataLoaders
        from iceberg_drift.data_processing import create_dataloaders

        sequence_length = config.model.ml_correction.sequence_length
        pred_horizon = config.model.ml_correction.prediction_horizon
        time_step = 6  # hours

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            sequence_length=sequence_length,
            prediction_horizon=pred_horizon,
            time_step_hours=time_step,
            batch_size=args.batch_size,
            num_workers=4,
            mode="sequence",
        )

        logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

        input_dim = len(feature_cols)
        output_dim = len(target_cols)

        if args.model_type == "pinn":
            model = PINNDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
                ml_model_type="lstm",
                loss_weights={
                    "data": 1.0,
                    "physics": config.model.ml_correction.loss_weights.physics,
                    "boundary": config.model.ml_correction.loss_weights.boundary,
                },
            )
        elif args.model_type == "hybrid":
            model = HybridDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
                ml_model_type="lstm",
            )
        else:
            model = create_model(
                args.model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
            )

        logger.info(f"Created {args.model_type} model with {model.get_num_params():,} parameters")

        # Training
        train_config = TrainingConfig(
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=15,
            output_dir=config.paths.models,
            metric_horizons=[6, 12, 24, 48, 72],
        )

        trained_model, state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=args.device,
        )

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best epoch: {state.best_epoch + 1}")
        logger.info(f"Best validation loss: {state.best_metric:.4f}")
        logger.info(f"Model saved to: {config.paths.models}")

        # Final Evaluation
        logger.info("=" * 60)
        logger.info("STEP 10: Final Test Evaluation")
        logger.info("=" * 60)

        val_config = ValidationConfig(
            metric_horizons=[6, 12, 24, 48, 72],
            output_dir="output/validation",
        )

        val_result = validate_model(
            trained_model,
            test_loader,
            config=val_config,
            device=args.device,
        )

        logger.info("Final Test Metrics:")
        logger.info(f"  Position RMSE: {val_result.metrics.rmse_position_km:.2f} km")
        logger.info(f"  Position MAE: {val_result.metrics.mean_position_error_km:.2f} km")
        logger.info(f"  Direction Error: {val_result.metrics.mean_direction_error_deg:.1f}°")
        logger.info(f"  Skill vs Persistence: {val_result.metrics.skill_score_vs_persistence:.3f}")
        logger.info(f"  Skill vs Physics: {val_result.metrics.skill_score_vs_physics:.3f}")


if __name__ == "__main__":
    main()