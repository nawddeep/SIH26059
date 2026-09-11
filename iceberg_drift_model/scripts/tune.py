#!/usr/bin/env python
"""
Hyperparameter tuning script for iceberg drift prediction models.

Uses Optuna for Bayesian optimization on validation set.

Usage:
    python scripts/tune.py --model-type xgboost --n-trials 50
    python scripts/tune.py --model-type pinn --n-trials 30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna

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
    MultiTargetGBM,
    PhysicsInformedGBM,
    create_gbm_config,
    create_model,
)
from iceberg_drift.evaluation import TrainingConfig, train_model, validate_model, ValidationConfig


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/tuning.log"),
        ],
    )


def load_and_prepare_data(config, logger):
    """Load and prepare data (same as train.py)."""
    logger.info("=" * 60)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 60)

    iceberg_df = download_iceberg_positions(
        output_dir=config.paths.raw_data,
        start_date="2015-01-01",
        end_date="2023-12-31",
        min_length_m=100,
        source="SYNTHETIC",
    )
    logger.info(f"Generated {len(iceberg_df)} iceberg position records")

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

    # Quality pipeline
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

    matched_df = match_environmental_data(iceberg_df, wind_ds, current_ds)
    features_df = engineer_features(
        matched_df,
        add_cyclical_time=True,
        add_physics_features=True,
        add_lag_features=True,
    )

    prediction_horizon = config.model.ml_correction.prediction_horizon
    targets_df = create_targets(
        features_df,
        prediction_horizon_hours=prediction_horizon,
        target_type="velocity",
    )

    train_df, val_df, test_df = split_trajectories(
        targets_df,
        train_frac=config.training.train_split,
        val_frac=config.training.val_split,
        test_frac=config.training.test_split,
        random_seed=config.training.random_seed,
        strategy="trajectory",
    )

    feature_cols = [c for c in train_df.columns if not c.startswith(('target_', 'iceberg_id', 'datetime', 'lat', 'lon'))]
    target_cols = [c for c in train_df.columns if c.startswith('target_')]

    train_df, val_df, test_df, scaler = scale_features(
        train_df, val_df, test_df, feature_cols, scaler_type="standard"
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_cols].values.astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, target_cols, scaler


# =============================================================================
# GBM Objective Functions
# =============================================================================

def objective_xgboost(trial, X_train, y_train, X_val, y_val, feature_cols, target_cols):
    """Optuna objective for XGBoost."""
    from iceberg_drift.models import MultiTargetGBM, create_gbm_config

    # Hyperparameters to tune
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
    }

    gbm_config = create_gbm_config(model_type="xgboost")
    gbm_config.xgb_params.update(params)

    model = MultiTargetGBM(config=gbm_config, target_names=target_cols)

    try:
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        # Evaluate on validation set
        val_pred = model.predict(X_val)
        from iceberg_drift.evaluation.metrics import regression_metrics
        val_metrics = regression_metrics(val_pred, y_val)

        return val_metrics["rmse"]

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Trial failed: {e}")
        return float('inf')


def objective_lightgbm(trial, X_train, y_train, X_val, y_val, feature_cols, target_cols):
    """Optuna objective for LightGBM."""
    from iceberg_drift.models import MultiTargetGBM, create_gbm_config

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
    }

    gbm_config = create_gbm_config(model_type="lightgbm")
    gbm_config.lgb_params.update(params)

    model = MultiTargetGBM(config=gbm_config, target_names=target_cols)

    try:
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        val_pred = model.predict(X_val)
        from iceberg_drift.evaluation.metrics import regression_metrics
        val_metrics = regression_metrics(val_pred, y_val)

        return val_metrics["rmse"]

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Trial failed: {e}")
        return float('inf')


def objective_physics_informed(trial, X_train, y_train, X_val, y_val, feature_cols, target_cols):
    """Optuna objective for Physics-Informed GBM."""
    from iceberg_drift.models import PhysicsInformedGBM, create_gbm_config, PhysicsDriftModel

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
    }

    gbm_config = create_gbm_config(model_type="xgboost")
    gbm_config.xgb_params.update(params)

    physics_model = PhysicsDriftModel()
    model = PhysicsInformedGBM(
        gbm_config=gbm_config,
        physics_model=physics_model,
        target_names=target_cols,
    )

    try:
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        val_pred = model.predict(X_val)
        from iceberg_drift.evaluation.metrics import regression_metrics
        val_metrics = regression_metrics(val_pred, y_val)

        return val_metrics["rmse"]

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Trial failed: {e}")
        return float('inf')


# =============================================================================
# Neural Network Objective (for PINN, LSTM, etc.)
# =============================================================================

def objective_nn(trial, train_loader, val_loader, target_cols, device, model_type, config):
    """Optuna objective for Neural Networks."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    if model_type == "pinn":
        from iceberg_drift.models import PINNDriftModel
        model = PINNDriftModel(
            input_dim=len(feature_cols),  # Need feature_cols from outer scope
            output_dim=len(target_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=config.model.ml_correction.prediction_horizon,
            ml_model_type="lstm",
            loss_weights={"data": 1.0, "physics": 0.5, "boundary": 0.1},
        )
    elif model_type == "lstm":
        model = create_model(
            "lstm",
            input_dim=len(feature_cols),
            output_dim=len(target_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=config.model.ml_correction.prediction_horizon,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(30):  # Fewer epochs for tuning
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()

            if isinstance(model, (PINNDriftModel)):
                output = model(x, targets=y, teacher_forcing_ratio=0.5)
                pred = output["predictions"]
                if hasattr(model, 'compute_loss'):
                    loss_dict = model.compute_loss(output, y, x, batch["meta"])
                    loss = loss_dict["total_loss"]
                else:
                    loss = criterion(pred, y)
            else:
                pred = model(x, targets=y, teacher_forcing_ratio=0.5)
                loss = criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                if isinstance(model, (PINNDriftModel)):
                    output = model(x, targets=None, teacher_forcing_ratio=0.0)
                    pred = output["predictions"]
                else:
                    pred = model(x, targets=None, teacher_forcing_ratio=0.0)

                loss = criterion(pred, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


# =============================================================================
# Main Tuning Function
# =============================================================================

def run_tuning(
    model_type: str,
    n_trials: int = 50,
    timeout: int = None,
    output_dir: str = "output/tuning",
    device: str = "auto",
):
    """Run hyperparameter tuning for specified model type."""
    logger = logging.getLogger(__name__)

    # Load config and data
    config = load_config("config/settings.yaml")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, target_cols, scaler = load_and_prepare_data(config, logger)

    # Setup output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create study
    study_name = f"iceberg_drift_{model_type}"
    storage = f"sqlite:///{output_dir}/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    logger.info(f"Starting {model_type} tuning with {n_trials} trials")

    # Select objective function
    if model_type == "xgboost":
        objective = lambda t: objective_xgboost(t, X_train, y_train, X_val, y_val, feature_cols, target_cols)
    elif model_type == "lightgbm":
        objective = lambda t: objective_lightgbm(t, X_train, y_train, X_val, y_val, feature_cols, target_cols)
    elif model_type == "physics_informed":
        objective = lambda t: objective_physics_informed(t, X_train, y_train, X_val, y_val, feature_cols, target_cols)
    else:
        raise ValueError(f"Unknown model_type for tuning: {model_type}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Results
    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (val RMSE): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best parameters
    best_params = study.best_params
    joblib.dump(best_params, Path(output_dir) / f"best_params_{model_type}.pkl")

    # Save study
    joblib.dump(study, Path(output_dir) / f"study_{model_type}.pkl")

    # Export trials to CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(Path(output_dir) / f"trials_{model_type}.csv", index=False)

    # Retrain best model on full train+val
    logger.info("Retraining best model on full train+val data...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.vstack([y_train, y_val])

    if model_type == "xgboost":
        gbm_config = create_gbm_config(model_type="xgboost")
        gbm_config.xgb_params.update(best_params)
        best_model = MultiTargetGBM(config=gbm_config, target_names=target_cols)
        best_model.fit(X_full, y_full, feature_names=feature_cols)
        best_model.save(Path(output_dir) / f"best_{model_type}_model")

    elif model_type == "lightgbm":
        gbm_config = create_gbm_config(model_type="lightgbm")
        gbm_config.lgb_params.update(best_params)
        best_model = MultiTargetGBM(config=gbm_config, target_names=target_cols)
        best_model.fit(X_full, y_full, feature_names=feature_cols)
        best_model.save(Path(output_dir) / f"best_{model_type}_model")

    elif model_type == "physics_informed":
        gbm_config = create_gbm_config(model_type="xgboost")
        gbm_config.xgb_params.update(best_params)
        physics_model = PhysicsDriftModel()
        best_model = PhysicsInformedGBM(
            gbm_config=gbm_config,
            physics_model=physics_model,
            target_names=target_cols,
        )
        best_model.fit(X_full, y_full, feature_names=feature_cols)
        best_model.save(Path(output_dir) / f"best_{model_type}_model")

    # Final test evaluation
    test_pred = best_model.predict(X_test)
    from iceberg_drift.evaluation.metrics import regression_metrics
    test_metrics = regression_metrics(test_pred, y_test)

    logger.info(f"Final Test Metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}")

    return study, best_model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for iceberg drift models")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["xgboost", "lightgbm", "physics_informed"])
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--output-dir", type=str, default="output/tuning", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    study, best_model, test_metrics = run_tuning(
        model_type=args.model_type,
        n_trials=args.n_trials,
        timeout=args.timeout,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()