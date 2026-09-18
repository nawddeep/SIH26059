"""Picklable predictor objects for the four navigation components.

Each class packages one component behind a `predict()` call and carries metadata
describing what it actually is. Two hold weights fitted from data; two hold the
constants of a published formula. `kind` says which, because a pickle that looks
like a trained model but contains a lookup table is the kind of thing that
should be labelled rather than glossed over.

Dependencies differ by artifact and are recorded in `requires`. Every one of
them needs THIS module importable as `app.model_artifacts`, because that is the
path pickle records for the classes below - ship it beside the .pkl files:
  polaris, fuel   numpy
  drift           scikit-learn (the fitted estimators are embedded)
  sea ice         torch, plus seaice_forecast on sys.path for the architecture

Build them with scripts/export_model_pickles.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class PolarisRiskPredictor:
    """IMO POLARIS navigational ice risk. Deterministic; no fitted parameters.

    Carries the Polar Class exponents so it evaluates without importing the
    project, which is what makes this artifact self-contained.
    """

    kind = "deterministic"
    name = "IMO POLARIS ice risk index"
    source = "IMO MSC.1/Circ.1519"
    requires = ["numpy", "app.model_artifacts on sys.path"]
    output = "navigational risk index in [0, 1]"

    def __init__(self, class_exponents: dict[str, float], open_water_threshold: float = 0.15,
                 open_water_risk: float = 0.0, max_risk: float = 1.0):
        self.class_exponents = dict(class_exponents)
        self.open_water_threshold = open_water_threshold
        self.open_water_risk = open_water_risk
        self.max_risk = max_risk

    def _key(self, polar_class) -> str:
        k = str(polar_class).strip().upper()
        if k.isdigit():
            k = f"PC{int(k)}" if 1 <= int(k) <= 7 else "UNCLASSED"
        return k if k in self.class_exponents else "UNCLASSED"

    def predict(self, sic, polar_class: str = "PC4"):
        """Risk for a scalar or array of sea-ice concentrations in [0, 1]."""
        s = np.clip(np.asarray(sic, dtype=float), 0.0, 1.0)
        e = self.class_exponents[self._key(polar_class)]
        t, r_open, r_max = self.open_water_threshold, self.open_water_risk, self.max_risk

        sub = r_open + 0.005 * (s / t) ** 2
        span = max(1.0 - t, 1e-6)
        ice = 0.005 + (r_max - 0.005) * (((s - t) / span).clip(0, 1) ** e)
        out = np.clip(np.where(s < t, sub, ice), 0.0, r_max)
        return float(out) if out.ndim == 0 else out


class FuelConsumptionPredictor:
    """Ice-aware fuel burn and attainable speed. Physical model, not fitted.

    Speed decays as a stretched exponential in concentration while delivered
    power ramps toward the vessel's maximum rating; both push burn per kilometre
    up, which is why the penalty in heavy ice is several-fold.
    """

    kind = "deterministic"
    name = "Ice-aware fuel consumption model"
    source = "Riska et al. (1997) speed degradation; POLARIS class ladder"
    requires = ["numpy", "app.model_artifacts on sys.path"]
    output = "tonnes per cell, attainable speed in knots"

    def __init__(self, retention: dict[str, float], base_rate_t_per_km: float = 0.035,
                 v_ref_kn: float = 12.0, max_power_ratio: float = 1.45,
                 power_ramp_k: float = 5.0, sic_exponent: float = 2.0,
                 heavy_ice_sic: float = 0.95):
        self.retention = dict(retention)
        self.base_rate_t_per_km = base_rate_t_per_km
        self.v_ref_kn = v_ref_kn
        self.max_power_ratio = max_power_ratio
        self.power_ramp_k = power_ramp_k
        self.sic_exponent = sic_exponent
        self.heavy_ice_sic = heavy_ice_sic

    def _key(self, polar_class) -> str:
        k = str(polar_class).strip().upper().replace(" ", "_").replace("-", "_")
        if k.isdigit():
            k = f"PC{int(k)}" if 1 <= int(k) <= 7 else "UNCLASSED"
        return k if k in self.retention else "UNCLASSED"

    def _speed_factor(self, s, polar_class):
        k = -np.log(self.retention[self._key(polar_class)]) / (self.heavy_ice_sic ** self.sic_exponent)
        return np.exp(-k * s ** self.sic_exponent)

    def predict(self, sic, polar_class: str = "PC4", distance_km: float = 25.0) -> dict[str, Any]:
        """Fuel, speed and transit time for a cell at this concentration."""
        s = np.clip(np.asarray(sic, dtype=float), 0.0, 1.0)
        sf = self._speed_factor(s, polar_class)
        power = 1.0 + (self.max_power_ratio - 1.0) * (1.0 - np.exp(-self.power_ramp_k * s))
        fuel_per_km = self.base_rate_t_per_km * power / sf
        speed_kn = self.v_ref_kn * sf
        return {
            "fuel_tonnes": distance_km * fuel_per_km,
            "speed_knots": speed_kn,
            "transit_hours": distance_km / (1.852 * speed_kn),
        }


class IcebergDriftPredictor:
    """Two-stage iceberg drift: a moving/stationary gate, then u/v regressors.

    Most tracked bergs are grounded or fast-locked on any given day, so a single
    regressor is dragged toward zero. Gating first lets the regressors fit the
    bergs that are actually moving.
    """

    kind = "trained"
    name = "Iceberg drift predictor"
    source = "US National Ice Center iceberg fixes"
    requires = ["numpy", "scikit-learn", "app.model_artifacts on sys.path"]
    output = "24 h displacement (u, v) in m/s, and whether the berg is moving"

    def __init__(self, clf, reg, features, metrics=None):
        self.clf = clf
        self.reg = reg
        self.features = list(features)
        self.metrics = metrics or {}

    def predict(self, X) -> dict[str, Any]:
        """X: array or DataFrame with columns matching `self.features`."""
        arr = np.asarray(X[self.features] if hasattr(X, "columns") else X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        moving = self.clf.predict(arr).astype(bool)
        u = np.zeros(len(arr))
        v = np.zeros(len(arr))
        if moving.any():
            u[moving] = self.reg["u_berg"].predict(arr[moving])
            v[moving] = self.reg["v_berg"].predict(arr[moving])
        return {
            "moving": moving,
            "u_ms": u,
            "v_ms": v,
            "drift_km_24h": np.hypot(u, v) * 86400.0 / 1000.0,
        }


class SeaIceForecastPredictor:
    """U-Net + ConvLSTM next-day sea-ice concentration.

    Holds the trained weights and the normalisation statistics. The architecture
    itself is rebuilt from seaice_forecast on first use, so that package must be
    importable - the same way an sklearn pickle needs sklearn.
    """

    kind = "trained"
    name = "Antarctic sea-ice forecaster"
    source = "NSIDC CDR v6, Southern Hemisphere 25 km"
    requires = ["numpy", "torch", "app.model_artifacts + seaice_forecast on sys.path"]
    output = "next-day sea-ice concentration, 332x316 grid in [0, 1]"
    variables = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]

    def __init__(self, state_dict, arch: dict, norm: dict, mask, metrics=None):
        self.state_dict = state_dict
        self.arch = dict(arch)
        self.norm = dict(norm)
        self.mask = mask
        self.metrics = metrics or {}
        self._model = None

    def _build(self):
        if self._model is not None:
            return self._model
        import torch
        from seaice_forecast.models.unet_convlstm import UNetConvLSTM
        m = UNetConvLSTM(**self.arch)
        m.load_state_dict(self.state_dict)
        m.eval()
        self._model = m
        return m

    def predict(self, history):
        """history: [7, 7, H, W] of raw daily fields -> [H, W] concentration."""
        import torch
        m = self._build()
        x = np.asarray(history, dtype=np.float32).copy()
        for c, v in enumerate(self.variables):
            if v != "sic" and v in self.norm and self.norm[v]["std"] > 1e-6:
                x[:, c] = (x[:, c] - self.norm[v]["mean"]) / self.norm[v]["std"]
        x[:, 1:, self.mask == 0] = 0.0
        with torch.no_grad():
            out = m(torch.from_numpy(x).unsqueeze(0).float()).squeeze().numpy()
        out = np.clip(out, 0.0, 1.0)
        out[self.mask == 0] = 0.0
        return out
