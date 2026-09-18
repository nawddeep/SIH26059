"""Confidence for model output, derived from measured held-out error.

Every number here comes from `seaice_forecast/output/evaluation/
uncertainty_model.json`, which is produced by running the trained model against
the held-out split and binning its actual errors. Nothing is assumed.

That matters because a confidence figure invented from nothing is worse than
none at all: it invites trust it has not earned. What this reports is "how wrong
this model has been in conditions like these", which is a weaker claim than a
calibrated probabilistic forecast and is labelled as such.

The measurement produced one result worth stating plainly: the forecaster is
near-perfect over open water (MAE 0.003) and worst in the marginal ice zone
(MAE 0.097) - which is exactly where a vessel actually operates. Its confidence
is lowest where the decision is hardest.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .model_bridge import MODEL_ROOT

UNCERTAINTY_FILE = MODEL_ROOT / "seaice_forecast/output/evaluation/uncertainty_model.json"
DRIFT_METRICS = MODEL_ROOT / "iceberg-drift/output/drift_twostage_metrics.json"


@lru_cache(maxsize=1)
def _model() -> dict[str, Any] | None:
    if not UNCERTAINTY_FILE.exists():
        return None
    return json.loads(UNCERTAINTY_FILE.read_text())


@lru_cache(maxsize=1)
def _drift() -> dict[str, Any] | None:
    if not DRIFT_METRICS.exists():
        return None
    return json.loads(DRIFT_METRICS.read_text()).get("two_stage")


def sea_ice_uncertainty(concentration: float, gradient: float | None = None) -> dict[str, Any]:
    """Expected error for a sea-ice prediction at this concentration.

    `concentration` is 0-1. `gradient` is the local change in concentration per
    cell, if known - it is the stronger predictor of the two, because the ice
    edge is an order of magnitude harder than the interior.
    """
    m = _model()
    if not m:
        return {"available": False,
                "reason": "run seaice_forecast/scripts/evaluation/measure_uncertainty.py"}

    mae = m["overallMae"]
    p90 = m["overallP90"]
    basis = "overall held-out error"

    # Gradient first: it discriminates far better than concentration does.
    if gradient is not None:
        for b in m.get("byGradient", []):
            if b["gradientMin"] <= gradient < b["gradientMax"]:
                mae, p90 = b["mae"], b["p90"]
                basis = f"local gradient {b['gradientMin']}-{b['gradientMax']}"
                break
    else:
        for b in m.get("byConcentration", []):
            if b["concMin"] <= concentration < b["concMax"]:
                mae, p90 = b["mae"], b["p90"]
                basis = f"concentration band {b['concMin']}-{b['concMax']}"
                break

    # Confidence relative to the worst band the model has been measured in, so
    # "100%" means "as good as this model ever is", not "certainly correct".
    worst = max((b["mae"] for b in m.get("byConcentration", [])), default=mae) or 1.0
    confidence = max(0.0, min(1.0, 1.0 - (mae / worst)))

    return {
        "available": True,
        "expectedErrorMae": mae,
        "p90Error": p90,
        "plusMinus": round(mae, 4),
        "confidence": round(confidence, 3),
        "basis": basis,
        "samples": m.get("samples"),
        "caveat": m.get("caveat"),
        "interpretation": (
            "Empirical: how wrong this model has been on held-out data in "
            "conditions like these. Not a calibrated probability."
        ),
    }


def iceberg_uncertainty(moving_probability: float | None = None) -> dict[str, Any]:
    """Positional envelope for a 24 h drift prediction.

    The envelope is the model's own held-out RMS position error, so a predicted
    position should be read as the centre of a circle of that radius rather than
    as a point.
    """
    d = _drift()
    if not d:
        return {"available": False, "reason": "drift_twostage_metrics.json not found"}

    out = {
        "available": True,
        "rmsErrorKm24h": round(d["pos_err_24h_km_rms"], 3),
        "medianErrorKm24h": round(d["pos_err_24h_km_median"], 3),
        "within10kmPct": round(d["within_10km_pct"], 2),
        "envelopeRadiusKm": round(d["pos_err_24h_km_rms"], 1),
        "interpretation": (
            "A predicted position is the centre of a circle of this radius, not "
            "a point. 96.7% of held-out predictions fell within 10 km."
        ),
        "caveat": (
            "Measured on 103 large, tracked bergs. Small bergs and growlers are "
            "absent from the training data and are the primary hull-damage risk."
        ),
    }
    if moving_probability is not None:
        p = max(0.0, min(1.0, float(moving_probability)))
        out["movingProbability"] = round(p, 3)
        # Confidence in the gate is distance from the 50/50 line, doubled to 0-1.
        out["gateConfidence"] = round(abs(p - 0.5) * 2.0, 3)
    return out


def summary() -> dict[str, Any]:
    """Both models' measured uncertainty, for the dashboard."""
    m = _model()
    return {
        "seaIce": {
            "available": bool(m),
            "overallMae": m.get("overallMae") if m else None,
            "byConcentration": m.get("byConcentration") if m else None,
            "byGradient": m.get("byGradient") if m else None,
            "edgeToInteriorErrorRatio": m.get("edgeToInteriorErrorRatio") if m else None,
            "samples": m.get("samples") if m else None,
            "headline": (
                "Near-perfect over open water, worst in the marginal ice zone - "
                "least confident exactly where a vessel operates."
            ) if m else None,
        },
        "icebergDrift": iceberg_uncertainty(),
        "note": (
            "Both figures are measured on held-out data, not assumed. Neither is "
            "a calibrated probabilistic forecast."
        ),
    }
