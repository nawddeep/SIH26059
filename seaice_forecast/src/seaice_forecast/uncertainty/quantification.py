"""
Phase E: Empirical Error-vs-Lead-Time Uncertainty Quantification Module.

Quantifies forecast uncertainty as a function of lead time:
- Computes per-horizon error statistics (mean, std, quantiles).
- Fits parametric expanding-uncertainty curves (linear, exponential, polynomial).
- Generates publication-ready diagnostic plots with confidence intervals.
- Generates JSON summary reports for downstream route planning and operational risk.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_horizon_error_stats(
    errors: Union[Dict[int, np.ndarray], Iterable[Tuple[int, float]], np.ndarray],
    horizons: Optional[Union[List[int], np.ndarray]] = None,
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-horizon error statistics (mean, std, quantiles).

    Args:
        errors: Either:
            - Dict mapping horizon (int) -> array-like of error values, OR
            - 1D array of error values (requires `horizons` parameter of same length), OR
            - Iterable of (horizon, error) tuples.
        horizons: Corresponding horizon array if `errors` is a 1D array.
        quantiles: Tuple of quantile values to evaluate in [0, 1].

    Returns:
        Dict mapping horizon (int) -> statistics dictionary:
            'mean', 'std', 'median', 'rmse', 'count', plus 'q{quantile*100:02d}'.
    """
    # Normalize input into dict of horizon -> np.ndarray
    horizon_groups: Dict[int, List[float]] = {}

    if isinstance(errors, dict):
        for h, errs in errors.items():
            arr = np.asarray(errs, dtype=np.float64).flatten()
            arr = arr[np.isfinite(arr)]
            horizon_groups[int(h)] = arr.tolist()
    elif horizons is not None:
        err_arr = np.asarray(errors, dtype=np.float64).flatten()
        h_arr = np.asarray(horizons, dtype=np.int64).flatten()
        if len(err_arr) != len(h_arr):
            raise ValueError(f"Length mismatch: errors ({len(err_arr)}) vs horizons ({len(h_arr)})")
        for h, e in zip(h_arr, err_arr):
            if np.isfinite(e):
                horizon_groups.setdefault(int(h), []).append(float(e))
    else:
        # Iterable of (horizon, error) pairs
        for h, e in errors:
            if np.isfinite(e):
                horizon_groups.setdefault(int(h), []).append(float(e))

    stats: Dict[int, Dict[str, float]] = {}
    for h in sorted(horizon_groups.keys()):
        vals = np.array(horizon_groups[h], dtype=np.float64)
        if len(vals) == 0:
            continue

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rmse_val = float(np.sqrt(np.mean(vals**2)))

        h_stats = {
            "mean": mean_val,
            "std": std_val,
            "rmse": rmse_val,
            "count": int(len(vals)),
        }

        # Quantile computation
        for q in quantiles:
            q_label = f"q{int(round(q * 100)):02d}"
            h_stats[q_label] = float(np.percentile(vals, q * 100))

        h_stats["median"] = h_stats.get("q50", float(np.median(vals)))
        stats[h] = h_stats

    return stats


def fit_uncertainty_curve(
    horizons: Union[List[int], np.ndarray],
    stds: Union[List[float], np.ndarray],
    method: str = "linear",
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Fit an expanding-uncertainty curve to empirical std vs forecast horizon.

    Args:
        horizons: Forecast horizons (e.g., [1, 3, 5, 7]).
        stds: Empirical standard deviations or uncertainty values per horizon.
        method: Fitting method ('linear', 'exponential', 'polynomial').

    Returns:
        Callable `predict_uncertainty(horizon)` returning estimated standard deviation.
    """
    h_arr = np.asarray(horizons, dtype=np.float64)
    s_arr = np.asarray(stds, dtype=np.float64)

    if len(h_arr) < 2:
        val = float(s_arr[0]) if len(s_arr) > 0 else 0.0
        return lambda h: np.full_like(np.asarray(h, dtype=np.float64), val)

    method = method.lower()

    if method == "linear":
        # Fit std(h) = slope * h + intercept (slope constrained >= 0)
        coeffs = np.polyfit(h_arr, s_arr, deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        slope = max(0.0, slope)  # Uncertainty must not decrease with time

        def predict_linear(h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            h_in = np.asarray(h, dtype=np.float64)
            pred = slope * h_in + intercept
            # Ensure strictly positive uncertainty
            return np.maximum(pred, 1e-6)

        return predict_linear

    elif method == "exponential":
        # Fit std(h) = a * exp(b * h) => log(std) = log(a) + b * h
        s_safe = np.maximum(s_arr, 1e-6)
        log_s = np.log(s_safe)
        coeffs = np.polyfit(h_arr, log_s, deg=1)
        b, log_a = coeffs[0], coeffs[1]
        a = np.exp(log_a)
        b = max(0.0, b)  # Monotonically non-decreasing variance growth

        def predict_exp(h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            h_in = np.asarray(h, dtype=np.float64)
            pred = a * np.exp(b * h_in)
            return np.maximum(pred, 1e-6)

        return predict_exp

    elif method == "polynomial":
        # Quadratic fit std(h) = a*h^2 + b*h + c
        deg = min(2, len(h_arr) - 1)
        coeffs = np.polyfit(h_arr, s_arr, deg=deg)
        poly = np.poly1d(coeffs)

        def predict_poly(h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            h_in = np.asarray(h, dtype=np.float64)
            pred = poly(h_in)
            return np.maximum(pred, 1e-6)

        return predict_poly

    else:
        raise ValueError(f"Unknown method '{method}'. Supported: 'linear', 'exponential', 'polynomial'.")


def plot_uncertainty_curve(
    horizons: Union[List[int], np.ndarray],
    stds: Union[List[float], np.ndarray],
    fit_fn: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    output_path_png: Union[str, Path],
    output_path_pdf: Optional[Union[str, Path]] = None,
    stats_dict: Optional[Dict[int, Dict[str, float]]] = None,
    is_smoke_test: bool = True,
    title_suffix: str = "",
):
    """
    Plot empirical error vs. lead time with expanding uncertainty confidence bands.

    Args:
        horizons: Array of integer horizons (e.g., [1, 3, 5, 7]).
        stds: Empirical standard deviations.
        fit_fn: Fitted predictive uncertainty function.
        output_path_png: Output PNG filepath.
        output_path_pdf: Optional output PDF filepath.
        stats_dict: Optional per-horizon statistics containing quantiles.
        is_smoke_test: True if running on synthetic / demo data.
        title_suffix: Additional text for the plot title.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    h_arr = np.asarray(horizons, dtype=np.float64)
    s_arr = np.asarray(stds, dtype=np.float64)

    # Dense curve for fitted model
    h_dense = np.linspace(max(0.5, float(h_arr.min())), float(h_arr.max()) + 0.5, 200)
    std_dense = fit_fn(h_dense)

    # Plot empirical points
    ax.scatter(h_arr, s_arr, color="#1f77b4", s=80, zorder=5, label=r"Empirical Std ($\sigma_h$)")

    # Plot fitted curve
    ax.plot(h_dense, std_dense, color="#d62728", linewidth=2.5, linestyle="-", label=r"Fitted Uncertainty Curve $\sigma(h)$")

    # If full quantile stats available, plot quantile intervals
    if stats_dict is not None:
        means = np.array([stats_dict[h]["mean"] for h in horizons])
        q05 = np.array([stats_dict[h].get("q05", means[i] - 1.96 * s_arr[i]) for i, h in enumerate(horizons)])
        q95 = np.array([stats_dict[h].get("q95", means[i] + 1.96 * s_arr[i]) for i, h in enumerate(horizons)])

        ax.fill_between(
            h_arr,
            np.maximum(0, q05),
            q95,
            color="#1f77b4",
            alpha=0.2,
            label="90% Empirical Error Interval (5th–95th %ile)",
        )

    # Fill +/- 1 sigma and +/- 2 sigma around fitted uncertainty
    ax.fill_between(
        h_dense,
        0,
        std_dense,
        color="#d62728",
        alpha=0.15,
        label=r"$\pm 1\sigma$ Expanding Uncertainty Band",
    )
    ax.fill_between(
        h_dense,
        std_dense,
        2 * std_dense,
        color="#ff7f0e",
        alpha=0.08,
        label=r"$\pm 2\sigma$ High-Uncertainty Boundary",
    )

    title = "Forecast Uncertainty vs. Lead Time"
    if is_smoke_test:
        title = "[SMOKE TEST ONLY] " + title
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(r"Forecast Lead Time $h$ (days)", fontsize=12)
    ax.set_ylabel(r"Error Standard Deviation $\sigma$ (SIC [0, 1])", fontsize=12)
    ax.set_xticks(h_arr)
    ax.set_xticklabels([f"+{int(h)}d" for h in h_arr])
    ax.set_xlim(min(h_dense) - 0.2, max(h_dense) + 0.2)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    out_png = Path(output_path_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if output_path_pdf:
        out_pdf = Path(output_path_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)


def generate_uncertainty_report(
    error_pairs: Union[Dict[int, np.ndarray], Iterable[Tuple[int, float]]],
    output_dir: Union[str, Path],
    method: str = "linear",
    is_smoke_test: bool = True,
) -> Dict[str, Union[dict, str]]:
    """
    Convenience wrapper: computes horizon stats, fits curve, saves plot and JSON report.

    Args:
        error_pairs: Iterable of (horizon, error) pairs or dict mapping horizon -> errors.
        output_dir: Directory to save generated artifacts.
        method: Uncertainty curve fitting method.
        is_smoke_test: Flag indicating synthetic/smoke-test execution.

    Returns:
        Dictionary containing summary statistics, curve parameters, and filepaths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_horizon_error_stats(error_pairs)
    horizons = sorted(stats.keys())
    stds = [stats[h]["std"] for h in horizons]

    fit_fn = fit_uncertainty_curve(horizons, stds, method=method)

    prefix = "SMOKE_TEST_ONLY_" if is_smoke_test else ""
    png_path = out_dir / f"{prefix}uncertainty_curve.png"
    pdf_path = out_dir / f"{prefix}uncertainty_curve.pdf"

    plot_uncertainty_curve(
        horizons=horizons,
        stds=stds,
        fit_fn=fit_fn,
        output_path_png=png_path,
        output_path_pdf=pdf_path,
        stats_dict=stats,
        is_smoke_test=is_smoke_test,
    )

    # Compute predicted uncertainty values at target horizons
    fitted_uncertainties = {int(h): float(fit_fn(h)) for h in horizons}

    report = {
        "status": "SMOKE_TEST_ONLY" if is_smoke_test else "TRAINED_EVALUATION",
        "method": method,
        "horizons": horizons,
        "empirical_stats": stats,
        "fitted_uncertainty_std": fitted_uncertainties,
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
    }

    json_path = out_dir / f"{prefix}uncertainty_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
