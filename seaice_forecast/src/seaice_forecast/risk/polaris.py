"""
Phase E: IMO POLARIS-based Sea-Ice Navigational Risk Function R_ice = f(SIC).

Translates Sea-Ice Concentration (SIC) into operational navigation risk / cost:
- Based on the principles of the IMO Polar Operational Limit Assessment Risk Indexing System (POLARIS, MSC.1/Circ.1519).
- Parameterized by vessel Polar Class (PC1 through PC7, or Unclassed).
- Independent of any trained machine learning model.
- Exposes configurable thresholds, exponents, and operational safety factors.

Stakeholder Note:
    The specific vessel Polar Class for the National Centre for Polar and Ocean
    Research (NCPOR) research expedition vessel must be confirmed by operations
    stakeholders (e.g., PC3/PC4 for dedicated ice-strengthened polar research vessels,
    or PC5/PC7 for chartered summer supply vessels). This module accepts the Polar
    Class as a parameter and avoids hardcoding any specific vessel classification.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import numpy as np


@dataclass
class PolarisRiskConfig:
    """
    Configuration for IMO POLARIS-based sea-ice navigational risk function.

    Attributes:
        open_water_threshold: SIC below which water is navigated as open water (default: 0.15).
        open_water_risk: Residual navigational risk in open water [0, 1] (default: 0.0).
        max_risk: Maximum navigational risk approaching full ice compaction (default: 1.0).
        class_exponents: Curvature exponents for Polar Classes PC1..PC7 & Unclassed.
            Larger exponent = slower risk escalation (higher ice-breaking capability).
        ice_type_riv_table: Optional IMO POLARIS Risk Index Values lookup by ice regime.
    """

    open_water_threshold: float = 0.15
    open_water_risk: float = 0.0
    max_risk: float = 1.0
    class_exponents: Dict[str, float] = field(
        default_factory=lambda: {
            "PC1": 2.8,  # Year-round polar operations in all polar waters
            "PC2": 2.4,  # Moderate multi-year ice conditions
            "PC3": 2.0,  # Second-year ice with multi-year inclusions
            "PC4": 1.7,  # Thick first-year ice
            "PC5": 1.4,  # Medium first-year ice
            "PC6": 1.1,  # Summer/autumn in medium first-year ice
            "PC7": 0.8,  # Summer/autumn in thin first-year ice
            "UNCLASSED": 0.5,  # Open-water hull / non-ice-strengthened
        }
    )


# Default global configuration instance
DEFAULT_POLARIS_CONFIG = PolarisRiskConfig()


def normalize_polar_class(polar_class: Union[int, str]) -> str:
    """
    Normalize Polar Class identifier into canonical string (e.g., 1 -> 'PC1', 'pc7' -> 'PC7').

    Args:
        polar_class: Integer 1-7 or string 'PC1'-'PC7', 'UNCLASSED'.

    Returns:
        Canonical uppercase string key.
    """
    if isinstance(polar_class, (int, np.integer)):
        if 1 <= polar_class <= 7:
            return f"PC{int(polar_class)}"
        elif polar_class > 7 or polar_class <= 0:
            return "UNCLASSED"
        else:
            return f"PC{int(polar_class)}"
    elif isinstance(polar_class, str):
        cleaned = polar_class.strip().upper()
        if cleaned.isdigit():
            val = int(cleaned)
            return f"PC{val}" if 1 <= val <= 7 else "UNCLASSED"
        if cleaned in {"PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "UNCLASSED"}:
            return cleaned
        # Strip prefix if e.g. "POLAR_CLASS_4"
        for pc in ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]:
            if pc in cleaned:
                return pc
        return "UNCLASSED"
    else:
        return "UNCLASSED"


def risk_ice(
    sic: float,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, PolarisRiskConfig]] = None,
) -> float:
    """
    Compute navigational risk R_ice as a function of Sea-Ice Concentration (SIC).

    Behavior:
    - For SIC < open_water_threshold (0.15): Risk is near-zero (negligible ice encounter).
    - For SIC >= open_water_threshold: Risk scales monotonically with SIC according to
      the vessel's Polar Class capability exponent.
    - As SIC -> 1.0: Risk asymptotically approaches max_risk (e.g., 1.0).
    - Lower Polar Class number (e.g., PC1 vs PC7) represents stronger icebreaker capability
      and results in lower risk for the same SIC.

    Args:
        sic: Sea-ice concentration, float in [0.0, 1.0].
        polar_class: Vessel Polar Class (1..7, "PC1".."PC7", or "UNCLASSED").
        config: Optional PolarisRiskConfig dataclass or dictionary.

    Returns:
        Navigation risk / impedance score in [0.0, 1.0].
    """
    # Parse configuration
    if config is None:
        cfg = DEFAULT_POLARIS_CONFIG
    elif isinstance(config, dict):
        cfg = PolarisRiskConfig(**config)
    elif isinstance(config, PolarisRiskConfig):
        cfg = config
    else:
        cfg = DEFAULT_POLARIS_CONFIG

    # Clip SIC to valid physical bounds [0, 1]
    sic_clamped = float(np.clip(sic, 0.0, 1.0))
    pc_key = normalize_polar_class(polar_class)
    exponent = cfg.class_exponents.get(pc_key, 1.0)

    t_open = cfg.open_water_threshold
    r_open = cfg.open_water_risk
    r_max = cfg.max_risk

    # Sub-threshold regime: open water (continuous quadratic ramp to avoid hard step)
    if sic_clamped < t_open:
        if t_open <= 0.0:
            return float(r_open)
        # Smooth transitional risk approaching threshold, remaining near zero
        ratio = sic_clamped / t_open
        sub_risk = r_open + 0.005 * (ratio**2)
        return float(np.clip(sub_risk, 0.0, r_max))

    # Ice pack regime: monotonic expanding risk
    # Normalized excess concentration s in [0, 1]
    span = 1.0 - t_open
    if span <= 1e-6:
        return float(r_max)

    s = (sic_clamped - t_open) / span
    # Power-law risk escalation based on Polar Class structural capacity
    # Add open-water transition offset (0.005) so the curve is strictly continuous
    r_base = 0.005
    scaled_risk = r_base + (r_max - r_base) * (s**exponent)

    return float(np.clip(scaled_risk, 0.0, r_max))


def risk_ice_array(
    sic_array: np.ndarray,
    polar_class: Union[int, str] = "PC4",
    config: Optional[Union[dict, PolarisRiskConfig]] = None,
) -> np.ndarray:
    """
    Vectorized navigational risk function R_ice = f(SIC) across 2D/3D grids.

    Args:
        sic_array: NumPy array of sea-ice concentration values in [0, 1].
        polar_class: Vessel Polar Class parameter.
        config: Optional PolarisRiskConfig configuration.

    Returns:
        NumPy array of risk values in [0.0, 1.0] matching input shape.
    """
    if config is None:
        cfg = DEFAULT_POLARIS_CONFIG
    elif isinstance(config, dict):
        cfg = PolarisRiskConfig(**config)
    elif isinstance(config, PolarisRiskConfig):
        cfg = config
    else:
        cfg = DEFAULT_POLARIS_CONFIG

    arr = np.asarray(sic_array, dtype=np.float64)
    clamped = np.clip(arr, 0.0, 1.0)

    pc_key = normalize_polar_class(polar_class)
    exponent = cfg.class_exponents.get(pc_key, 1.0)

    t_open = cfg.open_water_threshold
    r_open = cfg.open_water_risk
    r_max = cfg.max_risk

    out = np.zeros_like(clamped, dtype=np.float64)

    # Sub-threshold mask (open water)
    open_mask = clamped < t_open
    if t_open > 0.0:
        ratio = clamped[open_mask] / t_open
        out[open_mask] = r_open + 0.005 * (ratio**2)
    else:
        out[open_mask] = r_open

    # Pack ice mask
    ice_mask = ~open_mask
    span = max(1e-6, 1.0 - t_open)
    s = (clamped[ice_mask] - t_open) / span
    r_base = 0.005
    out[ice_mask] = r_base + (r_max - r_base) * (s**exponent)

    return np.clip(out, 0.0, r_max)
