# Phase E: Uncertainty Quantification & IMO POLARIS Navigational Risk Function

## Overview

Phase E establishes two core operational capabilities for the Antarctic Sea-Ice Forecasting system:
1. **Empirical Error-vs-Lead-Time Uncertainty Quantification**: Parametric expanding-variance models that capture forecast uncertainty growth from $+1\text{d}$ to $+7\text{d}$.
2. **IMO POLARIS-Based Navigational Risk Function ($R_{\text{ice}} = f(\text{SIC}, \text{PolarClass})$)**: Standalone, mathematically sound translation of forecasted sea-ice concentration into operational vessel risk, independent of any trained machine learning model.

---

## 1. Quick Start Commands

### Run Unit Tests
```bash
pytest tests/test_phase_e.py -v
```

### Run Phase E Demonstration
```bash
python demo_phase_e.py --config configs/phase_e.yaml
```

Generated outputs in `output/phase_e/`:
- `phase_e_uncertainty_demo.png` & `.pdf`: 90% confidence intervals and expanding $\pm 1\sigma$ and $\pm 2\sigma$ uncertainty bands.
- `phase_e_risk_vs_sic.png` & `.pdf`: IMO POLARIS risk curves across Polar Classes (PC1, PC3, PC5, PC7, Unclassed).
- `phase_e_risk_table.json`: Numerical risk values at key operational thresholds ($0.0, 0.15, 0.5, 0.85, 1.0$).
- `SMOKE_TEST_ONLY_uncertainty_report.json`: Per-horizon empirical error statistics (mean, std, RMSE, percentiles).

---

## 2. Uncertainty Quantification Methodology

Forecast uncertainty naturally expands as lead time increases. The uncertainty module ([`src/seaice_forecast/uncertainty/quantification.py`](src/seaice_forecast/uncertainty/quantification.py)) models this dispersion:

### Formulations
1. **Empirical Statistics**: Computes mean error, standard deviation ($\sigma_h$), RMSE, and quantiles ($5\text{th}, 25\text{th}, 50\text{th}, 75\text{th}, 95\text{th}$) per lead time $h \in \{1, 3, 5, 7\}$.
2. **Expanding Uncertainty Curves**:
   - **Linear Model**: $\sigma(h) = a \cdot h + b \quad (a \ge 0)$
   - **Exponential Model**: $\sigma(h) = a \cdot e^{b \cdot h} \quad (b \ge 0)$
   - **Polynomial Model**: $\sigma(h) = a \cdot h^2 + b \cdot h + c$

### Integration with Phase D Multi-Horizon Models
The uncertainty module is designed to directly ingest the multi-horizon test residuals from tomorrow's Phase D trained model:
```python
from seaice_forecast.uncertainty import generate_uncertainty_report

# error_dict maps horizon -> empirical test residuals (test_preds - test_targets)
report = generate_uncertainty_report(
    error_pairs=error_dict,
    output_dir="output/phase_e",
    method="linear",
    is_smoke_test=False
)
```

---

## 3. IMO POLARIS Navigational Risk Function

### IMO POLARIS Context
The International Maritime Organization (IMO) Polar Operational Limit Assessment Risk Indexing System (**POLARIS**, IMO Circular MSC.1/Circ.1519) assesses ship operational limits in ice. In POLARIS, the Risk Index Outcome (RIO) is computed as:

$$\text{RIO} = \sum_{i} \left( C_i \times \text{RIV}_i \right)$$

where $C_i$ is ice concentration in tenths and $\text{RIV}_i$ is the Risk Index Value assigned based on the vessel's Polar Class and ice type/thickness.

### Functional Approximation: $R_{\text{ice}} = f(\text{SIC}, \text{PolarClass})$
In downstream voyage routing and cost surface computation, sea-ice concentration is translated into an impedance metric $R_{\text{ice}} \in [0, 1]$:

$$
R_{\text{ice}}(\text{SIC}) =
\begin{cases}
R_{\text{open}} + 0.005 \left(\frac{\text{SIC}}{T_{\text{open}}}\right)^2, & \text{SIC} < T_{\text{open}} \\
R_{\text{base}} + (R_{\text{max}} - R_{\text{base}}) \left(\frac{\text{SIC} - T_{\text{open}}}{1.0 - T_{\text{open}}}\right)^{\gamma(\text{PC})}, & \text{SIC} \ge T_{\text{open}}
\end{cases}
$$

### Operational Properties
1. **Open-Water Threshold ($T_{\text{open}} = 0.15$)**: Water with $\text{SIC} < 0.15$ is navigated with near-zero ice impedance ($R_{\text{ice}} \le 0.005$).
2. **Strict Monotonicity**: Risk increases strictly as concentration rises past the ice edge.
3. **Pack Ice Asymptote**: Approaching full consolidation ($\text{SIC} \to 1.0$), risk approaches $1.0$ (maximum navigation impediment).
4. **Polar Class Hierarchy**: Stronger icebreaker hulls (e.g., PC1) have higher curvature exponents ($\gamma = 2.8$), allowing them to navigate moderate concentrations with minimal risk, while lower ice classes (e.g., PC7, $\gamma = 0.8$) face steep risk escalation immediately past the ice edge.

| SIC | PC1 (Heavy Icebreaker) | PC3 (Polar Research Vessel) | PC5 (Medium Icebreaker) | PC7 (Thin First-Year Ice) | Unclassed (Open Water Hull) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **0.15** | 0.0050 | 0.0050 | 0.0050 | 0.0050 | 0.0050 |
| **0.50** | 0.0880 | 0.1737 | 0.2923 | 0.4943 | 0.6435 |
| **0.85** | 0.5827 | 0.6798 | 0.7632 | 0.8569 | 0.9079 |
| **1.00** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Stakeholder Requirement Note: NCPOR Vessel Class
> [!IMPORTANT]
> The specific vessel Polar Class for the National Centre for Polar and Ocean Research (NCPOR) expedition vessel must be confirmed by expedition operations stakeholders (e.g., PC3/PC4 for dedicated ice-strengthened polar research vessels, or PC5/PC7 for chartered summer resupply vessels).
>
> The risk module intentionally accepts `polar_class` as a dynamic parameter:
> ```python
> from seaice_forecast.risk import risk_ice, risk_ice_array
>
> # Query scalar risk
> r = risk_ice(sic=0.45, polar_class="PC4")
>
> # Query full 2D spatial grid
> grid_risk = risk_ice_array(sic_grid_2d, polar_class="PC3")
> ```

---

## 4. How to Plug in Exact Discrete POLARIS RIV Tables

For operations requiring exact integer RIO scoring instead of the continuous impedance surface:
1. Define the RIV table mapping vessel class and ice type in `configs/phase_e.yaml`:
   ```yaml
   polaris_risk:
     ice_type_riv_table:
       PC4:
         ice_free: 3
         open_water: 3
         thin_first_year: 2
         medium_first_year: 1
         thick_first_year: 0
         second_year: -1
         multi_year: -2
   ```
2. Pass the custom table into `PolarisRiskConfig`:
   ```python
   from seaice_forecast.risk import PolarisRiskConfig, risk_ice

   custom_cfg = PolarisRiskConfig(open_water_threshold=0.15)
   risk = risk_ice(sic=0.7, polar_class="PC4", config=custom_cfg)
   ```
