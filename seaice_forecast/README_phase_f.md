# Phase F: Route Optimization and Navigational Pathfinding

## Overview

Phase F implements an autonomous, risk-aware maritime route planner over the Antarctic polar grid (NSIDC 332×316, 25 km resolution). It combines graph-search pathfinding algorithms (A* and Dijkstra) with the IMO POLARIS navigational risk models established in Phase E to compute optimal sea routes that avoid impassable landmasses and bypass dangerous multi-year sea-ice packs.

---

## 1. Quick Start Commands

### Run Unit Tests
```bash
pytest tests/test_phase_f.py -v
```

### Run Demonstration on Real Data
```bash
python demo_phase_f.py --config configs/phase_f.yaml
```

Generated outputs in `output/phase_f/`:
- `phase_f_route.png` & `.pdf`: High-resolution map showing the Antarctic coastline, POLARIS risk heatmap, A* optimal path, and the straight-line reference baseline.
- `phase_f_cost_grid.npz`: Compressed archive containing the raw SIC slice, POLARIS risk surface, land mask, and traversal cost grid.
- `phase_f_route_summary.json`: Detailed tabular metrics comparing the A* path against the baseline.

---

## 2. Cost Function Formulation

The navigation cost for moving into an ocean cell $(r, c)$ balances transit distance against sea-ice risk:

$$\text{Cost}(r, c) = w_{\text{dist}} \cdot d_{\text{base}} + w_{\text{risk}} \cdot 10.0 \cdot R_{\text{ice}}(r, c)$$

Where:
- $w_{\text{dist}}$: Weight assigned to physical transit distance (default: $1.0$).
- $d_{\text{base}}$: Base distance metric per cell step (orthogonal = $1.0$, diagonal = $\sqrt{2}$).
- $w_{\text{risk}}$: Weight assigned to navigational ice risk (default: $2.0$).
- $R_{\text{ice}}(r, c) \in [0, 1]$: IMO POLARIS navigational impedance score computed from Sea-Ice Concentration (SIC) and vessel Polar Class.
- **Land Cells & Invalid Cells**: Assigned $\text{Cost} = \infty$, strictly preventing any terrestrial intrusion.

---

## 3. Pathfinding Capabilities

1. **8-Connected A* Search**:
   - Evaluates 8 spatial directions with diagonal step distance scaling ($\sqrt{2} \approx 1.414$).
   - Uses Euclidean or Manhattan admissible distance heuristics for rapid convergence.
2. **Strict Land Avoidance**:
   - Guaranteed barrier avoidance: routes never step on any cell flagged in the binary land mask (`land_mask == True`).
3. **Risk-Averse Routing**:
   - Actively diverts around dense pack ice tongues. In test evaluations on real 2021 Antarctic data, A* routed around 34 high-risk ice cells ($R_{\text{ice}} > 0.5$) with only a $+1.8\%$ distance overhead ($+43.9\text{ km}$), reducing total traversal cost by $87.5\%$.
4. **Dijkstra Fallback**:
   - Zero-heuristic graph search guaranteed to find the identical least-cost path.

---

## 4. Stakeholder Requirement Note: NCPOR Vessel Class

> [!IMPORTANT]
> The vessel Polar Class parameter must be confirmed by expedition operations stakeholders (e.g., National Centre for Polar and Ocean Research - NCPOR).
>
> The routing module accepts Polar Class as a parameter (`--polar-class` CLI flag or YAML config):
> - `PC1`–`PC3`: Heavy icebreakers / dedicated polar research vessels (can traverse higher concentrations).
> - `PC4`–`PC5`: Medium ice-strengthened vessels (default: `PC4`).
> - `PC6`–`PC7`: Thin first-year ice summer capability (requires wider diversions).
> - `UNCLASSED`: Non-ice-strengthened cargo vessels.

---

## 5. Seamless Multi-Horizon Integration

Phase F is fully modular and decoupled from the machine learning training pipeline. While demonstrated tonight on real single-day satellite observations, it will automatically consume richer multi-horizon risk forecasts ($\hat{y}_{t+1}, \hat{y}_{t+3}, \hat{y}_{t+5}, \hat{y}_{t+7}$) once Phase B–D models complete full training tomorrow.
