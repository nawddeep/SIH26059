#!/usr/bin/env python3
"""Fail if any performance number in README.md disagrees with its source file.

Performance figures were previously typed into README.md, OPEN_PROBLEM.md and
two JSON files by hand. They drifted, and the drift was not cosmetic: one table
supported a conclusion ("the model beats persistence at +5d and +7d") that a
larger sample overturned. Numbers maintained in several places will always
diverge eventually, so this makes divergence a build failure instead of a thing
someone notices later.

The generated JSON under output/evaluation/ is authoritative. README.md and
OPEN_PROBLEM.md are checked against it.

    python scripts/check_claims.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLLOUT = ROOT / "seaice_forecast/output/evaluation/rollout_comparison.json"
BASELINE = ROOT / "seaice_forecast/output/evaluation/baseline_comparison.json"
DRIFT = ROOT / "iceberg-drift/output/drift_twostage_metrics.json"
README = ROOT / "README.md"
OPEN_PROBLEM = ROOT / "seaice_forecast/OPEN_PROBLEM.md"

TOL = 0.0006          # tables round to 3 decimals; allow half a unit in the last place

failures: list[str] = []
checks = 0


def check(label: str, expected: float, found: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    ok = abs(expected - found) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<44} source={expected:<8.4f} doc={found:.4f}")
    if not ok:
        failures.append(f"{label}: source says {expected:.4f}, document says {found:.4f}")


def md_rows(text: str, lead: str) -> list[list[str]]:
    """Every markdown table row whose first cell is `lead` (e.g. '+1d')."""
    out = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == lead:
            out.append(cells)
    return out


def num(cell: str) -> float | None:
    m = re.search(r"-?\d+\.\d+", cell)
    return float(m.group()) if m else None


def main() -> int:
    if not ROLLOUT.exists():
        print("!! rollout_comparison.json missing - run scripts/evaluation/rollout_comparison.py")
        return 1

    rollout = json.loads(ROLLOUT.read_text())["horizons"]
    readme = README.read_text()

    print("README.md multi-day table vs rollout_comparison.json")
    for lead, block in rollout.items():
        rows = md_rows(readme, lead)
        if not rows:
            continue
        # columns: lead | model | persistence | climatology | (n)
        row = rows[0]
        for idx, key in ((1, "model"), (2, "persistence"), (3, "climatology")):
            if idx < len(row):
                found = num(row[idx])
                if found is not None:
                    check(f"README {lead} {key}", block[key]["mae_ice"], found)

    if OPEN_PROBLEM.exists() and BASELINE.exists():
        # OPEN_PROBLEM.md carries two tables and they come from different
        # scripts: the +1d table is the single-pass baseline (the only horizon
        # where that method is valid), the multi-day table is the rollout. The
        # +1d lead therefore appears twice, once per source, and the order of
        # appearance is what tells them apart.
        print("\nOPEN_PROBLEM.md tables vs their source files")
        baseline = json.loads(BASELINE.read_text())
        op = OPEN_PROBLEM.read_text()

        def check_row(label, row, block):
            # columns: lead | persistence | climatology | model
            for idx, key in ((1, "persistence"), (2, "climatology"), (3, "model")):
                if idx < len(row) and block.get(key):
                    found = num(row[idx])
                    if found is not None:
                        check(f"{label} {key}", block[key]["mae_ice"], found)

        rows_1d = md_rows(op, "+1d")
        if rows_1d and baseline.get("+1d"):
            check_row("OPEN_PROBLEM +1d (baseline)", rows_1d[0], baseline["+1d"])
        if len(rows_1d) > 1 and rollout.get("+1d"):
            check_row("OPEN_PROBLEM +1d (rollout)", rows_1d[1], rollout["+1d"])

        for lead, block in rollout.items():
            if lead == "+1d":
                continue
            rows = md_rows(op, lead)
            if rows:
                check_row(f"OPEN_PROBLEM {lead} (rollout)", rows[0], block)

    if DRIFT.exists():
        print("\nREADME.md drift figure vs drift_twostage_metrics.json")
        two = json.loads(DRIFT.read_text())["two_stage"]
        m = re.search(r"\*\*([\d.]+) km RMS\*\*", readme)
        if m:
            check("README drift RMS km", two["pos_err_24h_km_rms"], float(m.group(1)), tol=0.006)
        m = re.search(r"([\d.]+)% within 10 km", readme)
        if m:
            check("README drift within-10km %", two["within_10km_pct"], float(m.group(1)), tol=0.06)

    print(f"\n{checks} claims checked")
    if failures:
        print(f"{len(failures)} MISMATCHED:")
        for f in failures:
            print(f"  - {f}")
        print("\nThe JSON under output/evaluation/ is authoritative. Update the")
        print("document to match it, or regenerate the JSON - never hand-edit both.")
        return 1
    print("every documented figure matches its source file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
