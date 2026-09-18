#!/usr/bin/env python3
"""Compare a freshly computed evaluation against the committed one.

check_claims.py proves the documents agree with the JSON. This proves the JSON
is what the model actually produces - without it, a stale or hand-edited result
file would pass every other check in the repository.

    python scripts/compare_recomputed.py <fresh.json> [committed.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "seaice_forecast/output/evaluation/rollout_comparison.json"
TOL = 0.002          # sampling differs slightly between runs; this is generous


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare_recomputed.py <fresh.json> [committed.json]")
        return 2
    fresh_p = Path(sys.argv[1])
    stored_p = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT

    if not fresh_p.exists():
        print(f"  recomputation produced nothing at {fresh_p}")
        return 1

    fresh = json.loads(fresh_p.read_text())["horizons"]
    stored = json.loads(stored_p.read_text())["horizons"]

    bad = 0
    for lead in fresh:
        if lead not in stored:
            continue
        for key in ("model", "persistence", "climatology"):
            a = fresh[lead][key]["mae_ice"]
            b = stored[lead][key]["mae_ice"]
            ok = abs(a - b) <= TOL
            print(f"  {'ok  ' if ok else 'FAIL'}  {lead} {key:<12} "
                  f"recomputed={a:.4f}  committed={b:.4f}")
            bad += not ok
    if bad:
        print(f"\n  {bad} figure(s) differ by more than {TOL} - the committed "
              f"results are not what the checkpoint produces.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
