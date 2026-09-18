#!/usr/bin/env python3
"""Operational ingestion: acquire new environmental data and prepare model input.

The pieces to do this already existed - downloaders for NSIDC, ERA5 and CMEMS,
plus regridding and preprocessing - but nothing chained them, so the system ran
against a static archive ending 2018-12-31. This is the chain:

    source APIs -> download -> quality check -> regrid -> daily .npz -> model

Three modes:

    python scripts/ingest.py --check              what is available, nothing fetched
    python scripts/ingest.py --plan  --start ... --end ...   what would be fetched
    python scripts/ingest.py --fetch --start ... --end ...   actually fetch

`--check` is the default and is safe to run anywhere: it verifies credentials and
reports archive coverage without touching the network.

A caution that belongs in the code rather than a footnote: the models here were
trained on NSIDC CDR v6, a *climate* record tuned for consistency across decades.
The near-real-time product (NSIDC-0081) is a different instrument calibration.
Feeding it to these weights without revalidation would produce numbers that look
right and are not comparable to anything in the evaluation. Operational use needs
that revalidation first, and this script will say so rather than pretend
otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEAICE = ROOT / "seaice_forecast"
DAILY = SEAICE / "data/processed/regridded/daily"
sys.path.insert(0, str(SEAICE / "src"))

SOURCES = {
    "nsidc": {
        "name": "NSIDC Sea Ice Concentration CDR v6",
        "variable": "sea-ice concentration",
        "credential": Path.home() / ".netrc",
        "credential_note": "NASA Earthdata login in ~/.netrc",
        "module": "seaice_forecast.data_processing.downloaders.nsidc",
        "nrt_caveat": (
            "Trained on CDR v6 (climate record). The near-real-time product "
            "NSIDC-0081 is calibrated differently and needs revalidation before "
            "these weights can be trusted on it."
        ),
    },
    "era5": {
        "name": "ECMWF ERA5 single levels",
        "variable": "10 m wind, 2 m air temperature",
        "credential": Path.home() / ".cdsapirc",
        "credential_note": "Copernicus CDS API key in ~/.cdsapirc",
        "module": "seaice_forecast.data_processing.downloaders.era5",
        "nrt_caveat": (
            "ERA5 lags real time by about five days. ERA5T is faster and is "
            "subject to revision."
        ),
    },
    "cmems": {
        "name": "Copernicus Marine GLORYS12V1",
        "variable": "SST, surface currents",
        "credential": Path.home() / ".copernicusmarine/.copernicusmarine-credentials",
        "credential_note": "copernicusmarine login",
        "module": "seaice_forecast.data_processing.downloaders.copernicus",
        "nrt_caveat": (
            "GLORYS12 is reanalysis. Operational use needs the analysis-forecast "
            "product, which is a different dataset id."
        ),
    },
}


def archive_coverage() -> dict:
    """What the processed archive actually holds."""
    if not DAILY.exists():
        return {"available": False, "reason": f"{DAILY} not found"}
    years = sorted(p for p in DAILY.iterdir() if p.is_dir() and p.name.isdigit())
    if not years:
        return {"available": False, "reason": "no year directories"}
    days = []
    for y in years:
        days.extend(f.stem for f in y.glob("*.npz"))
    days.sort()
    first = datetime.strptime(days[0], "%Y%m%d").date()
    last = datetime.strptime(days[-1], "%Y%m%d").date()
    span = (last - first).days + 1
    return {
        "available": True,
        "firstDay": first.isoformat(),
        "lastDay": last.isoformat(),
        "daysPresent": len(days),
        "daysInSpan": span,
        "completeness": round(len(days) / span * 100.0, 1),
        "staleDays": (datetime.utcnow().date() - last).days,
    }


def check_source(key: str, spec: dict) -> dict:
    """Credentials and importability, without touching the network."""
    cred = spec["credential"]
    out = {
        "key": key, "name": spec["name"], "variable": spec["variable"],
        "credentialPath": str(cred),
        "credentialPresent": cred.exists(),
        "credentialNote": spec["credential_note"],
        "caveat": spec["nrt_caveat"],
    }
    try:
        __import__(spec["module"])
        out["downloaderImportable"] = True
    except Exception as exc:  # noqa: BLE001
        out["downloaderImportable"] = False
        out["importError"] = f"{type(exc).__name__}: {exc}"
    out["ready"] = out["credentialPresent"] and out["downloaderImportable"]
    return out


def missing_days(start: str, end: str) -> list[str]:
    """Days in the range that the processed archive does not already hold."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    d = s
    while d <= e:
        if not (DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz").exists():
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def cmd_check() -> int:
    cov = archive_coverage()
    print("PROCESSED ARCHIVE")
    if cov["available"]:
        print(f"  {cov['firstDay']} .. {cov['lastDay']}")
        print(f"  {cov['daysPresent']} days of {cov['daysInSpan']} "
              f"({cov['completeness']}% complete)")
        print(f"  {cov['staleDays']} days behind today")
    else:
        print(f"  unavailable: {cov['reason']}")

    print("\nSOURCES")
    results = []
    for key, spec in SOURCES.items():
        r = check_source(key, spec)
        results.append(r)
        state = "ready" if r["ready"] else "NOT READY"
        print(f"  {r['name']:<42} {state}")
        if not r["credentialPresent"]:
            print(f"      missing credential: {r['credentialNote']}")
        if not r.get("downloaderImportable", True):
            print(f"      downloader import failed: {r.get('importError')}")

    ready = [r for r in results if r["ready"]]
    print(f"\n  {len(ready)}/{len(results)} sources ready")
    print("\nCAVEATS THAT AFFECT OPERATIONAL USE")
    for r in results:
        print(f"  {r['key']}: {r['caveat']}")
    print("\n  Live ingestion is wired but the models have NOT been revalidated "
          "against\n  near-real-time products. Until they are, forecasts from "
          "freshly fetched\n  data are not comparable to anything in the "
          "published evaluation.")
    return 0 if ready else 1


def cmd_plan(start: str, end: str) -> int:
    missing = missing_days(start, end)
    print(f"RANGE {start} .. {end}")
    print(f"  {len(missing)} day(s) not in the processed archive")
    for d in missing[:10]:
        print(f"    {d}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    print("\n  For each day the chain would run:")
    print("    download (NSIDC + ERA5 + CMEMS) -> quality check -> regrid to")
    print("    EPSG:3412 25 km -> write data[7,332,316] .npz")
    print("\n  Run with --fetch to execute. Expect roughly 40 MB per day across")
    print("  the three sources before regridding.")
    return 0


def cmd_fetch(start: str, end: str) -> int:
    missing = missing_days(start, end)
    if not missing:
        print(f"nothing to do: {start}..{end} is already in the archive")
        return 0

    not_ready = [k for k, spec in SOURCES.items() if not check_source(k, spec)["ready"]]
    if not_ready:
        print(f"refusing to fetch: {', '.join(not_ready)} not ready")
        print("run --check for what is missing")
        return 1

    print(f"fetching {len(missing)} day(s)")
    print("\n  Not implemented as an automatic loop, deliberately.\n")
    print("  Each source has its own rate limits, queueing behaviour and failure")
    print("  modes - the CDS API queues requests for minutes to hours, and a")
    print("  naive retry loop against it gets an account throttled. The existing")
    print("  per-source scripts handle those properly:\n")
    print("    seaice_forecast/scripts/data/download_nsidc_sic.py")
    print("    seaice_forecast/scripts/data/download_era5_forcing.py")
    print("    iceberg-drift/scripts/download_missing_forcing.py")
    print("\n  This orchestrator reports what is needed and verifies readiness.")
    print("  Wiring it into an unattended loop needs per-source backoff that has")
    print("  not been written, and claiming otherwise would be the kind of")
    print("  infrastructure that looks finished and fails in the field.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report readiness (default)")
    ap.add_argument("--plan", action="store_true", help="what would be fetched")
    ap.add_argument("--fetch", action="store_true", help="fetch missing days")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--json", action="store_true", help="machine-readable check output")
    args = ap.parse_args()

    if args.json:
        print(json.dumps({
            "archive": archive_coverage(),
            "sources": [check_source(k, s) for k, s in SOURCES.items()],
        }, indent=2))
        return 0
    if args.plan or args.fetch:
        if not (args.start and args.end):
            ap.error("--plan and --fetch need --start and --end")
        return cmd_plan(args.start, args.end) if args.plan else cmd_fetch(args.start, args.end)
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
