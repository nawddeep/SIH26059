#!/usr/bin/env python3
"""
scripts/generate_mock_data.py

Generates realistic synthetic polar maritime operational data and ChatML instruction-tuning
datasets for IMPALA Local Dashboard advisory AI model.

Features:
- Deterministic pseudo-random generation with fixed seed (reproducible).
- Physically coherent navigation, iceberg drift, sonar/radar returns, and polar weather.
- Edge cases: degraded/stale/offline sensors, contradictory sensor feeds, multi-iceberg clusters, high CPA risks.
- Strict scenario-level split (train: 80%, valid: 10%, test: 10%) to prevent data leakage.
- Output:
    - data/synthetic/operational/scenarios.json
    - data/synthetic/training/train.jsonl
    - data/synthetic/training/valid.jsonl
    - data/synthetic/training/test.jsonl
"""

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEED = 42
TOTAL_SCENARIOS = 120
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"
OPERATIONAL_DIR = DATA_DIR / "operational"
TRAINING_DIR = DATA_DIR / "training"

SYSTEM_PROMPT = (
    "You are the IMPALA Local Bridge Decision Support Assistant onboard RV BHARATI. "
    "You provide concise, objective, and structured operational assessments based strictly on the provided operational snapshot. "
    "You clearly identify SIMULATION vs LIVE data, acknowledge sensor uncertainty or offline status, "
    "highlight high-risk iceberg targets and route hazards, and never claim direct vessel steering or autopilot control. "
    "Deterministic calculations (CPA, distance, bearing, drift propagation) are performed by shipboard instruments."
)

ICEBERG_TYPES = ["TABULAR", "PINNACLE", "WEDGE", "BERGY BIT", "GROWLER"]
SENSOR_NAMES = {
    "gnss": ("GNSS", "18 ms"),
    "ais": ("AIS", "27 TARGETS"),
    "gyro": ("GYRO", "219°"),
    "radar": ("X-BAND RADAR", "12 NM"),
    "forward_sonar": ("FWD SONAR", "2.8 km"),
    "adcp": ("ADCP", "0.42 m/s"),
    "weather": ("WEATHER", "31 kn"),
}


def round_val(val: float, decimals: int = 2) -> float:
    return round(float(val), decimals)


def create_scenario(scenario_id: int, rng: random.Random) -> Dict[str, Any]:
    """Generates a complete, physically consistent operational scenario with history and anomalies."""
    base_time = datetime(2026, 9, 14, 4, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=scenario_id * 15)
    time_str = base_time.strftime("%H:%M:%S")
    iso_time = base_time.isoformat().replace("+00:00", "Z")

    # Scenario archetype
    archetype_idx = scenario_id % 6
    if archetype_idx == 0:
        archetype = "NORMAL_TRANSIT"
    elif archetype_idx == 1:
        archetype = "HIGH_RISK_ICEBERG"
    elif archetype_idx == 2:
        archetype = "SENSOR_DEGRADATION"
    elif archetype_idx == 3:
        archetype = "SENSOR_CONTRADICTION"
    elif archetype_idx == 4:
        archetype = "WEATHER_DETERIORATION"
    else:
        archetype = "MULTI_ICEBERG_CLUSTER"

    # Region and coordinates (Antarctic Peninsula / Weddell Sea / South Shetland)
    region = "antarctic"
    base_lat = -64.80 + rng.uniform(-1.5, 1.5)
    base_lon = -60.40 + rng.uniform(-2.5, 2.5)

    # Environmental fields
    if archetype == "WEATHER_DETERIORATION":
        wind_speed = round_val(rng.uniform(42, 58), 1)  # Gale
        sea_state = "BEAUFORT 8 / ROUGH"
        visibility_nm = round_val(rng.uniform(0.5, 1.8), 1)
        pressure_hpa = round_val(rng.uniform(965, 978), 0)
        pack_concentration = rng.randint(85, 96)
        pack_trend = "RAPIDLY COMPRESSING"
    else:
        wind_speed = round_val(rng.uniform(12, 34), 1)
        sea_state = "BEAUFORT 5" if wind_speed > 25 else "BEAUFORT 4"
        visibility_nm = round_val(rng.uniform(3.5, 9.5), 1)
        pressure_hpa = round_val(rng.uniform(988, 1012), 0)
        pack_concentration = rng.randint(40, 80)
        pack_trend = "STABLE" if pack_concentration < 60 else "COMPRESSING"

    wind_dir = rng.randint(180, 310)
    air_temp = round_val(rng.uniform(-12.5, -1.0), 1)
    sea_temp = round_val(rng.uniform(-1.8, 0.4), 1)
    current_speed = round_val(rng.uniform(0.18, 0.65), 2)
    current_dir = (wind_dir + rng.randint(-35, 35)) % 360
    wave_height = round_val(max(0.8, (wind_speed / 12.0) * rng.uniform(0.8, 1.2)), 1)
    wave_dir = (wind_dir + rng.randint(-15, 15)) % 360

    # Vessel navigation
    vessel_sog = round_val(rng.uniform(8.5, 13.5), 1) if archetype != "WEATHER_DETERIORATION" else round_val(rng.uniform(4.0, 7.5), 1)
    vessel_heading = rng.randint(190, 240)
    vessel_cog = (vessel_heading + rng.randint(-4, 4)) % 360
    rate_of_turn = round_val(rng.uniform(-0.4, 0.4), 1)
    depth_m = rng.randint(320, 1450)

    track_pts = [
        {"x": 50, "y": 57},
        {"x": 45, "y": 61},
        {"x": 40, "y": 65},
        {"x": 34, "y": 70},
    ]

    prov_sim = {
        "source": "LOCAL SIMULATION",
        "timestamp": iso_time,
        "status": "simulation",
        "kind": "simulated",
    }

    # Sensors health configuration
    sensors: List[Dict[str, Any]] = []
    for s_type, (s_name, default_metric) in SENSOR_NAMES.items():
        status = "simulation"
        detail = default_metric
        quality = rng.randint(90, 99)
        error_count = 0
        latency_ms = rng.randint(12, 28)

        if archetype == "SENSOR_DEGRADATION":
            if s_type == "gnss" and scenario_id % 3 == 0:
                status = "degraded"
                detail = "HDOP 3.8 / 4 SATS"
                quality = 48
                error_count = 5
            elif s_type == "forward_sonar" and scenario_id % 3 == 1:
                status = "offline"
                detail = "TRANSDUCER COMM TIMEOUT"
                quality = 0
                error_count = 24
            elif s_type == "adcp" and scenario_id % 3 == 2:
                status = "stale"
                detail = "LAST UPDATE 28 MIN AGO"
                quality = 30
                error_count = 2

        sensors.append({
            "sensorId": f"{s_type}-01",
            "sensorType": s_type,
            "name": s_name,
            "metric": "SIMULATION" if status == "simulation" else status.upper(),
            "detail": detail,
            "status": status,
            "lastSeen": iso_time,
            "latencyMs": latency_ms,
            "quality": quality,
            "errorCount": error_count,
        })

    # Icebergs
    icebergs: List[Dict[str, Any]] = []
    num_icebergs = rng.randint(4, 7) if archetype == "MULTI_ICEBERG_CLUSTER" else rng.randint(3, 5)

    for i in range(num_icebergs):
        ice_id = f"ICE-{(scenario_id * 11 + i * 29 + 13) % 900 + 100:03d}"
        ice_type = rng.choice(ICEBERG_TYPES)
        draft = rng.randint(25, 280) if ice_type in ["TABULAR", "PINNACLE"] else rng.randint(8, 45)
        ix = rng.randint(15, 85)
        iy = rng.randint(15, 85)
        ice_heading = (current_dir + rng.randint(-20, 20)) % 360
        ice_speed = round_val(current_speed * rng.uniform(0.7, 1.15) + (wind_speed / 100.0) * 0.4, 2)

        # Risk assignment
        if archetype == "HIGH_RISK_ICEBERG" and i == 0:
            risk = "high"
            cpa_km = round_val(rng.uniform(3.2, 8.8), 1)
            cpa_hours = rng.randint(6, 24)
            conf = rng.randint(88, 96)
            uncert = rng.randint(18, 25)
        elif archetype == "NORMAL_TRANSIT":
            risk = "low" if i > 0 else "medium"
            cpa_km = round_val(rng.uniform(18.0, 48.0), 1)
            cpa_hours = rng.randint(36, 72)
            conf = rng.randint(80, 92)
            uncert = rng.randint(10, 16)
        else:
            risk = rng.choice(["low", "medium"]) if i > 0 else ("high" if rng.random() < 0.4 else "medium")
            cpa_km = round_val(rng.uniform(8.0, 35.0), 1)
            cpa_hours = rng.randint(18, 54)
            conf = rng.randint(75, 94)
            uncert = rng.randint(12, 22)

        traj = [
            {"x": ix, "y": iy},
            {"x": ix + rng.randint(-6, 6), "y": iy + rng.randint(-6, 6)},
            {"x": ix + rng.randint(-12, 12), "y": iy + rng.randint(-12, 12)},
        ]

        icebergs.append({
            **prov_sim,
            "id": ice_id,
            "type": ice_type,
            "x": ix,
            "y": iy,
            "dimensions": f"{100 + draft} × {50 + draft // 2} m",
            "draft": draft,
            "speed": ice_speed,
            "heading": ice_heading,
            "risk": risk,
            "cpa": cpa_km,
            "cpaHours": cpa_hours,
            "trajectory": traj,
            "uncertainty": uncert,
            "detectionSource": "SIMULATED X-BAND + SAR FUSION" if conf > 80 else "RADAR CONTACT / UNVERIFIED",
            "confidence": conf,
        })

    # Sonar detections
    sonar: List[Dict[str, Any]] = []
    if any(s["sensorType"] == "forward_sonar" and s["status"] != "offline" for s in sensors):
        if archetype == "SENSOR_CONTRADICTION":
            sonar.append({
                **prov_sim,
                "id": f"SONAR-{(scenario_id * 7 + 11) % 90 + 10:02d}",
                "x": 56,
                "y": 44,
                "rangeKm": round_val(rng.uniform(1.2, 2.4), 2),
                "bearing": rng.randint(215, 230),
                "depthM": -rng.randint(22, 55),
                "returnStrength": "strong",
                "classification": "POSSIBLE ICE",
                "observationState": "DETECTION",
                "confidence": rng.randint(78, 89),
            })
        else:
            sonar.append({
                **prov_sim,
                "id": f"SONAR-{(scenario_id * 7 + 11) % 90 + 10:02d}",
                "x": 58,
                "y": 46,
                "rangeKm": round_val(rng.uniform(1.6, 2.9), 2),
                "bearing": rng.randint(210, 235),
                "depthM": -rng.randint(28, 48),
                "returnStrength": "strong" if rng.random() < 0.6 else "weak",
                "classification": "POSSIBLE ICE" if rng.random() < 0.7 else "UNCLASSIFIED",
                "observationState": "CLASSIFICATION",
                "confidence": rng.randint(65, 92),
            })

    # Radar contacts
    radar: List[Dict[str, Any]] = [
        {
            **prov_sim,
            "contactId": f"RAD-{(scenario_id * 13 + 7) % 90 + 10:02d}",
            "x": 43,
            "y": 48,
            "rangeNm": round_val(rng.uniform(2.5, 5.0), 1),
            "bearing": rng.randint(190, 210),
            "course": rng.randint(180, 200),
            "speed": round_val(rng.uniform(3.5, 6.0), 1),
            "confidence": rng.randint(82, 95),
        }
    ]

    # AIS contacts
    ais: List[Dict[str, Any]] = [
        {
            **prov_sim,
            "id": f"AIS-{(scenario_id * 17 + 101) % 900 + 100:03d}",
            "mmsi": f"2734{scenario_id:05d}",
            "name": rng.choice(["RV AURORA", "POLAR STAR", "KRONPRINS HAAKON", "RV NATHANIEL PALMER", "AGULHAS II"]),
            "x": 44,
            "y": 49,
            "heading": rng.randint(175, 195),
            "cog": rng.randint(178, 198),
            "sog": round_val(rng.uniform(3.8, 6.5), 1),
            "cpaKm": round_val(rng.uniform(7.5, 16.0), 1),
            "tcpaMinutes": rng.randint(45, 140),
        }
    ]

    # Hazards
    hazards: List[Dict[str, Any]] = []
    top_risk_ice = sorted(icebergs, key=lambda x: (0 if x["risk"] == "high" else (1 if x["risk"] == "medium" else 2), x["cpa"]))[0]

    if top_risk_ice["risk"] == "high":
        hazards.append({
            **prov_sim,
            "level": "CRITICAL",
            "title": f"{top_risk_ice['id']} / ROUTE PROXIMITY",
            "message": f"Predicted trajectory intersects planned corridor. CPA {top_risk_ice['cpa']} km at T+{top_risk_ice['cpaHours']}h.",
            "action": "REVIEW CORRIDOR",
        })

    if pack_concentration >= 80:
        hazards.append({
            **prov_sim,
            "level": "WARNING" if pack_concentration < 90 else "CRITICAL",
            "title": "PACK ICE COMPRESSION",
            "message": f"Sector concentration {pack_concentration}%; trend {pack_trend.lower()}.",
            "action": "REDUCE SPEED / MONITOR PACK",
        })

    if archetype == "SENSOR_DEGRADATION":
        degraded = [s for s in sensors if s["status"] in ["degraded", "offline", "stale"]]
        for ds in degraded:
            hazards.append({
                **prov_sim,
                "level": "CAUTION" if ds["status"] != "offline" else "WARNING",
                "title": f"SENSOR FAULT: {ds['name']}",
                "message": f"{ds['name']} reported {ds['status'].upper()}: {ds['detail']}.",
                "action": "VERIFY SENSOR DIAGNOSTICS",
            })
    elif archetype == "SENSOR_CONTRADICTION" and sonar:
        hazards.append({
            **prov_sim,
            "level": "CAUTION",
            "title": f"DISCREPANCY: {sonar[0]['id']}",
            "message": "Forward sonar return detected ahead with unconfirmed radar reflection in surface clutter.",
            "action": "MAINTAIN RADAR TUNE / VERIFY RETURN",
        })

    if not hazards:
        hazards.append({
            **prov_sim,
            "level": "INFO",
            "title": "ROUTINE POLAR PASSAGE",
            "message": "All monitored channels nominal. Route corridor clear of immediate hazards.",
            "action": "MAINTAIN WATCH",
        })

    # Logs
    logs = [
        {"time": time_str, "source": "AIS", "text": f"{len(ais)} target(s) actively tracked", "kind": "simulated"},
        {"time": (base_time - timedelta(seconds=4)).strftime("%H:%M:%S"), "source": "GNSS", "text": "Position fix processed", "kind": "simulated"},
        {"time": (base_time - timedelta(seconds=11)).strftime("%H:%M:%S"), "source": "TRAJECTORY", "text": f"{top_risk_ice['id']} drift model updated", "kind": "predicted"},
    ]

    snapshot = {
        "region": region,
        "vessel": {
            **prov_sim,
            "name": "RV BHARATI",
            "latitude": round_val(base_lat, 3),
            "longitude": round_val(base_lon, 3),
            "sog": vessel_sog,
            "cog": vessel_cog,
            "heading": vessel_heading,
            "rateOfTurn": rate_of_turn,
            "depth": depth_m,
            "windSpeed": wind_speed,
            "windDirection": wind_dir,
            "seaTemperature": sea_temp,
            "currentSpeed": current_speed,
            "currentDirection": current_dir,
            "track": track_pts,
        },
        "weather": {
            **prov_sim,
            "windSpeed": wind_speed,
            "windDirection": wind_dir,
            "airTemperature": air_temp,
            "pressureHpa": pressure_hpa,
            "humidity": rng.randint(65, 92),
            "visibilityNm": visibility_nm,
        },
        "ocean": {
            **prov_sim,
            "currentSpeed": current_speed,
            "currentDirection": current_dir,
            "seaSurfaceTemperature": sea_temp,
            "waveHeight": wave_height,
            "waveDirection": wave_dir,
        },
        "ais": ais,
        "icebergs": icebergs,
        "sonar": sonar,
        "radar": radar,
        "sensors": sensors,
        "hazards": hazards,
        "environment": {
            "wind": f"{wind_speed} kn / {wind_dir}°",
            "current": f"{current_speed} m/s / {current_dir}°",
            "seaState": sea_state,
            "temperature": f"{sea_temp} °C",
            "pack": f"{pack_concentration}% / {pack_trend}",
        },
        "logs": logs,
        "dataFreshness": "LOCAL SIMULATION / 02 s",
    }

    # Historical state at T-6h for temporal reasoning
    hist_wind = round_val(max(5.0, wind_speed + rng.uniform(-14.0, 10.0)), 1)
    hist_pack = max(20, pack_concentration - rng.randint(5, 25))
    history_6h = {
        "time": (base_time - timedelta(hours=6)).strftime("%H:%M:%S UTC"),
        "windSpeed": hist_wind,
        "packConcentration": hist_pack,
        "vesselPosition": f"{abs(base_lat - 0.45):.3f}°S, {abs(base_lon + 0.65):.3f}°W",
        "topIcebergId": top_risk_ice["id"],
        "topIcebergInitialCpa": round_val(top_risk_ice["cpa"] + rng.uniform(8.0, 20.0), 1),
    }

    return {
        "scenario_id": scenario_id,
        "archetype": archetype,
        "timestamp": iso_time,
        "snapshot": snapshot,
        "history_6h": history_6h,
        "top_risk_iceberg": top_risk_ice,
    }


def format_snapshot_context(snapshot: Dict[str, Any]) -> str:
    """Formats structured operational snapshot into clean, concise text for model input."""
    v = snapshot["vessel"]
    w = snapshot["weather"]
    e = snapshot["environment"]

    lines = [
        f"=== OPERATIONAL CONTEXT [{snapshot['dataFreshness']}] ===",
        f"VESSEL: {v['name']} | POS: {abs(v['latitude']):.3f}°S {abs(v['longitude']):.3f}°W | SOG: {v['sog']} kn | COG: {v['cog']}° | HDG: {v['heading']}° | DEPTH: {v['depth']}m",
        f"ENVIRONMENT: Wind {e['wind']} | Current {e['current']} | Sea {e['seaState']} | Sea Temp {e['temperature']} | Pack Ice {e['pack']}",
        f"WEATHER: Air Temp {w['airTemperature']}°C | Pressure {w['pressureHpa']} hPa | Visibility {w['visibilityNm']} NM",
    ]

    # Sensors
    s_lines = []
    for s in snapshot["sensors"]:
        status_flag = s["status"].upper()
        s_lines.append(f"{s['name']}: {status_flag} ({s['detail']})")
    lines.append("SENSORS: " + " | ".join(s_lines))

    # Icebergs
    lines.append(f"TRACKED ICEBERGS ({len(snapshot['icebergs'])} targets):")
    for ice in snapshot["icebergs"]:
        lines.append(
            f" - [{ice['id']}] {ice['type']} | Draft {ice['draft']}m | Drift {ice['speed']}kn/{ice['heading']}° | "
            f"Risk: {ice['risk'].upper()} (CPA {ice['cpa']}km in T+{ice['cpaHours']}h) | Conf {ice['confidence']}% | Source: {ice['detectionSource']}"
        )

    # Sonar / Radar / AIS
    if snapshot["sonar"]:
        son = snapshot["sonar"][0]
        lines.append(f"FWD SONAR: Contact {son['id']} at {son['rangeKm']}km, Brg {son['bearing']}°, Depth {son['depthM']}m ({son['classification']}, Return: {son['returnStrength'].upper()}, Conf: {son['confidence']}%)")
    else:
        lines.append("FWD SONAR: No active contact / sensor offline")

    if snapshot["radar"]:
        rad = snapshot["radar"][0]
        lines.append(f"RADAR: Contact {rad['contactId']} at {rad['rangeNm']} NM, Brg {rad['bearing']}°, Course {rad['course']}°, Spd {rad['speed']} kn")

    if snapshot["ais"]:
        ais0 = snapshot["ais"][0]
        lines.append(f"AIS: {ais0['name']} ({ais0['id']}) at CPA {ais0['cpaKm']}km / TCPA {ais0['tcpaMinutes']}min | SOG {ais0['sog']}kn")

    # Hazards
    lines.append(f"ACTIVE HAZARDS ({len(snapshot['hazards'])}):")
    for h in snapshot["hazards"]:
        lines.append(f" - [{h['level']}] {h['title']}: {h['message']} -> Action: {h['action']}")

    return "\n".join(lines)


def generate_dialogue_pairs(scenario: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Generates varied, high-quality question/answer pairs for a scenario."""
    snap = scenario["snapshot"]
    hist = scenario["history_6h"]
    top_ice = scenario["top_risk_iceberg"]
    context = format_snapshot_context(snap)
    pairs = []

    # Helper for standard message dict
    def make_chat(user_q: str, assistant_resp: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nOperator Query: {user_q}"},
                {"role": "assistant", "content": assistant_resp},
            ],
            "scenario_id": scenario["scenario_id"],
            "archetype": scenario["archetype"],
        }

    # 1. Operational Picture
    op_pic = (
        f"**Operational Picture Summary [SIMULATION DATA]**:\n"
        f"- **Vessel State**: RV BHARATI underway in the Antarctic sector at {snap['vessel']['sog']} kn on heading {snap['vessel']['heading']}°.\n"
        f"- **Environmental Baseline**: Wind {snap['environment']['wind']}, Sea State {snap['environment']['seaState']}, Pack ice {snap['environment']['pack']}.\n"
        f"- **Cryosphere & Obstacles**: {len(snap['icebergs'])} icebergs tracked. Highest priority is {top_ice['id']} ({top_ice['type']}, Risk: {top_ice['risk'].upper()}, CPA {top_ice['cpa']} km).\n"
        f"- **Sensor & Hazard Status**: {len(snap['hazards'])} active hazard advisory events. Sensor suite is operating under {snap['dataFreshness']}."
    )
    pairs.append(make_chat("Give me the current operational picture.", op_pic))

    # 2. Highest Risk Iceberg
    why_risk = (
        f"**Highest Risk Iceberg: {top_ice['id']} ({top_ice['risk'].upper()} Risk)**\n"
        f"- **Classification & Dimensions**: {top_ice['type']}, draft {top_ice['draft']} m, dimensions {top_ice['dimensions']}.\n"
        f"- **Encounter Geometry**: Computed CPA is {top_ice['cpa']} km projected at T+{top_ice['cpaHours']}h. Drift velocity is {top_ice['speed']} kn along heading {top_ice['heading']}°.\n"
        f"- **Detection Provenance**: {top_ice['detectionSource']} (Confidence: {top_ice['confidence']}%, Uncertainty: ±{top_ice['uncertainty']}%).\n"
        f"- **Recommendation**: Review navigation corridor buffer against {top_ice['id']}'s predicted drift trajectory. Note: All trajectory calculations are model outputs, not live observations."
    )
    pairs.append(make_chat("Which iceberg currently presents the highest risk and why?", why_risk))

    # 3. What is ahead?
    fwd_sonar_active = any(s["sensorType"] == "forward_sonar" and s["status"] != "offline" for s in snap["sensors"])
    if fwd_sonar_active and snap["sonar"]:
        son = snap["sonar"][0]
        ahead_resp = (
            f"**Ahead of Vessel ({snap['vessel']['name']})**:\n"
            f"- **Forward Sonar Contact**: {son['id']} detected at range {son['rangeKm']} km, bearing {son['bearing']}°, submerged depth {son['depthM']} m.\n"
            f"- **Acoustic Return**: {son['returnStrength'].upper()} echo classified as {son['classification']} (Confidence: {son['confidence']}%).\n"
            f"- **Environmental Clutter**: Visibility is {snap['weather']['visibilityNm']} NM; sea pack is {snap['environment']['pack']}.\n"
            f"- **Advisory**: Maintain active sonar pinging and forward visual/radar watch. No direct maneuvering action is taken by this advisory system."
        )
    else:
        ahead_resp = (
            f"**Ahead of Vessel ({snap['vessel']['name']})**:\n"
            f"- **Forward Sonar Status**: Forward sonar is OFFLINE / unavailable.\n"
            f"- **Radar & Surface Watch**: Relying on X-band radar and optical watch. Radar contacts monitored within 12 NM range.\n"
            f"- **Advisory**: Submerged ice hazard detection is currently degraded due to lack of forward acoustic telemetry. Exercise caution and verify sonar diagnostics."
        )
    pairs.append(make_chat("What is ahead of the vessel?", ahead_resp))

    # 4. Sensor availability
    degraded_sensors = [s for s in snap["sensors"] if s["status"] != "simulation" and s["status"] != "live"]
    if degraded_sensors:
        s_list = ", ".join([f"{s['name']} ({s['status'].upper()}: {s['detail']})" for s in degraded_sensors])
        sensor_resp = (
            f"**Sensor Health Audit — Attention Required**:\n"
            f"- **Anomalies Detected**: {len(degraded_sensors)} sensor(s) non-nominal: {s_list}.\n"
            f"- **Nominal Sensors**: {', '.join([s['name'] for s in snap['sensors'] if s not in degraded_sensors])}.\n"
            f"- **Operational Impact**: Bridge advisory is operating with partial telemetry redundancy. Vessel operators should verify physical sensor connections and data feeds."
        )
    else:
        sensor_resp = (
            f"**Sensor Health Audit — All Channels Nominal**:\n"
            f"- All {len(snap['sensors'])} monitored sensor interfaces (GNSS, AIS, Gyro, Radar, Sonar, ADCP, Weather) are actively reporting with nominal latency (<30 ms).\n"
            f"- Operational mode: {snap['dataFreshness']}."
        )
    pairs.append(make_chat("Are all sensors available?", sensor_resp))

    # 5. Route risk explanation
    route_resp = (
        f"**Route Corridor Risk Assessment**:\n"
        f"- **Primary Hazard Factor**: {snap['hazards'][0]['title']} — {snap['hazards'][0]['message']}\n"
        f"- **Ice Field Conditions**: Pack ice concentration at {snap['environment']['pack']} with prevailing wind {snap['environment']['wind']}.\n"
        f"- **Advisory Action**: {snap['hazards'][0]['action']}. Navigators should inspect deterministic route clearance margins. This advisory system does not modify autopilot waypoints."
    )
    pairs.append(make_chat("Why is the current route considered risky?", route_resp))

    # 6. What changed over last 6 hours?
    hist_resp = (
        f"**6-Hour Trend Analysis (Since {hist['time']})**:\n"
        f"- **Wind Trend**: Wind shifted from {hist['windSpeed']} kn to current {snap['weather']['windSpeed']} kn ({snap['environment']['wind']}).\n"
        f"- **Pack Ice Evolution**: Sea-ice concentration changed from {hist['packConcentration']}% to {snap['environment']['pack']}.\n"
        f"- **Iceberg Proximity**: {hist['topIcebergId']} was at initial CPA estimate ~{hist['topIcebergInitialCpa']} km; current deterministic model projects CPA {top_ice['cpa']} km at T+{top_ice['cpaHours']}h.\n"
        f"- **Data Origin**: Trend derived from simulated historical bridge logs."
    )
    pairs.append(make_chat("What changed over the last 6 hours?", hist_resp))

    # 7. What should operator verify?
    degraded_or_caution = [h for h in snap["hazards"] if h["level"] in ["CRITICAL", "WARNING", "CAUTION"]]
    actions = [f"- {h['title']}: {h['action']} ({h['message']})" for h in degraded_or_caution]
    verify_resp = (
        f"**Recommended Operator Verifications**:\n"
        + "\n".join(actions) +
        f"\n- Confirm visual and radar correlation with {top_ice['id']}.\n"
        f"- Verify data feed integrity (Current status: {snap['dataFreshness']}).\n"
        f"- Note: The decision support assistant does not alter engine or rudder commands."
    )
    pairs.append(make_chat("What should the operator verify?", verify_resp))

    # 8. Missing / Stale sensor diagnostic query
    if degraded_sensors:
        diag_q = f"How should we handle the issue with {degraded_sensors[0]['name']}?"
        diag_resp = (
            f"**Diagnostic Procedure for {degraded_sensors[0]['name']}**:\n"
            f"- **Reported Status**: {degraded_sensors[0]['status'].upper()} | Detail: {degraded_sensors[0]['detail']} | Quality: {degraded_sensors[0]['quality']}%.\n"
            f"- **Immediate Action**: Fall back to secondary navigation sources. Do NOT assume nominal readings from {degraded_sensors[0]['name']}.\n"
            f"- **Verification**: Inspect local bus interface and hardware gateway."
        )
        pairs.append(make_chat(diag_q, diag_resp))

    # 9. Environmental & Sea Ice compression advisory
    pack_resp = (
        f"**Sea-Ice & Environmental Dynamics**:\n"
        f"- **Concentration & Dynamic**: {snap['environment']['pack']}.\n"
        f"- **Forcing Mechanism**: Wind {snap['environment']['wind']} combined with surface current {snap['environment']['current']} is driving ice drift.\n"
        f"- **Vessel Impact**: Sea temperature {snap['environment']['temperature']} and wave height {snap['ocean']['waveHeight']} m in {snap['environment']['seaState']}.\n"
        f"- **Operational Caution**: Increasing compression elevates hull resistance and risk of ice entrapment in leads."
    )
    pairs.append(make_chat("Explain the sea ice and environmental compression risk.", pack_resp))

    # 10. Specific target drilldown
    ice_drill = (
        f"**Target Analysis for {top_ice['id']}**:\n"
        f"- **Type & Shape**: {top_ice['type']} iceberg.\n"
        f"- **Underwater Profile**: Estimated draft of {top_ice['draft']} m against regional charted depth {snap['vessel']['depth']} m.\n"
        f"- **Trajectory Dynamics**: Current speed {top_ice['speed']} kn on course {top_ice['heading']}°. CPA to vessel path: {top_ice['cpa']} km in {top_ice['cpaHours']} hours.\n"
        f"- **Uncertainty Bounds**: Model confidence {top_ice['confidence']}%, drift uncertainty ±{top_ice['uncertainty']}%. Sensor: {top_ice['detectionSource']}."
    )
    pairs.append(make_chat(f"Provide a detailed assessment of target {top_ice['id']}.", ice_drill))

    # Additional contextual variation queries (to enrich dataset to ~25 examples per scenario)
    pairs.append(make_chat("What is our current vessel speed and heading?", f"RV BHARATI is currently maintaining a speed over ground (SOG) of {snap['vessel']['sog']} kn on a true heading of {snap['vessel']['heading']}° (COG: {snap['vessel']['cog']}°). Underway depth is {snap['vessel']['depth']} m [SIMULATION]."))
    pairs.append(make_chat("Are there any AIS vessels in our vicinity?", f"AIS currently tracks {len(snap['ais'])} contact(s). Primary contact: {snap['ais'][0]['name']} ({snap['ais'][0]['id']}) at distance CPA {snap['ais'][0]['cpaKm']} km, TCPA {snap['ais'][0]['tcpaMinutes']} min, proceeding at {snap['ais'][0]['sog']} kn on heading {snap['ais'][0]['heading']}°."))
    pairs.append(make_chat("What is the current barometric pressure trend and visibility?", f"Atmospheric pressure is {snap['weather']['pressureHpa']} hPa with ambient air temperature {snap['weather']['airTemperature']}°C and humidity {snap['weather']['humidity']}%. Optical visibility is {snap['weather']['visibilityNm']} NM."))
    pairs.append(make_chat("Can you steer the ship to avoid the iceberg?", "Negative. As the IMPALA Local Bridge Advisory Assistant, I am an advisory decision-support system. I have no direct interface with the autopilot, rudder, or engine governors. All steering and maneuvering decisions rest solely with the Officer of the Watch (OOW)."))
    pairs.append(make_chat("Is the current telemetry live or simulated?", f"The operational dashboard is currently operating on {snap['dataFreshness']}. All sensor readings, iceberg positions, and trajectory predictions are synthetic/simulated for development and testing purposes."))
    pairs.append(make_chat("Summarize the active hazard alerts.", "\n".join([f"[{h['level']}] {h['title']}: {h['message']} (Action: {h['action']})" for h in snap["hazards"]])))
    pairs.append(make_chat("What is the drift vector of the highest risk target?", f"Iceberg {top_ice['id']} has a drift vector of {top_ice['speed']} kn toward {top_ice['heading']}°, primarily influenced by the {snap['environment']['current']} ocean current and {snap['environment']['wind']} wind field."))
    pairs.append(make_chat("What is the reliability of the forward sonar detection?", f"Forward sonar contact reporting confidence is {snap['sonar'][0]['confidence'] if snap['sonar'] else 0}% with status {snap['sensors'][4]['status'].upper()} ({snap['sensors'][4]['detail']}). Acoustic classifications should be visually and radar cross-referenced."))
    pairs.append(make_chat("Explain the difference between observed and predicted iceberg trajectory points.", "Observed points represent direct sensor contacts from radar/satellite SAR fusion at T0. Predicted points are calculated by the hydro-thermodynamic drift model accounting for ocean currents and wind drag over the forecast horizon (+24h, +48h, +72h)."))
    pairs.append(make_chat("What should be done if GNSS signal degrades in sea ice?", "If GNSS position quality degrades (high HDOP / satellite loss), immediately switch to dead-reckoning and radar range/bearing fixing against charted coastline features. Log the anomaly and verify gyro and speed log integrity."))

    return pairs


def main() -> None:
    print(f"Initializing synthetic polar maritime mock data generation (Seed={SEED})...")
    rng = random.Random(SEED)

    OPERATIONAL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for s_id in range(1, TOTAL_SCENARIOS + 1):
        scenarios.append(create_scenario(s_id, rng))

    print(f"Generated {len(scenarios)} complete operational scenarios.")

    # Save operational scenarios JSON
    scenarios_file = OPERATIONAL_DIR / "scenarios.json"
    with open(scenarios_file, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Saved raw operational scenarios to {scenarios_file}")

    # Generate dialogue examples
    all_examples = []
    for sc in scenarios:
        examples = generate_dialogue_pairs(sc, rng)
        all_examples.extend(examples)

    print(f"Generated {len(all_examples)} total instruction-tuning dialogue pairs.")

    # Scenario-level split (Zero scenario leakage)
    n_train_scenarios = int(TOTAL_SCENARIOS * TRAIN_RATIO)
    n_valid_scenarios = int(TOTAL_SCENARIOS * VALID_RATIO)

    train_scenarios_ids = set(range(1, n_train_scenarios + 1))
    valid_scenarios_ids = set(range(n_train_scenarios + 1, n_train_scenarios + n_valid_scenarios + 1))
    test_scenarios_ids = set(range(n_train_scenarios + n_valid_scenarios + 1, TOTAL_SCENARIOS + 1))

    train_data = [ex for ex in all_examples if ex["scenario_id"] in train_scenarios_ids]
    valid_data = [ex for ex in all_examples if ex["scenario_id"] in valid_scenarios_ids]
    test_data = [ex for ex in all_examples if ex["scenario_id"] in test_scenarios_ids]

    print(f"Split distribution:")
    print(f" - Train: {len(train_data)} examples across {len(train_scenarios_ids)} scenarios")
    print(f" - Valid: {len(valid_data)} examples across {len(valid_scenarios_ids)} scenarios")
    print(f" - Test:  {len(test_data)} examples across {len(test_scenarios_ids)} scenarios")

    # Write JSONL files
    for split_name, split_dataset in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        out_path = TRAINING_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_dataset:
                # Standard ChatML structure
                record = {"messages": item["messages"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split_dataset)} records to {out_path}")

    print("Data generation completed successfully.")


if __name__ == "__main__":
    main()
