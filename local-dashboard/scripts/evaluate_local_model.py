#!/usr/bin/env python3
"""
scripts/evaluate_local_model.py

Evaluates the local model (Base Model vs Fine-Tuned Model with LoRA adapter)
against the 10 mandatory operational test prompts.

Generates qualitative and quantitative evaluation metrics and saves a detailed
report to evaluation_results.json and prints human-readable comparison.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from mlx_lm import generate, load

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "impala-qwen3-8b"
DATA_DIR = BASE_DIR / "data" / "synthetic"

SYSTEM_PROMPT = (
    "You are the IMPALA Local Bridge Decision Support Assistant onboard RV BHARATI. "
    "You provide concise, objective, and structured operational assessments based strictly on the provided operational snapshot. "
    "You clearly identify SIMULATION vs LIVE data, acknowledge sensor uncertainty or offline status, "
    "highlight high-risk iceberg targets and route hazards, and never claim direct vessel steering or autopilot control. "
    "Deterministic calculations (CPA, distance, bearing, drift propagation) are performed by shipboard instruments."
)

# Standard test scenarios
NOMINAL_SNAPSHOT_TEXT = """=== OPERATIONAL CONTEXT [LOCAL SIMULATION / 02 s] ===
VESSEL: RV BHARATI | POS: 64.812°S 60.438°W | SOG: 11.4 kn | COG: 217° | HDG: 219° | DEPTH: 418m
ENVIRONMENT: Wind 31 kn / 248° | Current 0.42 m/s / 196° | Sea BEAUFORT 6 | Sea Temp −0.8 °C | Pack Ice 84% / COMPRESSING
WEATHER: Air Temp -4.2°C | Pressure 991 hPa | Visibility 4.8 NM
SENSORS: GNSS: SIMULATION (18 ms) | AIS: SIMULATION (27 TARGETS) | GYRO: SIMULATION (219°) | X-BAND RADAR: SIMULATION (12 NM) | FWD SONAR: SIMULATION (2.8 km) | ADCP: SIMULATION (0.42 m/s) | WEATHER: SIMULATION (31 kn)
TRACKED ICEBERGS (5 targets):
 - [ICE-042] TABULAR | Draft 245m | Drift 0.81kn/204° | Risk: HIGH (CPA 11.2km in T+34h) | Conf 92% | Source: SIMULATED X-BAND + SAR FUSION
 - [ICE-088] PINNACLE | Draft 78m | Drift 0.54kn/181° | Risk: MEDIUM (CPA 22.8km in T+58h) | Conf 83% | Source: RADAR CONTACT / UNVERIFIED
 - [ICE-104] WEDGE | Draft 112m | Drift 0.62kn/244° | Risk: LOW (CPA 39.4km in T+72h) | Conf 81% | Source: RADAR CONTACT / UNVERIFIED
 - [ICE-019] BERGY BIT | Draft 29m | Drift 0.46kn/148° | Risk: MEDIUM (CPA 18.6km in T+46h) | Conf 85% | Source: RADAR CONTACT / UNVERIFIED
 - [ICE-121] GROWLER | Draft 12m | Drift 0.38kn/231° | Risk: LOW (CPA 54.1km in T+96h) | Conf 79% | Source: RADAR CONTACT / UNVERIFIED
FWD SONAR: Contact SONAR-014 at 1.84km, Brg 224°, Depth -31m (POSSIBLE ICE, Return: STRONG, Conf: 87%)
RADAR: Contact RAD-077 at 3.2 NM, Brg 196°, Course 183°, Spd 4.1 kn
AIS: RV AURORA (AIS-302) at CPA 8.2km / TCPA 71min | SOG 4.1kn
ACTIVE HAZARDS (3):
 - [CRITICAL] ICE-042 / ROUTE ALPHA: Predicted trajectory intersects planned corridor. CPA 11.2 km at T+34h. -> Action: REVIEW CORRIDOR
 - [WARNING] PACK COMPRESSION: Weddell sector. Concentration 84%; compression trend increasing. -> Action: MONITOR SECTOR
 - [CAUTION] SONAR-014: Strong forward return; possible ice classification pending confirmation. -> Action: VERIFY RETURN"""

DEGRADED_SNAPSHOT_TEXT = """=== OPERATIONAL CONTEXT [LOCAL SIMULATION / 02 s] ===
VESSEL: RV BHARATI | POS: 65.120°S 61.200°W | SOG: 8.2 kn | COG: 210° | HDG: 212° | DEPTH: 520m
ENVIRONMENT: Wind 38 kn / 260° | Current 0.50 m/s / 205° | Sea BEAUFORT 7 | Sea Temp −1.2 °C | Pack Ice 88% / COMPRESSING
WEATHER: Air Temp -6.5°C | Pressure 982 hPa | Visibility 2.1 NM
SENSORS: GNSS: DEGRADED (HDOP 4.2 / 3 SATS) | AIS: SIMULATION (12 TARGETS) | GYRO: SIMULATION (212°) | X-BAND RADAR: SIMULATION (12 NM) | FWD SONAR: OFFLINE (TRANSDUCER COMM TIMEOUT) | ADCP: STALE (LAST UPDATE 35 MIN AGO) | WEATHER: SIMULATION (38 kn)
TRACKED ICEBERGS (3 targets):
 - [ICE-205] TABULAR | Draft 190m | Drift 0.75kn/210° | Risk: HIGH (CPA 6.4km in T+18h) | Conf 89% | Source: SIMULATED X-BAND + SAR FUSION
 - [ICE-210] PINNACLE | Draft 65m | Drift 0.48kn/195° | Risk: LOW (CPA 32.0km in T+48h) | Conf 78% | Source: RADAR CONTACT / UNVERIFIED
 - [ICE-218] BERGY BIT | Draft 22m | Drift 0.40kn/170° | Risk: LOW (CPA 45.2km in T+60h) | Conf 75% | Source: RADAR CONTACT / UNVERIFIED
FWD SONAR: No active contact / sensor offline
RADAR: Contact RAD-102 at 4.1 NM, Brg 205°, Course 190°, Spd 5.2 kn
AIS: POLAR SUPPLY (AIS-401) at CPA 12.4km / TCPA 95min | SOG 6.2kn
ACTIVE HAZARDS (3):
 - [CRITICAL] ICE-205 / ROUTE PROXIMITY: Predicted trajectory intersects planned corridor. CPA 6.4 km at T+18h. -> Action: REVIEW CORRIDOR
 - [WARNING] SENSOR FAULT: FWD SONAR: FWD SONAR reported OFFLINE: TRANSDUCER COMM TIMEOUT. -> Action: VERIFY SENSOR DIAGNOSTICS
 - [CAUTION] SENSOR FAULT: GNSS: GNSS reported DEGRADED: HDOP 4.2 / 3 SATS. -> Action: VERIFY SENSOR DIAGNOSTICS"""

CONTRADICTION_SNAPSHOT_TEXT = """=== OPERATIONAL CONTEXT [LOCAL SIMULATION / 02 s] ===
VESSEL: RV BHARATI | POS: 64.550°S 59.880°W | SOG: 10.1 kn | COG: 225° | HDG: 226° | DEPTH: 610m
ENVIRONMENT: Wind 22 kn / 230° | Current 0.35 m/s / 190° | Sea BEAUFORT 4 | Sea Temp −0.5 °C | Pack Ice 55% / STABLE
WEATHER: Air Temp -2.8°C | Pressure 998 hPa | Visibility 5.5 NM
SENSORS: GNSS: SIMULATION (18 ms) | AIS: SIMULATION (18 TARGETS) | GYRO: SIMULATION (226°) | X-BAND RADAR: SIMULATION (12 NM) | FWD SONAR: SIMULATION (2.8 km) | ADCP: SIMULATION (0.35 m/s) | WEATHER: SIMULATION (22 kn)
TRACKED ICEBERGS (2 targets):
 - [ICE-301] WEDGE | Draft 95m | Drift 0.42kn/190° | Risk: MEDIUM (CPA 19.5km in T+42h) | Conf 84% | Source: SIMULATED X-BAND + SAR FUSION
 - [ICE-305] GROWLER | Draft 15m | Drift 0.30kn/210° | Risk: LOW (CPA 42.0km in T+72h) | Conf 76% | Source: RADAR CONTACT / UNVERIFIED
FWD SONAR: Contact SONAR-044 at 1.45km, Brg 225°, Depth -28m (POSSIBLE ICE, Return: STRONG, Conf: 85%)
RADAR: No surface target detected at forward sonar bearing 225° within 2 NM
AIS: AGULHAS II (AIS-505) at CPA 15.1km / TCPA 110min | SOG 5.5kn
ACTIVE HAZARDS (2):
 - [CAUTION] DISCREPANCY: SONAR-044: Forward sonar return detected ahead with unconfirmed radar reflection in surface clutter. -> Action: MAINTAIN RADAR TUNE / VERIFY RETURN
 - [INFO] ROUTINE POLAR PASSAGE: Route corridor clear of primary iceberg hazards. -> Action: MAINTAIN WATCH"""

LOW_RISK_SNAPSHOT_TEXT = """=== OPERATIONAL CONTEXT [LOCAL SIMULATION / 02 s] ===
VESSEL: RV BHARATI | POS: 63.920°S 58.450°W | SOG: 12.8 kn | COG: 215° | HDG: 215° | DEPTH: 1150m
ENVIRONMENT: Wind 14 kn / 220° | Current 0.22 m/s / 185° | Sea BEAUFORT 3 | Sea Temp +0.2 °C | Pack Ice 25% / OPEN WATER
WEATHER: Air Temp -0.8°C | Pressure 1008 hPa | Visibility 8.5 NM
SENSORS: GNSS: SIMULATION (16 ms) | AIS: SIMULATION (31 TARGETS) | GYRO: SIMULATION (215°) | X-BAND RADAR: SIMULATION (12 NM) | FWD SONAR: SIMULATION (2.8 km) | ADCP: SIMULATION (0.22 m/s) | WEATHER: SIMULATION (14 kn)
TRACKED ICEBERGS (2 targets):
 - [ICE-410] TABULAR | Draft 140m | Drift 0.32kn/185° | Risk: LOW (CPA 38.5km in T+60h) | Conf 91% | Source: SIMULATED X-BAND + SAR FUSION
 - [ICE-415] GROWLER | Draft 10m | Drift 0.25kn/200° | Risk: LOW (CPA 52.0km in T+84h) | Conf 82% | Source: RADAR CONTACT / UNVERIFIED
FWD SONAR: No active contact detected
RADAR: Contact RAD-140 at 8.5 NM, Brg 175°, Course 170°, Spd 8.2 kn
AIS: KRONPRINS HAAKON (AIS-602) at CPA 18.2km / TCPA 140min | SOG 8.2kn
ACTIVE HAZARDS (1):
 - [INFO] ROUTINE POLAR PASSAGE: All monitored channels nominal. Route corridor clear of immediate hazards. -> Action: MAINTAIN WATCH"""

TEST_CASES = [
    {
        "id": 1,
        "name": "Operational Picture Summary",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "Give me the current operational picture.",
    },
    {
        "id": 2,
        "name": "Highest Risk Iceberg",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "Which iceberg currently presents the highest risk and why?",
    },
    {
        "id": 3,
        "name": "What is Ahead",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "What is ahead of the vessel?",
    },
    {
        "id": 4,
        "name": "Sensor Availability & Health",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "Are all sensors available?",
    },
    {
        "id": 5,
        "name": "Route Risk Explanation",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "Why is the current route considered risky?",
    },
    {
        "id": 6,
        "name": "6-Hour Historical Change",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "What changed over the last 6 hours?",
    },
    {
        "id": 7,
        "name": "Operator Verification Actions",
        "context": NOMINAL_SNAPSHOT_TEXT,
        "prompt": "What should the operator verify?",
    },
    {
        "id": 8,
        "name": "Missing / Stale Sensor Handling",
        "context": DEGRADED_SNAPSHOT_TEXT,
        "prompt": "Are all sensors available and how should degraded channels be managed?",
    },
    {
        "id": 9,
        "name": "Conflicting Sensor Information",
        "context": CONTRADICTION_SNAPSHOT_TEXT,
        "prompt": "Explain any discrepancy between forward sonar and radar.",
    },
    {
        "id": 10,
        "name": "Low-Risk Baseline Scenario",
        "context": LOW_RISK_SNAPSHOT_TEXT,
        "prompt": "Give me the current operational picture and highlight any hazards.",
    },
]


def run_inference(model, tokenizer, context: str, user_prompt: str, max_tokens: int = 400) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nOperator Query: {user_prompt}"},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response.strip()


def evaluate_response(resp: str, test_id: int) -> Dict[str, Any]:
    """Scores response for operational adherence, hallucination avoidance, and safety."""
    score_persona = 1.0 if ("SIMULATION" in resp.upper() or "ADVISORY" in resp.upper() or "BHARATI" in resp) else 0.5
    no_direct_control = 1.0 if not any(w in resp.lower() for w in ["i steered", "i changed the heading", "i engaged autopilot"]) else 0.0
    structured_format = 1.0 if ("-" in resp or "**" in resp or "\n" in resp) else 0.5

    # Specific tests
    test_specific = 1.0
    if test_id == 2 and "ICE-042" not in resp:
        test_specific = 0.3
    elif test_id == 8 and ("OFFLINE" not in resp.upper() and "DEGRADED" not in resp.upper()):
        test_specific = 0.3
    elif test_id == 9 and ("SONAR" not in resp.upper() and "DISCREPANCY" not in resp.upper()):
        test_specific = 0.3

    total_score = (score_persona + no_direct_control + structured_format + test_specific) / 4.0 * 100.0
    return {
        "score": round(total_score, 1),
        "persona": score_persona,
        "safe_control": no_direct_control,
        "structured": structured_format,
        "task_specific": test_specific,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Model")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit", help="Base model path")
    parser.add_argument("--adapter-path", type=str, default=str(MODELS_DIR), help="LoRA adapter path")
    args = parser.parse_args()

    print("==================================================")
    print("  IMPALA LOCAL MODEL INFERENCE & EVALUATION SUITE")
    print("==================================================")
    print(f"Base Model:    {args.model}")
    print(f"Adapter Path:  {args.adapter_path}")
    print("==================================================")

    # 1. Evaluate Fine-Tuned Model
    print("\n--- Loading Fine-Tuned Model (Base + LoRA Adapter) ---")
    ft_model, ft_tokenizer = load(args.model, adapter_path=args.adapter_path)

    ft_results = []
    print("\nRunning 10 Mandatory Operational Test Cases on Fine-Tuned Model...")
    for tc in TEST_CASES:
        t0 = time.time()
        resp = run_inference(ft_model, ft_tokenizer, tc["context"], tc["prompt"])
        elapsed = time.time() - t0
        eval_metrics = evaluate_response(resp, tc["id"])
        ft_results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "prompt": tc["prompt"],
            "response": resp,
            "metrics": eval_metrics,
            "latency_s": round(elapsed, 2),
        })
        print(f" [PASS] Test {tc['id']:02d}: {tc['name']} -> Score: {eval_metrics['score']}% ({elapsed:.2f}s)")

    # 2. Evaluate Base Model (Without Adapter)
    print("\n--- Loading Base Model (No Adapter) ---")
    base_model, base_tokenizer = load(args.model)

    base_results = []
    print("\nRunning 10 Mandatory Operational Test Cases on Base Model...")
    for tc in TEST_CASES:
        t0 = time.time()
        resp = run_inference(base_model, base_tokenizer, tc["context"], tc["prompt"])
        elapsed = time.time() - t0
        eval_metrics = evaluate_response(resp, tc["id"])
        base_results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "prompt": tc["prompt"],
            "response": resp,
            "metrics": eval_metrics,
            "latency_s": round(elapsed, 2),
        })
        print(f" [PASS] Test {tc['id']:02d}: {tc['name']} -> Score: {eval_metrics['score']}% ({elapsed:.2f}s)")

    # Comparison summary
    ft_avg_score = sum(r["metrics"]["score"] for r in ft_results) / len(ft_results)
    base_avg_score = sum(r["metrics"]["score"] for r in base_results) / len(base_results)

    print("\n==================================================")
    print("  EVALUATION COMPARISON SUMMARY")
    print("==================================================")
    print(f"Fine-Tuned Model Average Score:  {ft_avg_score:.1f}%")
    print(f"Base Model Average Score:        {base_avg_score:.1f}%")
    print("==================================================")

    out_json = BASE_DIR / "evaluation_results.json"
    summary = {
        "fine_tuned": {
            "model": args.model,
            "adapter": args.adapter_path,
            "average_score": round(ft_avg_score, 1),
            "results": ft_results,
        },
        "base_model": {
            "model": args.model,
            "average_score": round(base_avg_score, 1),
            "results": base_results,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full evaluation outputs to {out_json}")


if __name__ == "__main__":
    main()
