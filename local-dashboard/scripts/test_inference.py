#!/usr/bin/env python3
"""
scripts/test_inference.py

Tests the local inference server (http://127.0.0.1:8080/v1/chat/completions)
with operational queries and prints formatted output.
"""

import json
import urllib.request
import urllib.error

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are the IMPALA Local Bridge Decision Support Assistant onboard RV BHARATI. "
    "You provide concise, objective, and structured operational assessments based strictly on the provided operational snapshot. "
    "You clearly identify SIMULATION vs LIVE data, acknowledge sensor uncertainty or offline status, "
    "highlight high-risk iceberg targets and route hazards, and never claim direct vessel steering or autopilot control."
)

OPERATIONAL_PROMPT = (
    "=== OPERATIONAL CONTEXT [LOCAL SIMULATION / 02 s] ===\n"
    "VESSEL: RV BHARATI | POS: 64.812°S 60.438°W | SOG: 11.4 kn | COG: 217° | HDG: 219° | DEPTH: 418m\n"
    "ENVIRONMENT: Wind 31 kn / 248° | Current 0.42 m/s / 196° | Sea BEAUFORT 6 | Sea Temp −0.8 °C | Pack Ice 84% / COMPRESSING\n"
    "TRACKED ICEBERGS (1 targets):\n"
    " - [ICE-042] TABULAR | Draft 245m | Drift 0.81kn/204° | Risk: HIGH (CPA 11.2km in T+34h) | Conf 92%\n\n"
    "Operator Query: Which iceberg currently presents the highest risk and why?"
)

def main():
    payload = {
        "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OPERATIONAL_PROMPT},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            print("==================================================")
            print("  LOCAL INFERENCE SERVER RESPONSE (HTTP 200 OK)")
            print("==================================================")
            content = res_json["choices"][0]["message"]["content"]
            print(content)
            print("==================================================")
    except urllib.error.URLError as e:
        print(f"Error contacting local server at {ENDPOINT}: {e}")

if __name__ == "__main__":
    main()
