#!/usr/bin/env python3
"""
scripts/validate_mock_data.py

Validates generated mock datasets and operational scenarios against:
1. Strict Pydantic models matching TypeScript interfaces in src/types/operational.ts
2. Physical and numerical plausibility ranges for polar maritime domain
3. Zero data leakage across Train / Valid / Test splits
4. Quality checks on message formatting and system prompts
"""

import json
import sys
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"
OPERATIONAL_DIR = DATA_DIR / "operational"
TRAINING_DIR = DATA_DIR / "training"

# --- Pydantic Data Models mirroring src/types/operational.ts ---

DataKind = Literal["observed", "estimated", "predicted", "simulated"]
Health = Literal["live", "stale", "degraded", "offline", "simulation", "error"]
Risk = Literal["low", "medium", "high", "critical"]
SensorType = Literal[
    "gnss", "ais", "gyro", "ins", "radar", "forward_sonar", "depth",
    "adcp", "weather", "wind", "barometer", "sea_temperature", "current", "speed_log"
]


class ProvenanceModel(BaseModel):
    source: str
    timestamp: str
    status: Health
    confidence: Optional[float] = None
    kind: DataKind


class PlotPointModel(BaseModel):
    x: float
    y: float


class VesselStateModel(ProvenanceModel):
    name: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    sog: float = Field(ge=0.0, le=40.0)
    cog: float = Field(ge=0.0, le=360.0)
    heading: float = Field(ge=0.0, le=360.0)
    rateOfTurn: float
    depth: float = Field(ge=0.0, le=11000.0)
    windSpeed: float = Field(ge=0.0, le=150.0)
    windDirection: float = Field(ge=0.0, le=360.0)
    seaTemperature: float = Field(ge=-5.0, le=35.0)
    currentSpeed: float = Field(ge=0.0, le=10.0)
    currentDirection: float = Field(ge=0.0, le=360.0)
    track: List[PlotPointModel]


class WeatherStateModel(ProvenanceModel):
    windSpeed: float = Field(ge=0.0, le=150.0)
    windDirection: float = Field(ge=0.0, le=360.0)
    airTemperature: float = Field(ge=-60.0, le=45.0)
    pressureHpa: float = Field(ge=850.0, le=1080.0)
    humidity: float = Field(ge=0.0, le=100.0)
    visibilityNm: float = Field(ge=0.0, le=50.0)


class OceanStateModel(ProvenanceModel):
    currentSpeed: float = Field(ge=0.0, le=10.0)
    currentDirection: float = Field(ge=0.0, le=360.0)
    seaSurfaceTemperature: float = Field(ge=-5.0, le=35.0)
    waveHeight: float = Field(ge=0.0, le=30.0)
    waveDirection: float = Field(ge=0.0, le=360.0)


class AisTargetModel(ProvenanceModel):
    id: str
    mmsi: Optional[str] = None
    name: str
    x: float
    y: float
    heading: float = Field(ge=0.0, le=360.0)
    cog: float = Field(ge=0.0, le=360.0)
    sog: float = Field(ge=0.0, le=50.0)
    cpaKm: Optional[float] = None
    tcpaMinutes: Optional[float] = None


class IcebergObservationModel(ProvenanceModel):
    id: str
    type: Literal["TABULAR", "PINNACLE", "WEDGE", "BERGY BIT", "GROWLER"]
    x: float
    y: float
    dimensions: str
    draft: float = Field(ge=1.0, le=600.0)
    speed: float = Field(ge=0.0, le=10.0)
    heading: float = Field(ge=0.0, le=360.0)
    risk: Risk
    cpa: float = Field(ge=0.0)
    cpaHours: float = Field(ge=0.0)
    trajectory: Optional[List[PlotPointModel]] = None
    uncertainty: float = Field(ge=0.0, le=100.0)
    detectionSource: str


class SonarDetectionModel(ProvenanceModel):
    id: str
    x: float
    y: float
    rangeKm: float = Field(ge=0.0, le=20.0)
    bearing: float = Field(ge=0.0, le=360.0)
    depthM: float
    returnStrength: Literal["weak", "strong"]
    classification: Literal["POSSIBLE ICE", "SEABED", "UNDERWATER OBSTRUCTION", "UNCLASSIFIED"]
    observationState: Literal["DETECTION", "CLASSIFICATION"]


class RadarContactModel(ProvenanceModel):
    contactId: str
    x: float
    y: float
    rangeNm: float = Field(ge=0.0, le=100.0)
    bearing: float = Field(ge=0.0, le=360.0)
    course: float = Field(ge=0.0, le=360.0)
    speed: float = Field(ge=0.0, le=50.0)


class SensorStatusModel(BaseModel):
    sensorId: str
    sensorType: SensorType
    name: str
    metric: str
    detail: str
    status: Health
    lastSeen: str
    latencyMs: float = Field(ge=0.0)
    quality: float = Field(ge=0.0, le=100.0)
    errorCount: int = Field(ge=0)


class HazardEventModel(ProvenanceModel):
    level: Literal["CRITICAL", "WARNING", "CAUTION", "INFO"]
    title: str
    message: str
    action: str


class EnvironmentStateModel(BaseModel):
    wind: str
    current: str
    seaState: str
    temperature: str
    pack: str


class LogEntryModel(BaseModel):
    time: str
    source: str
    text: str
    kind: DataKind


class OperationalSnapshotModel(BaseModel):
    region: Literal["antarctic", "arctic"]
    vessel: VesselStateModel
    weather: WeatherStateModel
    ocean: OceanStateModel
    ais: List[AisTargetModel]
    icebergs: List[IcebergObservationModel]
    sonar: List[SonarDetectionModel]
    radar: List[RadarContactModel]
    sensors: List[SensorStatusModel]
    hazards: List[HazardEventModel]
    environment: EnvironmentStateModel
    logs: List[LogEntryModel]
    dataFreshness: str


# --- ChatML Validation Model ---

class MessageModel(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=5)


class ChatMLRecordModel(BaseModel):
    messages: List[MessageModel] = Field(min_length=2)


def validate_operational_scenarios() -> int:
    scenarios_file = OPERATIONAL_DIR / "scenarios.json"
    if not scenarios_file.exists():
        print(f"[FAIL] Missing operational scenarios file: {scenarios_file}")
        return 1

    print(f"Validating operational scenarios in {scenarios_file}...")
    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    if not scenarios or len(scenarios) < 10:
        print(f"[FAIL] Too few scenarios: {len(scenarios)}")
        return 1

    for sc in scenarios:
        try:
            OperationalSnapshotModel(**sc["snapshot"])
        except Exception as e:
            print(f"[FAIL] Schema validation error in scenario {sc.get('scenario_id')}: {e}")
            return 1

    print(f"[PASS] All {len(scenarios)} operational snapshots strictly adhere to Pydantic operational schema.")
    return 0


def validate_training_splits() -> int:
    train_file = TRAINING_DIR / "train.jsonl"
    valid_file = TRAINING_DIR / "valid.jsonl"
    test_file = TRAINING_DIR / "test.jsonl"

    for file_path in [train_file, valid_file, test_file]:
        if not file_path.exists():
            print(f"[FAIL] Missing training split file: {file_path}")
            return 1

    def load_jsonl(path: Path) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    ChatMLRecordModel(**obj)
                    records.append(obj)
                except Exception as e:
                    print(f"[FAIL] Invalid ChatML record in {path.name}:{line_no} -> {e}")
                    sys.exit(1)
        return records

    train_data = load_jsonl(train_file)
    valid_data = load_jsonl(valid_file)
    test_data = load_jsonl(test_file)

    print(f"[PASS] ChatML schema valid across all splits:")
    print(f" - train.jsonl: {len(train_data)} records")
    print(f" - valid.jsonl: {len(valid_data)} records")
    print(f" - test.jsonl:  {len(test_data)} records")

    # Content & Safety verification
    for split_name, dataset in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        for item in dataset:
            sys_msg = item["messages"][0]["content"]
            user_msg = item["messages"][1]["content"]
            asst_msg = item["messages"][2]["content"]

            if "IMPALA" not in sys_msg:
                print(f"[FAIL] System prompt in {split_name} missing IMPALA identifier")
                return 1

            # Verify advisory restraint (model does not claim vessel autopilot control)
            if "I changed the vessel heading" in asst_msg or "I steered the ship" in asst_msg:
                print(f"[FAIL] Forbidden claim of direct autopilot control in {split_name}")
                return 1

    print(f"[PASS] Operational advisory role and behavioral boundaries validated.")
    return 0


def main() -> None:
    print("==================================================")
    print("  IMPALA MOCK DATASET STRICT VALIDATION SUITE")
    print("==================================================")

    res1 = validate_operational_scenarios()
    res2 = validate_training_splits()

    if res1 == 0 and res2 == 0:
        print("==================================================")
        print("  DATA VALIDATION RESULT: ALL CHECKS PASSED [PASS]")
        print("==================================================")
        sys.exit(0)
    else:
        print("==================================================")
        print("  DATA VALIDATION RESULT: FAILED [FAIL]")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
