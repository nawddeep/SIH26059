"""End-to-end test: telemetry -> shore -> backend -> models -> risk -> route -> fuel.

Every other test in this repository exercises one component. This one asserts
the components actually connect, which is a different claim and the one that
tends to be false: individually healthy parts wired together wrongly is the
normal way a system of this shape fails.

The chain under test:

    gateway buffer  ->  shore listener  ->  telemetry API
                                                  |
    sea-ice model  ->  POLARIS risk  ->  route  ->  fuel + explainability

Run with the model environment, from the repository root:

    seaice_forecast/venv/bin/python -m pytest integration/ -v

Tests that need an artifact which is not present skip rather than fail, so a
clone without the 90 MB training checkpoints still reports usefully.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "shipNavigation/backend"
GATEWAY = ROOT / "telemetry-gateway/gateway"
SHORE = ROOT / "telemetry-gateway/shore"

for p in (str(BACKEND), str(ROOT / "seaice_forecast/src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------- telemetry
class TestTelemetryChain:
    """Gateway buffer -> shore wire format -> backend state."""

    def test_buffer_preserves_priority_order(self, tmp_path):
        """A queued position fix must outrank routine telemetry queued earlier."""
        sys.path.insert(0, str(GATEWAY))
        import buffer as buf

        conn = buf.connect(tmp_path / "q.db")
        buf.insert(conn, "engine", "2018-12-20T00:00:00Z", buf.PRIORITY_MEDIUM, {"rpm": 720})
        buf.insert(conn, "gps", "2018-12-20T00:00:01Z", buf.PRIORITY_HIGH,
                   {"lat": -65.0, "lon": -60.0})
        buf.insert(conn, "gps", "2018-12-20T00:00:02Z", buf.PRIORITY_LOW, {"hdop": 0.9})

        rows = buf.select_unsent(conn, 10)
        assert [r["priority"] for r in rows] == [1, 2, 3], "priority must beat arrival order"
        conn.close()

    def test_unacknowledged_records_are_not_marked_sent(self, tmp_path):
        """The guarantee that makes 'no data loss' true rather than hopeful."""
        sys.path.insert(0, str(GATEWAY))
        import buffer as buf

        conn = buf.connect(tmp_path / "q.db")
        ids = [buf.insert(conn, "gps", "t", buf.PRIORITY_HIGH, {"n": i}) for i in range(5)]

        # Shore acknowledged only the first three before the link died.
        buf.mark_sent(conn, ids[:3])

        remaining = [r["id"] for r in buf.select_unsent(conn, 10)]
        assert remaining == ids[3:], "unacknowledged records must stay queued"
        assert buf.stats(conn)["unsent"] == 2
        conn.close()

    def test_shore_wire_format_reaches_backend_state(self):
        """An envelope as sender.py emits it must populate the telemetry API."""
        from app import telemetry

        telemetry.reset()
        for seq, (source, payload) in enumerate([
            ("gps", {"type": "position_fix", "lat": -65.09, "lon": -60.27,
                     "speed_kn": 9.4, "course_deg": 141.0}),
            ("weather", {"wind_speed_kn": 41.2, "visibility_km": 0.8,
                         "alerts": ["gale_force_wind"]}),
            ("engine", {"rpm": 726.8, "fuel_remaining_pct": 62.0}),
        ], start=1):
            telemetry.ingest({"seq": seq, "source": source, "timestamp": "t",
                              "priority": 1, "payload": payload})

        live = telemetry.live()
        assert live["connected"]
        assert live["position"]["lat"] == pytest.approx(-65.09)
        assert live["weather"]["wind_speed_kn"] == pytest.approx(41.2)
        assert live["engine"]["rpm"] == pytest.approx(726.8)
        assert any(a["alert"] == "gale_force_wind" for a in live["alerts"]), \
            "an alert must survive the whole chain - it is the highest-priority traffic"

    def test_shore_listener_acknowledges_and_deduplicates(self):
        """Round-trip a record through the real shore listener over a socket."""
        pytest.importorskip("asyncio")
        sys.path.insert(0, str(SHORE))
        import asyncio

        import shore_listener as sl

        port = free_port()
        received: list = []

        async def run():
            server = await asyncio.start_server(sl.handle_client, "127.0.0.1", port)
            async with server:
                await asyncio.sleep(1.5)

        def serve():
            asyncio.run(run())

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        time.sleep(0.4)

        env = {"seq": 1, "source": "gps", "timestamp": "t", "priority": 1,
               "payload": {"lat": -65.0, "lon": -60.0, "speed_kn": 9.0}}
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall((json.dumps(env) + "\n").encode())
            s.sendall((json.dumps(env) + "\n").encode())   # deliberate duplicate
            s.settimeout(3)
            acks = b""
            while acks.count(b"\n") < 2:
                chunk = s.recv(1024)
                if not chunk:
                    break
                acks += chunk

        assert acks.count(b"ACK 1") == 2, "every record must be acknowledged, even a repeat"
        assert sl.duplicates >= 1, "shore must notice the duplicate rather than double-count"


# ------------------------------------------------------- models -> routing
@pytest.fixture(scope="module")
def bridge():
    mb = pytest.importorskip("app.model_bridge")
    if not mb.models_available():
        pytest.skip("model artifacts not present (90 MB checkpoints are not in git)")
    return mb


class TestModelToRouteChain:
    """Sea ice -> POLARIS risk -> route cost -> fuel, as the planner uses them."""

    def test_sea_ice_feeds_polaris_risk(self, bridge):
        from app.route_engine import polaris_risk_at

        sic = bridge.sic_at_point(-70.0, 0.0)
        assert 0.0 <= sic <= 100.0, "sic_at_point returns a percentage"

        weak = polaris_risk_at(-70.0, 0.0, ice_class="none")
        strong = polaris_risk_at(-70.0, 0.0, ice_class="pc3")
        assert 0.0 <= strong <= weak <= 1.0, \
            "a stronger hull must never be assessed as more at risk in the same ice"

    def test_risk_rises_with_ice_and_fuel_follows(self, bridge):
        from seaice_forecast.fuel import fuel_per_cell_array, speed_in_ice_array
        from seaice_forecast.risk.polaris import risk_ice

        light, heavy = 0.10, 0.90
        assert risk_ice(light, polar_class="PC4") < risk_ice(heavy, polar_class="PC4")

        f_light = float(fuel_per_cell_array(25.0, light, polar_class="PC4"))
        f_heavy = float(fuel_per_cell_array(25.0, heavy, polar_class="PC4"))
        s_light = float(speed_in_ice_array(light, polar_class="PC4"))
        s_heavy = float(speed_in_ice_array(heavy, polar_class="PC4"))

        assert f_heavy > f_light, "heavier ice must cost more fuel"
        assert s_heavy < s_light, "heavier ice must slow the vessel"
        # The two must move together: burn rises *because* speed collapses.
        assert (f_heavy / f_light) > 1.5

    def test_route_costs_against_the_model_not_a_substitute(self, bridge):
        """The ice field behind a route must be identified, never silently faked."""
        from app.route_engine import ice_data_source

        source = ice_data_source()
        assert "synthetic" not in source.lower(), \
            "a route must never be costed against simulated ice"
        assert "UNet" in source or "unavailable" in source

    def test_drift_model_predicts_on_real_fixes(self, bridge):
        import numpy as np
        import pandas as pd

        bundle, frame = bridge._drift_bundle()
        rows = frame.dropna(subset=bundle["features"]).head(50)
        assert len(rows) > 0

        moving = bundle["clf"].predict(rows[bundle["features"]].to_numpy(dtype=float))
        assert set(np.unique(moving)) <= {0, 1}
        u = bundle["reg"]["u_berg"].predict(rows[bundle["features"]].to_numpy(dtype=float))
        speed_kn = np.hypot(u, 0.0) * 1.94384
        assert (speed_kn < 3.0).all(), "icebergs do not drift at 3 knots"


# ------------------------------------------------------------- artifacts
class TestExportedArtifacts:
    """The four .pkl files are the deliverable; they must work standalone."""

    def test_every_artifact_loads_and_predicts(self):
        joblib = pytest.importorskip("joblib")
        manifest_p = ROOT / "model_exports/manifest.json"
        if not manifest_p.exists():
            pytest.skip("model_exports/manifest.json not present")

        manifest = json.loads(manifest_p.read_text())
        for entry in manifest["artifacts"]:
            path = ROOT / "model_exports" / entry["file"]
            if not path.exists():
                pytest.skip(f"{entry['file']} not present")
            obj = joblib.load(path)
            assert hasattr(obj, "predict"), f"{entry['file']} must expose predict()"
            assert obj.kind in ("trained", "deterministic")

    def test_deterministic_artifacts_are_self_consistent(self):
        joblib = pytest.importorskip("joblib")
        risk_p = ROOT / "model_exports/polaris_risk.pkl"
        fuel_p = ROOT / "model_exports/fuel_consumption.pkl"
        if not (risk_p.exists() and fuel_p.exists()):
            pytest.skip("deterministic artifacts not present")

        risk = joblib.load(risk_p)
        fuel = joblib.load(fuel_p)

        # The same physical story must hold across two independently exported
        # artifacts: more ice is riskier AND costlier, for the same hull.
        for pc in ("PC3", "PC7"):
            assert risk.predict(0.9, pc) > risk.predict(0.1, pc)
            assert fuel.predict(0.9, pc, 25.0)["fuel_tonnes"] > \
                   fuel.predict(0.1, pc, 25.0)["fuel_tonnes"]
            assert fuel.predict(0.9, pc, 25.0)["speed_knots"] < \
                   fuel.predict(0.1, pc, 25.0)["speed_knots"]


# --------------------------------------------------------- system status
class TestSystemStatus:
    def test_status_reports_every_component_and_is_not_realtime(self):
        pytest.importorskip("fastapi")
        from app import main as app_main
        import asyncio

        status = asyncio.run(app_main.system_status())
        for key in ("sea_ice", "iceberg", "polaris", "fuel", "routing", "telemetry"):
            assert key in status, f"{key} missing from /api/system-status"
        assert status["realtime"] is False, \
            "the archive ends in 2018; this must never claim to be real-time"
        if status["environment_data_date"]:
            datetime.strptime(status["environment_data_date"], "%Y-%m-%d")
