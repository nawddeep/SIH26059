"""Routing behaviour in the cases where being wrong is expensive.

Ordinary routing tests check that a plausible route comes back. These check what
happens at the edges - impassable ice, a hull that should not be there, an
iceberg on the track, no environmental data at all - because those are the
inputs where a navigation system either refuses honestly or quietly returns
something that looks fine and is not.

Expected behaviour is asserted explicitly, so a change in it is a test failure
rather than a surprise at sea.

    seaice_forecast/venv/bin/python -m pytest shipNavigation/backend/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
for p in (str(BACKEND), str(ROOT / "seaice_forecast/src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ------------------------------------------------------- POLARIS / ice risk
class TestIceRiskBoundaries:
    def test_risk_is_monotonic_in_concentration(self):
        """More ice is never safer. If this inverts, every route is suspect."""
        from seaice_forecast.risk.polaris import risk_ice

        prev = -1.0
        for sic in (0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0):
            r = float(risk_ice(sic, polar_class="PC4"))
            assert r >= prev, f"risk fell from {prev} to {r} as ice rose to {sic}"
            prev = r

    def test_extreme_ice_is_high_risk_for_every_hull(self):
        """At 100% concentration no hull class may be reported as low risk."""
        from seaice_forecast.risk.polaris import risk_ice

        for pc in ("PC1", "PC4", "PC7", "UNCLASSED"):
            assert float(risk_ice(1.0, polar_class=pc)) >= 0.5, \
                f"{pc} reported low risk in complete ice cover"

    def test_weaker_hull_is_never_safer_in_the_same_ice(self):
        from seaice_forecast.risk.polaris import risk_ice

        ladder = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "UNCLASSED"]
        for sic in (0.3, 0.6, 0.9):
            risks = [float(risk_ice(sic, polar_class=pc)) for pc in ladder]
            assert risks == sorted(risks), \
                f"risk is not monotonic across the hull ladder at SIC {sic}: {risks}"

    def test_open_water_carries_no_ice_risk(self):
        from seaice_forecast.risk.polaris import risk_ice
        assert float(risk_ice(0.0, polar_class="UNCLASSED")) == 0.0


# --------------------------------------------------------------- fuel model
class TestFuelBoundaries:
    def test_fuel_rises_and_speed_falls_without_exception(self):
        """Monotonic across the whole range - no crossover, no NaN, no blow-up."""
        import numpy as np
        from seaice_forecast.fuel import fuel_per_cell_array, speed_in_ice_array

        sic = np.linspace(0.0, 1.0, 101)
        fuel = fuel_per_cell_array(25.0, sic, polar_class="PC4")
        spd = speed_in_ice_array(sic, polar_class="PC4")

        assert np.all(np.isfinite(fuel)) and np.all(np.isfinite(spd))
        assert np.all(np.diff(fuel) > 0), "fuel must rise strictly with ice"
        assert np.all(np.diff(spd) < 0), "speed must fall strictly with ice"
        assert np.all(spd > 0), "attainable speed must never reach zero or invert"

    def test_extreme_fuel_penalty_is_bounded(self):
        """Heavy ice must cost much more, but a finite amount - not infinity."""
        from seaice_forecast.fuel import fuel_per_cell_array

        open_water = float(fuel_per_cell_array(25.0, 0.0, polar_class="UNCLASSED"))
        full_ice = float(fuel_per_cell_array(25.0, 1.0, polar_class="UNCLASSED"))
        ratio = full_ice / open_water
        assert 1.5 < ratio < 100.0, f"implausible ice fuel penalty: {ratio:.1f}x"

    def test_stronger_hull_burns_less_in_identical_ice(self):
        from seaice_forecast.fuel import fuel_per_cell_array
        strong = float(fuel_per_cell_array(25.0, 0.95, polar_class="PC1"))
        weak = float(fuel_per_cell_array(25.0, 0.95, polar_class="UNCLASSED"))
        assert strong < weak


# ------------------------------------------------------- degraded data paths
class TestDegradedData:
    def test_missing_ice_model_refuses_rather_than_inventing(self, monkeypatch):
        """With no sea-ice field, the cost lookup must raise - never substitute.

        This is the single most important assertion here. A route costed against
        a fabricated ice field is indistinguishable from a real one, and that is
        the failure most likely to put a vessel somewhere it should not be.
        """
        from app import route_engine

        def boom(*_a, **_k):
            raise RuntimeError("sea-ice model unavailable")

        monkeypatch.setattr(route_engine, "sic_at_point", boom, raising=False)
        import app.model_bridge as mb
        monkeypatch.setattr(mb, "sic_at_point", boom)

        with pytest.raises(Exception):
            route_engine.get_sea_ice_concentration(-70.0, 0.0)

    def test_ice_data_source_never_claims_synthetic(self):
        """The provenance string must name a real product or admit it has none."""
        from app.route_engine import ice_data_source
        src = ice_data_source().lower()
        assert "synthetic" not in src and "mock" not in src and "fake" not in src

    def test_no_bergs_yields_no_iceberg_term_not_fake_ones(self, monkeypatch):
        from app import route_engine

        route_engine.current_iceberg_positions.cache_clear()
        monkeypatch.setattr(
            route_engine, "_drift_bundle",
            lambda: (_ for _ in ()).throw(RuntimeError("drift model unavailable")),
            raising=False,
        )
        import app.model_bridge as mb
        monkeypatch.setattr(
            mb, "_drift_bundle",
            lambda: (_ for _ in ()).throw(RuntimeError("drift model unavailable")),
        )
        route_engine.current_iceberg_positions.cache_clear()
        assert route_engine.current_iceberg_positions() == (), \
            "an unavailable drift model must yield no bergs, never placeholder ones"
        route_engine.current_iceberg_positions.cache_clear()


# ------------------------------------------------------ degenerate requests
class TestDegenerateRequests:
    @pytest.fixture(scope="class")
    def engine(self):
        from app.datastore import Datastore
        from app.mask import LandMask
        from app.route_engine import RouteEngine
        try:
            store = Datastore()
            mask = LandMask()
            # Same construction main.py uses at startup, so these tests exercise
            # the engine the API actually serves rather than a simplified one.
            return RouteEngine(mask, store.ecas, store.chokepoints,
                               waterways=store.waterways)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"route engine unavailable: {exc}")

    @staticmethod
    def _wp(label, lat, lon):
        return {"label": label, "lat": lat, "lon": lon}

    def _route(self, engine, a, b, **kw):
        from datetime import datetime
        return engine.compute_route(
            [self._wp("A", *a), self._wp("B", *b)],
            speed_knots=12.0,
            departure_time_utc=datetime(2018, 12, 20),
            **kw,
        )

    def test_identical_start_and_destination(self, engine):
        """Zero-length route: near-zero distance, or a clean refusal. Never a crash."""
        from app.route_engine import RouteError
        try:
            res = self._route(engine, (-65.0, -60.0), (-65.0, -60.0))
        except RouteError:
            return                      # refusing is acceptable
        except Exception as exc:        # noqa: BLE001
            pytest.fail(f"degenerate request raised {type(exc).__name__}: {exc}")
        assert res["totalDistanceNm"] < 5.0

    def test_land_locked_destination_is_refused_not_faked(self, engine):
        """A destination deep inside a continent must fail loudly, not return a path."""
        from app.route_engine import RouteError
        with pytest.raises((RouteError, ValueError)):
            self._route(engine, (-65.0, -60.0), (-82.0, 0.0))   # interior Antarctica

    def test_every_route_carries_its_explanation_and_provenance(self, engine):
        """A returned route must always say what it was costed against."""
        from app.route_engine import RouteError
        try:
            res = self._route(engine, (-62.0, -58.0), (-65.0, -62.0),
                              optimize_for="safety", ice_class="pc5")
        except RouteError:
            pytest.skip("no route available between these points")
        assert "explanation" in res, "a route without reasoning is not decision support"
        exp = res["explanation"]
        assert exp["iceDataSource"], "the ice field behind a route must be named"
        assert "synthetic" not in exp["iceDataSource"].lower()
        assert exp["reason"] and exp["caveat"]
        assert 0.0 <= exp["maxIceRisk"] <= 1.0


# -------------------------------------------------------- explainability
class TestRouteExplanation:
    def test_explanation_reports_provenance_and_a_caveat(self):
        from app.route_engine import explain_route

        summary = {
            "totalDistanceNm": 500.0, "totalDistanceInSeaIceNm": 200.0,
            "maxPolarisRisk": 0.72, "meanPolarisRisk": 0.31,
            "maxIceFuelPenalty": 2.4, "estimatedFuelTons": 60.0,
            "totalDurationHours": 40.0, "maxSeaIceConcentrationPct": 80.0,
            "iceDataSource": "UNet+ConvLSTM forecast (NSIDC CDR v6)",
        }
        exp = explain_route(summary, "safety", "pc5", (), [])

        assert exp["iceDataSource"]
        assert "advisory" in exp["caveat"].lower()
        assert "not a live feed" in exp["caveat"]
        # High POLARIS risk must be named as the dominant term, not buried.
        assert "polaris" in exp["dominantCostTerm"].lower()

    def test_high_risk_route_is_not_described_as_safe(self):
        from app.route_engine import explain_route
        summary = {
            "totalDistanceNm": 500.0, "totalDistanceInSeaIceNm": 400.0,
            "maxPolarisRisk": 0.95, "meanPolarisRisk": 0.8,
            "maxIceFuelPenalty": 4.0, "estimatedFuelTons": 120.0,
            "totalDurationHours": 60.0, "maxSeaIceConcentrationPct": 98.0,
            "iceDataSource": "UNet+ConvLSTM forecast (NSIDC CDR v6)",
        }
        exp = explain_route(summary, "safety", "unclassed", (), [])
        assert exp["maxIceRisk"] >= 0.9
        assert "safe" not in exp["reason"].lower().replace("safety", "")


# ------------------------------------------- does each objective win its own metric?
class TestObjectivesWinTheirOwnMetric:
    """An objective that loses on the metric it optimises is mis-calibrated.

    This is the acceptance criterion for the routing layer: ask for the safest
    route and it should carry the lowest peak ice risk; ask for the cheapest and
    it should burn least. Encoded here so the requirement is visible and
    checkable rather than living in a review comment.

    The safety case is marked xfail because it genuinely fails today. Marking it
    rather than omitting it keeps the gap in the test output, where it belongs -
    a requirement nobody has written down is a requirement nobody will fix.
    """

    ROUTES = [
        ("Peninsula NE-SW", (-62.0, -58.0), (-67.0, -68.0)),
        ("Weddell approach", (-60.0, -45.0), (-70.0, -40.0)),
    ]

    @pytest.fixture(scope="class")
    def engine(self):
        from app.datastore import Datastore
        from app.mask import LandMask
        from app.route_engine import RouteEngine
        try:
            store = Datastore()
            mask = LandMask()
            return RouteEngine(mask, store.ecas, store.chokepoints,
                               waterways=store.waterways)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"route engine unavailable: {exc}")

    def _plan(self, engine, a, b, objective):
        from datetime import datetime
        from app.route_engine import RouteError
        try:
            return engine.compute_route(
                [{"label": "A", "lat": a[0], "lon": a[1]},
                 {"label": "B", "lat": b[0], "lon": b[1]}],
                speed_knots=12.0, departure_time_utc=datetime(2018, 12, 20),
                optimize_for=objective, vessel_type="research",
                draft_meters=7.0, ice_class="pc5",
            )
        except RouteError:
            return None

    @pytest.mark.xfail(
        reason=(
            "Known defect, smaller than the safety one but real. On the Peninsula "
            "passage the 'fuel' objective burns 37.6 t while 'time' burns 37.0 t - "
            "1.6% worse on the metric it exists to minimise. Both are grid-searched, "
            "so this is not the great-circle artefact that explains 'distance' "
            "winning: it is the fuel cost's own weighting. The ice fuel multiplier "
            "and the cubic water-speed term are applied to the same edge, and the "
            "combination appears to over-penalise light-ice cells that the time "
            "objective is happy to cross. Not yet isolated."
        ),
        strict=False,
    )
    def test_fuel_objective_burns_least(self, engine):
        """Compared only against other grid-searched objectives.

        'distance' is excluded because when the great-circle track is land-free,
        _leg_path returns it directly and never enters A*, so it is not
        grid-constrained while every other objective is. That exclusion is about
        comparing like with like - it does not rescue this test, which fails
        against 'time' regardless.
        """
        for name, a, b in self.ROUTES:
            routes = {o: self._plan(engine, a, b, o) for o in ("time", "fuel", "safety")}
            got = {o: r["estimatedFuelTons"] for o, r in routes.items() if r}
            if len(got) < 2:
                continue
            best = min(got, key=got.get)
            # A tenth of a percent is grid noise, not a calibration fault.
            assert got["fuel"] <= got[best] * 1.001, (
                f"{name}: 'fuel' burns {got['fuel']} t, '{best}' burns {got[best]} t"
            )

    @pytest.mark.xfail(
        reason=(
            "Known defect. The safety objective returns a track with HIGHER peak "
            "POLARIS risk than the shortest route - measured 0.541 vs 0.373 on the "
            "Peninsula passage. A grid path at 0.373 demonstrably exists: the "
            "distance, fuel and time objectives all find it in ~25 cells, while "
            "safety takes 198 cells to reach 0.843. Raising the risk weight makes "
            "it worse, not better (40x -> 0.541, 99999x -> 0.854), which is the "
            "signature of a min-sum objective being used for a minimax problem. "
            "See docs/ROUTE_COST.md."
        ),
        strict=False,
    )
    def test_safety_objective_carries_lowest_peak_risk(self, engine):
        for name, a, b in self.ROUTES:
            routes = {o: self._plan(engine, a, b, o)
                      for o in ("distance", "fuel", "safety")}
            got = {o: r["maxPolarisRisk"] for o, r in routes.items() if r}
            if len(got) < 2 or "safety" not in got:
                continue
            best = min(got, key=got.get)
            assert got["safety"] <= got[best] + 1e-6, (
                f"{name}: 'safety' peak risk {got['safety']}, "
                f"'{best}' achieves {got[best]}"
            )
