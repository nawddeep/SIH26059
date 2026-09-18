import React, { useState } from 'react'
import { Plus, Settings as SettingsIcon, Route as RouteIcon, Anchor } from 'lucide-react'
import WaypointRow from './WaypointRow'
import ResultsPanel from './ResultsPanel'
import VesselSelector from './VesselSelector'
import ModelStatusBar from './ModelStatusBar'
import { usePlanner } from '../store'

function PlannerTab() {
  const {
    waypoints, setWaypoint, addWaypoint, removeWaypoint, moveWaypoint,
    speedKnots, setSpeedKnots, departureTimeUTC, setDepartureTimeUTC,
    optimizeFor, setOptimizeFor, route, routeLoading, routeError,
  } = usePlanner()

  const [dragIndex, setDragIndex] = useState(null)

  function onDrop(index) {
    if (dragIndex !== null && dragIndex !== index) moveWaypoint(dragIndex, index)
    setDragIndex(null)
  }

  return (
    <div className="planner-body">
      <VesselSelector />

      <section className="section stops">
        <div className="section-head">
          <span className="section-title">Route points</span>
          <span className="section-hint">{waypoints.length} pts</span>
        </div>
        <div className="waypoint-list">
          {waypoints.map((w, i) => (
            <WaypointRow
              key={w.id}
              index={i}
              waypoint={w}
              canRemove={waypoints.length > 2 || typeof w.lat === 'number'}
              onChange={(patch) => setWaypoint(i, patch)}
              onRemove={() => removeWaypoint(i)}
              onDragStart={setDragIndex}
              onDragOver={setDragIndex}
              onDrop={onDrop}
              dragOver={dragIndex !== null && i !== dragIndex}
            />
          ))}
        </div>
        <button className="add-point" onClick={() => addWaypoint()}>
          <Plus size={15} /> Add new point
        </button>
      </section>

      <section className="section">
        <div className="section-head">
          <span className="section-title">Speed</span>
        </div>
        <div className="speed-row">
          <input
            type="range" min={1} max={24} step={1}
            value={speedKnots}
            onChange={(e) => setSpeedKnots(Number(e.target.value))}
          />
          <span className="speed-chip">{speedKnots} kn</span>
        </div>
        <div className="speed-scale"><span>1 kn</span><span>24 kn</span></div>
      </section>

      <section className="section">
        <div className="section-head">
          <span className="section-title">Departure (UTC)</span>
        </div>
        <div className="departure-row">
          <input
            type="datetime-local"
            value={departureTimeUTC}
            onChange={(e) => setDepartureTimeUTC(e.target.value)}
          />
          <button className="now-btn" onClick={() => setDepartureTimeUTC(new Date().toISOString().slice(0, 16))}>now</button>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <span className="section-title">Optimize for</span>
        </div>
        <select
          className="opt-select"
          value={optimizeFor}
          onChange={(e) => setOptimizeFor(e.target.value)}
        >
          <option value="distance">📏 Shortest distance</option>
          <option value="time">⚡ Fastest (Weather & Currents)</option>
          <option value="fuel">⛽ Most fuel efficient</option>
          <option value="safety">🛡️ Maximum safety (Extreme Weather Avoidance)</option>
        </select>
      </section>

      <ResultsPanel route={route} loading={routeLoading} error={routeError} waypoints={waypoints} />
    </div>
  )
}

function SettingsTab() {
  const {
    basemap, setBasemap,
    showSeamarks, setShowSeamarks,
    showWindFlow, setShowWindFlow,
    showEca, setShowEca,
    showChokepoints, setShowChokepoints,
    showChevrons, setShowChevrons,
    showSeaIce, setShowSeaIce,
  } = usePlanner()

  return (
    <div className="planner-body settings-body">
      <section className="section">
        <div className="section-head"><span className="section-title">Basemap</span></div>
        <div className="basemap-grid">
          {[
            ['google-sat', 'Google imagery', 'Satellite'],
            ['google-hybrid', 'Imagery + labels', 'Hybrid'],
            ['google-streets', 'Google roads', 'Streets'],
            ['standard', 'OpenStreetMap', 'Standard'],
            ['opentopo', 'OpenTopoMap relief', 'Topo'],
            ['light', 'CARTO light', 'Light'],
          ].map(([key, desc, label]) => (
            <button key={key} className={`basemap-card ${basemap === key ? 'active' : ''}`} onClick={() => setBasemap(key)}>
              <span className="bm-preview"><span className={`bm-dot ${key}`} /></span>
              <span className="bm-label">{label}</span>
              <span className="bm-desc">{desc}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-head"><span className="section-title">Overlays</span></div>
        <label className="toggle-row">
          <span>Seamarks (buoys, lights, depths)</span>
          <input type="checkbox" checked={showSeamarks} onChange={(e) => setShowSeamarks(e.target.checked)} />
        </label>
        <label className="toggle-row">
          <span>Animated wind flow</span>
          <input type="checkbox" checked={showWindFlow} onChange={(e) => setShowWindFlow(e.target.checked)} />
        </label>
        <label className="toggle-row">
          <span>Sea Ice concentration (cyan)</span>
          <input type="checkbox" checked={showSeaIce} onChange={(e) => setShowSeaIce(e.target.checked)} />
        </label>
        <label className="toggle-row">
          <span>Emission Control Areas (orange)</span>
          <input type="checkbox" checked={showEca} onChange={(e) => setShowEca(e.target.checked)} />
        </label>
        <label className="toggle-row">
          <span>Chokepoint crossings</span>
          <input type="checkbox" checked={showChokepoints} onChange={(e) => setShowChokepoints(e.target.checked)} />
        </label>
        <label className="toggle-row">
          <span>Directional route arrows</span>
          <input type="checkbox" checked={showChevrons} onChange={(e) => setShowChevrons(e.target.checked)} />
        </label>
      </section>

      <section className="section about-box">
        <p><strong>Ship Route Planner</strong> — v1</p>
        <p>
          Geodesic polyline routing with vessel-specific constraints and land avoidance on a 0.5° navigation grid.
        </p>
        <p className="attribution">
          Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ·
          CARTO · Ports: NGA World Port Index (public domain) · Land: Natural Earth.
        </p>
      </section>
    </div>
  )
}

export default function RoutePlannerPanel() {
  const [tab, setTab] = useState('planner')
  return (
    <aside className="panel">
      <header className="panel-header">
        <div className="brand">
          <Anchor size={18} className="brand-icon" />
          <span className="brand-name">Route Planner</span>
        </div>
        <nav className="tabs">
          <button className={`tab ${tab === 'planner' ? 'active' : ''}`} onClick={() => setTab('planner')}>
            <RouteIcon size={14} /> Planner
          </button>
          <button className={`tab ${tab === 'settings' ? 'active' : ''}`} onClick={() => setTab('settings')}>
            <SettingsIcon size={14} /> Settings
          </button>
        </nav>
        <ModelStatusBar />
      </header>
      {tab === 'planner' ? <PlannerTab /> : <SettingsTab />}
    </aside>
  )
}