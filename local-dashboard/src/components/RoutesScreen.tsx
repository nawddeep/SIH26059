import { useState } from 'react'
import { useSimulation } from '../data/simulationStore'

type Props = {
  onCenterOnMap?: () => void
}

export function RoutesScreen({ onCenterOnMap }: Props) {
  const { state, selectedRouteId, setSelectedRouteId } = useSimulation()
  const { routes, vessel } = state

  const [selectedWpId, setSelectedWpId] = useState<string>('WP-04')

  const activeRoute =
    routes.availableRoutes.find((r) => r.id === selectedRouteId) || routes.availableRoutes[0]

  return (
    <div className="demo-page routes-page">
      {/* Top Banner */}
      <div className="page-header-banner">
        <div>
          <span className="eyebrow">PASSAGE PLANNING & CORRIDOR DECISION SUPPORT · {vessel.name}</span>
          <h2>ACTIVE PASSAGE PLAN: {activeRoute.name}</h2>
        </div>
        <div className="banner-stats">
          <div>
            <small>ORIGIN</small>
            <strong>PORT BLAIR / CHENNAI</strong>
          </div>
          <div>
            <small>DESTINATION</small>
            <strong>BHARATI / LARSEMANN</strong>
          </div>
          <div>
            <small>TOTAL DISTANCE</small>
            <strong>{activeRoute.distanceNm.toLocaleString()} NM</strong>
          </div>
          <div>
            <small>CORRIDOR STATUS</small>
            <span
              className={`badge ${
                activeRoute.statusType === 'CRITICAL'
                  ? 'critical'
                  : activeRoute.statusType === 'WARNING'
                  ? 'warning'
                  : 'nominal'
              }`}
            >
              {activeRoute.status}
            </span>
          </div>
        </div>
      </div>

      <div className="demo-grid-layout">
        {/* Left Column: Waypoint Table & Corridor Clearance */}
        <div className="demo-column main-col">
          {/* Route Alternative Selector Cards */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">PASSAGE PLAN ALTERNATIVES · CLICK TO SWITCH ACTIVE CORRIDOR</span>
                <h2>ROUTE CANDIDATES EVALUATION MATRIX</h2>
              </div>
              <span className="fresh">{routes.availableRoutes.length} ROUTES EVALUATED</span>
            </div>
            <div className="route-cards-grid">
              {routes.availableRoutes.map((r) => {
                const isSel = r.id === activeRoute.id
                return (
                  <div
                    key={r.id}
                    className={`route-card clickable-card ${isSel ? 'active-route-card' : ''}`}
                    onClick={() => setSelectedRouteId(r.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="route-card-head">
                      <b>{r.id}</b>
                      <span
                        className={`status-pill ${
                          r.statusType === 'CRITICAL'
                            ? 'critical'
                            : r.statusType === 'WARNING'
                            ? 'warning'
                            : 'nominal'
                        }`}
                      >
                        {r.tag}
                      </span>
                    </div>
                    <h4>{r.name}</h4>
                    <div className="route-card-stats">
                      <div>
                        <small>DIST</small>
                        <b>{r.distanceNm} NM</b>
                      </div>
                      <div>
                        <small>ETA</small>
                        <b>{r.etaString}</b>
                      </div>
                      <div>
                        <small>FUEL</small>
                        <b>{r.fuelMt} MT</b>
                      </div>
                      <div>
                        <small>RISK</small>
                        <b className={r.riskScore > 75 ? 'critical' : r.riskScore > 60 ? 'warning' : 'nominal'}>
                          {r.riskScore}/100
                        </b>
                      </div>
                    </div>
                    <div className="route-card-exposure">
                      <span>ICE: {r.iceExposure}</span>
                      <span>MET: {r.weatherExposure}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>

          {/* Active Route Waypoint Table */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">WAYPOINT SEQUENCING · {activeRoute.id}</span>
                <h2>ACTIVE PASSAGE LEGS & NAVIGATION MILESTONES</h2>
              </div>
              <span className="fresh">CORRIDOR WIDTH: 12 NM</span>
            </div>
            <table className="demo-table clickable-table">
              <thead>
                <tr>
                  <th>WP ID</th>
                  <th>WAYPOINT NAME</th>
                  <th>COORDINATES</th>
                  <th>LEG DIST</th>
                  <th>ESTIMATED TIME</th>
                  <th>SAFETY MARGIN</th>
                  <th>LEG RISK</th>
                </tr>
              </thead>
              <tbody>
                {activeRoute.waypoints.map((wp) => {
                  const isSel = wp.id === selectedWpId
                  return (
                    <tr
                      key={wp.id}
                      className={isSel || wp.eta.includes('NOW') ? 'selected-row' : ''}
                      onClick={() => setSelectedWpId(wp.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <b>{wp.id}</b>
                      </td>
                      <td>{wp.name}</td>
                      <td>{wp.coords}</td>
                      <td>{wp.distNm ? `${wp.distNm} NM` : '—'}</td>
                      <td>{wp.eta}</td>
                      <td>
                        <span className={`status-pill ${wp.status.toLowerCase()}`}>{wp.status}</span>
                      </td>
                      <td>
                        <span className={`risk-tag ${wp.risk.toLowerCase()}`}>{wp.risk}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </section>

          {/* Route Profile Visualizer */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">PASSAGE SAFETY CORRIDOR PROFILE</span>
                <h2>CROSS-TRACK CLEARANCE MARGINS</h2>
              </div>
              <span className="model-note">HYDRODYNAMIC ENGINE · SIM</span>
            </div>
            <div className="route-profile-svg-box">
              <svg viewBox="0 0 700 120" className="route-profile-svg">
                {/* Safe corridor buffer */}
                <path
                  d="M 30 60 Q 180 30 350 60 T 670 60"
                  fill="none"
                  stroke="#2c524c"
                  strokeWidth="28"
                  strokeLinecap="round"
                  opacity="0.4"
                />
                {/* Route centerline */}
                <path
                  d="M 30 60 Q 180 30 350 60 T 670 60"
                  fill="none"
                  stroke="#79bba4"
                  strokeWidth="2.5"
                  strokeDasharray="5,3"
                />
                {/* Waypoints */}
                {[
                  { x: 30, y: 60, label: 'WP-01' },
                  { x: 140, y: 45, label: 'WP-02' },
                  { x: 250, y: 45, label: 'WP-03' },
                  { x: 350, y: 60, label: 'WP-04 (NOW)' },
                  { x: 470, y: 75, label: 'WP-05' },
                  { x: 570, y: 65, label: 'WP-06' },
                  { x: 670, y: 60, label: 'WP-07' },
                ].map((wp) => (
                  <g key={wp.label}>
                    <circle
                      cx={wp.x}
                      cy={wp.y}
                      r={wp.label.includes('NOW') ? 6 : 4}
                      fill={wp.label.includes('NOW') ? '#f0faf6' : '#79bba4'}
                      stroke="#061113"
                      strokeWidth="2"
                    />
                    <text
                      x={wp.x}
                      y={wp.y - 12}
                      fill={wp.label.includes('NOW') ? '#f0faf6' : '#99b8b3'}
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {wp.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </section>
        </div>

        {/* Right Column: Route Analysis & Decision Card */}
        <div className="demo-column side-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">CORRIDOR HAZARD ASSESSMENT</span>
                <h2>CONTRIBUTING HAZARD FACTORS</h2>
              </div>
              <span className="fresh">{activeRoute.id}</span>
            </div>
            <div className="risk-factor-list">
              {routes.riskFactors.map((rf) => (
                <div className="risk-factor-item" key={rf.factor}>
                  <div className="risk-factor-head">
                    <b>{rf.factor}</b>
                    <span className={`risk-tag ${rf.level.includes('HIGH') ? 'critical' : rf.level.includes('WARNING') ? 'warning' : 'nominal'}`}>
                      {rf.level}
                    </span>
                  </div>
                  <p>{rf.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Route Action Card */}
          <section className="module advisory-card">
            <div className="module-head">
              <div>
                <span className="eyebrow">PASSAGE PLAN ACTIONS</span>
                <h2>BRIDGE DECISION SUPPORT</h2>
              </div>
              <span className="alert-dot">▲</span>
            </div>
            <div className="advisory-body">
              <p>{activeRoute.description}</p>
              <div className="advisory-actions">
                <button onClick={onCenterOnMap} className="action-btn">
                  VIEW ROUTE ON OPERATIONAL MAP →
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
