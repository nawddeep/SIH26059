import { useSimulation } from '../data/simulationStore'

type Props = {
  selectedId: string
  onSelect: (id: string) => void
  onCenterOnMap: (id: string) => void
}

export function IcebergsScreen({ selectedId, onSelect, onCenterOnMap }: Props) {
  const { state, setSelectedIcebergId } = useSimulation()
  const { icebergs, vessel } = state

  const activeId = selectedId || state.icebergs[0].id
  const selected = icebergs.find((ice) => ice.id === activeId) ?? icebergs[0]

  const handleRowClick = (id: string) => {
    setSelectedIcebergId(id)
    onSelect(id)
  }

  return (
    <div className="demo-page icebergs-page">
      {/* Top Banner */}
      <div className="page-header-banner">
        <div>
          <span className="eyebrow">POLAR CRYOSPHERE TARGET TRACKING & DISPERSION · {vessel.name}</span>
          <h2>ICEBERG SITUATION REGISTRY & TRAJECTORY MATRIX</h2>
        </div>
        <div className="banner-stats">
          <div>
            <small>TRACKED TARGETS</small>
            <strong>{icebergs.length} OBJECTS</strong>
          </div>
          <div>
            <small>CRITICAL PROXIMITY</small>
            <strong className="critical">
              {icebergs[0].id} ({icebergs[0].cpa.toFixed(1)} km)
            </strong>
          </div>
          <div>
            <small>DETECTION SUITE</small>
            <strong>X-BAND + SAR FUSION</strong>
          </div>
          <div>
            <small>DATA MODE</small>
            <span className="mode-tag">● SIMULATION</span>
          </div>
        </div>
      </div>

      <div className="demo-grid-layout">
        {/* Left Column: Comprehensive Iceberg Table */}
        <div className="demo-column main-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">ACTIVE ICEBERG OBSERVATIONS · CLICK ROW TO SELECT</span>
                <h2>TARGET TRACKING TABLE (POLAR VICINITY)</h2>
              </div>
              <span className="fresh">{icebergs.length} TARGETS MONITORED</span>
            </div>
            <table className="demo-table clickable-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>CLASSIFICATION</th>
                  <th>DIMENSIONS</th>
                  <th>DRAFT</th>
                  <th>DRIFT SPEED / HDG</th>
                  <th>CPA DISTANCE</th>
                  <th>TCPA (HORIZON)</th>
                  <th>CONFIDENCE</th>
                  <th>RISK</th>
                </tr>
              </thead>
              <tbody>
                {icebergs.map((ice) => {
                  const isSel = ice.id === activeId
                  return (
                    <tr
                      key={ice.id}
                      className={isSel ? 'selected-row' : ''}
                      onClick={() => handleRowClick(ice.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <b>{ice.id}</b>
                      </td>
                      <td>{ice.type}</td>
                      <td>{ice.dimensions}</td>
                      <td>{ice.draft} m</td>
                      <td>
                        {ice.speed.toFixed(2)} kn / {ice.heading}°
                      </td>
                      <td>
                        <strong className={ice.risk === 'high' ? 'critical' : ''}>{ice.cpa.toFixed(1)} km</strong>
                      </td>
                      <td>T+{ice.cpaHours}h</td>
                      <td>{ice.confidence}%</td>
                      <td>
                        <span className={`status-pill ${ice.risk}`}>{ice.risk.toUpperCase()}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </section>

          {/* Historical & Predicted Trajectory Evolution Table */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">TRAJECTORY PROPAGATION TIMELINE</span>
                <h2>PROJECTED DRIFT POSITIONS FOR {selected.id}</h2>
              </div>
              <span className="model-note">HYDRO-DYNAMIC DRIFT MODEL · SIM</span>
            </div>
            <div className="trajectory-timeline-grid">
              <div className="timeline-node">
                <small>T−24h (HISTORICAL)</small>
                <strong>ORIGIN FIX</strong>
                <span>CPA: {(selected.cpa + 14.5).toFixed(1)} km</span>
              </div>
              <div className="timeline-node">
                <small>T−12h</small>
                <strong>TRACK INTERMEDIATE</strong>
                <span>CPA: {(selected.cpa + 8.2).toFixed(1)} km</span>
              </div>
              <div className="timeline-node active">
                <small>T+{selected.cpaHours}h (PREDICTED CPA)</small>
                <strong className={selected.risk === 'high' ? 'critical' : 'warning'}>CLOSEST POINT</strong>
                <span>CPA: {selected.cpa.toFixed(1)} km</span>
              </div>
              <div className="timeline-node">
                <small>T+48h</small>
                <strong>DOWN-DRIFT</strong>
                <span>CPA: {(selected.cpa + 18.0).toFixed(1)} km</span>
              </div>
              <div className="timeline-node">
                <small>T+72h</small>
                <strong>EXIT SECTOR</strong>
                <span>CPA: {(selected.cpa + 34.2).toFixed(1)} km</span>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Selected Target Detailed Dossier */}
        <div className="demo-column side-col">
          <section className="module target-dossier">
            <div className="module-head">
              <div>
                <span className="eyebrow">TARGET DOSSIER / SIMULATED</span>
                <h2>{selected.id} DETAILED PROFILE</h2>
              </div>
              <span className={`risk-tag ${selected.risk}`}>{selected.risk.toUpperCase()} RISK</span>
            </div>
            <div className="dossier-body">
              <div className="dossier-grid">
                <span>TARGET ID</span>
                <b>{selected.id}</b>
                <span>MORPHOLOGY</span>
                <b>{selected.type}</b>
                <span>DIMENSIONS</span>
                <b>{selected.dimensions}</b>
                <span>SUB-SURFACE DRAFT</span>
                <b>{selected.draft} m</b>
                <span>ESTIMATED MASS</span>
                <b>{selected.massMt || 48.2} MT</b>
                <span>DRIFT VELOCITY</span>
                <b>
                  {selected.speed.toFixed(2)} kn / {selected.heading}°
                </b>
                <span>CLOSEST POINT (CPA)</span>
                <b className={selected.risk === 'high' ? 'critical' : ''}>{selected.cpa.toFixed(1)} km</b>
                <span>TIME TO CPA (TCPA)</span>
                <b>T+{selected.cpaHours}h 00m</b>
                <span>UNCERTAINTY HALO</span>
                <b>±{selected.uncertainty} m</b>
                <span>SENSOR FUSION SOURCE</span>
                <b>{selected.detectionSource}</b>
                <span>MODEL CONFIDENCE</span>
                <b>{selected.confidence}%</b>
                <span>PREDICTION HORIZON</span>
                <b>{selected.predictionHorizon || '72 HOURS'}</b>
              </div>

              <div className="dossier-actions">
                <button
                  className="action-btn"
                  onClick={() => onCenterOnMap(selected.id)}
                  title="Inspect target on Map page"
                >
                  CENTER ON OPERATIONAL MAP →
                </button>
              </div>
            </div>
          </section>

          {/* Hydrodynamic Drift Advisory */}
          <section className="module advisory-card">
            <div className="module-head">
              <div>
                <span className="eyebrow">HYDRODYNAMIC DRIFT MODEL</span>
                <h2>OCEAN & WIND FORCING</h2>
              </div>
              <span className="nominal">● ACTIVE</span>
            </div>
            <div className="advisory-body">
              <p>
                <b>{selected.id}</b> drift is governed by 0.42 m/s ocean current ({state.ocean.currentDirection}°) with
                Ekman wind transport from {state.weather.windDirection}°. Projected collision risk with Route Alpha-26 corridor.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
