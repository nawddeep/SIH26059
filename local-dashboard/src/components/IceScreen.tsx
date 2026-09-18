import { useState } from 'react'
import { useSimulation } from '../data/simulationStore'

type Props = {
  utc: string
  onNavigate?: (screen: 'overview' | 'map' | 'ice' | 'icebergs' | 'routes' | 'risk' | 'sensors') => void
}

export function IceScreen({ utc, onNavigate }: Props) {
  const { state } = useSimulation()
  const { seaIce, vessel, ocean } = state

  const [selectedMetric, setSelectedMetric] = useState<'conc' | 'thickness' | 'drift' | 'dynamics'>('conc')
  const [selectedForecastIndex, setSelectedForecastIndex] = useState(0)
  const [selectedSectorName, setSelectedSectorName] = useState<string>('WEDDELL SECTOR')

  const activeForecast = seaIce.forecast[selectedForecastIndex] || seaIce.forecast[0]
  const activeSector = seaIce.sectors.find((s) => s.name === selectedSectorName) || seaIce.sectors[0]

  return (
    <div className="demo-page ice-page">
      {/* Page Top Banner */}
      <div className="page-header-banner">
        <div>
          <span className="eyebrow">CRYOSPHERE TELEMETRY & POLAR SEA-ICE DYNAMICS · UTC {utc}</span>
          <h2>SEA-ICE STATE & COMPRESSION ANALYSIS / {vessel.name} OPERATIONAL VICINITY</h2>
        </div>
        <div className="banner-stats">
          <div>
            <small>CONCENTRATION</small>
            <strong className={seaIce.concentration > 80 ? 'critical' : ''}>{seaIce.concentration}%</strong>
          </div>
          <div>
            <small>REGIONAL PACK</small>
            <strong>{seaIce.concentration > 82 ? 'COMPRESSING' : 'STABLE'}</strong>
          </div>
          <div>
            <small>MEAN THICKNESS</small>
            <strong>{seaIce.thickness.toFixed(2)} m</strong>
          </div>
          <div>
            <small>ICE-EDGE DISTANCE</small>
            <strong>{seaIce.edgeDistanceNm.toFixed(1)} NM</strong>
          </div>
          <div>
            <small>DATA SOURCE</small>
            <span className="mode-tag">● SIMULATION</span>
          </div>
        </div>
      </div>

      <div className="demo-grid-layout">
        {/* Left Column: Primary Ice Telemetry */}
        <div className="demo-column main-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">LOCAL VESSEL VICINITY · INTERACTIVE TELEMETRY CARDS</span>
                <h2>{vessel.name} OPERATIONAL ICE SECTOR</h2>
              </div>
              <span className={`badge ${seaIce.concentration > 80 ? 'critical' : 'warning'}`}>
                {seaIce.concentration > 80 ? 'HIGH COMPRESSION' : 'MODERATE COMPRESSION'}
              </span>
            </div>

            {/* Interactive Telemetry Cards */}
            <div className="ice-telemetry-grid">
              <div
                className={`telemetry-card clickable-card ${selectedMetric === 'conc' ? 'active-card' : ''}`}
                onClick={() => setSelectedMetric('conc')}
                title="Click to view Pack Concentration analysis"
              >
                <small>PACK CONCENTRATION</small>
                <strong>{seaIce.concentration} %</strong>
                <span>{seaIce.concentration > 80 ? 'HIGH CONCENTRATION' : 'MODERATE CONCENTRATION'}</span>
                <div className="progress-bar">
                  <div
                    className={`fill ${seaIce.concentration > 80 ? 'critical' : 'warning'}`}
                    style={{ width: `${seaIce.concentration}%` }}
                  />
                </div>
              </div>

              <div
                className={`telemetry-card clickable-card ${selectedMetric === 'thickness' ? 'active-card' : ''}`}
                onClick={() => setSelectedMetric('thickness')}
                title="Click to view Ice Thickness details"
              >
                <small>ICE THICKNESS (MEAN)</small>
                <strong>{seaIce.thickness.toFixed(2)} m</strong>
                <span>SECOND-YEAR FLOE PACK</span>
                <div className="progress-bar">
                  <div className="fill warning" style={{ width: `${Math.min(100, seaIce.thickness * 40)}%` }} />
                </div>
              </div>

              <div
                className={`telemetry-card clickable-card ${selectedMetric === 'drift' ? 'active-card' : ''}`}
                onClick={() => setSelectedMetric('drift')}
                title="Click to view Ice Drift Velocity"
              >
                <small>DRIFT VELOCITY</small>
                <strong>
                  {seaIce.driftSpeedKn.toFixed(2)} kn / {seaIce.driftDirectionDeg}°
                </strong>
                <span>WIND + SURFACE CURRENT DRIVEN</span>
                <div className="progress-bar">
                  <div className="fill nominal" style={{ width: `${Math.min(100, seaIce.driftSpeedKn * 120)}%` }} />
                </div>
              </div>

              <div
                className={`telemetry-card clickable-card ${selectedMetric === 'dynamics' ? 'active-card' : ''}`}
                onClick={() => setSelectedMetric('dynamics')}
                title="Click to view Freeze/Melt Dynamics"
              >
                <small>FREEZE / MELT DYNAMICS</small>
                <strong>+1.2 cm / 24h</strong>
                <span>THERMAL FREEZE-UP PHASE</span>
                <div className="progress-bar">
                  <div className="fill caution" style={{ width: '50%' }} />
                </div>
              </div>
            </div>

            {/* Selected Metric Detail Box */}
            <div className="ice-metric-inspector">
              <span className="eyebrow">DIAGNOSTIC INSPECTION: {selectedMetric.toUpperCase()}</span>
              {selectedMetric === 'conc' && (
                <p>
                  Satellite SAR + radiometer fusion indicates <b>{seaIce.concentration}%</b> floe concentration within
                  a 12 NM radius of <b>{vessel.name}</b>. Converging winds of {state.weather.windSpeed} kn from {state.weather.windDirection}° are generating active lead closures.
                </p>
              )}
              {selectedMetric === 'thickness' && (
                <p>
                  Electromagnetic sounder (EM-31) measures multi-year consolidated ridge thickness at <b>{seaIce.thickness.toFixed(2)} m</b>. Hull resistance factor: 1.42× open-water nominal.
                </p>
              )}
              {selectedMetric === 'drift' && (
                <p>
                  ADCP current ({ocean.currentSpeed} m/s @ {ocean.currentDirection}°) combined with 2% wind drag induces ice pack drift at <b>{seaIce.driftSpeedKn.toFixed(2)} kn</b> bearing <b>{seaIce.driftDirectionDeg}°</b>.
                </p>
              )}
              {selectedMetric === 'dynamics' && (
                <p>
                  Air temperature {state.weather.airTemperature}°C and sea surface temperature {ocean.seaSurfaceTemperature}°C maintain thermodynamic frazil freeze-up (+1.2 cm per day).
                </p>
              )}
            </div>

            {/* Ice Concentration Visual Profile */}
            <div className="ice-profile-box">
              <span className="eyebrow">RADIAL PACK DENSITY SPECTRUM (0 – 12 NM)</span>
              <div className="density-strip">
                <div className="band b1" title="0-3 NM: 84% Pack">0-3 NM ({seaIce.concentration}%)</div>
                <div className="band b2" title="3-6 NM: 88% Pack">3-6 NM ({Math.min(99, seaIce.concentration + 4)}%)</div>
                <div className="band b3" title="6-9 NM: 76% Pack">6-9 NM ({Math.max(40, seaIce.concentration - 8)}%)</div>
                <div className="band b4" title="9-12 NM: 65% Pack">9-12 NM ({Math.max(30, seaIce.concentration - 19)}%)</div>
                <div className="band b5" title="12+ NM: OPEN WATER LEADS">12+ NM (LEADS)</div>
              </div>
            </div>
          </section>

          {/* Regional Table (Clickable Rows) */}
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">REGIONAL SECTOR OVERVIEW · CLICK TO INSPECT</span>
                <h2>POLAR CRYOSPHERE MONITORING NETWORK</h2>
              </div>
              <span className="fresh">5 SECTORS SYNCHRONIZED</span>
            </div>
            <table className="demo-table clickable-table">
              <thead>
                <tr>
                  <th>SECTOR</th>
                  <th>CONCENTRATION</th>
                  <th>THICKNESS</th>
                  <th>ICE EDGE</th>
                  <th>DRIFT VECTOR</th>
                  <th>6H TREND</th>
                  <th>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {seaIce.sectors.map((r) => {
                  const isSel = r.name === selectedSectorName
                  return (
                    <tr
                      key={r.name}
                      className={isSel ? 'selected-row' : ''}
                      onClick={() => setSelectedSectorName(r.name)}
                    >
                      <td>
                        <b>{r.name}</b>
                      </td>
                      <td>{r.name === 'WEDDELL SECTOR' ? `${seaIce.concentration}%` : `${r.concentration}%`}</td>
                      <td>{r.name === 'WEDDELL SECTOR' ? `${seaIce.thickness.toFixed(2)} m` : r.thickness}</td>
                      <td>{r.edge}</td>
                      <td>{r.drift}</td>
                      <td>{r.trend}</td>
                      <td>
                        <span className={`status-pill ${r.status.toLowerCase()}`}>{r.status}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="forecast-detail-card" style={{ marginTop: '10px' }}>
              <span className="eyebrow">SECTOR INSPECTOR: {activeSector.name}</span>
              <div className="dossier-grid">
                <span>CONCENTRATION</span>
                <b>{activeSector.name === 'WEDDELL SECTOR' ? `${seaIce.concentration}%` : `${activeSector.concentration}%`}</b>
                <span>THICKNESS</span>
                <b>{activeSector.name === 'WEDDELL SECTOR' ? `${seaIce.thickness.toFixed(2)} m` : activeSector.thickness}</b>
                <span>EDGE DISTANCE</span>
                <b>{activeSector.edge}</b>
                <span>DRIFT / TREND</span>
                <b>{activeSector.drift} ({activeSector.trend})</b>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Interactive Forecast & Decision Advisory */}
        <div className="demo-column side-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">PREDICTIVE MODELING · CLICK PERIOD</span>
                <h2>48-HOUR COMPRESSION FORECAST</h2>
              </div>
              <span className="model-note">CONF {seaIce.forecastConfidence}% · SIM</span>
            </div>
            <div className="forecast-list">
              {seaIce.forecast.map((f, idx) => {
                const isSel = idx === selectedForecastIndex
                return (
                  <div
                    className={`forecast-row clickable-row ${isSel ? 'selected-forecast' : ''}`}
                    key={f.horizon}
                    onClick={() => setSelectedForecastIndex(idx)}
                  >
                    <div>
                      <b>{f.horizon}</b>
                      <small>
                        Drift {f.drift} | Press {f.pressure}
                      </small>
                    </div>
                    <div className="forecast-val">
                      <strong className={f.conc > 88 ? 'critical' : f.conc > 80 ? 'warning' : ''}>
                        {f.conc}%
                      </strong>
                      <span className={`status-pill ${f.risk.toLowerCase()}`}>{f.risk}</span>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Active Forecast Period Breakdown */}
            <div className="forecast-detail-card">
              <span className="eyebrow">FORECAST EVALUATION: {activeForecast.horizon}</span>
              <div className="dossier-grid">
                <span>PROJECTED PACK</span>
                <b>{activeForecast.conc}%</b>
                <span>PRESSURE FORCING</span>
                <b>{activeForecast.pressure}</b>
                <span>PREDICTED DRIFT</span>
                <b>{activeForecast.drift}</b>
                <span>ADVISORY RISK</span>
                <b className={activeForecast.risk === 'CRITICAL' ? 'critical' : ''}>{activeForecast.risk}</b>
              </div>
            </div>
          </section>

          {/* Tactical Advisory & Action Link */}
          <section className="module advisory-card">
            <div className="module-head">
              <div>
                <span className="eyebrow">BRIDGE DECISION ADVISORY</span>
                <h2>ICE TRANSIT RECOMMENDATIONS</h2>
              </div>
              <span className="alert-dot">▲</span>
            </div>
            <div className="advisory-body">
              <p>
                Lead closure rate projected to increase over the next 12 hours. Maintain minimum 8.5 kn headway to avoid floe adhesion.
              </p>
              <div className="advisory-actions">
                <button onClick={() => onNavigate?.('routes')} className="action-btn">
                  REVIEW ROUTE CORRIDOR →
                </button>
                <button onClick={() => onNavigate?.('map')} className="action-btn">
                  INSPECT ICE ON MAP →
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
