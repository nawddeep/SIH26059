import { useState } from 'react'
import { useSimulation } from '../data/simulationStore'

export function SensorsScreen() {
  const { state } = useSimulation()
  const { sensors, vessel } = state

  const [selectedSensorId, setSelectedSensorId] = useState<string>('gnss-01')
  const [pingStatus, setPingStatus] = useState<Record<string, string>>({})

  const selectedSensor =
    sensors.find((s) => s.id === selectedSensorId) || sensors[0]

  const handlePing = (id: string) => {
    setPingStatus((prev) => ({ ...prev, [id]: 'PINGING...' }))
    setTimeout(() => {
      setPingStatus((prev) => ({ ...prev, [id]: 'ACK (0 PACKET LOSS · RTT NOMINAL)' }))
      setTimeout(() => {
        setPingStatus((prev) => {
          const next = { ...prev }
          delete next[id]
          return next
        })
      }, 3000)
    }, 450)
  }

  return (
    <div className="demo-page sensors-page">
      {/* Top Banner */}
      <div className="page-header-banner">
        <div>
          <span className="eyebrow">BRIDGE INSTRUMENTATION & IN-PROCESS HARDWARE TRANSPORT BUS · {vessel.name}</span>
          <h2>SENSOR TELEMETRY, HEALTH MATRIX & HARDWARE BUS</h2>
        </div>
        <div className="banner-stats">
          <div>
            <small>ACTIVE CHANNELS</small>
            <strong>
              {sensors.filter((s) => s.status === 'live').length} / {sensors.length} ONLINE
            </strong>
          </div>
          <div>
            <small>DATA PROTOCOLS</small>
            <strong>NMEA / UDP / SERIAL</strong>
          </div>
          <div>
            <small>AVG BUS LATENCY</small>
            <strong>
              {Math.round(sensors.reduce((acc, s) => acc + s.latencyMs, 0) / sensors.length)} ms
            </strong>
          </div>
          <div>
            <small>DATA MODE</small>
            <span className="mode-tag">● SIMULATION</span>
          </div>
        </div>
      </div>

      <div className="demo-grid-layout">
        {/* Left Column: Interactive Sensor Cards Grid */}
        <div className="demo-column main-col">
          <section className="module">
            <div className="module-head">
              <div>
                <span className="eyebrow">HARDWARE ADAPTERS · CLICK CARD TO VIEW DIAGNOSTICS</span>
                <h2>INTEGRATED BRIDGE SENSOR SUITE</h2>
              </div>
              <span className="fresh">ALL TRANSPORTS NOMINAL</span>
            </div>

            <div className="sensors-grid">
              {sensors.map((s) => {
                const isSel = s.id === selectedSensorId
                return (
                  <div
                    key={s.id}
                    className={`sensor-card clickable-card ${isSel ? 'active-sensor-card' : ''}`}
                    onClick={() => setSelectedSensorId(s.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="sensor-card-head">
                      <div>
                        <small>{s.type}</small>
                        <h3>{s.name}</h3>
                      </div>
                      <span className={`status-pill ${s.status === 'live' ? 'nominal' : s.status}`}>
                        {s.status === 'live' ? 'ONLINE' : s.status.toUpperCase()}
                      </span>
                    </div>

                    <div className="sensor-metric-val">
                      <strong>{s.metric}</strong>
                      <span>{s.unit}</span>
                    </div>

                    <div className="sensor-card-stats">
                      <div>
                        <small>LATENCY</small>
                        <b>{s.latencyMs} ms</b>
                      </div>
                      <div>
                        <small>QUALITY</small>
                        <b>{s.quality}%</b>
                      </div>
                      <div>
                        <small>ERRORS</small>
                        <b>{s.errorCount}</b>
                      </div>
                    </div>

                    <div className="sensor-card-footer">
                      <small>{s.detail}</small>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>
        </div>

        {/* Right Column: Selected Sensor Diagnostic Inspector */}
        <div className="demo-column side-col">
          <section className="module sensor-diagnostics">
            <div className="module-head">
              <div>
                <span className="eyebrow">DIAGNOSTIC INSPECTOR</span>
                <h2>{selectedSensor.name}</h2>
              </div>
              <span className={`status-pill ${selectedSensor.status === 'live' ? 'nominal' : 'warning'}`}>
                {selectedSensor.status.toUpperCase()}
              </span>
            </div>

            <div className="diagnostics-body">
              <div className="dossier-grid">
                <span>DEVICE ID</span>
                <b>{selectedSensor.id}</b>
                <span>SUBSYSTEM</span>
                <b>{selectedSensor.type}</b>
                <span>CURRENT VALUE</span>
                <b>{selectedSensor.metric}</b>
                <span>UNIT OF MEASURE</span>
                <b>{selectedSensor.unit}</b>
                <span>SIGNAL QUALITY</span>
                <b>{selectedSensor.quality}% NOMINAL</b>
                <span>BUS LATENCY</span>
                <b>{selectedSensor.latencyMs} ms</b>
                <span>PACKET LOSS</span>
                <b>{selectedSensor.packetLoss || '0.01%'}</b>
                <span>LAST PACKET</span>
                <b>{selectedSensor.lastSeen}</b>
                <span>TRANSPORT BUS</span>
                <b>{selectedSensor.transport}</b>
                {selectedSensor.satelliteCount && (
                  <>
                    <span>SATELLITES</span>
                    <b>{selectedSensor.satelliteCount} TRACKED</b>
                  </>
                )}
              </div>

              <div className="diag-text-box">
                <span className="eyebrow">SUBSYSTEM DIAGNOSTIC STRING:</span>
                <code>{selectedSensor.diagnostics || selectedSensor.detail}</code>
              </div>

              {pingStatus[selectedSensor.id] && (
                <div className="ping-response-box">
                  <small>DIAGNOSTIC PROBE STATUS:</small>
                  <b>{pingStatus[selectedSensor.id]}</b>
                </div>
              )}

              <div className="diagnostics-actions">
                <button
                  className="action-btn"
                  onClick={() => handlePing(selectedSensor.id)}
                  title="Send bus verification packet"
                >
                  SEND DIAGNOSTIC PROBE (PING) ⚡
                </button>
              </div>
            </div>
          </section>

          {/* Transport Bus Architecture */}
          <section className="module advisory-card">
            <div className="module-head">
              <div>
                <span className="eyebrow">IN-PROCESS DATA BUS ARCHITECTURE</span>
                <h2>HARDWARE ABSTRACTION LAYER</h2>
              </div>
              <span className="nominal">● CONNECTED</span>
            </div>
            <div className="advisory-body">
              <p>
                In-process transport adapters consume simulated sensor datagrams at 18ms latency with zero memory leaks.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
