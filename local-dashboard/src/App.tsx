import { useState, useEffect } from 'react'
import {
  SimulationProvider,
  useSimulation,
  useDataFreshness,
  formatUtcTime,
} from './data/simulationStore'
import { NavigationRenderer, type MapReadout } from './visualization/NavigationRenderer'
import { OperationalRadar } from './visualization/OperationalRadar'
import { IceScreen } from './components/IceScreen'
import { IcebergsScreen } from './components/IcebergsScreen'
import { RoutesScreen } from './components/RoutesScreen'
import { RiskScreen } from './components/RiskScreen'
import { SensorsScreen } from './components/SensorsScreen'
import type { Risk, SonarDetection } from './types/operational'
import { mapLayers, validateLayer, type PolarRegion } from './data/mapLayers'

export type ActiveScreen = 'overview' | 'map' | 'ice' | 'icebergs' | 'routes' | 'risk' | 'sensors'

const riskMark: Record<Risk, string> = { low: '◇', medium: '△', high: '▲', critical: '✕' }
const formatLat = (v: number) => `${Math.abs(v).toFixed(3)}°${v < 0 ? 'S' : 'N'}`
const formatLon = (v: number) => `${Math.abs(v).toFixed(3)}°${v < 0 ? 'W' : 'E'}`

export default function App() {
  return (
    <SimulationProvider>
      <AppConsole />
    </SimulationProvider>
  )
}

function AppConsole() {
  const {
    state,
    isSyncing,
    lastSyncStatus,
    backendOnline,
    selectedIcebergId,
    setSelectedIcebergId,
    refreshTelemetry,
    setRegion,
  } = useSimulation()

  const freshnessLabel = useDataFreshness()

  const [screen, setScreen] = useState<ActiveScreen>('overview')
  const [showChangeLog, setShowChangeLog] = useState(false)
  const [horizon, setHorizon] = useState('NOW')
  const [sonarOpen, setSonarOpen] = useState(true)
  const [liveUtc, setLiveUtc] = useState(() => formatUtcTime())

  useEffect(() => {
    const timer = window.setInterval(() => {
      setLiveUtc(formatUtcTime())
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])

  const selectedIceberg =
    state.icebergs.find((ice) => ice.id === selectedIcebergId) ?? state.icebergs[0]

  const handleCenterOnMap = (id: string) => {
    setSelectedIcebergId(id)
    setScreen('map')
  }

  const handleRegionChange = (newRegion: PolarRegion) => {
    setRegion(newRegion)
  }

  return (
    <main className="console">
      {/* Global Status Bar with Interactive LATEST DATA Control */}
      <header className="status-bar">
        <div className="brand">
          IMPALA<span>POLAR BRIDGE / DECISION SUPPORT</span>
        </div>

        <div className="mode" title={backendOnline ? "FastAPI Backend Active (http://localhost:8600)" : "Local Simulation Mode"}>
          <b style={{ color: backendOnline ? '#00ffcc' : '#ffaa00' }}>●</b> {backendOnline ? 'BACKEND CONNECTED' : 'LOCAL SIMULATION'}
        </div>

        <Status label="VESSEL" value={state.vessel.name} onClick={() => setScreen('sensors')} />
        <Status label="UTC" value={liveUtc} />
        <Status
          label="POSITION"
          value={`${formatLat(state.vessel.latitude)} ${formatLon(state.vessel.longitude)}`}
          onClick={() => setScreen('map')}
        />
        <Status
          label="SOG / COG"
          value={`${state.vessel.sog.toFixed(1)} kn / ${state.vessel.cog.toFixed(0)}°`}
        />

        {/* Global Interactive LATEST DATA / REFRESH Control */}
        <div className="status freshness-status-control">
          <div className="freshness-row-top">
            <small>DATA FRESHNESS</small>
            <button
              className="changelog-toggle-btn"
              onClick={() => setShowChangeLog(!showChangeLog)}
              title="Toggle Recent Data Updates Log"
            >
              {showChangeLog ? 'LOG [✕]' : 'LOG [▾]'}
            </button>
          </div>
          <div className="freshness-action-row">
            <span className="freshness-tag">● {freshnessLabel}</span>
            <button
              className={`latest-data-btn ${isSyncing ? 'syncing' : lastSyncStatus === 'updated' ? 'updated' : ''}`}
              onClick={refreshTelemetry}
              disabled={isSyncing}
              title="Trigger global telemetry synchronization"
            >
              {lastSyncStatus === 'syncing'
                ? '↻ SYNCING...'
                : lastSyncStatus === 'updated'
                ? '✓ DATA UPDATED'
                : '↻ LATEST DATA'}
            </button>
          </div>

          {/* Recent Updates Dropdown Popover */}
          {showChangeLog && (
            <div className="recent-updates-popover">
              <div className="popover-head">
                <span className="eyebrow">RECENT DATA UPDATES / AUDIT BUS</span>
                <button className="close-mini-btn" onClick={() => setShowChangeLog(false)}>
                  ✕
                </button>
              </div>
              <div className="popover-list">
                {state.changeLog.length === 0 ? (
                  <div className="popover-empty">Awaiting telemetry delta...</div>
                ) : (
                  state.changeLog.map((log) => (
                    <div className="changelog-item" key={log.id}>
                      <div className="log-time-field">
                        <time>{log.timestamp}</time>
                        <b>{log.field}</b>
                      </div>
                      <div className="log-values">
                        <span>{log.oldValue}</span>
                        <i>→</i>
                        <strong>{log.newValue}</strong>
                        {log.delta && <small className="delta-badge">{log.delta}</small>}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Global Navigation Bar */}
      <Nav active={screen} onNavigate={setScreen} />

      <div className="subbar">
        <span>
          {screen === 'map'
            ? `INTERACTIVE OPERATIONAL VECTOR MAP · REGION: ${state.region.toUpperCase()} · ${state.vessel.name}`
            : screen === 'ice'
            ? 'POLAR SEA-ICE TELEMETRY · CONCENTRATION & PACK DYNAMICS'
            : screen === 'icebergs'
            ? 'CRYOSPHERE TARGET TRACKING MATRIX · HYDRODYNAMIC DRIFT MODELS'
            : screen === 'routes'
            ? 'PASSAGE PLAN: ROUTE-ALPHA-26 · CORRIDOR SAFETY ADVISORY'
            : screen === 'risk'
            ? 'INTEGRATED OPERATIONAL RISK MATRIX · PROBABILISTIC HAZARD INDEX'
            : screen === 'sensors'
            ? 'BRIDGE TELEMETRY & IN-PROCESS HARDWARE TRANSPORT BUS'
            : 'ANTARCTIC POLAR STEREOGRAPHIC VIEW · SOUTH SHETLAND / WEDDELL APPROACH'}
        </span>
        <span>DECISION-SUPPORT SIMULATION · NOT CERTIFIED FOR NAVIGATION</span>
      </div>

      {/* Screen Router */}
      {screen === 'overview' && (
        <OverviewScreen
          selected={selectedIceberg}
          selectedId={selectedIcebergId}
          setSelectedId={setSelectedIcebergId}
          horizon={horizon}
          setHorizon={setHorizon}
          sonarOpen={sonarOpen}
          setSonarOpen={setSonarOpen}
          onNavigate={setScreen}
        />
      )}

      {screen === 'map' && (
        <MapScreen
          utc={liveUtc}
          selected={selectedIceberg}
          selectedId={selectedIcebergId}
          onSelect={setSelectedIcebergId}
          region={state.region}
          setRegion={handleRegionChange}
        />
      )}

      {screen === 'ice' && <IceScreen utc={liveUtc} onNavigate={setScreen} />}

      {screen === 'icebergs' && (
        <IcebergsScreen
          selectedId={selectedIcebergId}
          onSelect={setSelectedIcebergId}
          onCenterOnMap={handleCenterOnMap}
        />
      )}

      {screen === 'routes' && <RoutesScreen onCenterOnMap={() => setScreen('map')} />}

      {screen === 'risk' && <RiskScreen />}

      {screen === 'sensors' && <SensorsScreen />}
    </main>
  )
}

function OverviewScreen({
  selected,
  selectedId,
  setSelectedId,
  horizon,
  setHorizon,
  sonarOpen,
  setSonarOpen,
  onNavigate,
}: {
  selected: typeof import('./data/simulationStore').useSimulation extends () => { state: infer S }
    ? S extends { icebergs: (infer I)[] }
      ? I
      : never
    : never
  selectedId: string
  setSelectedId: (id: string) => void
  horizon: string
  setHorizon: (val: string) => void
  sonarOpen: boolean
  setSonarOpen: (val: boolean) => void
  onNavigate: (s: ActiveScreen) => void
}) {
  const { state } = useSimulation()

  return (
    <>
      <div className="workstation">
        {/* Left Sensor Rail (Clickable -> Navigates to Sensors) */}
        <aside className="sensor-rail">
          <SensorStrip onNavigate={() => onNavigate('sensors')} />
          <div className="rail-note" onClick={() => onNavigate('sensors')} style={{ cursor: 'pointer' }}>
            <b>DATA BUS</b>
            <span>IN-PROCESS / LATEST VALUE</span>
            <span>SIMULATED TRANSPORT →</span>
          </div>
        </aside>

        {/* PROTECTED OVERVIEW RADAR — UNTOUCHED & FROZEN */}
        <section className="plot-module module">
          <div className="module-head">
            <div>
              <span className="eyebrow">IMPALA X-BAND RADAR / SIMULATION</span>
              <h2>POLAR PLOT / RANGE 24 NM / GRID 2 NM</h2>
            </div>
            <div className="plot-tools">
              <button className="active">ALL LAYERS</button>
              <button onClick={() => onNavigate('map')}>FULL MAP →</button>
            </div>
          </div>
          <div className="plot-wrap">
            <OperationalRadar state={state} selectedId={selectedId} onSelect={setSelectedId} />
            <div className="plot-overlay bottom-left">
              SOUTH ↓
              <br />
              <span>RANGE RINGS 6 / 12 / 18 / 24 NM</span>
            </div>
            <div className="plot-overlay bottom-right">
              {horizon} FORECAST
              <br />
              <span>TRAJECTORY MODEL / VARIABLE CONFIDENCE</span>
            </div>
          </div>
          <div className="selection-readout" onClick={() => onNavigate('icebergs')} style={{ cursor: 'pointer' }}>
            <span className={`risk-tag ${selected.risk}`}>
              {riskMark[selected.risk]} {selected.risk.toUpperCase()}
            </span>
            <strong>{selected.id}</strong>
            <span>
              {selected.type} · {selected.dimensions} · DRAFT {selected.draft} m
            </span>
            <span>
              CPA {selected.cpa.toFixed(1)} km / T+{selected.cpaHours}h / CONF {selected.confidence}%
            </span>
            <span className="prediction-horizon">INSPECT ICEBERGS →</span>
          </div>
        </section>

        {/* Right Rail: Hazards & Target Matrix (Clickable) */}
        <aside className="right-rail">
          <Hazards onNavigate={() => onNavigate('risk')} />
          <Targets selectedId={selectedId} onSelect={setSelectedId} onNavigate={() => onNavigate('icebergs')} />
        </aside>
      </div>

      <footer className="bottom-row">
        <SonarPanel open={sonarOpen} setOpen={setSonarOpen} onNavigate={() => onNavigate('sensors')} />
        <Timeline horizon={horizon} setHorizon={setHorizon} onNavigate={() => onNavigate('ice')} />
        <SystemLog onNavigate={() => onNavigate('risk')} />
      </footer>
    </>
  )
}

function MapScreen({
  utc,
  selected,
  selectedId,
  onSelect,
  region,
  setRegion,
}: {
  utc: string
  selected: typeof import('./data/simulationStore').useSimulation extends () => { state: infer S }
    ? S extends { icebergs: (infer I)[] }
      ? I
      : never
    : never
  selectedId: string
  onSelect: (id: string) => void
  region: PolarRegion
  setRegion: (r: PolarRegion) => void
}) {
  const { state } = useSimulation()
  const [rangeNm, setRangeNm] = useState(12)
  const [inspectVessel, setInspectVessel] = useState(false)
  const [readout, setReadout] = useState<MapReadout>({
    latitude: state.vessel.latitude,
    longitude: state.vessel.longitude,
    rangeNm: 12,
    zoom: 7.2,
    bearing: 0,
  })
  const [visible, setVisible] = useState(() => new Set(mapLayers.filter((l) => l.defaultVisible).map((l) => l.id)))

  const layersForRegion = mapLayers.filter((layer) => layer.region === 'both' || layer.region === region)

  const toggle = (id: string) =>
    setVisible((previous) => {
      const next = new Set(previous)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  const toggleGroup = (ids: string[]) =>
    setVisible((previous) => {
      const next = new Set(previous)
      const turnOn = ids.some((id) => !next.has(id))
      ids.forEach((id) => (turnOn ? next.add(id) : next.delete(id)))
      return next
    })

  const handleTargetSelect = (id: string) => {
    setInspectVessel(false)
    onSelect(id)
  }

  const handleVesselClick = () => {
    setInspectVessel(true)
  }

  const vesselTelemetry = {
    name: state.vessel.name,
    latitude: state.vessel.latitude,
    longitude: state.vessel.longitude,
    sog: state.vessel.sog,
    cog: state.vessel.cog,
    heading: state.vessel.heading,
    rateOfTurn: state.vessel.rateOfTurn,
    stw: state.vessel.stw,
    windDirection: state.weather.windDirection,
    windSpeed: state.weather.windSpeed,
    depth: state.vessel.depth,
    aisContactCount: state.ais.length,
    radarContactCount: state.radar.length,
    draft: state.vessel.draft,
    engineRpm: state.vessel.engineRpm,
    provenance: 'LOCAL SIMULATION',
  }

  return (
    <div className="map-console-body">
      <div className="map-stage">
        {/* Left Map Layers Panel */}
        <section className="layer-panel">
          <div className="module-head">
            <div>
              <span className="eyebrow">MAP LAYERS</span>
              <h2>LOCAL REGISTRY</h2>
            </div>
            <span className="fresh">{region.toUpperCase()}</span>
          </div>
          {(
            [
              'BASE',
              'NAVIGATION',
              'CRYOSPHERE',
              'OCEAN',
              'WEATHER',
              'BATHYMETRY',
              'VESSEL',
              'AIS',
              'ICEBERG',
              'SENSOR',
              'ROUTE',
            ] as const
          ).map((category) => {
            const items = layersForRegion.filter((layer) => layer.category === category)
            return items.length ? (
              <div className="layer-group" key={category}>
                <b>{category}</b>
                {items.map((layer) => (
                  <label key={layer.id} className={!layer.enabled ? 'unavailable' : ''}>
                    <input
                      type="checkbox"
                      checked={visible.has(layer.id)}
                      disabled={!layer.enabled}
                      onChange={() => toggle(layer.id)}
                    />
                    <span>{layer.name}</span>
                    {!layer.enabled && <i title={validateLayer(layer)}>!</i>}
                  </label>
                ))}
              </div>
            ) : null
          })}
        </section>

        {/* Center Interactive Operational Map */}
        <section className="full-map">
          <NavigationRenderer
            state={state}
            selectedId={selectedId}
            onSelect={handleTargetSelect}
            onSelectVessel={handleVesselClick}
            visibleLayers={visible}
            region={region}
            rangeNm={rangeNm}
            onReadout={setReadout}
            vesselTelemetry={vesselTelemetry}
          />

          <div className="map-overlay map-title">
            POLAR & MARITIME VECTOR PLOT / RANGE {Math.max(1, Math.round(readout.rangeNm))} NM
            <small>
              LOCAL VECTOR BASE · DRAG / WHEEL / ROTATE · {formatLat(readout.latitude)}{' '}
              {formatLon(readout.longitude)} · HDG {readout.bearing.toFixed(0)}°
            </small>
            <div className="map-layer-actions">
              <button onClick={() => toggleGroup(mapLayers.filter((l) => l.enabled).map((l) => l.id))}>
                ALL LAYERS
              </button>
              <button onClick={() => toggleGroup(['sonar'])}>SONAR</button>
              <button onClick={() => toggleGroup(['radar'])}>RADAR</button>
              <button onClick={() => toggleGroup(['safe-corridor'])}>ROUTE</button>
              <button onClick={() => toggleGroup(['wind', 'current'])}>METOCEAN</button>
            </div>
          </div>

          {/* Operational Theater Selector */}
          <div className="map-overlay region-select">
            <span>OPERATIONAL THEATER</span>
            <button className={region === 'antarctic' ? 'active' : ''} onClick={() => setRegion('antarctic')}>
              ANTARCTIC
            </button>
            <button className={region === 'indo-pacific' ? 'active' : ''} onClick={() => setRegion('indo-pacific')}>
              INDO-PACIFIC
            </button>
            <button className={region === 'arctic' ? 'active' : ''} onClick={() => setRegion('arctic')}>
              ARCTIC
            </button>
          </div>

          {/* Range Rings Selector */}
          <div className="map-overlay range-select">
            <span>RANGE RINGS</span>
            {[6, 12, 24, 48].map((range) => (
              <button key={range} className={range === rangeNm ? 'active' : ''} onClick={() => setRangeNm(range)}>
                {range} NM
              </button>
            ))}
          </div>

          {/* Bottom-left: Selected Target Dossier OR Vessel Telemetry Panel */}
          {inspectVessel ? (
            <VesselTelemetryModal telemetry={vesselTelemetry} onClose={() => setInspectVessel(false)} />
          ) : (
            <Inspection selected={selected} onInspectVessel={() => setInspectVessel(true)} />
          )}
        </section>
      </div>

      {/* Map Status Bar */}
      <div className="map-status">
        <span>
          MAP SOURCE <b>LOCAL DEMO VECTOR (NATURAL EARTH)</b>
        </span>
        <span>
          DATA MODE <b>SIMULATION</b>
        </span>
        <span>
          REGION <b>{region.toUpperCase()}</b>
        </span>
        <span>
          AIS <b>{state.ais.length} CONTACTS</b>
        </span>
        <span>
          RADAR <b>{state.radar.length} CONTACTS / 12 NM</b>
        </span>
        <span>
          SONAR <b>SIM / 2.8 KM</b>
        </span>
        <span>
          WIND <b>{state.weather.windSpeed} KT / {state.weather.windDirection}°</b>
        </span>
        <span>
          CAMERA <b>Z{readout.zoom.toFixed(1)} / {Math.round(readout.rangeNm)} NM</b>
        </span>
        <span>
          UTC <b>{utc}</b>
        </span>
      </div>
    </div>
  )
}

function Nav({
  active,
  onNavigate,
}: {
  active: ActiveScreen
  onNavigate: (s: ActiveScreen) => void
}) {
  return (
    <nav className="console-nav">
      <button className={active === 'overview' ? 'active' : ''} onClick={() => onNavigate('overview')}>
        OVERVIEW
      </button>
      <button className={active === 'map' ? 'active' : ''} onClick={() => onNavigate('map')}>
        MAP
      </button>
      <button className={active === 'ice' ? 'active' : ''} onClick={() => onNavigate('ice')}>
        ICE
      </button>
      <button className={active === 'icebergs' ? 'active' : ''} onClick={() => onNavigate('icebergs')}>
        ICEBERGS
      </button>
      <button className={active === 'routes' ? 'active' : ''} onClick={() => onNavigate('routes')}>
        ROUTES
      </button>
      <button className={active === 'risk' ? 'active' : ''} onClick={() => onNavigate('risk')}>
        RISK
      </button>
      <button className={active === 'sensors' ? 'active' : ''} onClick={() => onNavigate('sensors')}>
        SENSORS
      </button>
    </nav>
  )
}

function Inspection({
  selected,
  onInspectVessel,
}: {
  selected: any
  onInspectVessel?: () => void
}) {
  return (
    <aside className="inspection" aria-label="Selected target readout">
      <div className="inspection-head">
        <span className="eyebrow">SELECTED TARGET / SIMULATED</span>
        <button className="text-btn" onClick={onInspectVessel} title="Inspect Own Vessel">
          OWN SHIP ⚓
        </button>
      </div>
      <h2>{selected.id}</h2>
      <div>
        <span>CLASS</span>
        <b>{selected.type}</b>
        <span>SPEED / HDG</span>
        <b>
          {selected.speed.toFixed(2)} KT / {selected.heading}°
        </b>
        <span>DRAFT</span>
        <b>{selected.draft} m</b>
        <span>CPA</span>
        <b>
          {selected.cpa.toFixed(1)} km / T+{selected.cpaHours}h
        </b>
        <span>CONFIDENCE</span>
        <b>{selected.confidence}%</b>
        <span>HORIZON</span>
        <b>72 HOURS</b>
        <span>SOURCE</span>
        <b>{selected.detectionSource}</b>
      </div>
    </aside>
  )
}

function VesselTelemetryModal({
  telemetry,
  onClose,
}: {
  telemetry: any
  onClose: () => void
}) {
  return (
    <aside className="inspection vessel-telemetry-hud" aria-label="Vessel Telemetry Inspector">
      <div className="inspection-head">
        <span className="eyebrow">VESSEL TELEMETRY · SIMULATED</span>
        <button className="text-btn" onClick={onClose} title="Return to target dossier">
          CLOSE [✕]
        </button>
      </div>
      <h2>{telemetry.name}</h2>
      <div className="vessel-grid">
        <span>GNSS FIX</span>
        <b>
          {formatLat(telemetry.latitude)} {formatLon(telemetry.longitude)}
        </b>
        <span>SOG (GROUND)</span>
        <b>{telemetry.sog.toFixed(1)} kn</b>
        <span>COG (TRACK)</span>
        <b>{telemetry.cog.toFixed(1)}°</b>
        <span>TRUE HEADING</span>
        <b>{telemetry.heading.toFixed(1)}°</b>
        <span>RATE OF TURN</span>
        <b>
          {telemetry.rateOfTurn >= 0 ? `+${telemetry.rateOfTurn.toFixed(1)}` : telemetry.rateOfTurn.toFixed(1)}
          °/min
        </b>
        <span>SPEED (WATER)</span>
        <b>{telemetry.stw.toFixed(1)} kn</b>
        <span>TRUE WIND</span>
        <b>
          {telemetry.windDirection}° / {telemetry.windSpeed} kn
        </b>
        <span>UNDER-KEEL DEPTH</span>
        <b>{telemetry.depth} m</b>
        <span>AIS CONTACTS</span>
        <b>{telemetry.aisContactCount} TRACKED</b>
        <span>RADAR CONTACTS</span>
        <b>{telemetry.radarContactCount} TRACKED</b>
        <span>VESSEL DRAFT</span>
        <b>{telemetry.draft.toFixed(1)} m</b>
        <span>PROPULSION</span>
        <b>{telemetry.engineRpm} RPM</b>
      </div>
      <div className="sim-disclaimer">
        <span>SIMULATED / LOCAL BRIDGE BUS</span>
      </div>
    </aside>
  )
}

function Status({
  label,
  value,
  onClick,
}: {
  label: string
  value: string
  onClick?: () => void
}) {
  return (
    <div
      className={`status ${onClick ? 'clickable-status' : ''}`}
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      <small>{label}</small>
      <span>{value}</span>
    </div>
  )
}

function SensorStrip({ onNavigate }: { onNavigate?: () => void }) {
  const { state } = useSimulation()
  return (
    <section className="sensor-strip" onClick={onNavigate} style={{ cursor: 'pointer' }}>
      <div className="module-head">
        <span className="eyebrow">SENSOR STATUS · CLICK TO INSPECT</span>
        <span className="fresh">07 / 07</span>
      </div>
      {state.sensors.map((sensor) => (
        <div className="sensor" key={sensor.id}>
          <i className={sensor.status} />
          <b>{sensor.name}</b>
          <span>● SIM</span>
          <small>{sensor.detail}</small>
        </div>
      ))}
    </section>
  )
}

function Hazards({ onNavigate }: { onNavigate?: () => void }) {
  const { state } = useSimulation()
  return (
    <section className="module hazards">
      <div className="module-head" onClick={onNavigate} style={{ cursor: 'pointer' }}>
        <div>
          <span className="eyebrow">ACTIVE HAZARDS · RISK {state.compositeRiskScore}/100</span>
          <h2>{state.hazards.length} OPEN EVENTS ({state.compositeRiskTrend})</h2>
        </div>
        <span className="alert-dot">▲</span>
      </div>
      {state.hazards.map((hazard) => (
        <article className={`hazard ${hazard.level.toLowerCase()}`} key={hazard.title} onClick={onNavigate} style={{ cursor: 'pointer' }}>
          <div>
            <b>{hazard.level}</b>
            <strong>{hazard.title}</strong>
          </div>
          <p>{hazard.message}</p>
          <button>{hazard.action} →</button>
        </article>
      ))}
    </section>
  )
}

function Targets({
  selectedId,
  onSelect,
  onNavigate,
}: {
  selectedId: string
  onSelect: (id: string) => void
  onNavigate?: () => void
}) {
  const { state } = useSimulation()
  return (
    <section className="module targets">
      <div className="module-head" onClick={onNavigate} style={{ cursor: 'pointer' }}>
        <div>
          <span className="eyebrow">ICEBERG TARGETS · MATRIX</span>
          <h2>{state.icebergs.length} TRACKED / 01 CRITICAL</h2>
        </div>
        <span className="fresh">INSPECT →</span>
      </div>
      {state.icebergs.map((ice) => (
        <button
          className={`target-row ${selectedId === ice.id ? 'selected' : ''}`}
          key={ice.id}
          onClick={() => onSelect(ice.id)}
        >
          <span className={`target-symbol ${ice.risk}`}>{riskMark[ice.risk]}</span>
          <span>
            <b>{ice.id}</b>
            <small>
              {ice.type} · {ice.speed.toFixed(2)} kn
            </small>
          </span>
          <span>
            <b>{ice.cpa.toFixed(1)} km</b>
            <small>CPA / T+{ice.cpaHours}h</small>
          </span>
        </button>
      ))}
    </section>
  )
}

function SonarPanel({
  open,
  setOpen,
  onNavigate,
}: {
  open: boolean
  setOpen: (value: boolean) => void
  onNavigate?: () => void
}) {
  const { state } = useSimulation()
  const detection = state.sonar[0]
  return (
    <section className="sonar-panel module">
      <div className="module-head">
        <div onClick={onNavigate} style={{ cursor: 'pointer' }}>
          <span className="eyebrow">FORWARD LOOKING SONAR / SIMULATION</span>
          <h2>WHAT IS AHEAD / RANGE 2.8 km</h2>
        </div>
        <button className="switch" onClick={() => setOpen(!open)}>
          {open ? '● ACTIVE' : '○ OFFLINE'}
        </button>
      </div>
      {open ? (
        <div className="sonar-body">
          <SonarSector detection={detection} />
          <div className="waterfall">
            <span>SONAR HISTORY / RANGE × TIME</span>
            <div className="waterfall-lines" />
          </div>
          <div className="sonar-detail">
            <b>{detection.id}</b>
            <span>
              RANGE {detection.rangeKm} km / BRG {detection.bearing}°
            </span>
            <span>
              DEPTH {detection.depthM} m / RETURN {detection.returnStrength.toUpperCase()}
            </span>
            <span>
              {detection.observationState} / {detection.classification}
            </span>
            <strong>CONF {detection.confidence}% · SIMULATED</strong>
          </div>
        </div>
      ) : (
        <div className="offline-state">
          SONAR OFFLINE · LAST DATA {state.lastSyncTimestampUtc} UTC
          <br />
          <span>OTHER NAVIGATION LAYERS REMAIN AVAILABLE</span>
        </div>
      )}
    </section>
  )
}

function SonarSector({ detection }: { detection: SonarDetection }) {
  return (
    <svg className="sonar-sector-panel" viewBox="0 0 160 100" aria-label="Forward-looking sonar simulation">
      <path d="M80 92 L24 15 A96 96 0 0 1 136 15 Z" className="sector-bg" />
      <path d="M80 92 L38 34 A72 72 0 0 1 122 34 Z" className="sector-line" />
      <path d="M80 92 L51 51 A50 50 0 0 1 109 51 Z" className="sector-line" />
      <circle cx="91" cy="49" r="4" className="sonar-return strong" />
      <circle cx="76" cy="38" r="2" className="sonar-return weak" />
      <path d="M80 92l-5-10h10z" className="vessel-icon" />
      <text x="85" y="46">
        {detection.id}
      </text>
      <text x="75" y="99">
        VESSEL
      </text>
    </svg>
  )
}

function Timeline({
  horizon,
  setHorizon,
  onNavigate,
}: {
  horizon: string
  setHorizon: (value: string) => void
  onNavigate?: () => void
}) {
  return (
    <section className="timeline">
      <div className="module-head" onClick={onNavigate} style={{ cursor: 'pointer', padding: '4px 0 2px' }}>
        <span className="eyebrow">TRAJECTORY TIMELINE / UTC · OPEN ICE MODEL →</span>
      </div>
      <div className="time-controls">
        {['NOW', '+24H', '+48H', '+72H', '+7D'].map((step) => (
          <button onClick={() => setHorizon(step)} className={horizon === step ? 'active' : ''} key={step}>
            {step}
          </button>
        ))}
      </div>
      <span className="model-note">MODEL OUTPUT — NOT AN OBSERVATION</span>
    </section>
  )
}

function SystemLog({ onNavigate }: { onNavigate?: () => void }) {
  const { state } = useSimulation()
  return (
    <section className="system-log" onClick={onNavigate} style={{ cursor: 'pointer' }}>
      <span className="eyebrow">SYSTEM LOG / LOCAL BUS</span>
      <div>
        {state.logs.slice(0, 3).map((log, idx) => (
          <p key={`${log.time}-${idx}`}>
            <time>{log.time}</time>
            <b>{log.source}</b>
            <span>{log.text}</span>
          </p>
        ))}
      </div>
    </section>
  )
}
