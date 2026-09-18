export type DataKind = 'observed' | 'estimated' | 'predicted' | 'simulated'
export type Health = 'live' | 'stale' | 'degraded' | 'offline' | 'simulation' | 'error'
export type Risk = 'low' | 'medium' | 'high' | 'critical'
export type SensorType = 'gnss' | 'ais' | 'gyro' | 'ins' | 'radar' | 'forward_sonar' | 'depth' | 'adcp' | 'weather' | 'wind' | 'barometer' | 'sea_temperature' | 'current' | 'speed_log'
export type Transport = 'simulated' | 'serial' | 'tcp' | 'udp' | 'can_gateway' | 'file_replay' | 'websocket'

export interface Provenance {
  source: string
  timestamp: string
  status: Health
  confidence?: number
  kind: DataKind
}

export interface PlotPoint {
  x: number
  y: number
}

export interface SensorReading<T = number | string> {
  sensorId: string
  sensorType: SensorType
  sequence: number
  timestampUtc: string
  source: string
  value: T
  unit: string
  quality: number
  status: Health
}

export interface SensorHealth {
  sensorId: string
  sensorType: SensorType
  status: Health
  lastSeen: string
  latencyMs: number
  quality: number
  errorCount: number
  detail: string
}

export type PolarRegion = 'antarctic' | 'arctic' | 'indo-pacific'

export interface VesselState extends Provenance {
  name: string
  latitude: number
  longitude: number
  sog: number
  cog: number
  heading: number
  rateOfTurn: number
  stw: number
  draft: number
  depth: number
  status: Health
  vesselStatus: string
  destination: string
  eta: string
  fuelLevel: number
  fuelConsumption: number
  engineRpm: number
  engineStatus: string
  windSpeed: number
  windDirection: number
  seaTemperature: number
  currentSpeed: number
  currentDirection: number
  track: PlotPoint[]
}

export interface WeatherState extends Provenance {
  windSpeed: number
  windDirection: number
  airTemperature: number
  pressureHpa: number
  humidity: number
  visibilityNm: number
  waveHeight: number
  waveDirection: number
  weatherRisk: string
}

export interface OceanState extends Provenance {
  currentSpeed: number
  currentDirection: number
  seaSurfaceTemperature: number
  waveHeight: number
  waveDirection: number
  salinityPsu: number
  layerDepthM: number
}

export interface AisTarget extends Provenance {
  id: string
  mmsi?: string
  name: string
  x: number
  y: number
  heading: number
  cog: number
  sog: number
  cpaKm?: number
  tcpaMinutes?: number
}

export interface IcebergObservation extends Provenance {
  id: string
  type: 'TABULAR' | 'PINNACLE' | 'WEDGE' | 'BERGY BIT' | 'GROWLER'
  x: number
  y: number
  dimensions: string
  draft: number
  massMt?: number
  speed: number
  heading: number
  risk: Risk
  cpa: number
  cpaHours: number
  trajectory?: PlotPoint[]
  uncertainty: number
  detectionSource: string
  predictionHorizon?: string
}

export interface SonarDetection extends Provenance {
  id: string
  x: number
  y: number
  rangeKm: number
  bearing: number
  depthM: number
  returnStrength: 'weak' | 'strong'
  classification: 'POSSIBLE ICE' | 'SEABED' | 'UNDERWATER OBSTRUCTION' | 'UNCLASSIFIED'
  observationState: 'DETECTION' | 'CLASSIFICATION'
}

export interface RadarContact extends Provenance {
  contactId: string
  x: number
  y: number
  rangeNm: number
  bearing: number
  course: number
  speed: number
}

export interface SensorDiagnosticDetail {
  id: string
  name: string
  type: string
  sensorId?: string
  sensorType?: SensorType
  status: Health
  metric: string
  unit: string
  latencyMs: number
  quality: number
  lastSeen: string
  errors: number
  errorCount?: number
  detail?: string
  transport: string
  packetCount?: number
  packetLoss?: string
  satelliteCount?: number
  diagnostics?: string
}

export interface SensorStatus extends SensorHealth {
  name: string
  metric: string
}

export interface HazardEvent extends Provenance {
  level: 'CRITICAL' | 'WARNING' | 'CAUTION' | 'INFO'
  title: string
  message: string
  action: string
  category?: string
  delta?: string
}

export interface EnvironmentState {
  wind: string
  current: string
  seaState: string
  temperature: string
  pack: string
}

export interface LogEntry {
  time: string
  source: string
  text: string
  kind: DataKind
}

export interface DataChangeLogEntry {
  id: string
  timestamp: string
  field: string
  oldValue: string
  newValue: string
  delta?: string
  category: 'VESSEL' | 'WEATHER' | 'SEA ICE' | 'ICEBERGS' | 'RISK' | 'SENSORS' | 'ROUTES'
}

export interface IceSectorData {
  name: string
  concentration: number
  thickness: string
  edge: string
  drift: string
  trend: string
  status: 'CRITICAL' | 'WARNING' | 'CAUTION' | 'NOMINAL'
}

export interface IceForecastPeriod {
  horizon: string
  conc: number
  drift: string
  pressure: string
  risk: 'CRITICAL' | 'WARNING' | 'HIGH' | 'CAUTION' | 'NOMINAL'
}

export interface RouteWaypoint {
  id: string
  name: string
  coords: string
  distNm: number
  eta: string
  status: 'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL'
  risk: 'LOW' | 'MEDIUM' | 'HIGH'
}

export interface RouteAlternative {
  id: string
  name: string
  tag: string
  distanceNm: number
  etaHours: number
  etaString: string
  fuelMt: number
  riskScore: number
  iceExposure: string
  weatherExposure: string
  status: string
  statusType: 'CRITICAL' | 'WARNING' | 'NOMINAL' | 'CAUTION'
  description: string
  waypoints: RouteWaypoint[]
}

export interface RiskCategoryDetail {
  id: string
  name: string
  score: number
  previousScore?: number
  level: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW'
  trend: string
  desc: string
  weight: number
  contributingFactors: string[]
  mitigation: string
}

export interface SeaIceState extends Provenance {
  concentration: number
  thickness: number
  edgeDistanceNm: number
  driftSpeedKn: number
  driftDirectionDeg: number
  pressureHpa: number
  compressionTrend: string
  forecastConfidence: number
  sectors: IceSectorData[]
  forecast: IceForecastPeriod[]
}

export interface OperationalSnapshot {
  vessel: VesselState
  weather: WeatherState
  ocean: OceanState
  seaIce: SeaIceState
  ais: AisTarget[]
  icebergs: IcebergObservation[]
  sonar: SonarDetection[]
  radar: RadarContact[]
  sensors: SensorDiagnosticDetail[]
  hazards: HazardEvent[]
  riskCategories: RiskCategoryDetail[]
  compositeRiskScore: number
  compositeRiskTrend: 'INCREASING' | 'STABLE' | 'DECREASING'
  compositeRiskDelta: string
  routes: {
    activeRouteId: string
    availableRoutes: RouteAlternative[]
    riskFactors: { factor: string; level: string; score: number; desc: string }[]
  }
  environment: EnvironmentState
  logs: LogEntry[]
  changeLog: DataChangeLogEntry[]
  dataFreshness: string
  lastSyncTimestampUtc: string
  secondsSinceLastSync: number
  region: PolarRegion
}

export interface TransportAdapter {
  readonly transport: Transport
  connect(): Promise<void>
  close(): Promise<void>
  onPacket(callback: (packet: Uint8Array | string) => void): () => void
}

export interface ProtocolParser {
  readonly protocol: string
  parse(packet: Uint8Array | string, source: string): SensorReading[]
}

export interface SensorAdapter {
  readonly sourceId: string
  connect(): Promise<void>
  latest(): Promise<OperationalSnapshot>
  disconnect(): Promise<void>
}

export interface LocalDataBus {
  publish(reading: SensorReading): void
  subscribe(callback: (reading: SensorReading) => void): () => void
  latest(sensorId: string): SensorReading | undefined
}

export interface HardwareConfig {
  enabled: boolean
  transport: Transport
  endpoint?: string
  protocol?: 'nmea0183' | 'ais' | 'raw'
}

export interface HardwareConfiguration {
  hardware: Partial<Record<SensorType, HardwareConfig>>
}
