import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { Map as MapLibreMap } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { OperationalSnapshot } from '../types/operational'
import type { PolarRegion } from '../data/mapLayers'

export type LngLat = [number, number]

export interface MapReadout {
  latitude: number
  longitude: number
  rangeNm: number
  zoom: number
  bearing: number
}

export interface VesselTelemetryData {
  name: string
  latitude: number
  longitude: number
  sog: number
  cog: number
  heading: number
  rateOfTurn: number
  stw: number
  windDirection: number
  windSpeed: number
  depth: number
  aisContactCount: number
  radarContactCount: number
  draft: number
  engineRpm: number
  provenance: string
}

interface Props {
  state: OperationalSnapshot
  selectedId: string
  onSelect: (id: string) => void
  onSelectVessel?: () => void
  visibleLayers?: Set<string>
  region?: PolarRegion
  rangeNm?: number
  onReadout?: (readout: MapReadout) => void
  vesselTelemetry: VesselTelemetryData
}

const REGION_CONFIGS: Record<PolarRegion, { center: LngLat; zoom: number; vessel: LngLat }> = {
  antarctic: { center: [170.0, -69.5], zoom: 7.2, vessel: [170.0, -69.5] },
  'indo-pacific': { center: [84.5, 12.0], zoom: 5.6, vessel: [84.5, 12.0] },
  arctic: { center: [-18.0, 78.5], zoom: 6.2, vessel: [-18.0, 78.5] },
}

const EARTH_RADIUS_M = 6371008.8
const toRad = (d: number) => (d * Math.PI) / 180
const toDeg = (r: number) => (r * 180) / Math.PI

/** Calculates destination coordinate given starting LngLat, bearing in degrees, and distance in metres */
export function destination([lon, lat]: LngLat, bearingDeg: number, metres: number): LngLat {
  const d = metres / EARTH_RADIUS_M
  const b = toRad(bearingDeg)
  const p1 = toRad(lat)
  const l1 = toRad(lon)
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b))
  const l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2))
  return [((toDeg(l2) + 540) % 360) - 180, toDeg(p2)]
}

export function NavigationRenderer({
  selectedId,
  onSelect,
  onSelectVessel,
  visibleLayers,
  region = 'antarctic',
  rangeNm = 12,
  onReadout,
  vesselTelemetry,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [cameraVersion, setCameraVersion] = useState(0)
  const lastReadoutTime = useRef(0)

  const vesselPos: LngLat = useMemo(
    () => [vesselTelemetry.longitude, vesselTelemetry.latitude],
    [vesselTelemetry.longitude, vesselTelemetry.latitude]
  )

  const isVisible = useCallback(
    (id: string) => visibleLayers?.has(id) ?? true,
    [visibleLayers]
  )

  // Initialize MapLibre GL Map
  useEffect(() => {
    if (!containerRef.current) return

    const initial = REGION_CONFIGS[region] || REGION_CONFIGS.antarctic
    let map: MapLibreMap

    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        center: initial.center,
        zoom: initial.zoom,
        bearing: 0,
        pitch: 0,
        maxPitch: 60,
        attributionControl: false,
        style: {
          version: 8,
          sources: {
            'ne-land': {
              type: 'geojson',
              data: '/map-data/natural-earth/land.geojson',
            },
            'ne-coastline': {
              type: 'geojson',
              data: '/map-data/natural-earth/coastline.geojson',
            },
            'ne-glaciated': {
              type: 'geojson',
              data: '/map-data/natural-earth/glaciated.geojson',
            },
            'ne-shelves': {
              type: 'geojson',
              data: '/map-data/natural-earth/ice-shelves.geojson',
            },
            'ne-graticule': {
              type: 'geojson',
              data: '/map-data/natural-earth/graticule.geojson',
            },
          },
          layers: [
            {
              id: 'layer-ocean',
              type: 'background',
              paint: { 'background-color': '#061113' },
            },
            {
              id: 'layer-land',
              type: 'fill',
              source: 'ne-land',
              paint: {
                'fill-color': '#112225',
                'fill-outline-color': '#2a4c50',
              },
            },
            {
              id: 'layer-glaciated',
              type: 'fill',
              source: 'ne-glaciated',
              paint: {
                'fill-color': '#6c9699',
                'fill-opacity': 0.22,
              },
            },
            {
              id: 'layer-ice-shelves',
              type: 'fill',
              source: 'ne-shelves',
              paint: {
                'fill-color': '#8eb8b5',
                'fill-opacity': 0.28,
                'fill-outline-color': '#5f9491',
              },
            },
            {
              id: 'layer-coastline',
              type: 'line',
              source: 'ne-coastline',
              paint: {
                'line-color': '#578f8e',
                'line-width': ['interpolate', ['linear'], ['zoom'], 2, 0.8, 6, 1.6, 10, 2.6],
              },
            },
            {
              id: 'layer-graticule',
              type: 'line',
              source: 'ne-graticule',
              paint: {
                'line-color': '#1c3437',
                'line-width': 0.7,
                'line-dasharray': [2, 3],
              },
            },
          ],
        },
      })
    } catch (err) {
      console.warn('MapLibre GL initialization fallback:', err)
      return
    }

    map.addControl(new maplibregl.ScaleControl({ maxWidth: 140, unit: 'nautical' }), 'bottom-right')

    map.on('load', () => {
      setMapLoaded(true)
      map.resize()
    })

    const handleCameraChange = () => {
      setCameraVersion((v) => v + 1)

      const now = performance.now()
      if (now - lastReadoutTime.current < 80) return
      lastReadoutTime.current = now

      const center = map.getCenter()
      const bounds = map.getBounds()
      const latSpanM = EARTH_RADIUS_M * Math.abs(toRad(bounds.getNorth() - bounds.getSouth()))
      const computedRangeNm = latSpanM / 3704

      onReadout?.({
        latitude: center.lat,
        longitude: center.lng,
        rangeNm: Math.max(1, computedRangeNm),
        zoom: map.getZoom(),
        bearing: map.getBearing(),
      })
    }

    map.on('move', handleCameraChange)
    map.on('zoom', handleCameraChange)
    map.on('rotate', handleCameraChange)
    map.on('pitch', handleCameraChange)
    map.on('resize', handleCameraChange)

    mapRef.current = map

    const resizeObserver = new ResizeObserver(() => {
      map.resize()
      setCameraVersion((v) => v + 1)
    })
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Sync region flyTo
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const target = REGION_CONFIGS[region] || REGION_CONFIGS.antarctic
    map.flyTo({
      center: target.center,
      zoom: target.zoom,
      bearing: 0,
      pitch: 0,
      speed: 1.2,
      essential: true,
    })
  }, [region])

  // Sync layer visibility to MapLibre basemap layers
  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapLoaded) return

    const baseMapLayers: Record<string, string[]> = {
      'polar-base': ['layer-land', 'layer-glaciated'],
      coastline: ['layer-coastline'],
      'ice-shelves': ['layer-ice-shelves'],
      graticule: ['layer-graticule'],
    }

    Object.entries(baseMapLayers).forEach(([key, layerIds]) => {
      const show = isVisible(key)
      layerIds.forEach((id) => {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, 'visibility', show ? 'visible' : 'none')
        }
      })
    })
  }, [visibleLayers, mapLoaded, isVisible])

  // Camera toolbar actions
  const handleCameraAction = (action: string) => {
    const map = mapRef.current
    if (!map) return

    if (action === '+') map.zoomIn({ duration: 300 })
    else if (action === '−') map.zoomOut({ duration: 300 })
    else if (action === 'NORTH UP') map.rotateTo(0, { duration: 400 })
    else if (action === 'CENTER VESSEL') {
      map.easeTo({
        center: vesselPos,
        zoom: Math.max(map.getZoom(), 7.2),
        essential: true,
      })
    } else if (action === 'FOLLOW VESSEL') {
      map.easeTo({
        center: vesselPos,
        bearing: vesselTelemetry.heading,
        zoom: 7.8,
        essential: true,
      })
    }
  }

  // Project geographic coordinates to screen pixel coordinates
  const project = useCallback(
    (coords: LngLat): { x: number; y: number } | null => {
      const map = mapRef.current
      if (!map) return null
      try {
        const p = map.project(coords)
        return { x: p.x, y: p.y }
      } catch {
        return null
      }
    },
    // cameraVersion triggers recalculation on pan/zoom
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cameraVersion]
  )

  // Radar sweep animation frame
  const [sweepAngle, setSweepAngle] = useState(0)
  useEffect(() => {
    let frame = 0
    const animate = (time: number) => {
      setSweepAngle((time / 55) % 360)
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [])

  // Icebergs data
  const icebergs = useMemo(() => getRegionIcebergs(region, vesselPos), [region, vesselPos])
  const ice042 = useMemo(() => icebergs.find((i) => i.id === 'ICE-042') || icebergs[0], [icebergs])

  // AIS contacts data (at least 8 contacts)
  const aisContacts = useMemo(() => getRegionAisContacts(region, vesselPos), [region, vesselPos])

  // Radar contacts data (at least 5 contacts)
  const radarContacts = useMemo(() => getRegionRadarContacts(region, vesselPos), [region, vesselPos])

  // Route waypoints
  const waypoints = useMemo(() => getRegionRouteWaypoints(region, vesselPos), [region, vesselPos])

  // Geographic landmark labels
  const landmarkLabels = useMemo(() => getRegionLandmarks(region), [region])

  // Screen projected coordinates
  const vesselPt = project(vesselPos)
  const vesselHeadingTip = project(destination(vesselPos, vesselTelemetry.heading, 18000))

  return (
    <div className="maplibre-shell" aria-label="Interactive polar vector navigation map">
      {/* MapLibre GL Canvas Container (Basemap & GPU Vector Tiles) */}
      <div ref={containerRef} className="maplibre-map" />

      {/* Synchronized Tactical Operational SVG Overlay (100% Reliable Offline Vector Layers) */}
      <svg className="tactical-operational-overlay" aria-hidden="true">
        <defs>
          <radialGradient id="vessel-halo-grad">
            <stop offset="0%" stopColor="#78bda6" stopOpacity="0.4" />
            <stop offset="70%" stopColor="#78bda6" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#78bda6" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="ice42-halo-grad">
            <stop offset="0%" stopColor="#e8784d" stopOpacity="0.5" />
            <stop offset="65%" stopColor="#e8784d" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#e8784d" stopOpacity="0" />
          </radialGradient>
          <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* 1. Dynamic Local Latitude/Longitude Grid Lines */}
        {isVisible('grid') && (
          <g className="layer-tactical-grid" opacity="0.6">
            {generateLocalGridLines(region, vesselPos).map((line, idx) => {
              const p1 = project(line[0])
              const p2 = project(line[1])
              return p1 && p2 ? (
                <line
                  key={`grid-${idx}`}
                  x1={p1.x}
                  y1={p1.y}
                  x2={p2.x}
                  y2={p2.y}
                  stroke="#264549"
                  strokeWidth="0.8"
                  strokeDasharray="3 3"
                />
              ) : null
            })}
          </g>
        )}

        {/* 2. Concentric Range Rings centered on RV BHARATI */}
        {isVisible('range-rings') && vesselPt && (
          <g className="layer-range-rings">
            {[0.25, 0.5, 0.75, 1.0].map((frac) => {
              const ringCoords = generateCircleCoords(vesselPos, rangeNm * frac * 1852, 48)
              const pathD = buildSvgPath(ringCoords, project)
              const topPoint = project(destination(vesselPos, 0, rangeNm * frac * 1852))
              return pathD ? (
                <g key={`ring-${frac}`}>
                  <path d={pathD} fill="none" stroke="#3b696e" strokeWidth="1.1" strokeDasharray="4 3" />
                  {topPoint && (
                    <text
                      x={topPoint.x + 4}
                      y={topPoint.y - 4}
                      fill="#7da5a7"
                      fontSize="8.5"
                      fontFamily="monospace"
                      letterSpacing="0.5"
                    >
                      {(rangeNm * frac).toFixed(0)} NM
                    </text>
                  )}
                </g>
              ) : null
            })}
          </g>
        )}

        {/* 3. Safe Navigation Corridor & Route Waypoints */}
        {isVisible('safe-corridor') && (
          <g className="layer-safe-corridor">
            {/* Corridor Swath Buffer Polygon */}
            {(() => {
              const swathCoords = generateCorridorSwath(waypoints, 9260)
              const swathPath = buildSvgPath(swathCoords, project)
              return swathPath ? (
                <path d={swathPath} fill="#325d55" fillOpacity="0.18" stroke="#5f9d86" strokeWidth="1.2" />
              ) : null
            })()}

            {/* Route Centerline */}
            {(() => {
              const routePath = buildSvgPath(waypoints, project, false)
              return routePath ? (
                <path
                  d={routePath}
                  fill="none"
                  stroke="#79bfa7"
                  strokeWidth="2.2"
                  strokeDasharray="5 2.5"
                  filter="url(#glow-cyan)"
                />
              ) : null
            })()}

            {/* Route Waypoints */}
            {waypoints.map((wp, idx) => {
              const pt = project(wp)
              return pt ? (
                <g key={`wp-${idx}`} transform={`translate(${pt.x}, ${pt.y})`}>
                  <polygon points="0,-5 5,0 0,5 -5,0" fill="#79bfa7" stroke="#061113" strokeWidth="1.5" />
                  <text x="7" y="3" fill="#cae5dc" fontSize="8.5" fontFamily="monospace">
                    WP-0{idx + 1}
                  </text>
                </g>
              ) : null
            })}
          </g>
        )}

        {/* 4. Environmental Wind Vectors (242° / 23 kn) */}
        {isVisible('wind') && (
          <g className="layer-wind-vectors" opacity="0.85">
            {generateVectorFieldGrid(region, vesselPos).map((root, idx) => {
              const tip = destination(root, vesselTelemetry.windDirection, vesselTelemetry.windSpeed * 380)
              const barb = destination(tip, vesselTelemetry.windDirection + 145, 2200)
              const pRoot = project(root)
              const pTip = project(tip)
              const pBarb = project(barb)
              return pRoot && pTip ? (
                <g key={`wind-${idx}`}>
                  <line x1={pRoot.x} y1={pRoot.y} x2={pTip.x} y2={pTip.y} stroke="#689e9a" strokeWidth="1.3" />
                  {pBarb && <line x1={pTip.x} y1={pTip.y} x2={pBarb.x} y2={pBarb.y} stroke="#689e9a" strokeWidth="1.3" />}
                </g>
              ) : null
            })}
          </g>
        )}

        {/* 5. Ocean Current Vectors (196° / 0.42 m/s) */}
        {isVisible('current') && (
          <g className="layer-current-vectors" opacity="0.75">
            {generateVectorFieldGrid(region, vesselPos, 0.2, 0.15).map((root, idx) => {
              const tip = destination(root, 196, 6800)
              const pRoot = project(root)
              const pTip = project(tip)
              return pRoot && pTip ? (
                <line
                  key={`curr-${idx}`}
                  x1={pRoot.x}
                  y1={pRoot.y}
                  x2={pTip.x}
                  y2={pTip.y}
                  stroke="#4b8e96"
                  strokeWidth="1.4"
                  strokeDasharray="4 2"
                />
              ) : null
            })}
          </g>
        )}

        {/* 6. Forward Sonar Acoustic Cone */}
        {isVisible('sonar') && vesselPt && (
          <g className="layer-sonar-cone">
            {(() => {
              const coneCoords = [
                vesselPos,
                destination(vesselPos, vesselTelemetry.heading - 28, 4800),
                destination(vesselPos, vesselTelemetry.heading + 28, 4800),
                vesselPos,
              ]
              const pathD = buildSvgPath(coneCoords, project)
              return pathD ? (
                <path d={pathD} fill="#6cb2b7" fillOpacity="0.14" stroke="#78bec3" strokeWidth="1.2" />
              ) : null
            })()}
          </g>
        )}

        {/* 7. Animated Radar Sweep Sector */}
        {isVisible('radar') && vesselPt && (
          <g className="layer-radar-sweep">
            {(() => {
              const sweepCoords = [
                vesselPos,
                ...Array.from({ length: 12 }, (_, i) =>
                  destination(vesselPos, sweepAngle - 20 + (i * 40) / 11, 12 * 1852)
                ),
                vesselPos,
              ]
              const pathD = buildSvgPath(sweepCoords, project)
              const leadTip = project(destination(vesselPos, sweepAngle, 12 * 1852))
              return (
                <>
                  {pathD && <path d={pathD} fill="#64a9aa" fillOpacity="0.18" stroke="#82cac9" strokeWidth="1" />}
                  {leadTip && (
                    <line
                      x1={vesselPt.x}
                      y1={vesselPt.y}
                      x2={leadTip.x}
                      y2={leadTip.y}
                      stroke="#a6f0eb"
                      strokeWidth="1.8"
                      opacity="0.9"
                    />
                  )}
                </>
              )
            })()}
          </g>
        )}

        {/* 8. ICE-042 Uncertainty Envelope, 72h Prediction Horizon, and Historical Drift */}
        {ice042 && (
          <g className="layer-ice42-prediction">
            {/* Uncertainty Corridor Swath */}
            {isVisible('uncertainty') &&
              (() => {
                const swathCoords = generateIce42UncertaintySwath(ice042.coordinates)
                const pathD = buildSvgPath(swathCoords, project)
                return pathD ? (
                  <path d={pathD} fill="#d87e50" fillOpacity="0.18" stroke="#e08a5b" strokeWidth="1.2" />
                ) : null
              })()}

            {/* Historical 24h Track */}
            {isVisible('trajectories') &&
              (() => {
                const histPoints = [
                  destination(ice042.coordinates, 24, 18000),
                  destination(ice042.coordinates, 22, 12000),
                  destination(ice042.coordinates, 25, 6000),
                  ice042.coordinates,
                ]
                const pathD = buildSvgPath(histPoints, project, false)
                return pathD ? (
                  <path d={pathD} fill="none" stroke="#9e7354" strokeWidth="1.8" strokeDasharray="2 3" />
                ) : null
              })()}

            {/* 72h Predicted Trajectory Line */}
            {isVisible('trajectories') &&
              (() => {
                const milestones = getIce42Milestones(ice042.coordinates)
                const pathD = buildSvgPath(
                  milestones.map((m) => m.coord),
                  project,
                  false
                )
                return pathD ? (
                  <path
                    d={pathD}
                    fill="none"
                    stroke="#ea995c"
                    strokeWidth="2.4"
                    strokeDasharray="4 2"
                    filter="url(#glow-cyan)"
                  />
                ) : null
              })()}

            {/* Future Milestone Waypoint Tags */}
            {isVisible('trajectories') &&
              getIce42Milestones(ice042.coordinates)
                .slice(1)
                .map((m, idx) => {
                  const pt = project(m.coord)
                  return pt ? (
                    <g key={`ice42-m-${idx}`} transform={`translate(${pt.x}, ${pt.y})`}>
                      <circle r="3.5" fill="#e89e67" stroke="#061012" strokeWidth="1.2" />
                      <text x="6" y="3" fill="#e0ab85" fontSize="8" fontFamily="monospace">
                        {m.t}
                      </text>
                    </g>
                  ) : null
                })}
          </g>
        )}

        {/* 9. Other Iceberg Trajectories */}
        {isVisible('trajectories') && (
          <g className="layer-iceberg-trajectories">
            {icebergs
              .filter((i) => i.id !== 'ICE-042' && i.trajectory.length > 1)
              .map((ice, idx) => {
                const pathD = buildSvgPath(ice.trajectory, project, false)
                return pathD ? (
                  <path
                    key={`ice-traj-${idx}`}
                    d={pathD}
                    fill="none"
                    stroke="#cca064"
                    strokeWidth="1.6"
                    strokeDasharray="3 2"
                  />
                ) : null
              })}
          </g>
        )}

        {/* 10. Radar Contacts */}
        {isVisible('radar') && (
          <g className="layer-radar-contacts">
            {radarContacts.map((rad) => {
              const pt = project(rad.coordinates)
              return pt ? (
                <g key={rad.contactId} transform={`translate(${pt.x}, ${pt.y})`}>
                  <circle r="4.5" fill="none" stroke="#75d0d1" strokeWidth="1.5" />
                  <circle r="1.5" fill="#75d0d1" />
                  <text x="7" y="3" fill="#9de6e7" fontSize="8" fontFamily="monospace">
                    {rad.contactId} ({rad.speed} KT)
                  </text>
                </g>
              ) : null
            })}
          </g>
        )}

        {/* 11. AIS Contacts */}
        {isVisible('ais') && (
          <g className="layer-ais-contacts">
            {aisContacts.map((ais) => {
              const pt = project(ais.coordinates)
              const headingTip = project(destination(ais.coordinates, ais.heading, ais.sog * 1852))
              return pt ? (
                <g key={ais.id} transform={`translate(${pt.x}, ${pt.y})`}>
                  {headingTip && (
                    <line
                      x1="0"
                      y1="0"
                      x2={headingTip.x - pt.x}
                      y2={headingTip.y - pt.y}
                      stroke="#86cde0"
                      strokeWidth="1.5"
                    />
                  )}
                  <polygon
                    points="0,-6 4.5,5 0,2.5 -4.5,5"
                    transform={`rotate(${ais.heading})`}
                    fill="#7fc2d3"
                    stroke="#091517"
                    strokeWidth="1.2"
                  />
                  <text x="8" y="3" fill="#b9e4ef" fontSize="8.5" fontFamily="monospace">
                    {ais.name} ({ais.sog} KT)
                  </text>
                </g>
              ) : null
            })}
          </g>
        )}

        {/* 12. Iceberg Target Markers (Clickable) */}
        {isVisible('icebergs') && (
          <g className="layer-iceberg-markers">
            {icebergs.map((ice) => {
              const pt = project(ice.coordinates)
              const isSelected = ice.id === selectedId
              if (!pt) return null

              return (
                <g
                  key={ice.id}
                  className="interactive-svg-target"
                  transform={`translate(${pt.x}, ${pt.y})`}
                  onClick={() => onSelect(ice.id)}
                  style={{ cursor: 'pointer' }}
                >
                  {/* High Risk / Selected Halo */}
                  {(isSelected || ice.risk === 'high') && (
                    <circle r={isSelected ? 16 : 12} fill="url(#ice42-halo-grad)" />
                  )}

                  {/* Marker Shape */}
                  {ice.type === 'TABULAR' ? (
                    <rect
                      x="-6"
                      y="-4.5"
                      width="12"
                      height="9"
                      fill={isSelected ? '#f08658' : ice.risk === 'high' ? '#d46b4e' : '#d4a657'}
                      stroke="#061012"
                      strokeWidth="1.8"
                    />
                  ) : (
                    <circle
                      r={isSelected ? 7 : 5.5}
                      fill={isSelected ? '#f08658' : ice.risk === 'medium' ? '#d4a657' : '#79adb1'}
                      stroke="#061012"
                      strokeWidth="1.8"
                    />
                  )}

                  {/* Label */}
                  <text
                    x="10"
                    y="3"
                    fill={isSelected ? '#f7d0ba' : '#c8dedb'}
                    fontSize={isSelected ? '9.5' : '8.5'}
                    fontWeight={isSelected ? 'bold' : 'normal'}
                    fontFamily="monospace"
                  >
                    {ice.id} · {ice.type} ({ice.speed} KT)
                  </text>
                </g>
              )
            })}
          </g>
        )}

        {/* 13. Own Vessel: RV BHARATI (Clickable -> Opens Vessel Telemetry Panel) */}
        {isVisible('own-ship') && vesselPt && (
          <g
            className="layer-own-vessel"
            transform={`translate(${vesselPt.x}, ${vesselPt.y})`}
            onClick={onSelectVessel}
            style={{ cursor: 'pointer' }}
          >
            {/* Heading Vector */}
            {vesselHeadingTip && (
              <line
                x1="0"
                y1="0"
                x2={vesselHeadingTip.x - vesselPt.x}
                y2={vesselHeadingTip.y - vesselPt.y}
                stroke="#eef8f5"
                strokeWidth="2.2"
                strokeDasharray="6 2"
              />
            )}

            {/* Glowing Halo */}
            <circle r="16" fill="url(#vessel-halo-grad)" />

            {/* Ship Chevron Icon oriented with vessel.heading */}
            <g transform={`rotate(${vesselTelemetry.heading})`}>
              <polygon
                points="0,-12 8,8 0,4 -8,8"
                fill="#f2fcf8"
                stroke="#061012"
                strokeWidth="2"
                filter="url(#glow-cyan)"
              />
            </g>

            {/* Vessel Label */}
            <text x="14" y="3" fill="#eefaf5" fontSize="10" fontWeight="bold" fontFamily="monospace">
              {vesselTelemetry.name} [{vesselTelemetry.sog.toFixed(1)} KT]
            </text>
            <text x="14" y="14" fill="#8cbab3" fontSize="8" fontFamily="monospace">
              HDG {vesselTelemetry.heading.toFixed(1)}° · COG {vesselTelemetry.cog.toFixed(0)}°
            </text>
          </g>
        )}

        {/* 14. Landmark Geographic Labels */}
        {isVisible('graticule') && (
          <g className="layer-geographic-landmarks" opacity="0.85">
            {landmarkLabels.map((lm, idx) => {
              const pt = project(lm.coords)
              return pt ? (
                <text
                  key={`lm-${idx}`}
                  x={pt.x}
                  y={pt.y}
                  fill={lm.type === 'ocean' ? '#3d6c71' : lm.type === 'sea' ? '#528388' : '#7ba5a3'}
                  fontSize={lm.type === 'ocean' ? '13' : lm.type === 'sea' ? '11' : '9.5'}
                  fontFamily="monospace"
                  letterSpacing="1.8"
                  textAnchor="middle"
                >
                  {lm.name}
                </text>
              ) : null
            })}
          </g>
        )}
      </svg>

      {/* Map Camera Toolbar Controls */}
      <div className="map-camera-controls" aria-label="Map Camera Controls">
        {['+', '−', 'NORTH UP', 'CENTER VESSEL', 'FOLLOW VESSEL'].map((label) => (
          <button key={label} onClick={() => handleCameraAction(label)}>
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

// Helpers

function generateCircleCoords(center: LngLat, radiusM: number, points = 48): LngLat[] {
  const coords: LngLat[] = []
  for (let i = 0; i <= points; i++) {
    coords.push(destination(center, (i * 360) / points, radiusM))
  }
  return coords
}

function buildSvgPath(
  coords: LngLat[],
  project: (c: LngLat) => { x: number; y: number } | null,
  close = true
): string | null {
  const points = coords.map(project).filter((p): p is { x: number; y: number } => p !== null)
  if (points.length < 2) return null
  const d = points.reduce((acc, p, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`, '')
  return close ? `${d} Z` : d
}

function generateCorridorSwath(waypoints: LngLat[], bufferM: number): LngLat[] {
  const left: LngLat[] = []
  const right: LngLat[] = []
  waypoints.forEach((wp) => {
    left.push(destination(wp, 305, bufferM))
    right.push(destination(wp, 125, bufferM))
  })
  return [...left, ...right.reverse(), left[0]]
}

function getIce42Milestones(p0: LngLat) {
  return [
    { t: 'T+0h', coord: p0, bufferM: 1200 },
    { t: 'T+12h', coord: destination(p0, 204, 4500), bufferM: 1800 },
    { t: 'T+24h', coord: destination(p0, 203, 9000), bufferM: 2600 },
    { t: 'T+34h [CPA 11.2km]', coord: destination(p0, 202, 12800), bufferM: 3400 },
    { t: 'T+48h', coord: destination(p0, 200, 18000), bufferM: 4500 },
    { t: 'T+60h', coord: destination(p0, 198, 22500), bufferM: 5600 },
    { t: 'T+72h', coord: destination(p0, 196, 27000), bufferM: 6800 },
  ]
}

function generateIce42UncertaintySwath(p0: LngLat): LngLat[] {
  const milestones = getIce42Milestones(p0)
  const left: LngLat[] = []
  const right: LngLat[] = []
  milestones.forEach((m) => {
    left.push(destination(m.coord, 204 - 90, m.bufferM))
    right.push(destination(m.coord, 204 + 90, m.bufferM))
  })
  return [...left, ...right.reverse(), left[0]]
}

function generateLocalGridLines(region: PolarRegion, vPos: LngLat): [LngLat, LngLat][] {
  const lines: [LngLat, LngLat][] = []
  const latSpan = region === 'indo-pacific' ? 6 : 2.5
  const lonSpan = region === 'indo-pacific' ? 8 : 4
  const step = region === 'indo-pacific' ? 2 : 0.5

  for (let lat = Math.floor(vPos[1] - latSpan); lat <= Math.ceil(vPos[1] + latSpan); lat += step) {
    lines.push([
      [vPos[0] - lonSpan, lat],
      [vPos[0] + lonSpan, lat],
    ])
  }
  for (let lon = Math.floor(vPos[0] - lonSpan); lon <= Math.ceil(vPos[0] + lonSpan); lon += step) {
    lines.push([
      [lon, vPos[1] - latSpan],
      [lon, vPos[1] + latSpan],
    ])
  }
  return lines
}

function generateVectorFieldGrid(
  region: PolarRegion,
  vPos: LngLat,
  dlonOff = 0,
  dlatOff = 0
): LngLat[] {
  const points: LngLat[] = []
  const step = region === 'indo-pacific' ? 1.5 : 0.6
  for (let dlat = -2; dlat <= 2; dlat++) {
    for (let dlon = -2; dlon <= 2; dlon++) {
      points.push([
        vPos[0] + dlon * step + dlonOff,
        vPos[1] + dlat * (step * 0.6) + dlatOff,
      ])
    }
  }
  return points
}

function getRegionIcebergs(region: PolarRegion, vPos: LngLat) {
  if (region === 'antarctic') {
    return [
      {
        id: 'ICE-042',
        type: 'TABULAR',
        speed: 0.37,
        heading: 204,
        draft: 245,
        risk: 'high',
        cpa: 11.2,
        coordinates: [170.16, -69.34] as LngLat,
        trajectory: [],
      },
      {
        id: 'ICE-088',
        type: 'PINNACLE',
        speed: 0.52,
        heading: 181,
        draft: 78,
        risk: 'medium',
        cpa: 22.8,
        coordinates: [169.65, -69.62] as LngLat,
        trajectory: [
          [169.65, -69.62],
          destination([169.65, -69.62], 181, 10000),
          destination([169.65, -69.62], 180, 20000),
        ] as LngLat[],
      },
      {
        id: 'ICE-104',
        type: 'WEDGE',
        speed: 0.41,
        heading: 244,
        draft: 112,
        risk: 'low',
        cpa: 39.4,
        coordinates: [170.48, -69.68] as LngLat,
        trajectory: [
          [170.48, -69.68],
          destination([170.48, -69.68], 244, 9000),
          destination([170.48, -69.68], 240, 18000),
        ] as LngLat[],
      },
      {
        id: 'ICE-019',
        type: 'BERGY BIT',
        speed: 0.65,
        heading: 148,
        draft: 29,
        risk: 'medium',
        cpa: 18.6,
        coordinates: [169.78, -69.38] as LngLat,
        trajectory: [],
      },
      {
        id: 'ICE-121',
        type: 'GROWLER',
        speed: 0.48,
        heading: 231,
        draft: 12,
        risk: 'low',
        cpa: 54.1,
        coordinates: [170.35, -69.22] as LngLat,
        trajectory: [],
      },
      {
        id: 'ICE-205',
        type: 'TABULAR',
        speed: 0.29,
        heading: 198,
        draft: 190,
        risk: 'medium',
        cpa: 28.3,
        coordinates: [169.42, -69.75] as LngLat,
        trajectory: [],
      },
    ]
  }

  return [
    {
      id: 'ICE-042',
      type: 'TABULAR',
      speed: 0.37,
      heading: 204,
      draft: 245,
      risk: 'high',
      cpa: 11.2,
      coordinates: destination(vPos, 35, 16000),
      trajectory: [],
    },
    {
      id: 'ICE-088',
      type: 'PINNACLE',
      speed: 0.52,
      heading: 181,
      draft: 78,
      risk: 'medium',
      cpa: 22.8,
      coordinates: destination(vPos, 220, 18000),
      trajectory: [destination(vPos, 220, 18000), destination(vPos, 200, 30000)],
    },
    {
      id: 'ICE-104',
      type: 'WEDGE',
      speed: 0.41,
      heading: 244,
      draft: 112,
      risk: 'low',
      cpa: 39.4,
      coordinates: destination(vPos, 110, 24000),
      trajectory: [destination(vPos, 110, 24000), destination(vPos, 130, 36000)],
    },
    {
      id: 'ICE-019',
      type: 'BERGY BIT',
      speed: 0.65,
      heading: 148,
      draft: 29,
      risk: 'medium',
      cpa: 18.6,
      coordinates: destination(vPos, 310, 14000),
      trajectory: [],
    },
    {
      id: 'ICE-121',
      type: 'GROWLER',
      speed: 0.48,
      heading: 231,
      draft: 12,
      risk: 'low',
      cpa: 54.1,
      coordinates: destination(vPos, 60, 28000),
      trajectory: [],
    },
    {
      id: 'ICE-205',
      type: 'TABULAR',
      speed: 0.29,
      heading: 198,
      draft: 190,
      risk: 'medium',
      cpa: 28.3,
      coordinates: destination(vPos, 250, 22000),
      trajectory: [],
    },
  ]
}

function getRegionAisContacts(region: PolarRegion, vPos: LngLat) {
  if (region === 'antarctic') {
    return [
      { id: 'AIS-302', name: 'RV AURORA AUSTRALIS', sog: 10.2, heading: 210, coordinates: [169.85, -69.42] as LngLat },
      { id: 'AIS-611', name: 'POLAR SUPPLY', sog: 8.1, heading: 195, coordinates: [170.25, -69.60] as LngLat },
      { id: 'AIS-408', name: 'NATHANIEL B. PALMER', sog: 9.5, heading: 180, coordinates: [169.52, -69.30] as LngLat },
      { id: 'AIS-512', name: 'ARA ALMIRANTE IRIZAR', sog: 11.0, heading: 225, coordinates: [170.38, -69.75] as LngLat },
      { id: 'AIS-720', name: 'SA AGULHAS II', sog: 12.4, heading: 205, coordinates: [169.90, -69.70] as LngLat },
      { id: 'AIS-815', name: 'RRS SIR DAVID ATTENBOROUGH', sog: 10.8, heading: 218, coordinates: [170.50, -69.45] as LngLat },
      { id: 'AIS-930', name: 'RV KOREA ARAON', sog: 8.9, heading: 190, coordinates: [169.40, -69.55] as LngLat },
      { id: 'AIS-104', name: 'RV KRONPRINS HAAKON', sog: 11.2, heading: 230, coordinates: [170.60, -69.25] as LngLat },
    ]
  }

  return [
    { id: 'AIS-302', name: 'RV AURORA', sog: 10.2, heading: 183, coordinates: destination(vPos, 340, 12000) },
    { id: 'AIS-611', name: 'POLAR SUPPLY', sog: 8.1, heading: 245, coordinates: destination(vPos, 120, 16000) },
    { id: 'AIS-408', name: 'OCEAN DISCOVERY', sog: 9.5, heading: 190, coordinates: destination(vPos, 200, 22000) },
    { id: 'AIS-512', name: 'PACIFIC ENDEAVOUR', sog: 11.0, heading: 220, coordinates: destination(vPos, 45, 26000) },
    { id: 'AIS-720', name: 'SOUTHERN RESEARCH', sog: 12.4, heading: 210, coordinates: destination(vPos, 280, 18000) },
    { id: 'AIS-815', name: 'ARCTIC EXPLORER', sog: 10.8, heading: 215, coordinates: destination(vPos, 160, 24000) },
    { id: 'AIS-930', name: 'VICTORIA TRADER', sog: 8.9, heading: 175, coordinates: destination(vPos, 25, 20000) },
    { id: 'AIS-104', name: 'EASTERN DAWN', sog: 11.2, heading: 235, coordinates: destination(vPos, 230, 29000) },
  ]
}

function getRegionRadarContacts(region: PolarRegion, vPos: LngLat) {
  if (region === 'antarctic') {
    return [
      { contactId: 'RAD-081', speed: 4.2, coordinates: [169.92, -69.45] as LngLat },
      { contactId: 'RAD-084', speed: 6.8, coordinates: [170.12, -69.58] as LngLat },
      { contactId: 'RAD-092', speed: 0.4, coordinates: [169.75, -69.55] as LngLat },
      { contactId: 'RAD-099', speed: 11.1, coordinates: [170.18, -69.44] as LngLat },
      { contactId: 'RAD-112', speed: 0.4, coordinates: [170.16, -69.34] as LngLat },
    ]
  }

  return [
    { contactId: 'RAD-081', speed: 4.2, coordinates: destination(vPos, 198, 3.4 * 1852) },
    { contactId: 'RAD-084', speed: 6.8, coordinates: destination(vPos, 235, 5.1 * 1852) },
    { contactId: 'RAD-092', speed: 0.4, coordinates: destination(vPos, 172, 7.8 * 1852) },
    { contactId: 'RAD-099', speed: 11.1, coordinates: destination(vPos, 260, 4.2 * 1852) },
    { contactId: 'RAD-112', speed: 0.4, coordinates: destination(vPos, 210, 11.2 * 1852) },
  ]
}

function getRegionRouteWaypoints(region: PolarRegion, _vPos?: LngLat): LngLat[] {
  if (region === 'antarctic') {
    return [
      [171.2, -68.4],
      [170.8, -69.0],
      [170.0, -69.5],
      [169.1, -70.1],
      [167.8, -71.2],
      [166.5, -73.5],
    ]
  }

  if (region === 'indo-pacific') {
    return [
      [80.3, 13.08],
      [82.5, 12.0],
      [87.5, 9.5],
      [92.8, 11.6],
      [95.5, 5.8],
    ]
  }

  return [
    [-15.0, 79.5],
    [-18.0, 78.5],
    [-21.0, 77.2],
    [-24.0, 75.8],
  ]
}

function getRegionLandmarks(region: PolarRegion): { name: string; type: 'sea' | 'ocean' | 'landmark'; coords: LngLat }[] {
  if (region === 'antarctic') {
    return [
      { name: 'ROSS SEA', type: 'sea', coords: [170.0, -71.2] },
      { name: 'CAPE ADARE', type: 'landmark', coords: [170.2, -68.9] },
      { name: 'VICTORIA LAND', type: 'landmark', coords: [165.5, -70.5] },
      { name: 'BALLENY ISLANDS', type: 'sea', coords: [163.0, -66.5] },
      { name: 'SOUTHERN OCEAN', type: 'ocean', coords: [174.0, -65.5] },
      { name: 'MCMURDO SOUND APPROACH', type: 'sea', coords: [166.5, -73.0] },
    ]
  }

  if (region === 'indo-pacific') {
    return [
      { name: 'BAY OF BENGAL', type: 'sea', coords: [88.0, 14.5] },
      { name: 'ARABIAN SEA', type: 'sea', coords: [68.0, 16.0] },
      { name: 'INDIAN OCEAN', type: 'ocean', coords: [80.0, 2.0] },
      { name: 'ANDAMAN SEA', type: 'sea', coords: [94.0, 10.0] },
      { name: 'INDIA', type: 'landmark', coords: [78.9, 18.5] },
      { name: 'SRI LANKA', type: 'landmark', coords: [80.7, 7.8] },
    ]
  }

  return [
    { name: 'ARCTIC OCEAN', type: 'ocean', coords: [0.0, 80.0] },
    { name: 'BARENTS SEA', type: 'sea', coords: [30.0, 72.0] },
    { name: 'GREENLAND SEA', type: 'sea', coords: [-10.0, 76.0] },
  ]
}
