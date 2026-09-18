import React, { useEffect, useRef, useState } from 'react'
import VesselTelemetry from './VesselTelemetry'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { fetchEcas, fetchChokepoints, fetchSeaIce, fetchIceRisk, fetchOceanCurrents, fetchWind, fetchWeatherHeatmap, geocode } from '../api'
import { usePlanner } from '../store'
import LayerControlPanel from './LayerControlPanel'
import WindFlowLayer from './WindFlowLayer'
import { unwrapLongitudes, splitIntoRenderedParts } from '../geo'
import { getShipPositionAtTime, getIcebergPositionAtTime } from '../utils'

const MAX_WAYPOINTS = 8

const PIN_BLUE = '#1452c4'
const PIN_WHITE = '#ffffff'

// Every selectable basemap layer. The seamark overlay is deliberately absent:
// it draws on top of a basemap rather than being one.
const BASEMAP_LAYER_IDS = [
  'basemap-google-sat',
  'basemap-google-streets',
  'basemap-google-hybrid',
  'basemap-standard',
  'basemap-opentopo',
  'basemap-light',
]

function basemapStyle() {
  return {
    version: 8,
    glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
    sources: {
      'basemap-google-sat': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        ],
        tileSize: 256,
        attribution: '© Google Maps',
        maxzoom: 20,
      },
      'basemap-google-streets': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        ],
        tileSize: 256,
        attribution: '© Google Maps',
        maxzoom: 20,
      },
      'basemap-google-hybrid': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
        ],
        tileSize: 256,
        attribution: '© Google Maps',
        maxzoom: 20,
      },
      'basemap-standard': {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenStreetMap contributors',
        maxzoom: 19,
      },
      'basemap-opentopo': {
        type: 'raster',
        tiles: ['https://a.tile.opentopomap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenTopoMap contributors',
        maxzoom: 17,
      },
      'overlay-seamarks': {
        type: 'raster',
        tiles: ['https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenSeaMap contributors',
        maxzoom: 18,
      },
      'basemap-light': {
        type: 'raster',
        tiles: ['https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© CARTO · © OpenStreetMap contributors',
        maxzoom: 20,
      },
    },
    layers: [
      { id: 'basemap-google-sat', type: 'raster', source: 'basemap-google-sat', layout: { visibility: 'visible' } },
      { id: 'basemap-standard', type: 'raster', source: 'basemap-standard', layout: { visibility: 'none' } },
      { id: 'basemap-opentopo', type: 'raster', source: 'basemap-opentopo', layout: { visibility: 'none' } },
      { id: 'basemap-light', type: 'raster', source: 'basemap-light', layout: { visibility: 'none' } },
      { id: 'basemap-google-streets', type: 'raster', source: 'basemap-google-streets', layout: { visibility: 'none' } },
      { id: 'basemap-google-hybrid', type: 'raster', source: 'basemap-google-hybrid', layout: { visibility: 'none' } },
      { id: 'overlay-seamarks', type: 'raster', source: 'overlay-seamarks', layout: { visibility: 'none' } },
    ],
  }
}

function transformCoord(lon, lat, projection) {
  if (typeof lon !== 'number' || typeof lat !== 'number') return [lon, lat]

  if (projection === 'arctic') {
    const r = (90 - lat) * 0.18
    const rad = (lon * Math.PI) / 180
    const x = r * Math.sin(rad)
    const y = -r * Math.cos(rad)
    return [x, 80 + y]
  }

  if (projection === 'antarctic') {
    const r = (90 + lat) * 0.18
    const rad = (lon * Math.PI) / 180
    const x = r * Math.sin(rad)
    const y = r * Math.cos(rad)
    return [x, -82 + y]
  }

  return [lon, lat]
}

function transformGeoJSON(geojson, projection) {
  if (!geojson || !geojson.features || projection === 'mercator' || projection === 'wgs84') {
    return geojson
  }
  const transformedFeatures = geojson.features.map(feat => {
    if (!feat.geometry) return feat
    const geom = feat.geometry
    if (geom.type === 'Point') {
      const [tLon, tLat] = transformCoord(geom.coordinates[0], geom.coordinates[1], projection)
      return { ...feat, geometry: { ...geom, coordinates: [tLon, tLat] } }
    }
    if (geom.type === 'Polygon') {
      const rings = geom.coordinates.map(ring =>
        ring.map(([lon, lat]) => transformCoord(lon, lat, projection))
      )
      return { ...feat, geometry: { ...geom, coordinates: rings } }
    }
    if (geom.type === 'LineString') {
      const coords = geom.coordinates.map(([lon, lat]) => transformCoord(lon, lat, projection))
      return { ...feat, geometry: { ...geom, coordinates: coords } }
    }
    return feat
  })
  return { ...geojson, features: transformedFeatures }
}

function getCardinalDirection(deg) {
  if (typeof deg !== 'number' || isNaN(deg)) return ''
  const val = Math.floor((deg / 22.5) + 0.5)
  const arr = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
  return arr[val % 16]
}

function createArrowImageData(fillColor, strokeColor = '#ffffff') {
  const canvas = document.createElement('canvas')
  canvas.width = 32
  canvas.height = 32
  const ctx = canvas.getContext('2d')

  ctx.fillStyle = fillColor
  ctx.strokeStyle = strokeColor
  ctx.lineWidth = 1.0

  ctx.beginPath()
  ctx.moveTo(16, 3)   // Arrow tip pointing North (0 deg)
  ctx.lineTo(23, 25)  // Right wing (sleek & narrow)
  ctx.lineTo(16, 20)  // Inner notch
  ctx.lineTo(9, 25)   // Left wing
  ctx.closePath()

  ctx.fill()
  ctx.stroke()

  return ctx.getImageData(0, 0, 32, 32)
}

export default function MapView() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  // Mirrored into state so the telemetry marker mounts once the map is ready;
  // a ref alone would not re-render the child when the map finishes loading.
  const [mapReady, setMapReady] = useState(null)
  const hoverPopupRef = useRef(null)
  const {
    waypoints, addWaypoint, route, routeLoading,
    basemap, mapProjection,
    showSeamarks, showWindFlow,
    showEca, showChokepoints, showChevrons, showSeaIce, showIceRisk, polarClass, showOceanCurrents, showWind, showWeatherHeatmap, showStorms, weatherSubFilters,
    icebergs, routeWeather, shipWeatherAtSliderTime, closestApproaches, sliderTimeISO, departureTimeUTC,
  } = usePlanner()

  const addRef = useRef(addWaypoint)
  addRef.current = addWaypoint
  const wpLenRef = useRef(waypoints.length)
  wpLenRef.current = waypoints.length

  const rawSeaIceRef = useRef(null)

  const rawIceRiskRef = useRef(null)
  const rawCurrentsRef = useRef(null)
  const rawWindRef = useRef(null)
  const rawWeatherHeatmapRef = useRef(null)
  const rawEcasRef = useRef(null)
  const rawChokepointsRef = useRef(null)

  const syncVectorLayers = () => {
    const map = mapRef.current
    if (!map) return

    if (rawIceRiskRef.current && map.getSource('ice-risk')) {
      map.getSource('ice-risk').setData(transformGeoJSON(rawIceRiskRef.current, mapProjection))
    }
    if (rawSeaIceRef.current && map.getSource('sea-ice')) {
      map.getSource('sea-ice').setData(transformGeoJSON(rawSeaIceRef.current, mapProjection))
    }
    if (rawCurrentsRef.current && map.getSource('ocean-currents')) {
      map.getSource('ocean-currents').setData(transformGeoJSON(rawCurrentsRef.current, mapProjection))
    }
    if (rawWindRef.current && map.getSource('wind')) {
      map.getSource('wind').setData(transformGeoJSON(rawWindRef.current, mapProjection))
    }
    if (rawWeatherHeatmapRef.current && map.getSource('weather-heatmap')) {
      map.getSource('weather-heatmap').setData(transformGeoJSON(rawWeatherHeatmapRef.current, mapProjection))
    }
    if (rawEcasRef.current && map.getSource('ecas')) {
      map.getSource('ecas').setData(transformGeoJSON(rawEcasRef.current, mapProjection))
    }
    if (rawChokepointsRef.current && map.getSource('chokepoints')) {
      map.getSource('chokepoints').setData(transformGeoJSON(rawChokepointsRef.current, mapProjection))
    }
    if (routeWeather && routeWeather.samples && map.getSource('storms-hazard')) {
      // Every sample is rendered, not just the ones a sub-filter matches.
      // A calm stretch is a forecast result, not missing data: dropping those
      // points left holes in the track that read as gaps in coverage. The
      // filters now set emphasis instead, so calm legs stay visible as small
      // muted dots.
      const matches = (s) => {
        if (!weatherSubFilters) return true
        if (weatherSubFilters.hazards && s.hasHazard) return true
        if (weatherSubFilters.storms && s.hasStorms) return true
        if (weatherSubFilters.rain && s.hasRain) return true
        if (weatherSubFilters.wind && s.hasWind) return true
        if (weatherSubFilters.waves && s.hasWaves) return true
        return false
      }
      const features = routeWeather.samples.map(s => {
        const [tLon, tLat] = transformCoord(s.lon, s.lat, mapProjection)
        return {
          type: 'Feature',
          properties: { ...s, emphasised: matches(s) },
          geometry: { type: 'Point', coordinates: [tLon, tLat] },
        }
      })
      map.getSource('storms-hazard').setData({ type: 'FeatureCollection', features })
    }
  }

  const syncRef = useRef(syncVectorLayers)
  syncRef.current = syncVectorLayers

  const syncLayerVisibility = React.useCallback(() => {
    const map = mapRef.current
    if (!map) return

    if (map.getLayer('sea-ice-fill')) {
      map.setLayoutProperty('sea-ice-fill', 'visibility', showSeaIce ? 'visible' : 'none')
    }
    if (map.getLayer('ice-risk-fill')) {
      map.setLayoutProperty('ice-risk-fill', 'visibility', showIceRisk ? 'visible' : 'none')
    }
    if (map.getLayer('ocean-currents-vectors')) {
      map.setLayoutProperty('ocean-currents-vectors', 'visibility', showOceanCurrents ? 'visible' : 'none')
    }
    if (map.getLayer('wind-vectors')) {
      map.setLayoutProperty('wind-vectors', 'visibility', showWind ? 'visible' : 'none')
    }
    if (map.getLayer('weather-heatmap-layer')) {
      map.setLayoutProperty('weather-heatmap-layer', 'visibility', showWeatherHeatmap ? 'visible' : 'none')
    }
    if (map.getLayer('storms-hazard-layer-forecast')) {
      map.setLayoutProperty('storms-hazard-layer-forecast', 'visibility', showStorms ? 'visible' : 'none')
    }
    if (map.getLayer('storms-hazard-layer-modeled')) {
      map.setLayoutProperty('storms-hazard-layer-modeled', 'visibility', showStorms ? 'visible' : 'none')
    }
    if (map.getLayer('eca-outline') && map.getLayer('eca-fill')) {
      map.setLayoutProperty('eca-fill', 'visibility', showEca ? 'visible' : 'none')
      map.setLayoutProperty('eca-outline', 'visibility', showEca ? 'visible' : 'none')
    }
    if (map.getLayer('chokepoints')) {
      map.setLayoutProperty('chokepoints', 'visibility', showChokepoints ? 'visible' : 'none')
    }
    if (map.getLayer('route-chevrons')) {
      map.setLayoutProperty('route-chevrons', 'visibility', (showChevrons && !!route) ? 'visible' : 'none')
    }
  }, [showSeaIce, showIceRisk, polarClass, showOceanCurrents, showWind, showWeatherHeatmap, showStorms, showEca, showChokepoints, showChevrons, mapProjection, route])


  const syncVisibilityRef = useRef(syncLayerVisibility)
  syncVisibilityRef.current = syncLayerVisibility

  // Always holds a fn drawing the *current* route into the map source
  const drawRouteRef = useRef(() => {})
  const [mapInstance, setMapInstance] = useState(null)

  // ---- init once ----
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapStyle(),
      center: [66, 12],
      zoom: 2.4,
      attributionControl: { compact: false },
    })
    mapRef.current = map

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'nautical' }), 'bottom-left')
    map.addControl(new maplibregl.AttributionControl({ compact: false }))
    setMapReady(map)

    map.on('load', () => {
      // A ref alone cannot tell the overlay the map exists, so publish it as
      // state: the wind canvas needs the instance to project coordinates.
      setMapInstance(map)
      map.addSource('route', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'route-outline',
        type: 'line',
        source: 'route',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': 'rgba(20,40,80,0.25)', 'line-width': 7 },
      })
      map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': PIN_BLUE, 'line-width': 2.6 },
      })
      map.addLayer({
        id: 'route-chevrons',
        type: 'symbol',
        source: 'route',
        layout: {
          'symbol-placement': 'line',
          'symbol-spacing': 72,
          'text-field': ['literal', '»'],
          'text-size': 13,
          'text-font': ['Noto Sans Bold'],
          'text-rotation-alignment': 'auto',
          'icon-rotation-alignment': 'map',
          'text-allow-overlap': false,
        },
        paint: { 'text-color': PIN_BLUE },
      })

      map.addSource('waypoints', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'wp-halo',
        type: 'circle',
        source: 'waypoints',
        paint: {
          'circle-radius': 11,
          'circle-color': PIN_WHITE,
          'circle-stroke-width': 2,
          'circle-stroke-color': PIN_BLUE,
        },
      })
      map.addLayer({
        id: 'wp-dot',
        type: 'circle',
        source: 'waypoints',
        paint: { 'circle-radius': 8.5, 'circle-color': PIN_BLUE },
      })
      map.addLayer({
        id: 'wp-number',
        type: 'symbol',
        source: 'waypoints',
        layout: {
          'text-field': ['get', 'rank'],
          'text-size': 11,
          'text-font': ['Noto Sans Bold'],
          'text-allow-overlap': true,
          'text-padding': 0,
        },
        paint: { 'text-color': PIN_WHITE },
      })

      map.addSource('ecas', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'eca-fill',
        type: 'fill',
        source: 'ecas',
        paint: { 'fill-color': '#ff7a00', 'fill-opacity': 0.08 },
      })
      map.addLayer({
        id: 'eca-outline',
        type: 'line',
        source: 'ecas',
        paint: { 'line-color': '#ff7a00', 'line-width': 1.6, 'line-opacity': 0.95 },
      })

      map.addSource('chokepoints', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'chokepoints',
        type: 'line',
        source: 'chokepoints',
        paint: { 'line-color': '#d33', 'line-width': 1.5, 'line-dasharray': [3, 2] },
      })

      map.addSource('sea-ice', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'sea-ice-fill',
        type: 'fill',
        source: 'sea-ice',
        paint: {
          'fill-color': [
            'match',
            ['get', 'category'],
            'heavy', '#0077b6',
            'medium', '#00b4d8',
            '#90e0ef'
          ],
          'fill-opacity': 0.35,
        },
      })

      // IMO POLARIS navigational ice risk, derived from the SIC forecast.
      // Severity buckets mirror the backend: low / elevated / severe.
      map.addSource('ice-risk', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'ice-risk-fill',
        type: 'fill',
        source: 'ice-risk',
        layout: { visibility: 'none' },
        paint: {
          'fill-color': [
            'match',
            ['get', 'severity'],
            'severe', '#b00020',
            'elevated', '#f18701',
            '#f7d154'
          ],
          'fill-opacity': 0.45,
        },
      })

      if (!map.hasImage('ocean-current-arrow')) {
        map.addImage('ocean-current-arrow', createArrowImageData('#00b4d8', '#004b6e'))
      }
      if (!map.hasImage('wind-arrow')) {
        map.addImage('wind-arrow', createArrowImageData('#a855f7', '#4c1d95'))
      }

      map.addSource('ocean-currents', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'ocean-currents-vectors',
        type: 'symbol',
        source: 'ocean-currents',
        layout: {
          'icon-image': 'ocean-current-arrow',
          'icon-size': ['interpolate', ['linear'], ['zoom'], 1, 0.35, 4, 0.55, 8, 0.8],
          'icon-rotate': ['get', 'dirDeg'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'icon-padding': 2,
        },
      })

      map.addSource('wind', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'wind-vectors',
        type: 'symbol',
        source: 'wind',
        layout: {
          'icon-image': 'wind-arrow',
          'icon-size': ['interpolate', ['linear'], ['zoom'], 1, 0.35, 4, 0.55, 8, 0.8],
          'icon-rotate': ['get', 'dirDeg'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'icon-padding': 2,
        },
      })

      map.addSource('weather-heatmap', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'weather-heatmap-layer',
        type: 'heatmap',
        source: 'weather-heatmap',
        maxzoom: 12,
        layout: { visibility: 'none' },
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'riskIndex'], 0, 0, 100, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(0, 0, 0, 0)',
            0.2, 'rgba(59, 130, 246, 0.45)',
            0.45, 'rgba(234, 179, 8, 0.65)',
            0.75, 'rgba(249, 115, 22, 0.85)',
            1.0, 'rgba(239, 68, 68, 0.95)'
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 14, 9, 40],
          'heatmap-opacity': 0.75
        }
      })

      map.addSource('storms-hazard', { type: 'geojson', data: emptyFeatureCollection() })
      map.addLayer({
        id: 'storms-hazard-layer-forecast',
        type: 'circle',
        source: 'storms-hazard',
        filter: ['==', ['get', 'dataSource'], 'forecast'],
        paint: {
          'circle-radius': ['case', ['get', 'emphasised'], 7, 3.2],
          'circle-color': [
            'match',
            ['get', 'hazardLevel'],
            'severe', '#ef4444',
            'moderate', '#eab308',
            '#22c55e'
          ],
          'circle-stroke-width': ['case', ['get', 'emphasised'], 2, 0.8],
          'circle-stroke-color': '#ffffff',
          'circle-opacity': ['case', ['get', 'emphasised'], 0.9, 0.5],
        },
      })
      map.addLayer({
        id: 'storms-hazard-layer-modeled',
        type: 'circle',
        source: 'storms-hazard',
        filter: ['==', ['get', 'dataSource'], 'modeled_estimate'],
        paint: {
          'circle-radius': ['case', ['get', 'emphasised'], 6, 3],
          'circle-color': [
            'match',
            ['get', 'hazardLevel'],
            'severe', '#f87171',
            'moderate', '#fde047',
            '#4ade80'
          ],
          'circle-stroke-width': ['case', ['get', 'emphasised'], 1.5, 0.6],
          'circle-stroke-color': '#d97706',
          'circle-opacity': ['case', ['get', 'emphasised'], 0.5, 0.3],
        },
      })

      const vectorPopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        className: 'vector-hover-popup',
        offset: 10,
      })
      hoverPopupRef.current = vectorPopup

      const registerStormHover = (layerId) => {
        map.on('mousemove', layerId, (e) => {
          if (!e.features || !e.features.length) return
          map.getCanvas().style.cursor = 'pointer'
          const feat = e.features[0]
          const props = feat.properties || {}
          const coords = feat.geometry.coordinates

          const isForecast = props.dataSource === 'forecast'
          const badgeClass = isForecast ? 'forecast' : 'modeled_estimate'
          const badgeLabel = isForecast ? 'FORECAST' : 'MODELED ESTIMATE'
          const hazardClass = props.hazardLevel || 'none'

          const html = `
            <div class="vector-popup-card wind">
              <div class="vector-popup-header">
                <div class="vector-title-box">
                  <span class="vector-icon">⛈️</span>
                  <span class="vector-title">Weather Sample</span>
                </div>
                <span class="vector-badge ${badgeClass}">${badgeLabel}</span>
              </div>
              <div class="vector-popup-grid">
                <div class="vector-stat-box">
                  <span class="stat-label">Wind Speed</span>
                  <span class="stat-value"><strong>${props.windSpeedKn ?? '--'}</strong> kn</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Wind Gusts</span>
                  <span class="stat-value"><strong>${props.windGustsKn ?? '--'}</strong> kn</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Wave Height</span>
                  <span class="stat-value"><strong>${props.waveHeightM ?? '--'}</strong> m</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Hazard Level</span>
                  <span class="stat-value"><strong class="hazard-tag ${hazardClass}">${(props.hazardLevel || 'none').toUpperCase()}</strong></span>
                </div>
                <div class="vector-stat-box full-row">
                  <span class="stat-label">Timestamp (UTC)</span>
                  <span class="stat-value">${(props.timestamp || '').replace('T', ' ').replace('Z', '')}</span>
                </div>
              </div>
            </div>
          `
          vectorPopup.setLngLat(coords).setHTML(html).addTo(map)
        })

        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = ''
          vectorPopup.remove()
        })
      }

      registerStormHover('storms-hazard-layer-forecast')
      registerStormHover('storms-hazard-layer-modeled')

      const registerVectorHover = (layerId, title, icon, cardClass) => {
        map.on('mousemove', layerId, (e) => {
          if (!e.features || !e.features.length) return
          map.getCanvas().style.cursor = 'pointer'
          const feat = e.features[0]
          const props = feat.properties || {}
          const coords = feat.geometry.coordinates

          const speed = props.speedKnots ?? '--'
          const dir = props.dirDeg ?? '--'
          const cardinal = typeof props.dirDeg === 'number' ? getCardinalDirection(props.dirDeg) : ''
          const u = typeof props.u === 'number' ? props.u : 0
          const v = typeof props.v === 'number' ? props.v : 0
          const intensity = props.intensity || 'moderate'

          const html = `
            <div class="vector-popup-card ${cardClass}">
              <div class="vector-popup-header">
                <div class="vector-title-box">
                  <span class="vector-icon">${icon}</span>
                  <span class="vector-title">${title}</span>
                </div>
                <span class="vector-badge ${intensity}">${intensity}</span>
              </div>
              <div class="vector-popup-grid">
                <div class="vector-stat-box">
                  <span class="stat-label">Speed</span>
                  <span class="stat-value"><strong>${speed}</strong> kn</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Heading</span>
                  <span class="stat-value"><strong>${dir}°</strong> ${cardinal ? `(${cardinal})` : ''}</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Latitude</span>
                  <span class="stat-value"><strong>${coords[1].toFixed(2)}°</strong></span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Longitude</span>
                  <span class="stat-value"><strong>${coords[0].toFixed(2)}°</strong></span>
                </div>
                <div class="vector-stat-box full-row">
                  <span class="stat-label">Vector (u, v)</span>
                  <span class="stat-value">${u >= 0 ? '+' : ''}${u}, ${v >= 0 ? '+' : ''}${v} m/s</span>
                </div>
              </div>
            </div>
          `
          vectorPopup.setLngLat(coords).setHTML(html).addTo(map)
        })

        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = ''
          vectorPopup.remove()
        })
      }

      registerVectorHover('ocean-currents-vectors', 'Ocean Current', '🌊', 'ocean')
      registerVectorHover('wind-vectors', 'Surface Wind (10m)', '💨', 'wind')

      drawRouteRef.current()
      syncRef.current()
      syncVisibilityRef.current()
    })

    return () => {
      Object.values(icebergMarkersRef.current).forEach(m => m.remove())
      icebergMarkersRef.current = {}
      if (shipMarkerRef.current) {
        shipMarkerRef.current.remove()
        shipMarkerRef.current = null
      }
      map.remove()
      mapRef.current = null
    }
  }, [])

  // ---- route geometry (transformed for current projection) ----
  useEffect(() => {
    drawRouteRef.current = () => {
      const map = mapRef.current
      if (!map || !map.getSource('route')) return
      let geo
      if (!route || !route.legs || !route.legs.length) {
        geo = emptyFeatureCollection()
      } else {
        let pts = []
        for (const leg of route.legs) {
          const legPts = leg.path.map(([lat, lon]) => [lat, lon])
          if (pts.length && samePoint(pts[pts.length - 1], legPts[0])) legPts.shift()
          pts.push(...legPts)
        }
        if (pts.length < 2) {
          geo = emptyFeatureCollection()
        } else {
          const unwrapped = unwrapLongitudes(pts)
          const parts = splitIntoRenderedParts(unwrapped)

          const transformedParts = parts.map(part =>
            part.map(([lon, lat]) => transformCoord(lon, lat, mapProjection))
          )

          geo = transformedParts.length === 1
            ? { type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: transformedParts[0] } }
            : { type: 'Feature', properties: {}, geometry: { type: 'MultiLineString', coordinates: transformedParts } }
        }
      }
      map.getSource('route').setData(geo)
    }
    const map = mapRef.current
    if (map && map.getSource('route')) drawRouteRef.current()
    return () => { drawRouteRef.current = () => {} }
  }, [route, mapProjection])

  // ---- waypoint pins (transformed for current projection) ----
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.getSource('waypoints')) return
    const features = waypoints
      .filter(w => typeof w.lat === 'number')
      .map((w, i) => {
        const [tLon, tLat] = transformCoord(w.lon, w.lat, mapProjection)
        return {
          type: 'Feature',
          properties: { rank: String(i + 1), label: w.label },
          geometry: { type: 'Point', coordinates: [tLon, tLat] },
        }
      })
    map.getSource('waypoints').setData({ type: 'FeatureCollection', features })
  }, [waypoints, mapProjection])

  // ---- iceberg DOM markers synced to slider time ----
  const icebergMarkersRef = useRef({})

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (!icebergs || !icebergs.length) {
      Object.values(icebergMarkersRef.current).forEach(m => m.remove())
      icebergMarkersRef.current = {}
      return
    }

    const targetTime = sliderTimeISO || departureTimeUTC
    const currentIds = new Set(icebergs.map(b => b.id))

    Object.keys(icebergMarkersRef.current).forEach(id => {
      if (!currentIds.has(id)) {
        icebergMarkersRef.current[id].remove()
        delete icebergMarkersRef.current[id]
      }
    })

    icebergs.forEach(berg => {
      const pos = getIcebergPositionAtTime(berg, targetTime)
      const [tLon, tLat] = transformCoord(pos.lon, pos.lat, mapProjection)
      const heading = typeof pos.headingDeg === 'number' ? pos.headingDeg : (berg.headingDeg || 0)
      const speed = typeof pos.speedKnots === 'number' ? pos.speedKnots : (berg.speedKnots || 0.5)

      const attachHoverListeners = (el) => {
        el.onmouseenter = () => {
          const popup = hoverPopupRef.current
          const mapInstance = mapRef.current
          if (!popup || !mapInstance) return
          const cardinal = getCardinalDirection(heading)
          const currentStr = berg.currentSpeedKnots !== undefined ? `${berg.currentSpeedKnots} kn @ ${berg.currentDirDeg}°` : ''
          const windStr = berg.windSpeedKnots !== undefined ? `${berg.windSpeedKnots} kn @ ${berg.windDirDeg}°` : ''
          const html = `
            <div class="vector-popup-card iceberg">
              <div class="vector-popup-header">
                <div class="vector-title-box">
                  <span class="vector-icon">🔺</span>
                  <span class="vector-title">${berg.name || berg.id}</span>
                </div>
                <span class="vector-badge iceberg">${berg.sizeCategory || 'iceberg'}</span>
              </div>
              <div class="vector-popup-grid">
                <div class="vector-stat-box">
                  <span class="stat-label">Drift Speed</span>
                  <span class="stat-value"><strong>${speed}</strong> kn</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Drift Heading</span>
                  <span class="stat-value"><strong>${Math.round(heading)}°</strong> ${cardinal ? `(${cardinal})` : ''}</span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Latitude</span>
                  <span class="stat-value"><strong>${pos.lat.toFixed(4)}°</strong></span>
                </div>
                <div class="vector-stat-box">
                  <span class="stat-label">Longitude</span>
                  <span class="stat-value"><strong>${pos.lon.toFixed(4)}°</strong></span>
                </div>
                ${currentStr ? `
                  <div class="vector-stat-box full-row">
                    <span class="stat-label">Current Forcing</span>
                    <span class="stat-value"><strong>${currentStr}</strong></span>
                  </div>
                ` : ''}
                ${windStr ? `
                  <div class="vector-stat-box full-row">
                    <span class="stat-label">Wind Forcing</span>
                    <span class="stat-value"><strong>${windStr}</strong></span>
                  </div>
                ` : ''}
              </div>
            </div>
          `
          popup.setLngLat([tLon, tLat]).setHTML(html).addTo(mapInstance)
        }

        el.onmouseleave = () => {
          if (hoverPopupRef.current) hoverPopupRef.current.remove()
        }
      }

      if (icebergMarkersRef.current[berg.id]) {
        icebergMarkersRef.current[berg.id].setLngLat([tLon, tLat])
        const markerEl = icebergMarkersRef.current[berg.id].getElement()
        const arrowEl = markerEl.querySelector('.iceberg-drift-arrow')
        if (arrowEl) arrowEl.style.transform = `rotate(${heading}deg)`
        attachHoverListeners(markerEl)
      } else {
        const el = document.createElement('div')
        el.className = 'iceberg-marker-el'
        el.style.pointerEvents = 'auto'
        el.style.cursor = 'pointer'
        el.innerHTML = `
          <div class="iceberg-marker-wrap">
            <div class="iceberg-drift-arrow" style="transform: rotate(${heading}deg);" title="Drift Heading: ${heading}°, Speed: ${speed} kn">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L18 10H14V22H10V10H6L12 2Z" fill="#ff7a00" stroke="#8c3e00" stroke-width="1.5" stroke-linejoin="round"/>
              </svg>
            </div>
            <div class="iceberg-svg-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <polygon points="12,2 22,21 2,21" fill="#00d2ff" stroke="#005b96" stroke-width="2" stroke-linejoin="round"/>
                <polygon points="12,2 15,10 12,14 8,11" fill="#ffffff" opacity="0.65"/>
              </svg>
            </div>
          </div>
          <div class="iceberg-label">${berg.name || berg.id} (${speed} kn ➔ ${Math.round(heading)}°)</div>
        `
        attachHoverListeners(el)
        const marker = new maplibregl.Marker({ element: el, anchor: 'bottom' })
          .setLngLat([tLon, tLat])
          .addTo(map)
        icebergMarkersRef.current[berg.id] = marker
      }
    })
  }, [icebergs, sliderTimeISO, departureTimeUTC, mapProjection])

  // ---- moving ship DOM marker synced to slider time ----
  const shipMarkerRef = useRef(null)

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (!route || !sliderTimeISO) {
      if (shipMarkerRef.current) {
        shipMarkerRef.current.remove()
        shipMarkerRef.current = null
      }
      return
    }

    const shipPos = getShipPositionAtTime(route, departureTimeUTC, sliderTimeISO)
    if (!shipPos) {
      if (shipMarkerRef.current) {
        shipMarkerRef.current.remove()
        shipMarkerRef.current = null
      }
      return
    }

    const [tLon, tLat] = transformCoord(shipPos.lon, shipPos.lat, mapProjection)

    if (shipMarkerRef.current) {
      shipMarkerRef.current.setLngLat([tLon, tLat])
    } else {
      const el = document.createElement('div')
      el.className = 'ship-marker-el'
      el.innerHTML = `
        <div class="ship-pulse-halo"></div>
        <div class="ship-svg-icon">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 15L2 9H22L20 15H4Z" fill="#ff7a00" stroke="#b35400" stroke-width="1.5" stroke-linejoin="round"/>
            <path d="M7 9V5H12V9" fill="#ffffff" stroke="#b35400" stroke-width="1.5" stroke-linejoin="round"/>
            <path d="M14 9V6H17V9" fill="#ffffff" stroke="#b35400" stroke-width="1.5" stroke-linejoin="round"/>
            <path d="M2 18C5 18 7 16 10 18C13 20 15 18 18 18C20 18 22 17 23 16.5" stroke="#ff7a00" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
      `
      shipMarkerRef.current = new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([tLon, tLat])
        .addTo(map)
    }
  }, [route, sliderTimeISO, departureTimeUTC, mapProjection])

  // ---- ECA + chokepoints + sea-ice + currents + wind data ----
  useEffect(() => {
    let alive = true
    fetchEcas().then(fc => {
      if (!alive) return
      rawEcasRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchChokepoints().then(fc => {
      if (!alive) return
      rawChokepointsRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchSeaIce().then(fc => {
      if (!alive) return
      rawSeaIceRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchIceRisk(polarClass).then(fc => {
      if (!alive) return
      rawIceRiskRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchOceanCurrents().then(fc => {
      if (!alive) return
      rawCurrentsRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchWind().then(fc => {
      if (!alive) return
      rawWindRef.current = fc
      syncRef.current()
    }).catch(() => {})

    fetchWeatherHeatmap().then(fc => {
      if (!alive) return
      rawWeatherHeatmapRef.current = fc
      syncRef.current()
    }).catch(() => {})

    return () => { alive = false }
  }, [])

  useEffect(() => {
    syncVectorLayers()
  }, [mapProjection, routeWeather, weatherSubFilters, syncVectorLayers])

  useEffect(() => {
    syncLayerVisibility()
  }, [syncLayerVisibility])

  // ---- basemap switch ----
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.getLayer('basemap-standard')) return

    const target = `basemap-${basemap}`
    for (const id of BASEMAP_LAYER_IDS) {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, 'visibility', id === target ? 'visible' : 'none')
      }
    }
    // Seamarks are buoys, lights and depth contours drawn on transparent tiles,
    // so they stack on whichever basemap is active rather than replacing it.
    if (map.getLayer('overlay-seamarks')) {
      map.setLayoutProperty('overlay-seamarks', 'visibility', showSeamarks ? 'visible' : 'none')
    }
  }, [basemap, showSeamarks])

  // ---- fit to route once computed (not on every keystroke) ----
  const fittedRef = useRef('')

  useEffect(() => {
    const map = mapRef.current
    if (!map || !route) return
    const sig = route.legs.map(l => `${l.from.label}:${l.to.label}`).join('|')
    if (fittedRef.current === sig) return
    fittedRef.current = sig
    const pts = route.legs.flatMap(l => l.path)
    if (!pts.length) return
    const lats = pts.map(p => p[0])
    const lons = pts.map(p => p[1])
    const pad = 3
    map.fitBounds(
      [[Math.min(...lons) - pad, Math.min(...lats) - pad], [Math.max(...lons) + pad, Math.max(...lats) + pad]],
      { duration: 800, padding: 24 },
    )
  }, [route])

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map" />
      <WindFlowLayer map={mapInstance} visible={showWindFlow} />

      <LayerControlPanel />

      {route && (showStorms || (showSeaIce && icebergs.length > 0)) && (
        <div className="live-hud-card">
          <div className="hud-title">
            Live Route Conditions & ETA Timeline ({new Date(sliderTimeISO || departureTimeUTC).toUTCString().slice(0, 22)})
          </div>
          <div className="hud-grid">
            {showStorms && shipWeatherAtSliderTime && (
              <div className="hud-section storm-section">
                <div className="hud-header">
                  <span>⛈️ Sea & Weather</span>
                  <span className={`hud-badge ${shipWeatherAtSliderTime.dataSource}`}>
                    {shipWeatherAtSliderTime.dataSource === 'forecast' ? 'Forecast' : 'Modeled Estimate'}
                  </span>
                </div>
                <div className="hud-row">
                  <span className="lbl">Wind:</span>
                  <span className="val"><strong>{shipWeatherAtSliderTime.windSpeedKn}</strong> kn (gusts {shipWeatherAtSliderTime.windGustsKn} kn)</span>
                </div>
                <div className="hud-row">
                  <span className="lbl">Wave Height:</span>
                  <span className="val"><strong>{shipWeatherAtSliderTime.waveHeightM}</strong> m</span>
                </div>
                <div className="hud-row">
                  <span className="lbl">Hazard Tier:</span>
                  <span className={`val hazard-tag ${shipWeatherAtSliderTime.hazardLevel}`}>
                    {shipWeatherAtSliderTime.hazardLevel.toUpperCase()}
                  </span>
                </div>
              </div>
            )}
            {showSeaIce && closestApproaches.length > 0 && (
              <div className="hud-section ice-section">
                <div className="hud-header">
                  <span>🧊 Nearest Iceberg</span>
                  <span className="hud-badge info">Tracked</span>
                </div>
                <div className="hud-row">
                  <span className="lbl">Iceberg:</span>
                  <span className="val"><strong>{closestApproaches[0].icebergName}</strong></span>
                </div>
                <div className="hud-row">
                  <span className="lbl">Min Distance:</span>
                  <span className="val"><strong>{closestApproaches[0].minDistanceNm}</strong> NM</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}


      <VesselTelemetry map={mapReady} />

      {routeLoading && <div className="routing-indicator"><span className="spinner" /> Routing…</div>}
    </div>
  )
}

function emptyFeatureCollection() {
  return { type: 'FeatureCollection', features: [] }
}

function samePoint(a, b) {
  return Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9
}