import React, { createContext, useContext, useCallback, useEffect, useMemo, useState } from 'react'
import { computeRoute, fetchIcebergs, fetchRouteWeather } from './api'
import { computeClosestApproaches, getShipPositionAtTime } from './utils'

const PlannerContext = createContext(null)

export function usePlanner() {
  return useContext(PlannerContext)
}

export function defaultDeparture() {
  const d = new Date()
  d.setSeconds(0, 0)
  return d.toISOString().slice(0, 16).replace('T', 'T')
}

export function makeWaypoint(overrides = {}) {
  return {
    id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2),
    label: '',
    countryCode: '',
    lat: null,
    lon: null,
    isPort: true,
    ...overrides,
  }
}

export function validWaypoints(wps) {
  return wps.every(w => typeof w.lat === 'number' && typeof w.lon === 'number')
}

export function PlannerProvider({ children }) {
  const [waypoints, setWaypoints] = useState(() => [makeWaypoint({}), makeWaypoint({})])
  const [speedKnots, setSpeedKnots] = useState(15)
  const [departureTimeUTC, setDepartureTimeUTC] = useState(defaultDeparture())
  const [optimizeFor, setOptimizeFor] = useState('distance')
  const [vesselType, setVesselType] = useState('cargo')
  const [draftMeters, setDraftMeters] = useState(10.5)
  const [iceClass, setIceClass] = useState('none')
  // Kept as a fixed value: the coordinate transforms still take it, but the
  // projection picker is gone because it never called setProjection - it only
  // moved the camera.
  const mapProjection = 'mercator'

  const [basemap, setBasemap] = useState('google-sat')
  const [showSeamarks, setShowSeamarks] = useState(false)
  const [showWindFlow, setShowWindFlow] = useState(false)
  const [showEca, setShowEca] = useState(true)
  const [showChokepoints, setShowChokepoints] = useState(false)
  const [showChevrons, setShowChevrons] = useState(true)
  const [showSeaIce, setShowSeaIce] = useState(true)
  const [showIceRisk, setShowIceRisk] = useState(false)
  const [polarClass, setPolarClass] = useState('PC4')
  const [showOceanCurrents, setShowOceanCurrents] = useState(true)
  const [showWind, setShowWind] = useState(false)
  const [showWeatherHeatmap, setShowWeatherHeatmap] = useState(false)
  const [showStorms, setShowStorms] = useState(true)
  const [weatherSubFilters, setWeatherSubFilters] = useState({
    wind: true,
    waves: true,
    storms: true,
    rain: true,
    hazards: true,
  })

  const toggleWeatherSubFilter = useCallback((key) => {
    setWeatherSubFilters(prev => ({ ...prev, [key]: !prev[key] }))
  }, [])

  const [route, setRoute] = useState(null)
  const [routeLoading, setRouteLoading] = useState(false)
  const [routeError, setRouteError] = useState(null)

  const [icebergs, setIcebergs] = useState([])
  const [routeWeather, setRouteWeather] = useState(null)
  const [sliderTimeISO, setSliderTimeISO] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)

  const setWaypoint = useCallback((index, patch) => {
    setWaypoints(prev => prev.map((w, i) => (i === index ? { ...w, ...patch } : w)))
  }, [])

  const addWaypoint = useCallback((patch = {}) => {
    setWaypoints(prev => [...prev, makeWaypoint(patch)])
  }, [])

  const removeWaypoint = useCallback((index) => {
    setWaypoints(prev => {
      if (prev.length <= 2) {
        const w = prev[index]
        if (w && typeof w.lat === 'number') {
          return prev.map((p, i) =>
            i === index ? { ...p, label: '', countryCode: '', lat: null, lon: null, isPort: true } : p)
        }
        return prev
      }
      return prev.filter((_, i) => i !== index)
    })
  }, [])

  const moveWaypoint = useCallback((from, to) => {
    setWaypoints(prev => {
      if (from === to || from < 0 || to < 0 || to >= prev.length) return prev
      const next = [...prev]
      const [item] = next.splice(from, 1)
      next.splice(to, 0, item)
      return next
    })
  }, [])

  const valid = validWaypoints(waypoints)

  useEffect(() => {
    if (!valid) {
      setRoute(null)
      setRouteError(null)
      setIcebergs([])
      setRouteWeather(null)
      return
    }

    let active = true
    let timer = null

    const run = async () => {
      setRouteLoading(true)
      setRouteError(null)
      try {
        const payload = {
          waypoints: waypoints.map(w => ({ id: w.id, label: w.label, countryCode: w.countryCode, lat: w.lat, lon: w.lon, isPort: w.isPort })),
          speedKnots,
          departureTimeUTC,
          optimizeFor,
          vesselType,
          draftMeters,
          iceClass,
        }
        const res = await computeRoute(
          payload.waypoints,
          payload.speedKnots,
          payload.departureTimeUTC,
          payload.optimizeFor,
          payload.vesselType,
          payload.draftMeters,
          payload.iceClass,
        )
        if (active) {
          setRoute(res)
          setRouteLoading(false)
          setSliderTimeISO(departureTimeUTC)

          // Fetch iceberg trajectories covering the route window
          try {
            const bergs = await fetchIcebergs(departureTimeUTC, res.etaUTC, 1.0)
            if (active) setIcebergs(bergs)
          } catch {
            if (active) setIcebergs([])
          }

          // Fetch Open-Meteo route weather and fallback estimates
          try {
            const rw = await fetchRouteWeather(res.routeId || '', departureTimeUTC, speedKnots)
            if (active) setRouteWeather(rw)
          } catch {
            if (active) setRouteWeather(null)
          }
        }
      } catch (err) {
        if (active) {
          setRoute(null)
          setRouteError(err instanceof Error ? err.message : 'Routing failed')
          setRouteLoading(false)
          setIcebergs([])
          setRouteWeather(null)
        }
      }
    }

    timer = setTimeout(run, 350)
    return () => { active = false; clearTimeout(timer) }
  }, [waypoints, speedKnots, departureTimeUTC, optimizeFor, vesselType, draftMeters, iceClass, valid])

  const closestApproaches = useMemo(() => {
    if (!route || !icebergs.length) return []
    return computeClosestApproaches(route, departureTimeUTC, icebergs)
  }, [route, departureTimeUTC, icebergs])

  // Interpolated ship weather conditions at current sliderTimeISO
  const shipWeatherAtSliderTime = useMemo(() => {
    if (!routeWeather || !routeWeather.samples || !routeWeather.samples.length || !sliderTimeISO || !departureTimeUTC) {
      return null
    }
    const targetMs = new Date(sliderTimeISO).getTime()
    const samples = routeWeather.samples

    const t0Ms = new Date(samples[0].timestamp).getTime()
    if (targetMs <= t0Ms) return samples[0]

    const tEndMs = new Date(samples[samples.length - 1].timestamp).getTime()
    if (targetMs >= tEndMs) return samples[samples.length - 1]

    for (let i = 0; i < samples.length - 1; i++) {
      const s1 = samples[i]
      const s2 = samples[i + 1]
      const t1 = new Date(s1.timestamp).getTime()
      const t2 = new Date(s2.timestamp).getTime()
      if (targetMs >= t1 && targetMs <= t2) {
        const frac = t2 > t1 ? (targetMs - t1) / (t2 - t1) : 0
        const wind = round1(s1.windSpeedKn + frac * (s2.windSpeedKn - s1.windSpeedKn))
        const gusts = round1(s1.windGustsKn + frac * (s2.windGustsKn - s1.windGustsKn))
        const wave = round1(s1.waveHeightM + frac * (s2.waveHeightM - s1.waveHeightM))
        const source = (s1.dataSource === 'modeled_estimate' || s2.dataSource === 'modeled_estimate')
          ? 'modeled_estimate'
          : 'forecast'
        const hazard = (wind >= 34 || gusts >= 40 || wave >= 5.0) ? 'severe' : (wind >= 22 || wave >= 3.0) ? 'moderate' : 'none'
        return {
          lat: round4(s1.lat + frac * (s2.lat - s1.lat)),
          lon: round4(s1.lon + frac * (s2.lon - s1.lon)),
          timestamp: sliderTimeISO,
          windSpeedKn: wind,
          windGustsKn: gusts,
          waveHeightM: wave,
          weatherCode: s1.weatherCode,
          hazardLevel: hazard,
          dataSource: source,
        }
      }
    }
    return samples[samples.length - 1]
  }, [routeWeather, sliderTimeISO, departureTimeUTC])

  const value = useMemo(() => ({
    waypoints, setWaypoint, addWaypoint, removeWaypoint, moveWaypoint,
    speedKnots, setSpeedKnots,
    departureTimeUTC, setDepartureTimeUTC,
    optimizeFor, setOptimizeFor,
    vesselType, setVesselType,
    draftMeters, setDraftMeters,
    iceClass, setIceClass,
    mapProjection,
    basemap, setBasemap,
    showSeamarks, setShowSeamarks,
    showWindFlow, setShowWindFlow,
    showEca, setShowEca,
    showChokepoints, setShowChokepoints,
    showChevrons, setShowChevrons,
    showSeaIce, setShowSeaIce,
    showIceRisk, setShowIceRisk,
    polarClass, setPolarClass,
    showOceanCurrents, setShowOceanCurrents,
    showWind, setShowWind,
    showWeatherHeatmap, setShowWeatherHeatmap,
    showStorms, setShowStorms,
    weatherSubFilters, toggleWeatherSubFilter,
    route, routeLoading, routeError,
    icebergs, routeWeather,
    sliderTimeISO, setSliderTimeISO,
    isPlaying, setIsPlaying,
    closestApproaches, shipWeatherAtSliderTime,
    valid,
  }), [waypoints, setWaypoint, addWaypoint, removeWaypoint, moveWaypoint, speedKnots,
       departureTimeUTC, optimizeFor, vesselType, draftMeters, iceClass, mapProjection,
       basemap, showSeamarks, showWindFlow,
       showEca, showChokepoints, showChevrons, showSeaIce, showIceRisk, polarClass, showOceanCurrents, showWind,
       showWeatherHeatmap, showStorms, weatherSubFilters, toggleWeatherSubFilter, route, routeLoading, routeError, icebergs, routeWeather,
       sliderTimeISO, isPlaying, closestApproaches, shipWeatherAtSliderTime, valid])

  return <PlannerContext.Provider value={value}>{children}</PlannerContext.Provider>
}

function round1(val) { return Math.round(val * 10) / 10 }
function round4(val) { return Math.round(val * 10000) / 10000 }