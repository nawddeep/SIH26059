# IMPALA Local Dashboard — development note

This is an isolated React + TypeScript + Vite application. It intentionally does not import from, route through, or alter the existing IMPALA presentation app.

`src/data/simulatedProvider.ts` implements the local data-service boundary. UI modules consume its normalized `OperationalSnapshot`, so a localhost API/WebSocket adapter can replace the provider without rewriting the plot or panels. `src/types/operational.ts` is the transport-neutral contract for future NMEA/AIS/sonar adapters, native computation, and local ML/LLM services. `src/data/localBus.ts` provides a bounded latest-value in-process data bus, plus explicit simulated/serial/TCP/UDP/CAN/file-replay transport configuration.

`src/visualization/NavigationRenderer.tsx` is the rendering boundary: it draws the local polar plot, sonar sector, iceberg/radar/sonar targets, and a requestAnimationFrame-owned radar sweep. The sweep mutates its SVG group directly and does not cause a React render loop. Place future locally-vetted GeoJSON/PMTiles/MBTiles-derived assets in `public/map-data/`; no online tile fallback is permitted.

No remote fonts, images, map tiles, API calls, or runtime dependencies are used. The navigation plot is local SVG and deterministic mock data. This is a decision-support simulation, not a certified navigation system.
