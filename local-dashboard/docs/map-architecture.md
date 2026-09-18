# Offline map architecture

## Discovered map data

All supplied geospatial archives are local ZIP files in `vector maps/`. Each Natural Earth shapefile package includes `.shp`, `.shx`, `.dbf`, `.cpg`, and `.prj`; the inspected `.prj` files declare WGS 84 geographic coordinates. They are vectors, not tiles, and are not directly browser-renderable without a shapefile reader or offline conversion.

| Dataset | Archive size | Coverage / feature | CRS | Decision |
|---|---:|---|---|---|
| `ne_50m_coastline.zip` | 445 KB | Global coastline, 1:50m | WGS 84 geographic | Registered; preprocess before use |
| `ne_50m_land.zip` | 446 KB | Global land polygons, 1:50m | WGS 84 geographic | Not selected |
| `ne_50m_ocean.zip` | 451 KB | Global ocean polygon, 1:50m | WGS 84 geographic | Not selected |
| `ne_50m_antarctic_ice_shelves_lines.zip` | 38 KB | Antarctic shelf lines, 1:50m | WGS 84 geographic | Registered; preprocess before use |
| `ne_50m_antarctic_ice_shelves_polys.zip` | 79 KB | Antarctic shelf polygons, 1:50m | WGS 84 geographic | Registered; preprocess before use |
| `ne_50m_glaciated_areas.zip` | 211 KB | Global glaciated areas, 1:50m | WGS 84 geographic | Not selected |
| `ne_50m_geographic_lines.zip` | 31 KB | Global geographic lines, 1:50m | WGS 84 geographic | Not selected |
| `ne_110m_glaciated_areas.zip` | 19 KB | Global glaciated areas, 1:110m | WGS 84 geographic | Not selected |
| `ne_110m_graticules_1.zip` | 1.4 MB | Global 1° graticule, 1:110m | WGS 84 geographic | Not selected; too detailed for startup |
| `ne_110m_graticules_30.zip` | 42 KB | Global 30° graticule, 1:110m | WGS 84 geographic | Not selected; SVG grid used instead |
| `ne_110m_graticules_all.zip` | 1.9 MB | Combined global graticules | WGS 84 geographic | Not selected; redundant |
| `ne_110m_wgs84_bounding_box.zip` | 9 KB | Global extent | WGS 84 geographic | Not selected |
| `AQ-EPS-01-0001.zip` / `WRLD-EPS-01-0006.zip` | 1.1 / 3.1 MB | Illustration PDF/EPS/AI/JPG/PNG | Unknown | Not geospatial; excluded |

The MP3 in `vector maps/` is not geospatial and is excluded. No PMTiles, MBTiles, GeoPackage, GeoJSON, raster, or web vector-tile data was found.

## Renderer and layers

The project retains its existing lightweight local SVG renderer rather than adding MapLibre for non-tiled source ZIPs. `src/data/mapLayers.ts` is the typed layer registry. It drives layer controls and visibility; disabled layers are not passed into the renderer. Static SVG supplies the immediate operational polar plot; normalized vessel, AIS, iceberg, trajectory, radar, sonar, wind, and current state are layered dynamically. The map supports local pointer pan and wheel zoom without a remote tile service.

The Antarctic and Arctic switches select distinct polar display configurations (including polar orientation and base geometry). The source packages remain WGS84 geographic and must be reprojected during an explicit offline preprocessing step before their geometry is used in a polar operational view. This dashboard does not claim that its bundled schematic coastline is authoritative chart data.

## Lazy/offline strategy

The browser loads no source archive at startup. Shapefile ZIP layers are marked `LAYER UNAVAILABLE` because loading them directly would require a parser and projection treatment that is intentionally not bundled. A future offline build step should generate region-specific GeoJSON for small layers or PMTiles for large layers under `public/map-data/`, then mark the matching registry layer available. Those assets should be fetched only after region/layer selection and according to viewport/zoom.

There is no remote basemap, tile service, font, image, or API. All overlays come from the in-process normalized sensor state. The sonar sector follows vessel heading, sonar detections are local map objects, radar contacts and an animation-owned sweep are separate from React state, and iceberg paths/uncertainty remain explicitly predicted rather than observed.
