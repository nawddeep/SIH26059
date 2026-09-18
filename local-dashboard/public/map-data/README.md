# Offline vector map data

The map runs entirely from local assets. `natural-earth/` is generated from the reviewed Natural Earth 1:50m Shapefile ZIPs in `vector maps/` by `npm run prepare-map-data`; these WGS 84 GeoJSON files provide land, coastline, glaciated areas, Antarctic ice shelves, and graticule geometry.

The MapLibre renderer has no style URL, remote fonts, remote JavaScript, or remote tile fallback. The current source files are compact enough for local GeoJSON rendering. A future operational deployment should replace the broad-area GeoJSON sources with locally hosted PMTiles or MVT for true viewport tile streaming; scientific layers must remain separate and be delivered as subset features or tiles.
