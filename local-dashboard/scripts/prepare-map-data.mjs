import { readFile, mkdir, writeFile } from 'node:fs/promises'
import shp from 'shpjs'

const datasets = [
  ['land', 'vector maps/ne_50m_land.zip'],
  ['coastline', 'vector maps/ne_50m_coastline.zip'],
  ['ocean', 'vector maps/ne_50m_ocean.zip'],
  ['glaciated', 'vector maps/ne_50m_glaciated_areas.zip'],
  ['ice-shelves', 'vector maps/ne_50m_antarctic_ice_shelves_polys.zip'],
  ['graticule', 'vector maps/ne_50m_geographic_lines.zip'],
]

await mkdir('public/map-data/natural-earth', { recursive: true })
for (const [name, input] of datasets) {
  const archive = await readFile(input)
  const data = await shp(archive.buffer.slice(archive.byteOffset, archive.byteOffset + archive.byteLength))
  await writeFile(`public/map-data/natural-earth/${name}.geojson`, JSON.stringify(data))
  console.log(`${name}: ${data.features.length} features`)
}
