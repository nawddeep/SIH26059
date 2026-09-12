"""
Data downloaders for real data sources.

Each downloader module handles a specific data source:
- nsidc: NSIDC sea-ice concentration data
- era5: ERA5 reanalysis (wind, temperature)
- copernicus: Copernicus Marine (SST, ocean currents)
"""

__all__ = ['nsidc', 'era5', 'copernicus']
