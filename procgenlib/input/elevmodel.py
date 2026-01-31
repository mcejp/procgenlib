from enum import Enum
import logging
from pathlib import Path
from typing import Callable, Tuple
import zipfile

import numpy as np


logger = logging.getLogger(__name__)


class Format(Enum):
    """Supported elevation data file formats."""

    HGT = "hgt"
    ZIP = "zip"


class HgtTileProvider:
    """Base class for loading 1-arcsecond elevation tiles in HGT format.

    Instances are callable: given a tile's integer latitude and longitude,
    they return a (3601, 3601) int16 NumPy array of elevation values in meters.
    Data is returned in south-to-north order per ISO 6709.

    Tiles can optionally be zipped.
    """

    def __init__(self, path, *, format, case_convention, zip_pattern):
        self._path = Path(path)
        self._format = Format[format.upper()]
        self._case_convention = case_convention
        self._zip_pattern = zip_pattern

    def __call__(self, lat: int, lon: int) -> np.ndarray:
        if lat >= 0:
            lat_str = f"n{lat:02d}"
        else:
            lat_str = f"s{-lat + 1:02d}"

        if lon >= 0:
            lon_str = f"e{lon:03d}"
        else:
            lon_str = f"w{-lon + 1:03d}"

        if self._case_convention == "upper":
            lat_str = lat_str.upper()
            lon_str = lon_str.upper()

        filename = f"{lat_str}{lon_str}.hgt"

        if self._format is Format.ZIP:
            zip_path = self._path / self._zip_pattern.format(lat=lat_str, lon=lon_str)

            # AAAAAAAAAAAAAAAA
            if not zip_path.exists():
                return np.zeros((3601, 3601), dtype=np.int16)

            logger.debug("Load %s", zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(filename) as f:
                    raw_data = f.read()
        elif self._format is Format.HGT:
            logger.debug("Load %s", self._path / filename)
            with open(self._path / filename, 'rb') as f:
                raw_data = f.read()

        tile_data = np.frombuffer(raw_data, dtype='>i2').reshape((3601, 3601))
        # file is stored from north-to-south, we prefer south-to-north in line with ISO 6709
        return np.flip(tile_data, axis=0)


# https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf
class NASADEM_HGT(HgtTileProvider):
    """Tile provider for `NASADEM <https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf>`_ HGT data."""

    def __init__(self, path, *, format):
        super().__init__(path,
                         format=format,
                         case_convention="lower",
                         zip_pattern="NASADEM_HGT_{lat}{lon}.zip")


class SRTMGL1(HgtTileProvider):
    """Tile provider for SRTM Global 1-arcsecond (SRTMGL1) data."""

    def __init__(self, path, *, format):
        super().__init__(path,
                         format=format,
                         case_convention="upper",
                         zip_pattern="{lat}{lon}.SRTMGL1.hgt.zip")


def extract(tile_provider_fn: Callable[[int, int], np.ndarray],
            min_latitude: float,
            min_longitude: float,
            max_latitude: float,
            max_longitude: float) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Extract elevation data for the given latitude and longitude spans (inclusive of endpoints).

    The convention is as per ISO 6709: north latitude is positive, east longitude is positive, decimal degrees are used

    Returns a tuple (height, latitude_span, longitude_span) where:
        - height is a 2D numpy array of int16 with elevation data in meters
    """

    # compute span in terms of tiles (1x1 degree each)
    # these are *inclusive*
    tile_min_lat = int(np.floor(min_latitude))
    tile_min_lon = int(np.floor(min_longitude))
    tile_max_lat = int(np.floor(max_latitude))
    tile_max_lon = int(np.floor(max_longitude))

    # prepare working buffer
    # if we were smart, we would crop it to the region of interest only to reduce transient memory usage (which is ~24.7 MB per tile)
    # (at the expense of more complicated logic for piecing the tiles together)
    height = np.zeros(((tile_max_lat - tile_min_lat + 1) * 3600 + 1,
                       (tile_max_lon - tile_min_lon + 1) * 3600 + 1), dtype=np.int16)

    logger.debug("Extracting data for lat <%d; %d> lon <%d; %d>, buffer shape %dx%d (%.1f MB)",
                    tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon, height.shape[0], height.shape[1], height.nbytes / 1024**2)

    # fetch and merge tiles
    # fetching could be parallelized if latency is a problem
    for lat in range(tile_min_lat, tile_max_lat + 1):
        for lon in range(tile_min_lon, tile_max_lon + 1):
            tile_height = tile_provider_fn(lat, lon)
            height[(lat - tile_min_lat) * 3600:(lat - tile_min_lat + 1) * 3600 + 1,
                   (lon - tile_min_lon) * 3600:(lon - tile_min_lon + 1) * 3600 + 1] = tile_height

    # crop to region of interest
    # indexes still inclusive
    lat_start = int(np.floor((min_latitude - tile_min_lat) * 3600))
    lon_start = int(np.floor((min_longitude - tile_min_lon) * 3600))
    lat_end = int(np.ceil((max_latitude - tile_min_lat) * 3600))     # ceil because we just want to round up to the next whole arcsecond
    lon_end = int(np.ceil((max_longitude - tile_min_lon) * 3600))
    roi_height = height[lat_start:lat_end + 1, lon_start:lon_end + 1].copy()
    del height

    output_min_lat = tile_min_lat + lat_start / 3600
    output_min_lon = tile_min_lon + lon_start / 3600
    output_max_lat = tile_min_lat + lat_end / 3600
    output_max_lon = tile_min_lon + lon_end / 3600

    logger.debug("Returning lat <%f, %f> lon <%f, %f>, output shape %dx%d (%.1f MB)",
                    output_min_lat, output_max_lat, output_min_lon, output_max_lon,
                    roi_height.shape[0], roi_height.shape[1], roi_height.nbytes / 1024**2)

    return roi_height, output_min_lat, output_min_lon, output_max_lat, output_max_lon
