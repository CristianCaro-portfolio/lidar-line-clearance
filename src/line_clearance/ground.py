"""Ground classification with a progressive morphological filter.

The lowest return in each grid cell is a first guess of the terrain. Opening
that raster with growing windows removes anything that sticks out more than
the slope allows (trees, towers), and what survives is interpolated into a
terrain model. Height above ground is what every later stage works with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import GroundConfig


@dataclass
class Terrain:
    origin: np.ndarray
    cell: float
    surface: np.ndarray
    occupied_cells: int = 0

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear terrain elevation at arbitrary positions."""
        coords = ((xy - self.origin) / self.cell - 0.5).T
        return ndimage.map_coordinates(self.surface, coords, order=1, mode="nearest")


def _lowest_per_cell(index: np.ndarray, z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = index[:, 0] * shape[1] + index[:, 1]
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    first = np.ones(len(flat_sorted), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    raster = np.full(shape[0] * shape[1], np.nan)
    raster[flat_sorted[first]] = z[order][first]
    return raster.reshape(shape)


def _median_per_cell(index: np.ndarray, z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = index[:, 0] * shape[1] + index[:, 1]
    order = np.lexsort((z, flat))
    flat_sorted, z_sorted = flat[order], z[order]
    cells, start, count = np.unique(flat_sorted, return_index=True, return_counts=True)
    raster = np.full(shape[0] * shape[1], np.nan)
    raster[cells] = z_sorted[start + count // 2]
    return raster.reshape(shape)


def _fill(raster: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells from their neighbours, nearest value as last resort."""
    weights = ndimage.gaussian_filter(valid.astype(float), 2.0)
    values = ndimage.gaussian_filter(np.where(valid, raster, 0.0), 2.0)
    smooth = np.divide(values, weights, out=np.zeros_like(values), where=weights > 1e-3)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    fallback = raster[tuple(nearest)]
    filled = np.where(weights > 0.05, smooth, fallback)
    return np.where(valid, raster, filled)


def classify_ground(xyz: np.ndarray, cfg: GroundConfig) -> tuple[np.ndarray, np.ndarray, Terrain]:
    """Return (is_ground, height_above_ground, terrain)."""
    origin = xyz[:, :2].min(axis=0)
    index = np.floor((xyz[:, :2] - origin) / cfg.cell_m).astype(np.int64)
    shape = (int(index[:, 0].max()) + 1, int(index[:, 1].max()) + 1)

    lowest = _lowest_per_cell(index, xyz[:, 2], shape)
    has_data = ~np.isnan(lowest)
    surface = _fill(lowest, has_data)

    is_object = np.zeros(shape, dtype=bool)
    window, previous = 3, 1
    max_window = max(int(cfg.max_window_m / cfg.cell_m), 3)
    while window <= max_window:
        opened = ndimage.grey_opening(surface, size=(window, window))
        threshold = min(
            cfg.slope * (window - previous) * cfg.cell_m + cfg.base_threshold_m,
            cfg.max_threshold_m,
        )
        is_object |= (surface - opened) > threshold
        surface = opened
        previous, window = window, 2 * window - 1

    terrain_cells = has_data & ~is_object
    if not terrain_cells.any():
        terrain_cells = has_data
    dtm = _fill(np.where(terrain_cells, lowest, np.nan), terrain_cells)
    dtm = ndimage.uniform_filter(dtm, size=3, mode="nearest")
    terrain = Terrain(origin, cfg.cell_m, dtm, int(has_data.sum()))

    # The lowest return sits below the real surface by about two sigma of the
    # sensor noise. Re-estimate each terrain cell as the median of the returns
    # in a thin band above that first surface, which removes the bias.
    hag = xyz[:, 2] - terrain.height_at(xyz[:, :2])
    band = hag <= cfg.refine_band_m
    median = _median_per_cell(index[band], xyz[band, 2], shape)
    refined_cells = terrain_cells & ~np.isnan(median)
    if refined_cells.any():
        dtm = _fill(np.where(refined_cells, median, np.nan), refined_cells)
        terrain.surface = ndimage.uniform_filter(dtm, size=3, mode="nearest")
        hag = xyz[:, 2] - terrain.height_at(xyz[:, :2])
    return np.abs(hag) <= cfg.ground_tolerance_m, hag, terrain
