"""Tower detection and span layout.

A tower is tall, fills its whole column with returns, has wires running
through it and is a place where those wires kink upwards. A tall tree next to
the line can pass the first tests but not the last one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import TowerConfig


@dataclass(frozen=True)
class Span:
    """Local frame between two consecutive towers: s along the line, d to the left."""

    span_id: str
    start: np.ndarray
    end: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end[:2] - self.start[:2]))

    @property
    def axis(self) -> np.ndarray:
        return (self.end[:2] - self.start[:2]) / self.length

    @property
    def normal(self) -> np.ndarray:
        u = self.axis
        return np.array([-u[1], u[0]])

    def to_local(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = xy - self.start[:2]
        return rel @ self.axis, rel @ self.normal


def detect_towers(
    xyz: np.ndarray, hag: np.ndarray, wire_candidate: np.ndarray, cfg: TowerConfig
) -> np.ndarray:
    """Return tower positions as (n, 3) rows of x, y, ground elevation."""
    tall = hag > 2.0
    if not tall.any() or not wire_candidate.any():
        return np.zeros((0, 3))
    pts, heights, is_wire = xyz[tall], hag[tall], wire_candidate[tall]
    origin = pts[:, :2].min(axis=0)
    cell = np.floor((pts[:, :2] - origin) / cfg.cell_m).astype(np.int64)
    shape = (int(cell[:, 0].max()) + 1, int(cell[:, 1].max()) + 1)
    flat = cell[:, 0] * shape[1] + cell[:, 1]

    # A lattice tower leans across several cells, so occupancy is measured over
    # the 3 x 3 block around each cell: the share of one metre height bins
    # between 2 m and the top that hold at least one return.
    top = np.zeros(shape)
    np.maximum.at(top, (cell[:, 0], cell[:, 1]), heights)
    top = ndimage.maximum_filter(top, size=3)
    bins = np.unique(np.column_stack([cell, heights.astype(np.int64)]), axis=0)
    spread = np.vstack([bins + np.array([dx, dy, 0]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    inside = (
        (spread[:, 0] >= 0)
        & (spread[:, 0] < shape[0])
        & (spread[:, 1] >= 0)
        & (spread[:, 1] < shape[1])
    )
    spread = np.unique(spread[inside], axis=0)
    filled = np.zeros(shape)
    np.add.at(filled, (spread[:, 0], spread[:, 1]), 1.0)
    occupancy = filled / np.maximum(top - 2.0, 1.0)

    column = (top >= cfg.min_height_m) & (occupancy >= cfg.min_column_occupancy)
    has_points = np.zeros(shape, dtype=bool)
    has_points[cell[:, 0], cell[:, 1]] = True
    column &= has_points
    groups, count = ndimage.label(column, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros((0, 3))

    wires, wire_hag = xyz[wire_candidate], hag[wire_candidate]
    wire_tree = cKDTree(wires[:, :2])
    group_of_point = groups.reshape(-1)[flat]
    towers = []
    for group in range(1, count + 1):
        member = group_of_point == group
        # the steel at the peak sits above the cross arm and any neighbouring
        # tree, so its mean is a centre that vegetation at the base cannot pull away
        top_of_steel = np.percentile(heights[member], 98)
        peak = member & (heights >= top_of_steel - 3.0) & ~is_wire
        if not peak.any():
            peak = member
        centre = pts[peak, :2].mean(axis=0)
        near = wire_tree.query_ball_point(centre, cfg.wire_search_radius_m)
        if len(near) < cfg.min_wire_points:
            continue
        if heights[member].max() < wire_hag[near].max() - 0.5:
            continue
        if not _is_support(centre, wires[near, 2].max(), wires, wire_tree, cfg):
            continue
        ground = float(np.median(pts[member, 2] - heights[member]))
        towers.append([centre[0], centre[1], ground])
    towers = np.array(towers).reshape(-1, 3)
    return _merge_close(towers, cfg.radius_m)


def _is_support(
    centre: np.ndarray, peak: float, wires: np.ndarray, wire_tree: cKDTree, cfg: TowerConfig
) -> bool:
    """Wires hang convex inside a span and kink upwards only where they are held.

    Compare the wire elevation at the candidate with the elevation a short
    distance away on each side. Above the chord means a support, below means
    the candidate sits somewhere along a span. Using elevation on both sides
    makes the test independent of terrain slope.
    """
    outer = np.array(wire_tree.query_ball_point(centre, cfg.kink_outer_m), dtype=np.int64)
    if len(outer) == 0:
        return True
    rel = wires[outer, :2] - centre
    dist = np.linalg.norm(rel, axis=1)
    ring = dist >= cfg.kink_inner_m
    if ring.sum() < 4:
        return True
    _, _, vt = np.linalg.svd(rel[ring], full_matrices=False)
    side = rel[ring] @ vt[0] > 0
    heights = wires[outer[ring], 2]
    if side.all() or not side.any():
        return True  # line end, wires on one side only
    chord = (heights[side].max() + heights[~side].max()) / 2.0
    return peak - chord >= cfg.kink_min_rise_m


def _merge_close(towers: np.ndarray, radius: float) -> np.ndarray:
    """A lattice tower can split into several columns, keep one centre each."""
    if len(towers) < 2:
        return towers
    pairs = cKDTree(towers[:, :2]).query_pairs(radius)
    parent = list(range(len(towers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(towers))])
    return np.array([towers[roots == r].mean(axis=0) for r in np.unique(roots)])


def build_spans(towers: np.ndarray, cfg: TowerConfig) -> list[Span]:
    """Order towers along the line and pair neighbours into spans."""
    if len(towers) < 2:
        return []
    centred = towers[:, :2] - towers[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    ordered = towers[np.argsort(centred @ vt[0])]
    spans = []
    for a, b in zip(ordered[:-1], ordered[1:], strict=True):
        length = float(np.linalg.norm(b[:2] - a[:2]))
        if cfg.min_span_m <= length <= cfg.max_span_m:
            spans.append(Span(f"S{len(spans) + 1:02d}", a, b))
    return spans
