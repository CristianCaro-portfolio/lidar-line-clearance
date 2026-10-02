"""Clearance between vegetation and conductors, as flown and at design condition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .conductors import FittedConductor
from .config import ClearanceConfig
from .features import dbscan
from .towers import Span


@dataclass
class Finding:
    span_id: str
    conductor_id: str
    kind: str  # grow_in, fall_in or coverage_gap
    severity: str  # critical, design, watch
    clearance_m: float | None
    design_clearance_m: float | None
    x: float
    y: float
    z: float
    tree_height_m: float | None
    n_points: int


def detect_trees(veg: np.ndarray, hag: np.ndarray, cfg: ClearanceConfig) -> np.ndarray:
    """Tree tops as (n, 4) rows of x, y, z, height.

    Local maxima of a one metre canopy height model, the standard way to find
    individual trees in airborne LiDAR.
    """
    tall = hag >= cfg.tree_min_height_m
    if not tall.any():
        return np.zeros((0, 4))
    pts, height = veg[tall], hag[tall]
    origin = pts[:, :2].min(axis=0)
    cell = np.floor(pts[:, :2] - origin).astype(np.int64)
    shape = (int(cell[:, 0].max()) + 1, int(cell[:, 1].max()) + 1)
    flat = cell[:, 0] * shape[1] + cell[:, 1]
    order = np.lexsort((height, flat))
    last = np.ones(len(order), dtype=bool)
    last[:-1] = flat[order][1:] != flat[order][:-1]
    highest = order[last]

    canopy = np.zeros(shape[0] * shape[1])
    canopy[flat[highest]] = height[highest]
    canopy = canopy.reshape(shape)
    window = int(2 * cfg.tree_window_m) + 1
    smooth = ndimage.gaussian_filter(canopy, 1.0)
    peak = (smooth == ndimage.maximum_filter(smooth, size=window)) & (canopy > 0)
    point_of_cell = np.full(shape[0] * shape[1], -1, dtype=np.int64)
    point_of_cell[flat[highest]] = highest
    picks = point_of_cell[np.flatnonzero(peak.reshape(-1))]
    picks = picks[picks >= 0]
    return np.column_stack([pts[picks], height[picks]])


def _tree_index(trees: np.ndarray) -> cKDTree | None:
    return cKDTree(trees[:, :2]) if len(trees) else None


def measure_span(
    span: Span,
    conductors: list[FittedConductor],
    veg: np.ndarray,
    trees: np.ndarray,
    cfg: ClearanceConfig,
    include_shield: bool,
) -> list[Finding]:
    """Findings for one span: grow-in at both conditions, then fall-in risk."""
    active = [c for c in conductors if include_shield or c.role != "shield"]
    if not active:
        return []
    flown = [c.model.sample(cfg.curve_step_m) for c in active]
    design = [
        c.model.envelope(cfg.curve_step_m, cfg.design_sag_factor, cfg.design_blowout_deg)
        for c in active
    ]
    flown_owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(flown)])
    flown_tree = cKDTree(np.vstack(flown))
    design_tree = cKDTree(np.vstack(design))

    s, d = span.to_local(veg[:, :2])
    # distances use the raw returns: averaging them into voxels would pull the
    # crown surface inwards and report more clearance than there is
    local = veg[(s > -5.0) & (s < span.length + 5.0) & (np.abs(d) < 40.0)]
    findings: list[Finding] = []
    tree_index = _tree_index(trees)

    if len(local):
        flown_dist, flown_hit = flown_tree.query(local)
        design_dist, _ = design_tree.query(local)
        close = np.minimum(flown_dist, design_dist) < cfg.warning_m
        idx = np.flatnonzero(close)
        if len(idx):
            groups = dbscan(local[idx], cfg.finding_cluster_eps_m, 1)
            # touching crowns form one cluster, so split it by the nearest tree top
            if tree_index is not None:
                dist, nearest = tree_index.query(local[idx, :2])
                groups = np.where(dist < cfg.tree_assign_radius_m, -2 - nearest, groups)
            for group in np.unique(groups):
                members = idx[groups == group]
                if len(members) < cfg.min_finding_points:
                    continue
                worst = members[np.argmin(flown_dist[members])]
                flown_min = float(flown_dist[members].min())
                design_min = float(design_dist[members].min())
                # the margin covers what the sensor cannot see: the outermost
                # leaves are rarely hit, so measured clearance errs on the high side
                limit = cfg.violation_m + cfg.measurement_margin_m
                if flown_min < limit:
                    severity = "critical"
                elif design_min < limit:
                    severity = "design"
                else:
                    severity = "watch"
                height = None
                if tree_index is not None:
                    dist, nearest = tree_index.query(local[worst, :2])
                    if dist < 6.0:
                        height = float(trees[nearest, 3])
                findings.append(
                    Finding(
                        span_id=span.span_id,
                        conductor_id=active[flown_owner[flown_hit[worst]]].conductor_id,
                        kind="grow_in",
                        severity=severity,
                        clearance_m=flown_min,
                        design_clearance_m=design_min,
                        x=float(local[worst, 0]),
                        y=float(local[worst, 1]),
                        z=float(local[worst, 2]),
                        tree_height_m=height,
                        n_points=int(len(members)),
                    )
                )

    # a tree is a strike risk when it is taller than its distance to the wire
    if len(trees):
        ts, td = span.to_local(trees[:, :2])
        for tree in trees[(ts >= 0) & (ts < span.length) & (np.abs(td) < 40.0)]:
            base = np.array([tree[0], tree[1], tree[2] - tree[3]])
            reach, hit = flown_tree.query(base)
            if tree[3] + cfg.measurement_margin_m >= reach:
                findings.append(
                    Finding(
                        span_id=span.span_id,
                        conductor_id=active[flown_owner[hit]].conductor_id,
                        kind="fall_in",
                        severity="watch",
                        clearance_m=float(reach),
                        design_clearance_m=None,
                        x=float(tree[0]),
                        y=float(tree[1]),
                        z=float(tree[2]),
                        tree_height_m=float(tree[3]),
                        n_points=1,
                    )
                )
    return findings
