"""Turn wire candidate points into one catenary model per conductor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .catenary import Conductor, fit_catenary
from .config import QualityConfig, WireConfig
from .features import dbscan
from .towers import Span


@dataclass
class FittedConductor:
    conductor_id: str
    span_id: str
    model: Conductor
    rmse: float
    n_points: int
    coverage: float
    role: str = "phase"
    status: str = "ok"
    issues: list[str] = field(default_factory=list)


def _fit(span: Span, pts: np.ndarray) -> tuple[Conductor, float]:
    """Fit the plan line first, then the catenary inside that vertical plane."""
    s, d = span.to_local(pts[:, :2])
    k, d0 = np.polyfit(s, d, 1)
    direction = span.axis + k * span.normal
    stretch = float(np.linalg.norm(direction))
    direction = direction / stretch
    origin = span.start[:2] + d0 * span.normal
    catenary, rmse = fit_catenary(s * stretch, pts[:, 2])
    model = Conductor(
        (float(origin[0]), float(origin[1])),
        (float(direction[0]), float(direction[1])),
        span.length * stretch,
        catenary,
    )
    return model, rmse


def _coverage(span: Span, pts: np.ndarray, bin_m: float = 5.0) -> float:
    s, _ = span.to_local(pts[:, :2])
    n_bins = max(int(span.length / bin_m), 1)
    hit = np.unique(np.clip((s / bin_m).astype(int), 0, n_bins - 1))
    return len(hit) / n_bins


def fit_span_conductors(
    span: Span,
    candidates: np.ndarray,
    cfg: WireConfig,
    quality: QualityConfig,
    tower_radius: float,
) -> tuple[list[FittedConductor], np.ndarray]:
    """Fit every conductor in a span.

    Returns the fitted conductors and, for each candidate point, the index of
    the conductor it belongs to (-1 when it belongs to none).

    DBSCAN splits the candidates into connected pieces. A wire with a gap in
    its returns comes out as several pieces, so pieces are merged whenever
    one catenary explains them together.
    """
    owner = np.full(len(candidates), -1, dtype=np.int64)
    s, d = span.to_local(candidates[:, :2])
    inside = (s > tower_radius) & (s < span.length - tower_radius) & (np.abs(d) < 15.0)
    index = np.flatnonzero(inside)
    if len(index) < cfg.min_fragment_points:
        return [], owner
    pts = candidates[index]
    labels = dbscan(pts, cfg.cluster_eps_m, cfg.cluster_min_points)
    pieces = [np.flatnonzero(labels == lab) for lab in np.unique(labels) if lab >= 0]
    pieces = sorted((p for p in pieces if len(p) >= cfg.min_fragment_points), key=len, reverse=True)

    groups: list[np.ndarray] = []
    pending: list[np.ndarray] = []

    def joins(piece: np.ndarray) -> int:
        """Index of the wire this piece belongs to, or -1.

        The piece is tested against a fit of both sets together rather than
        against the existing model, because a catenary fitted to a short piece
        extrapolates badly and would reject the rest of its own wire.
        """
        for i, group in enumerate(groups):
            try:
                model, _ = _fit(span, pts[np.concatenate([group, piece])])
            except ValueError:
                continue
            dist, _ = cKDTree(model.sample(0.5)).query(pts[piece])
            if np.mean(dist < cfg.inlier_tolerance_m) > 0.8:
                return i
        return -1

    for piece in pieces:
        hit = joins(piece)
        if hit >= 0:
            groups[hit] = np.concatenate([groups[hit], piece])
        elif _coverage(span, pts[piece]) < cfg.min_seed_coverage:
            pending.append(piece)
        else:
            try:
                _fit(span, pts[piece])
            except ValueError:
                continue
            groups.append(piece)
    for piece in pending:
        hit = joins(piece)
        if hit >= 0:
            groups[hit] = np.concatenate([groups[hit], piece])
    fits = []
    for group in groups:
        try:
            model, rmse = _fit(span, pts[group])
        except ValueError:
            continue
        lateral = float(span.to_local(np.array([model.origin_xy]))[1][0])
        fits.append((round(lateral, 1), model.catenary.z0, group, model, rmse))
    fits.sort(key=lambda fit: fit[:2])

    fitted: list[FittedConductor] = []
    for rank, (_, _, group, model, rmse) in enumerate(fits, start=1):
        conductor = FittedConductor(
            conductor_id=f"{span.span_id}-C{rank}",
            span_id=span.span_id,
            model=model,
            rmse=rmse,
            n_points=len(group),
            coverage=_coverage(span, pts[group]),
        )
        if conductor.coverage < quality.min_wire_coverage:
            conductor.issues.append("low_coverage")
        if conductor.rmse > quality.max_fit_rmse_m:
            conductor.issues.append("poor_fit")
        if conductor.issues:
            conductor.status = "degraded"
        owner[index[group]] = len(fitted)
        fitted.append(conductor)

    _tag_shield_wires(fitted, cfg)
    return fitted, owner


def _tag_shield_wires(conductors: list[FittedConductor], cfg: WireConfig) -> None:
    """Shield wires run on top of the tower, well above the energised phases."""
    if len(conductors) < 3:
        return
    mid = np.array([float(c.model.catenary.z(c.model.length / 2)) for c in conductors])
    reference = np.median(mid)
    for conductor, height in zip(conductors, mid, strict=True):
        if height - reference > cfg.shield_height_gap_m:
            conductor.role = "shield"
