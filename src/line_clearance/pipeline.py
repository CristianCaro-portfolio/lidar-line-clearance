"""End to end analysis of one survey and persistence of its results."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
from scipy.spatial import cKDTree
from shapely import LineString, Point

from . import __version__, lakehouse
from .clearance import Finding, detect_trees, measure_span
from .conductors import FittedConductor, fit_span_conductors
from .config import Config, PipelineConfig
from .features import isolated_points, shape_features
from .ground import classify_ground
from .io import file_sha256, read_las, write_las
from .towers import Span, build_spans, detect_towers

log = logging.getLogger(__name__)

UNCLASSIFIED, GROUND, LOW_VEG, MED_VEG, HIGH_VEG, NOISE, WIRE, TOWER = 1, 2, 3, 4, 5, 7, 14, 15


@dataclass
class Analysis:
    classification: np.ndarray
    towers: np.ndarray
    spans: list[Span]
    conductors: list[FittedConductor]
    findings: list[Finding]
    trees: np.ndarray
    status: str
    issues: list[str]
    points_per_m2: float
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]


def analyse(xyz: np.ndarray, cfg: PipelineConfig) -> Analysis:
    """Classify the cloud, model the line and measure clearances.

    Works in coordinates shifted to the corner of the survey so that
    covariance and fitting maths keep full precision, then shifts results back.
    """
    timings: dict[str, float] = {}
    issues: list[str] = []
    shift = np.array([*np.floor(xyz[:, :2].min(axis=0)), 0.0])
    pts = xyz - shift

    tick = time.perf_counter()
    is_ground, hag, terrain = classify_ground(pts, cfg.ground)
    timings["ground"] = time.perf_counter() - tick
    density = len(pts) / max(terrain.occupied_cells * cfg.ground.cell_m**2, 1.0)
    if density < cfg.quality.min_points_per_m2:
        issues.append("low_point_density")

    classification = np.full(len(pts), UNCLASSIFIED, dtype=np.uint8)
    classification[is_ground] = GROUND
    above = hag > cfg.ground.ground_tolerance_m

    # --- stray returns
    tick = time.perf_counter()
    high = np.flatnonzero(hag > 2.0)
    noise = np.zeros(len(pts), dtype=bool)
    noise[high[isolated_points(pts[high], cfg.outlier_radius_m, cfg.outlier_min_neighbors)]] = True
    classification[noise] = NOISE

    # --- wire candidates from neighbourhood shape
    elevated = np.flatnonzero((hag >= cfg.wires.min_height_m) & ~noise)
    linearity, verticality = shape_features(
        pts[elevated], cfg.wires.feature_radius_m, cfg.wires.feature_max_nn
    )
    is_candidate = (linearity >= cfg.wires.min_linearity) & (
        verticality <= cfg.wires.max_verticality
    )
    candidate = np.zeros(len(pts), dtype=bool)
    candidate[elevated[is_candidate]] = True
    timings["features"] = time.perf_counter() - tick

    # --- towers and spans
    tick = time.perf_counter()
    towers = detect_towers(pts[~noise], hag[~noise], candidate[~noise], cfg.towers)
    spans = build_spans(towers, cfg.towers)
    near_tower = np.zeros(len(pts), dtype=bool)
    if len(towers):
        dist, _ = cKDTree(towers[:, :2]).query(pts[:, :2])
        near_tower = (dist < cfg.towers.radius_m) & above & ~noise
    classification[near_tower] = TOWER
    timings["towers"] = time.perf_counter() - tick

    # --- conductors
    tick = time.perf_counter()
    conductors: list[FittedConductor] = []
    candidate_idx = np.flatnonzero(candidate)
    for span in spans:
        try:
            fitted, _ = fit_span_conductors(
                span, pts[candidate_idx], cfg.wires, cfg.quality, cfg.towers.radius_m
            )
        except Exception:  # one bad span must not sink the whole survey
            log.exception("conductor fitting failed in span %s", span.span_id)
            issues.append(f"{span.span_id}:fit_error")
            fitted = []
        conductors.extend(fitted)
    if conductors:
        curve = np.vstack([c.model.sample(0.25) for c in conductors])
        elevated_all = np.flatnonzero((hag >= cfg.wires.min_height_m - 1.0) & ~noise)
        dist, _ = cKDTree(curve).query(pts[elevated_all])
        classification[elevated_all[dist < cfg.wires.inlier_tolerance_m]] = WIRE
    # wire like points no model explains are left unclassified, never vegetation
    orphan = candidate & (classification != WIRE) & (classification != TOWER)
    classification[orphan] = UNCLASSIFIED
    timings["conductors"] = time.perf_counter() - tick

    # --- vegetation
    rest = above & (classification == UNCLASSIFIED) & ~orphan
    # second isolation pass: a stray return right next to a wire has neighbours,
    # but none of them are vegetation
    lone = np.flatnonzero(rest & (hag > 2.0))
    lone = lone[isolated_points(pts[lone], cfg.outlier_radius_m, cfg.outlier_min_neighbors)]
    classification[lone] = NOISE
    rest[lone] = False
    classification[rest & (hag <= 0.5)] = LOW_VEG
    classification[rest & (hag > 0.5) & (hag <= 2.0)] = MED_VEG
    classification[rest & (hag > 2.0)] = HIGH_VEG
    veg_mask = rest & (hag > cfg.clearance.min_vegetation_height_m)
    veg, veg_hag = pts[veg_mask], hag[veg_mask]

    tick = time.perf_counter()
    trees = detect_trees(veg, veg_hag, cfg.clearance)
    findings: list[Finding] = []
    for span in spans:
        in_span = [c for c in conductors if c.span_id == span.span_id]
        expected = cfg.quality.expected_wires_per_span
        reasons = []
        if not in_span or (expected is not None and len(in_span) != expected):
            reasons.append("conductor_count")
        if any(c.status != "ok" for c in in_span):
            reasons.append("conductor_quality")
        if reasons:
            # Fail closed: a span whose wires are not modelled well is reported
            # for review. Measuring against a bad model would either hide real
            # encroachments or invent them.
            issues.append(f"{span.span_id}:unmeasurable({'+'.join(reasons)})")
            mid = (span.start + span.end) / 2.0
            findings.append(
                Finding(
                    span.span_id,
                    "",
                    "coverage_gap",
                    "review",
                    None,
                    None,
                    float(mid[0]),
                    float(mid[1]),
                    float(mid[2]),
                    None,
                    len(in_span),
                )
            )
            continue
        findings.extend(
            measure_span(
                span,
                in_span,
                veg,
                trees,
                cfg.clearance,
                include_shield=not cfg.wires.ignore_shield_wires,
            )
        )
    timings["clearance"] = time.perf_counter() - tick

    for conductor in conductors:
        issues.extend(f"{conductor.conductor_id}:{issue}" for issue in conductor.issues)
    if not spans:
        status = "no_line_detected"
    elif issues:
        status = "needs_review"
    else:
        status = "ok"

    return _to_world(
        Analysis(
            classification,
            towers,
            spans,
            conductors,
            findings,
            trees,
            status,
            issues,
            float(density),
            timings,
        ),
        shift,
    )


def _to_world(result: Analysis, shift: np.ndarray) -> Analysis:
    xy = shift[:2]
    if len(result.towers):
        result.towers[:, :2] += xy
    if len(result.trees):
        result.trees[:, :2] += xy
    result.spans = [replace(s, start=s.start + shift, end=s.end + shift) for s in result.spans]
    for conductor in result.conductors:
        origin = conductor.model.origin_xy
        conductor.model = replace(conductor.model, origin_xy=(origin[0] + xy[0], origin[1] + xy[1]))
    result.findings = [replace(f, x=f.x + xy[0], y=f.y + xy[1]) for f in result.findings]
    return result


# --------------------------------------------------------------------------- persistence


def _finding_id(survey_id: str, f: Finding) -> str:
    key = f"{survey_id}|{f.kind}|{f.span_id}|{round(f.x)}|{round(f.y)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def frames(
    result: Analysis, survey_id: str, run_id: str, cfg: PipelineConfig, started: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Conductor and finding rows for the lakehouse."""
    conductor_rows = []
    for c in result.conductors:
        line = c.model.sample(5.0)
        hot = c.model.sample(5.0, cfg.clearance.design_sag_factor)
        chord = np.linspace(line[0, 2], line[-1, 2], len(line))
        conductor_rows.append(
            {
                "survey_id": survey_id,
                "run_id": run_id,
                "span_id": c.span_id,
                "conductor_id": c.conductor_id,
                "role": c.role,
                "status": c.status,
                "issues": ",".join(c.issues),
                "length_m": c.model.length,
                "sag_m": c.model.sag,
                "design_sag_m": float((chord - hot[:, 2]).max()),
                "catenary_c": c.model.catenary.c,
                "rmse_m": c.rmse,
                "n_points": c.n_points,
                "coverage": c.coverage,
                "geom": LineString(line).wkb,
            }
        )
    finding_rows = [
        {
            "survey_id": survey_id,
            "run_id": run_id,
            "finding_id": _finding_id(survey_id, f),
            "span_id": f.span_id,
            "conductor_id": f.conductor_id,
            "kind": f.kind,
            "severity": f.severity,
            "clearance_m": f.clearance_m,
            "design_clearance_m": f.design_clearance_m,
            "tree_height_m": f.tree_height_m,
            "n_points": f.n_points,
            "easting": f.x,
            "northing": f.y,
            "elevation": f.z,
            "geom": Point(f.x, f.y, f.z).wkb,
            "detected_at": started,
        }
        for f in result.findings
    ]
    schemas = {
        name: {
            f.name: pl.from_arrow(_empty(f.type)).dtype for f in lakehouse.SCHEMAS[name].as_arrow()
        }
        for name in ("conductors", "findings")
    }
    return (
        pl.DataFrame(conductor_rows, schema=schemas["conductors"]),
        pl.DataFrame(finding_rows, schema=schemas["findings"]),
    )


def _empty(arrow_type: pa.DataType) -> pa.Array:
    return pa.array([], type=arrow_type)


def run_survey(
    las_path: str | Path,
    cfg: Config,
    survey_id: str | None = None,
    classified_out: str | Path | None = None,
) -> tuple[Analysis, str]:
    """Analyse a LAS file and write runs, conductors and findings to Iceberg."""
    las_path = Path(las_path)
    survey_id = survey_id or las_path.stem
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()

    xyz = read_las(las_path)
    digest = file_sha256(las_path)
    run_id = hashlib.sha256(
        f"{digest}|{cfg.pipeline.fingerprint()}|{__version__}".encode()
    ).hexdigest()[:12]
    result = analyse(xyz, cfg.pipeline)
    duration = time.perf_counter() - tick

    if classified_out is not None:
        write_las(classified_out, xyz, result.classification, cfg.lakehouse.survey_crs)

    conductors, findings = frames(result, survey_id, run_id, cfg.pipeline, started)
    run = pl.DataFrame(
        [
            {
                "survey_id": survey_id,
                "run_id": run_id,
                "input_file": las_path.name,
                "input_sha256": digest,
                "config_hash": cfg.pipeline.fingerprint(),
                "code_version": __version__,
                "crs": cfg.lakehouse.survey_crs,
                "n_points": len(xyz),
                "points_per_m2": result.points_per_m2,
                "n_towers": len(result.towers),
                "n_spans": len(result.spans),
                "n_conductors": len(result.conductors),
                "n_findings": len(result.findings),
                "n_critical": len(result.critical),
                "status": result.status,
                "issues": ",".join(result.issues),
                "duration_s": duration,
                "started_at": started,
            }
        ]
    )
    catalog = lakehouse.open_catalog(cfg.lakehouse)
    for name, frame in (("runs", run), ("conductors", conductors), ("findings", findings)):
        lakehouse.replace_survey(lakehouse.table(catalog, cfg.lakehouse, name), frame, survey_id)
    return result, run_id
