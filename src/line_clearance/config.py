"""Typed configuration for the simulator and the pipeline.

Everything that changes a result lives here so a run can be reproduced from
the config hash stored in the lakehouse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class SceneConfig(BaseModel):
    """Synthetic corridor: terrain, towers, wires, trees and sensor model."""

    seed: int = 7
    n_spans: int = Field(6, ge=1)
    span_length_m: tuple[float, float] = (220.0, 300.0)
    max_bend_deg: float = 6.0
    corridor_half_width_m: float = 30.0
    tower_height_m: float = 28.0
    attach_height_m: float = 21.0
    phase_offsets_m: tuple[float, ...] = (-4.5, 0.0, 4.5)
    shield_wire: bool = True
    sag_ratio: tuple[float, float] = (0.025, 0.035)
    ground_density_pts_m2: float = 25.0
    wire_density_pts_m: float = 12.0
    vegetation_density_pts_m2: float = 60.0
    trees_per_100m: float = 9.0
    encroaching_trees_per_span: float = 2.0
    shrubs_per_100m: float = 12.0
    noise_sigma_m: float = 0.03
    outlier_fraction: float = 0.0004
    wire_dropouts_per_wire: int = 2
    # second survey: growth applied to every tree, plus a share that got trimmed
    growth_m: float = 0.0
    trimmed_fraction: float = 0.0
    origin_easting: float = 600_000.0
    origin_northing: float = 500_000.0
    crs: str = "EPSG:32618"


class GroundConfig(BaseModel):
    cell_m: float = 1.0
    max_window_m: float = 17.0
    slope: float = 0.2
    base_threshold_m: float = 0.3
    max_threshold_m: float = 2.5
    ground_tolerance_m: float = 0.25
    refine_band_m: float = 0.4


class WireConfig(BaseModel):
    min_height_m: float = 5.0
    feature_radius_m: float = 2.5
    feature_max_nn: int = 40
    min_linearity: float = 0.9
    max_verticality: float = 0.5
    cluster_eps_m: float = 2.5
    cluster_min_points: int = 3
    min_fragment_points: int = 12
    min_seed_coverage: float = 0.15
    inlier_tolerance_m: float = 0.35
    shield_height_gap_m: float = 3.0
    ignore_shield_wires: bool = True


class TowerConfig(BaseModel):
    cell_m: float = 2.0
    min_height_m: float = 14.0
    min_column_occupancy: float = 0.75
    wire_search_radius_m: float = 4.0
    kink_inner_m: float = 22.0
    kink_outer_m: float = 28.0
    kink_min_rise_m: float = 0.3
    min_wire_points: int = 20
    radius_m: float = 7.0
    min_span_m: float = 60.0
    max_span_m: float = 700.0


class ClearanceConfig(BaseModel):
    """Distances are engineering defaults for the demo, not regulatory values."""

    violation_m: float = 3.0
    warning_m: float = 5.0
    design_sag_factor: float = 1.25
    design_blowout_deg: float = 20.0
    curve_step_m: float = 0.25
    min_vegetation_height_m: float = 0.5
    finding_cluster_eps_m: float = 2.0
    min_finding_points: int = 3
    tree_min_height_m: float = 3.0
    tree_window_m: float = 3.0
    tree_assign_radius_m: float = 6.0
    measurement_margin_m: float = 0.1

    @model_validator(mode="after")
    def _ordered(self) -> ClearanceConfig:
        if self.warning_m < self.violation_m:
            raise ValueError("warning_m must be greater than or equal to violation_m")
        return self


class QualityConfig(BaseModel):
    min_points_per_m2: float = 4.0
    min_wire_coverage: float = 0.6
    max_fit_rmse_m: float = 0.15
    expected_wires_per_span: int | None = None


class LakehouseConfig(BaseModel):
    warehouse: str = "warehouse"
    namespace: str = "grid"
    # CRS of the survey coordinates, recorded with every run. Must be projected, in metres.
    survey_crs: str = "EPSG:32618"


class PipelineConfig(BaseModel):
    outlier_radius_m: float = 2.0
    outlier_min_neighbors: int = 3
    ground: GroundConfig = GroundConfig()
    wires: WireConfig = WireConfig()
    towers: TowerConfig = TowerConfig()
    clearance: ClearanceConfig = ClearanceConfig()
    quality: QualityConfig = QualityConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


class Config(BaseModel):
    scene: SceneConfig = SceneConfig()
    pipeline: PipelineConfig = PipelineConfig()
    lakehouse: LakehouseConfig = LakehouseConfig()


def load_config(path: str | Path | None) -> Config:
    """Load a YAML config; missing keys fall back to the defaults above."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(raw)
