from __future__ import annotations

import pytest

from line_clearance.config import Config
from line_clearance.pipeline import analyse
from line_clearance.simulate import build_scene


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Two short spans with a thin point cloud, enough to exercise every stage."""
    config = Config()
    config.scene.seed = 11
    config.scene.n_spans = 2
    config.scene.span_length_m = (150.0, 180.0)
    config.scene.ground_density_pts_m2 = 8.0
    config.scene.vegetation_density_pts_m2 = 30.0
    config.scene.wire_density_pts_m = 8.0
    config.scene.encroaching_trees_per_span = 3.0
    config.pipeline.quality.expected_wires_per_span = 4
    return config


@pytest.fixture(scope="session")
def scene(cfg):
    return build_scene(cfg.scene)


@pytest.fixture(scope="session")
def analysis(cfg, scene):
    return analyse(scene.points, cfg.pipeline)
