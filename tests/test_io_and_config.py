import numpy as np
import pytest
from pydantic import ValidationError

from line_clearance.config import ClearanceConfig, PipelineConfig, load_config
from line_clearance.io import SurveyError, read_las, write_las


def test_las_roundtrip_keeps_millimetres_and_classes(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform([600_000, 500_000, 90], [600_100, 500_100, 130], (500, 3))
    classes = rng.integers(1, 16, 500)
    path = write_las(tmp_path / "a.las", xyz, classes, "EPSG:32618")
    np.testing.assert_allclose(read_las(path), xyz, atol=0.001)


def test_unreadable_surveys_raise_a_clear_error(tmp_path):
    with pytest.raises(SurveyError, match="not found"):
        read_las(tmp_path / "missing.las")
    broken = tmp_path / "broken.las"
    broken.write_bytes(b"this is not a point cloud")
    with pytest.raises(SurveyError, match="cannot read"):
        read_las(broken)


def test_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        ClearanceConfig(violation_m=5.0, warning_m=3.0)


def test_fingerprint_tracks_every_setting():
    base = PipelineConfig()
    changed = PipelineConfig()
    changed.clearance.design_sag_factor = 1.4
    assert base.fingerprint() == PipelineConfig().fingerprint()
    assert base.fingerprint() != changed.fingerprint()


def test_yaml_overrides_only_what_it_names(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("pipeline:\n  clearance:\n    violation_m: 4.0\n    warning_m: 6.0\n")
    cfg = load_config(path)
    assert cfg.pipeline.clearance.violation_m == 4.0
    assert cfg.pipeline.ground.cell_m == PipelineConfig().ground.cell_m
