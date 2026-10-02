import numpy as np
from scipy.spatial import cKDTree

from line_clearance.evaluate import evaluate
from line_clearance.pipeline import analyse
from line_clearance.simulate import TOWER, WIRE, build_scene


def test_simulation_is_deterministic(cfg, scene):
    again = build_scene(cfg.scene)
    np.testing.assert_array_equal(again.points, scene.points)
    np.testing.assert_array_equal(again.labels, scene.labels)


def test_towers_are_found_where_they_stand(scene, analysis):
    assert len(analysis.towers) == len(scene.towers)
    dist, _ = cKDTree(analysis.towers[:, :2]).query(scene.towers[:, :2])
    assert dist.max() < 0.5


def test_every_wire_gets_a_catenary_with_the_right_sag(cfg, scene, analysis):
    metrics = evaluate(analysis, scene.labels, scene.truth(), cfg.pipeline)["conductors"]
    assert metrics["fitted"] == metrics["true"] == metrics["matched"] == 8
    assert metrics["mean_curve_error_m"] < 0.05
    assert metrics["max_sag_error_m"] < 0.15
    assert sum(c.role == "shield" for c in analysis.conductors) == 2


def test_point_classification_quality(cfg, scene, analysis):
    scores = evaluate(analysis, scene.labels, scene.truth(), cfg.pipeline)["classification"]
    assert scores["ground"]["f1"] > 0.99
    assert scores["conductor"]["f1"] > 0.98
    assert scores["tower"]["f1"] > 0.95
    assert scores["vegetation"]["f1"] > 0.95


def test_encroachments_are_found_and_measured(cfg, scene, analysis):
    metrics = evaluate(analysis, scene.labels, scene.truth(), cfg.pipeline)["clearance"]
    assert analysis.status == "ok"
    assert metrics["critical"]["true"] >= 2
    assert metrics["critical"]["recall"] == 1.0
    assert metrics["action_list"]["recall"] >= 0.85
    assert metrics["action_list"]["precision"] >= 0.85
    assert metrics["clearance_mae_m"] < 0.15


def test_survey_of_the_wrong_place_reports_no_line(cfg, scene):
    terrain_only = ~np.isin(scene.labels, [WIRE, TOWER])
    result = analyse(scene.points[terrain_only], cfg.pipeline)
    assert result.status == "no_line_detected"
    assert result.findings == []


def test_towers_without_wires_yield_gaps_not_clearances(cfg, scene):
    result = analyse(scene.points[scene.labels != WIRE], cfg.pipeline)
    assert result.status == "needs_review"
    assert {f.kind for f in result.findings} == {"coverage_gap"}
    assert len(result.findings) == len(result.spans) == 2


def test_span_with_a_lost_conductor_is_reported_not_measured(cfg, scene):
    """Fail closed: no clearance numbers come out of a span that is not fully modelled."""
    wires = scene.truth()["wires"]
    lost = next(w for w in wires if w["span"] == 0 and w["role"] == "phase")
    origin, direction = np.array(lost["origin_xy"]), np.array(lost["direction_xy"])
    rel = scene.points[:, :2] - origin
    along, across = rel @ direction, rel @ np.array([-direction[1], direction[0]])
    on_wire = (
        (scene.labels == WIRE) & (np.abs(across) < 1.0) & (along > 0) & (along < lost["length"])
    )

    result = analyse(scene.points[~on_wire], cfg.pipeline)

    assert result.status == "needs_review"
    gaps = [f for f in result.findings if f.kind == "coverage_gap"]
    assert len(gaps) == 1
    measured = {f.span_id for f in result.findings if f.kind != "coverage_gap"}
    assert gaps[0].span_id not in measured
    assert measured, "the healthy span is still measured"
