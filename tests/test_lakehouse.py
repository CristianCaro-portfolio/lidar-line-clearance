import polars as pl
import pytest
from shapely import from_wkb

from line_clearance import lakehouse
from line_clearance.compare import compare_findings, summarise
from line_clearance.pipeline import run_survey
from line_clearance.simulate import build_scene, write_scene


@pytest.fixture(scope="module")
def store(cfg, scene, tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    config = cfg.model_copy(deep=True)
    config.lakehouse.warehouse = str(root / "warehouse")
    las = write_scene(scene, root / "survey_a.las")
    return config, las, root


def _table(config, name):
    return lakehouse.table(lakehouse.open_catalog(config.lakehouse), config.lakehouse, name)


def test_rerun_replaces_rows_and_keeps_history(store):
    config, las, _ = store
    first, run_a = run_survey(las, config)
    _, run_b = run_survey(las, config)
    assert run_a == run_b, "same input and config give the same run id"

    findings = lakehouse.read(_table(config, "findings"), "survey_a")
    assert findings.height == len(first.findings)
    assert findings["finding_id"].n_unique() == findings.height
    assert lakehouse.read(_table(config, "runs")).height == 1
    assert lakehouse.history(_table(config, "findings")).height >= 2


def test_geometry_is_valid_wkb_in_survey_coordinates(store):
    config, las, _ = store
    run_survey(las, config)
    conductors = lakehouse.read(_table(config, "conductors"), "survey_a")
    line = from_wkb(conductors["geom"][0])
    assert line.geom_type == "LineString" and line.has_z
    assert line.length == pytest.approx(conductors["length_m"][0], rel=0.01)

    findings = lakehouse.read(_table(config, "findings"), "survey_a")
    point = from_wkb(findings["geom"][0])
    assert point.x == pytest.approx(findings["easting"][0])
    assert 590_000 < point.x < 610_000


def test_second_survey_shows_growth(store):
    config, las, root = store
    run_survey(las, config)
    later = config.scene.model_copy(update={"growth_m": 1.0})
    run_survey(write_scene(build_scene(later), root / "survey_b.las"), config)

    tbl = _table(config, "findings")
    changes = compare_findings(lakehouse.read(tbl, "survey_a"), lakehouse.read(tbl, "survey_b"))
    summary = summarise(changes)
    assert summary["persisting"] > 0
    assert summary["median_clearance_change_m"] < 0
    assert lakehouse.read(tbl)["survey_id"].n_unique() == 2


def test_compare_labels_new_resolved_and_persisting():
    def frame(rows):
        return pl.DataFrame(
            rows, schema=["easting", "northing", "severity", "clearance_m"], orient="row"
        ).with_columns(pl.lit("grow_in").alias("kind"))

    base = frame([(0.0, 0.0, "watch", 4.5), (100.0, 0.0, "critical", 2.0)])
    head = frame([(0.5, 0.5, "critical", 2.8), (200.0, 0.0, "design", 3.5)])
    changes = compare_findings(base, head)
    assert sorted(changes["change"].to_list()) == ["new", "persisting", "resolved"]
    assert summarise(changes) == {
        "new": 1,
        "resolved": 1,
        "persisting": 1,
        "newly_critical": 1,
        "median_clearance_change_m": pytest.approx(-1.7),
    }
