import json

from typer.testing import CliRunner

from line_clearance.cli import app

runner = CliRunner()


def test_simulate_then_run_writes_results(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "scene:\n  seed: 11\n  n_spans: 2\n  span_length_m: [150.0, 180.0]\n"
        "  ground_density_pts_m2: 8.0\n  vegetation_density_pts_m2: 30.0\n"
        "  wire_density_pts_m: 8.0\n"
    )
    las = tmp_path / "s.las"
    assert runner.invoke(app, ["simulate", str(las), "-c", str(config)]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "run",
            str(las),
            "-c",
            str(config),
            "--warehouse",
            str(tmp_path / "wh"),
            "--figure",
            str(tmp_path / "s.png"),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["spans"] == 2 and summary["conductors"] == 8
    assert (tmp_path / "s.png").stat().st_size > 10_000


def test_missing_file_exits_with_an_error(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.las")])
    assert result.exit_code == 1


def test_warehouse_can_come_from_the_environment(tmp_path, monkeypatch):
    from line_clearance.cli import _config

    monkeypatch.setenv("LINECLEAR_WAREHOUSE", str(tmp_path / "from_env"))
    assert _config(None).lakehouse.warehouse == str(tmp_path / "from_env")
    assert _config(None, tmp_path / "flag").lakehouse.warehouse == str(tmp_path / "flag")
