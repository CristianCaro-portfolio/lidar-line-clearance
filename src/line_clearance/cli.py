"""Command line interface."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import polars as pl
import typer

from . import lakehouse
from .benchmark import run_benchmark
from .compare import compare_findings, summarise
from .config import Config, load_config
from .evaluate import evaluate as score
from .evaluate import flatten
from .io import SurveyError, read_las
from .pipeline import run_survey
from .report import architecture_figure, benchmark_figure, overview_figure
from .simulate import build_scene, load_truth, write_scene

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
ConfigOption = typer.Option(None, "--config", "-c", help="YAML config, defaults apply if omitted")


def _config(path: Path | None, warehouse: Path | None = None) -> Config:
    cfg = load_config(path)
    override = warehouse or os.environ.get("LINECLEAR_WAREHOUSE")
    if override:
        cfg.lakehouse.warehouse = str(override)
    return cfg


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )


@app.command()
def simulate(
    out: Path = typer.Argument(..., help="LAS file to write"),
    config: Path | None = ConfigOption,
    seed: int | None = typer.Option(None, help="override the scene seed"),
    growth_m: float = typer.Option(0.0, help="grow every tree by this much, for a later survey"),
    trimmed: float = typer.Option(0.0, help="share of trees removed, for a later survey"),
) -> None:
    """Generate a synthetic corridor survey with ground truth next to it."""
    cfg = _config(config)
    if seed is not None:
        cfg.scene.seed = seed
    cfg.scene.growth_m, cfg.scene.trimmed_fraction = growth_m, trimmed
    scene = build_scene(cfg.scene)
    write_scene(scene, out)
    typer.echo(
        f"{out}: {len(scene.points):,} points, {len(scene.towers)} towers, {len(scene.trees)} trees"
    )


@app.command()
def run(
    survey: Path = typer.Argument(..., help="LAS or LAZ file"),
    config: Path | None = ConfigOption,
    survey_id: str | None = typer.Option(None, help="defaults to the file name"),
    warehouse: Path | None = typer.Option(None, help="Iceberg warehouse folder"),
    classified_out: Path | None = typer.Option(None, help="write a classified LAS copy"),
    figure: Path | None = typer.Option(None, help="write the overview PNG"),
) -> None:
    """Analyse a survey and write runs, conductors and findings to Iceberg."""
    cfg = _config(config, warehouse)
    try:
        result, run_id = run_survey(survey, cfg, survey_id, classified_out)
    except SurveyError as exc:
        _fail(str(exc))
    if figure is not None:
        overview_figure(read_las(survey), result, cfg.pipeline, figure, survey_id or survey.stem)
    by_severity: dict[str, int] = {}
    for finding in result.findings:
        key = f"{finding.kind}/{finding.severity}"
        by_severity[key] = by_severity.get(key, 0) + 1
    typer.echo(
        json.dumps(
            {
                "survey_id": survey_id or survey.stem,
                "run_id": run_id,
                "status": result.status,
                "issues": result.issues,
                "towers": len(result.towers),
                "spans": len(result.spans),
                "conductors": len(result.conductors),
                "findings": by_severity,
                "seconds": {k: round(v, 2) for k, v in result.timings.items()},
            },
            indent=2,
        )
    )
    if result.status == "no_line_detected":
        raise typer.Exit(code=2)


@app.command()
def evaluate(
    survey: Path = typer.Argument(..., help="simulated LAS file with truth next to it"),
    config: Path | None = ConfigOption,
    out: Path | None = typer.Option(None, help="write the metrics as JSON"),
) -> None:
    """Score the pipeline against the ground truth of a simulated survey."""
    from .pipeline import analyse

    cfg = _config(config)
    labels, truth = load_truth(survey)
    metrics = score(analyse(read_las(survey), cfg.pipeline), labels, truth, cfg.pipeline)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
    for key, value in flatten(metrics).items():
        typer.echo(f"{key:45s} {value:10.4f}")


@app.command()
def findings(
    survey_id: str = typer.Argument(...),
    config: Path | None = ConfigOption,
    warehouse: Path | None = typer.Option(None),
    kind: str = typer.Option("grow_in"),
) -> None:
    """Print the work list of one survey, most urgent first."""
    cfg = _config(config, warehouse)
    tbl = lakehouse.table(lakehouse.open_catalog(cfg.lakehouse), cfg.lakehouse, "findings")
    frame = (
        lakehouse.read(tbl, survey_id)
        .filter(pl.col("kind") == kind)
        .sort("clearance_m")
        .select(
            "finding_id",
            "span_id",
            "conductor_id",
            "severity",
            "clearance_m",
            "design_clearance_m",
            "tree_height_m",
            "easting",
            "northing",
        )
    )
    with pl.Config(tbl_rows=50, tbl_cols=12, float_precision=2):
        typer.echo(str(frame))


@app.command()
def compare(
    base: str = typer.Argument(..., help="earlier survey id"),
    head: str = typer.Argument(..., help="later survey id"),
    config: Path | None = ConfigOption,
    warehouse: Path | None = typer.Option(None),
) -> None:
    """What changed between two surveys of the same corridor."""
    cfg = _config(config, warehouse)
    tbl = lakehouse.table(lakehouse.open_catalog(cfg.lakehouse), cfg.lakehouse, "findings")
    changes = compare_findings(lakehouse.read(tbl, base), lakehouse.read(tbl, head))
    typer.echo(json.dumps(summarise(changes), indent=2))


@app.command()
def history(
    table: str = typer.Argument("findings"),
    config: Path | None = ConfigOption,
    warehouse: Path | None = typer.Option(None),
) -> None:
    """Snapshot log of a table."""
    cfg = _config(config, warehouse)
    tbl = lakehouse.table(lakehouse.open_catalog(cfg.lakehouse), cfg.lakehouse, table)
    typer.echo(str(lakehouse.history(tbl)))


@app.command()
def benchmark(
    config: Path | None = ConfigOption,
    seeds: int = typer.Option(3, help="surveys per condition"),
    out: Path = typer.Option(Path("docs/benchmark.json")),
    figure: Path | None = typer.Option(Path("docs/benchmark.png")),
) -> None:
    """Accuracy and speed across survey conditions, from clean to degraded."""
    rows = run_benchmark(_config(config), seeds, echo=typer.echo)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    if figure is not None:
        benchmark_figure(rows, figure)


@app.command()
def demo(
    workdir: Path = typer.Option(Path("demo"), help="where surveys, warehouse and figures go"),
    config: Path | None = ConfigOption,
) -> None:
    """Two surveys of one corridor, a year apart: analyse, score and compare them."""
    cfg = _config(config, workdir / "warehouse")
    surveys = {"survey_2025": (0.0, 0.0), "survey_2026": (0.6, 0.1)}
    for name, (growth, trimmed) in surveys.items():
        cfg.scene.growth_m, cfg.scene.trimmed_fraction = growth, trimmed
        las = write_scene(build_scene(cfg.scene), workdir / f"{name}.las")
        result, _ = run_survey(las, cfg, name, workdir / f"{name}.classified.las")
        labels, truth = load_truth(las)
        metrics = flatten(score(result, labels, truth, cfg.pipeline))
        overview_figure(read_las(las), result, cfg.pipeline, workdir / f"{name}.png", name)
        typer.echo(
            f"{name}: status={result.status} spans={len(result.spans)} "
            f"critical={len(result.critical)} "
            f"action_recall={metrics['clearance.action_list.recall']:.2f} "
            f"action_precision={metrics['clearance.action_list.precision']:.2f} "
            f"clearance_mae_m={metrics.get('clearance.clearance_mae_m', float('nan')):.3f}"
        )
    tbl = lakehouse.table(lakehouse.open_catalog(cfg.lakehouse), cfg.lakehouse, "findings")
    changes = compare_findings(
        lakehouse.read(tbl, "survey_2025"), lakehouse.read(tbl, "survey_2026")
    )
    typer.echo(f"changes 2025 -> 2026: {json.dumps(summarise(changes))}")


@app.command()
def diagram(out: Path = typer.Argument(Path("docs/architecture.png"))) -> None:
    """Render the architecture diagram."""
    architecture_figure(out)
    typer.echo(str(out))


if __name__ == "__main__":
    app()
