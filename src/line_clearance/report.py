"""Figures: corridor overview, benchmark chart and architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .config import PipelineConfig  # noqa: E402
from .pipeline import GROUND, HIGH_VEG, MED_VEG, TOWER, WIRE, Analysis  # noqa: E402

SURFACE, INK, INK_SOFT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEVERITY = {
    "critical": ("#d03b3b", "X", "critical: inside the limit today"),
    "design": ("#ec835a", "D", "design: inside the limit at max sag and wind"),
    "watch": ("#fab219", "o", "watch: inside the warning distance"),
}
CLASS_COLOR = {
    GROUND: "#c9c4b6",
    MED_VEG: "#7fbf9b",
    HIGH_VEG: "#3d9a6b",
    TOWER: "#6f6d68",
    WIRE: INK,
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def overview_figure(
    xyz: np.ndarray, result: Analysis, cfg: PipelineConfig, out: str | Path, survey_id: str = ""
) -> Path:
    """Profile of the worst span on top, plan view of the corridor below."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    grow = [f for f in result.findings if f.kind == "grow_in"]
    fig = plt.figure(figsize=(12, 7.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.38)
    top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    if result.spans:
        worst = min(grow, key=lambda f: f.clearance_m).span_id if grow else result.spans[0].span_id
        span = next(s for s in result.spans if s.span_id == worst)
        s, d = span.to_local(xyz[:, :2])
        near = np.flatnonzero((s > -8) & (s < span.length + 8) & (np.abs(d) < 14))
        near = rng.choice(near, min(len(near), 60_000), replace=False)
        for code, colour in CLASS_COLOR.items():
            pick = near[result.classification[near] == code]
            top.scatter(
                s[pick],
                xyz[pick, 2],
                s=1.2 if code != WIRE else 2.0,
                c=colour,
                linewidths=0,
                rasterized=True,
            )
        first = True
        for conductor in (c for c in result.conductors if c.span_id == worst and c.role == "phase"):
            flown = conductor.model.sample(1.0)
            hot = conductor.model.sample(1.0, cfg.clearance.design_sag_factor)
            top.plot(
                span.to_local(flown[:, :2])[0],
                flown[:, 2],
                color=BLUE,
                linewidth=2,
                label="fitted catenary, as flown" if first else None,
            )
            top.plot(
                span.to_local(hot[:, :2])[0],
                hot[:, 2],
                color=ORANGE,
                linewidth=2,
                linestyle=(0, (4, 2)),
                label="design sag" if first else None,
            )
            first = False
        for f in (f for f in grow if f.span_id == worst):
            colour, marker, _ = SEVERITY[f.severity]
            fs = span.to_local(np.array([[f.x, f.y]]))[0][0]
            top.scatter(
                fs, f.z, s=90, c=colour, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=5
            )
            if f.severity == "critical":
                top.annotate(
                    f"{f.clearance_m:.1f} m",
                    (fs, f.z),
                    xytext=(0, 11),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK,
                )
        _style(
            top,
            f"Span {worst} profile: fitted conductors and vegetation findings",
            "distance along span (m)",
            "elevation (m)",
        )
        top.legend(loc="lower right", ncol=2, fontsize=8.5, frameon=False, labelcolor=INK_SOFT)

    # plan view in line coordinates, so a diagonal corridor still fills the panel
    sample = rng.choice(len(xyz), min(len(xyz), 150_000), replace=False)
    if len(result.towers) >= 2:
        start, end = result.spans[0].start[:2], result.spans[-1].end[:2]
        axis = (end - start) / np.linalg.norm(end - start)
    else:
        start, axis = xyz[:, :2].min(axis=0), np.array([1.0, 0.0])
    normal = np.array([-axis[1], axis[0]])

    def line_coords(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = np.atleast_2d(xy) - start
        return rel @ axis, rel @ normal

    for code, colour in CLASS_COLOR.items():
        pick = sample[result.classification[sample] == code]
        along, across = line_coords(xyz[pick, :2])
        bottom.scatter(along, across, s=0.5, c=colour, linewidths=0, rasterized=True)
    for severity, (colour, marker, label) in SEVERITY.items():
        hits = [f for f in grow if f.severity == severity]
        along, across = line_coords(np.array([[f.x, f.y] for f in hits]).reshape(-1, 2))
        bottom.scatter(
            along,
            across,
            s=70,
            c=colour,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=5,
            label=f"{label} ({len(hits)})",
        )
    _style(
        bottom,
        "Corridor plan view: classified points and grow-in findings",
        "distance along the line (m)",
        "offset from line (m)",
    )
    bottom.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        labelcolor=INK_SOFT,
    )

    phases = sum(c.role == "phase" for c in result.conductors)
    fig.suptitle(
        f"Vegetation clearance from drone LiDAR   {survey_id}   "
        f"{len(xyz) / 1e6:.1f} M points, {len(result.spans)} spans, {phases} phase conductors",
        x=0.07,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def benchmark_figure(rows: list[dict], out: str | Path) -> Path:
    """One small panel per metric, one bar per survey condition."""
    out = Path(out)
    panels = [
        ("classification.conductor.f1", "Conductor point F1"),
        ("clearance.action_list.recall", "Action list recall"),
        ("clearance.action_list.precision", "Action list precision"),
        ("clearance.clearance_mae_m", "Clearance error (m, mean abs)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.6), facecolor=SURFACE)
    names = [r["condition"] for r in rows]
    for ax, (key, title) in zip(axes, panels, strict=True):
        values = [r.get(key, 0.0) or 0.0 for r in rows]
        bars = ax.barh(names, values, color=BLUE, height=0.55)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=INK_SOFT)
        _style(ax, title, "", "")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(values) * 1.18, 0.1))
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def architecture_figure(out: str | Path) -> Path:
    """Left to right diagram of the pipeline stages."""
    out = Path(out)
    stages = [
        ("LAS survey", "laspy\ndrone LiDAR tile"),
        ("Ground model", "morphological filter\nheight above ground"),
        ("Shape features", "Open3D covariance\nlinearity, verticality"),
        ("Towers and spans", "column occupancy\nwire kink test"),
        ("Catenary fit", "DBSCAN pieces\nrobust least squares"),
        ("Clearance", "as flown and design\ngrow-in, fall-in"),
        ("Iceberg tables", "runs, conductors,\nfindings with WKB"),
    ]
    pitch, width = 2.15, 1.85
    fig, ax = plt.subplots(figsize=(15.5, 3.9), facecolor=SURFACE)
    ax.set_xlim(0, len(stages) * pitch)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, colour: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#ffffff",
                edgecolor=colour,
                linewidth=1.4,
            )
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=12, color=MUTED, linewidth=1.2
            )
        )

    for i, (title, detail) in enumerate(stages):
        x = i * pitch + 0.12
        box(x, 1.55, width, 1.4, BLUE)
        ax.text(
            x + width / 2,
            2.55,
            title,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
        ax.text(x + width / 2, 1.98, detail, ha="center", va="center", fontsize=8.5, color=INK_SOFT)
        if i < len(stages) - 1:
            arrow((x + width + 0.04, 2.25), (x + pitch - 0.04, 2.25))

    gates_x, gates_w = pitch + 0.12, 4 * pitch + width
    box(gates_x, 0.3, gates_w, 0.8, ORANGE)
    ax.text(
        gates_x + gates_w / 2,
        0.7,
        "Quality gates: point density, wire coverage, fit error, conductor count.\n"
        "A span that cannot be measured is reported for review, never passed as clear.",
        ha="center",
        va="center",
        fontsize=9,
        color=INK_SOFT,
    )
    last_x = (len(stages) - 1) * pitch + 0.12
    box(last_x, 0.3, width, 0.8, BLUE)
    ax.text(
        last_x + width / 2,
        0.7,
        "Polars: work list,\nsurvey to survey diff",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK_SOFT,
    )
    arrow((last_x + width / 2, 1.5), (last_x + width / 2, 1.16))
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
