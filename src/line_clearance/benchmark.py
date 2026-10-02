"""Accuracy and throughput across survey conditions.

Each condition changes what the sensor delivers, not the pipeline settings,
so the numbers show how the same configuration degrades as data gets worse.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .config import Config
from .evaluate import evaluate, flatten
from .pipeline import analyse
from .simulate import build_scene

CONDITIONS: dict[str, dict] = {
    "baseline": {},
    "noisy sensor (8 cm)": {"noise_sigma_m": 0.08},
    "sparse (4 pts/m wire)": {
        "wire_density_pts_m": 4.0,
        "ground_density_pts_m2": 8.0,
        "vegetation_density_pts_m2": 20.0,
    },
    "very sparse (1.5 pts/m wire)": {
        "wire_density_pts_m": 1.5,
        "ground_density_pts_m2": 4.0,
        "vegetation_density_pts_m2": 10.0,
    },
    "sharp bends (15 deg)": {"max_bend_deg": 15.0},
    "dense vegetation": {"trees_per_100m": 22.0, "encroaching_trees_per_span": 4.0},
}

COUNTED = ("clearance.critical", "clearance.action_list", "clearance.fall_in")
SUMMED = (".tp", ".true", ".predicted", ".matched", ".detected", ".fitted", ".false")


def run_benchmark(cfg: Config, seeds: int, echo: Callable[[str], None] = print) -> list[dict]:
    rows = []
    for name, overrides in CONDITIONS.items():
        runs, points, seconds, review = [], 0, 0.0, 0
        for seed in range(seeds):
            scene_cfg = cfg.scene.model_copy(update={"seed": 100 + seed, **overrides})
            scene = build_scene(scene_cfg)
            tick = time.perf_counter()
            result = analyse(scene.points, cfg.pipeline)
            seconds += time.perf_counter() - tick
            points += len(scene.points)
            review += result.status != "ok"
            runs.append(flatten(evaluate(result, scene.labels, scene.truth(), cfg.pipeline)))

        row: dict = {
            "condition": name,
            "surveys": seeds,
            "needs_review": review,
            "points": points,
            "points_per_second": points / seconds,
        }
        keys = sorted({k for r in runs for k in r})
        for key in keys:
            values = [r[key] for r in runs if key in r]
            row[key] = float(np.sum(values) if key.endswith(SUMMED) else np.mean(values))
        # detection metrics are pooled over all trees, not averaged per survey
        for prefix in COUNTED:
            tp, pred, true = (row.get(f"{prefix}.{k}", 0.0) for k in ("tp", "predicted", "true"))
            row[f"{prefix}.precision"] = tp / pred if pred else 1.0
            row[f"{prefix}.recall"] = tp / true if true else 1.0
        rows.append(row)
        echo(
            f"{name:30s} conductor_f1={row['classification.conductor.f1']:.3f} "
            f"towers={row['towers.matched']:.0f}/{row['towers.true']:.0f} "
            f"action P={row['clearance.action_list.precision']:.2f} "
            f"R={row['clearance.action_list.recall']:.2f} "
            f"critical P={row['clearance.critical.precision']:.2f} "
            f"R={row['clearance.critical.recall']:.2f} "
            f"mae={row.get('clearance.clearance_mae_m', float('nan')):.3f} "
            f"review={review}/{seeds} {row['points_per_second'] / 1e3:.0f}k pts/s"
        )
    return rows
