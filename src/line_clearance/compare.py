"""Compare the findings of two surveys of the same corridor."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import cKDTree


def compare_findings(
    base: pl.DataFrame, head: pl.DataFrame, match_radius_m: float = 4.0
) -> pl.DataFrame:
    """Label every grow-in finding as new, resolved or persisting.

    Findings are matched by position because a tree does not move between
    surveys, while span and conductor identifiers can change if a tower is
    added or the flight covers a different extent.
    """
    base = base.filter(pl.col("kind") == "grow_in")
    head = head.filter(pl.col("kind") == "grow_in")
    columns = ["easting", "northing", "severity", "clearance_m"]
    rows: list[dict] = []
    matched: set[int] = set()
    base_xy = base.select("easting", "northing").to_numpy()
    tree = cKDTree(base_xy) if len(base_xy) else None
    base_rows = base.select(columns).to_dicts()

    for row in head.select(columns).to_dicts():
        partner = None
        if tree is not None:
            dist, i = tree.query([row["easting"], row["northing"]])
            if dist <= match_radius_m and int(i) not in matched:
                partner = int(i)
                matched.add(partner)
        before = base_rows[partner] if partner is not None else None
        rows.append(
            {
                "change": "persisting" if before else "new",
                "easting": row["easting"],
                "northing": row["northing"],
                "severity_before": before["severity"] if before else None,
                "severity_after": row["severity"],
                "clearance_before_m": before["clearance_m"] if before else None,
                "clearance_after_m": row["clearance_m"],
            }
        )
    for i, row in enumerate(base_rows):
        if i not in matched:
            rows.append(
                {
                    "change": "resolved",
                    "easting": row["easting"],
                    "northing": row["northing"],
                    "severity_before": row["severity"],
                    "severity_after": None,
                    "clearance_before_m": row["clearance_m"],
                    "clearance_after_m": None,
                }
            )
    schema = {
        "change": pl.String,
        "easting": pl.Float64,
        "northing": pl.Float64,
        "severity_before": pl.String,
        "severity_after": pl.String,
        "clearance_before_m": pl.Float64,
        "clearance_after_m": pl.Float64,
    }
    frame = pl.DataFrame(rows, schema=schema)
    return frame.with_columns(
        (pl.col("clearance_after_m") - pl.col("clearance_before_m")).alias("clearance_change_m")
    )


def summarise(changes: pl.DataFrame) -> dict[str, int | float | None]:
    counts = dict(changes.group_by("change").len().iter_rows())
    persisting = changes.filter(pl.col("change") == "persisting")
    delta = persisting["clearance_change_m"].to_numpy()
    return {
        "new": int(counts.get("new", 0)),
        "resolved": int(counts.get("resolved", 0)),
        "persisting": int(counts.get("persisting", 0)),
        "newly_critical": int(
            changes.filter(
                (pl.col("severity_after") == "critical")
                & (pl.col("severity_before").is_null() | (pl.col("severity_before") != "critical"))
            ).height
        ),
        "median_clearance_change_m": float(np.median(delta)) if len(delta) else None,
    }
