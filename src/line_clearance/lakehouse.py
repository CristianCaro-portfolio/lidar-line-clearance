"""Iceberg tables for runs, conductors and findings.

The catalog is a SQLite file and the warehouse a local folder, so the whole
lakehouse runs embedded. Swapping to a REST catalog and object storage only
changes ``open_catalog``.

Geometry is stored as WKB in a binary column on a format version 2 table.
Iceberg v3 has native geometry types and PyIceberg 0.12 can read them, but it
cannot write v3 metadata yet, so v2 plus WKB is what works from Python today.
See docs/adr-0003-iceberg-geometry.md.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.expressions import EqualTo
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BinaryType,
    DoubleType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)

from .config import LakehouseConfig


def _schema(*fields: tuple[str, object, bool]) -> Schema:
    return Schema(
        *[
            NestedField(i, name, kind, required=required)
            for i, (name, kind, required) in enumerate(fields, start=1)
        ]
    )


S, D, L, B, T = StringType(), DoubleType(), LongType(), BinaryType(), TimestamptzType()

SCHEMAS: dict[str, Schema] = {
    "runs": _schema(
        ("survey_id", S, True),
        ("run_id", S, True),
        ("input_file", S, False),
        ("input_sha256", S, False),
        ("config_hash", S, False),
        ("code_version", S, False),
        ("crs", S, False),
        ("n_points", L, False),
        ("points_per_m2", D, False),
        ("n_towers", L, False),
        ("n_spans", L, False),
        ("n_conductors", L, False),
        ("n_findings", L, False),
        ("n_critical", L, False),
        ("status", S, False),
        ("issues", S, False),
        ("duration_s", D, False),
        ("started_at", T, False),
    ),
    "conductors": _schema(
        ("survey_id", S, True),
        ("run_id", S, True),
        ("span_id", S, True),
        ("conductor_id", S, True),
        ("role", S, False),
        ("status", S, False),
        ("issues", S, False),
        ("length_m", D, False),
        ("sag_m", D, False),
        ("design_sag_m", D, False),
        ("catenary_c", D, False),
        ("rmse_m", D, False),
        ("n_points", L, False),
        ("coverage", D, False),
        ("geom", B, False),
    ),
    "findings": _schema(
        ("survey_id", S, True),
        ("run_id", S, True),
        ("finding_id", S, True),
        ("span_id", S, False),
        ("conductor_id", S, False),
        ("kind", S, False),
        ("severity", S, False),
        ("clearance_m", D, False),
        ("design_clearance_m", D, False),
        ("tree_height_m", D, False),
        ("n_points", L, False),
        ("easting", D, False),
        ("northing", D, False),
        ("elevation", D, False),
        ("geom", B, False),
        ("detected_at", T, False),
    ),
}

BY_SURVEY = PartitionSpec(
    PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="survey_id")
)


def open_catalog(cfg: LakehouseConfig) -> Catalog:
    root = Path(cfg.warehouse).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "local", uri=f"sqlite:///{root / 'catalog.db'}", warehouse=f"file://{root}"
    )
    try:
        catalog.create_namespace(cfg.namespace)
    except NamespaceAlreadyExistsError:
        pass
    return catalog


def table(catalog: Catalog, cfg: LakehouseConfig, name: str) -> Table:
    identifier = f"{cfg.namespace}.{name}"
    try:
        return catalog.load_table(identifier)
    except NoSuchTableError:
        return catalog.create_table(
            identifier,
            schema=SCHEMAS[name],
            partition_spec=BY_SURVEY,
            properties={"format-version": "2", "geo.column": "geom", "geo.encoding": "WKB"},
        )


def replace_survey(tbl: Table, frame: pl.DataFrame, survey_id: str) -> None:
    """Atomically replace every row of one survey.

    Running the same survey twice leaves one copy of its rows and two
    snapshots, so reruns are safe and the history stays auditable.
    """
    target = tbl.schema().as_arrow()
    mine = EqualTo("survey_id", survey_id)
    exists = tbl.scan(row_filter=mine, limit=1).to_arrow().num_rows > 0
    if frame.is_empty():
        if exists:
            tbl.delete(mine)
        return
    data = frame.select([f.name for f in target]).to_arrow()
    data = pa.Table.from_arrays(
        [data[f.name].cast(f.type) for f in target],
        schema=pa.schema([pa.field(f.name, f.type, f.nullable) for f in target]),
    )
    if exists:
        tbl.overwrite(data, overwrite_filter=mine)
    else:
        tbl.append(data)


def read(tbl: Table, survey_id: str | None = None) -> pl.DataFrame:
    scan = pl.scan_iceberg(tbl)
    if survey_id is not None:
        scan = scan.filter(pl.col("survey_id") == survey_id)
    return scan.collect()


def history(tbl: Table) -> pl.DataFrame:
    """Snapshot log of a table, the audit trail of who wrote what and when."""
    rows = [
        {
            "snapshot_id": s.snapshot_id,
            "committed_at_ms": s.timestamp_ms,
            "operation": s.summary.operation.value if s.summary else None,
            "added_records": int((s.summary or {}).get("added-records", 0) or 0),
            "deleted_records": int((s.summary or {}).get("deleted-records", 0) or 0),
        }
        for s in tbl.snapshots()
    ]
    return pl.DataFrame(rows)
