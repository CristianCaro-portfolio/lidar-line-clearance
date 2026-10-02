# ADR 0003: Results in Iceberg, geometry as WKB on format version 2

Status: accepted

## Context

A clearance finding is a compliance record. A utility has to show what was measured, on
which survey, with which settings, and what changed since the last flight. Results also
have to be queryable by position.

Iceberg format version 3 adds native `geometry` and `geography` types. Iceberg 1.12.0
(end of September 2026) maps them to Parquet logical types, and PyIceberg 0.12.0
(3 September 2026) added the types to the Python library.

## What I found

PyIceberg 0.12.0 can declare a `GeometryType` column and read it, but it cannot create or
commit a v3 table. `create_table` with `format-version: 3` stops with
`NotImplementedError: Writing V3 is not yet supported`. So from Python, native geometry is
read only today.

## Decision

- Three tables: `runs`, `conductors`, `findings`, partitioned by `survey_id`.
- Format version 2. Geometry is a `binary` column holding WKB (PointZ for findings,
  LineStringZ for conductors) in the survey CRS, with `geo.column` and `geo.encoding` set as
  table properties. Easting, northing and elevation are also stored as plain doubles so
  range filters work without decoding.
- The catalog is SQLite and the warehouse a local folder, so the lakehouse is embedded and
  needs no service. `open_catalog` is the only function that changes for a REST catalog.
- Writing a survey replaces its partition in one atomic commit. Rerunning the same file
  with the same config gives the same `run_id` and the same rows, and the replace shows up
  in the history as a delete snapshot plus an append snapshot.
- Polars reads the tables with `scan_iceberg`.

## Why Iceberg and not files or PostGIS

- Plain GeoParquet files have no atomic replace and no history. Two reruns leave duplicates
  or a half written folder.
- PostGIS has the spatial functions but is a server, and the point of the lakehouse is that
  the same tables are readable by Spark, Trino, DuckDB or Snowflake without an export.
- Snapshots give the audit trail for free: `lineclear history findings` lists every commit.

## Migration path

When PyIceberg can write v3: add a `geometry('EPSG:32618')` column, backfill it from the
WKB column and drop the binary one. The bytes are the same WKB, so readers of the new column
see the same values and bounds based pruning becomes available.
