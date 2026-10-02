"""LAS input and output."""

from __future__ import annotations

import hashlib
from pathlib import Path

import laspy
import numpy as np


class SurveyError(ValueError):
    """The input file cannot be used as a corridor survey."""


def read_las(path: str | Path) -> np.ndarray:
    """Return the points of a LAS or LAZ file as a float64 (n, 3) array."""
    path = Path(path)
    if not path.exists():
        raise SurveyError(f"survey file not found: {path}")
    try:
        las = laspy.read(path)
    except Exception as exc:  # laspy raises several unrelated types on bad files
        raise SurveyError(f"cannot read {path.name}: {exc}") from exc
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    if len(xyz) == 0:
        raise SurveyError(f"{path.name} has no points")
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        raise SurveyError(f"{path.name} has {int((~finite).sum())} points with invalid coordinates")
    return xyz


def write_las(
    path: str | Path,
    xyz: np.ndarray,
    classification: np.ndarray | None = None,
    crs: str | None = None,
) -> Path:
    """Write points as LAS 1.4 with millimetre resolution."""
    path = Path(path)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0))
    if crs:
        header.system_identifier = crs[:32]
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    las.write(path)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
