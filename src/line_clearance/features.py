"""Open3D helpers: neighbourhood shape features, outlier removal, clustering."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    return cloud


def shape_features(xyz: np.ndarray, radius: float, max_nn: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearity and verticality of the neighbourhood around each point.

    Linearity is (l1 - l2) / l1 over the eigenvalues of the local covariance.
    It is close to one on a wire and low inside a tree crown. Verticality is
    the z component of the main direction, which separates wires from trunks
    and tower legs.
    """
    if len(xyz) == 0:
        return np.zeros(0), np.zeros(0)
    cloud = _cloud(xyz)
    cloud.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    values, vectors = np.linalg.eigh(np.asarray(cloud.covariances))
    largest = np.maximum(values[:, 2], 1e-12)
    linearity = (values[:, 2] - values[:, 1]) / largest
    # isolated points come back with an identity covariance
    linearity[np.isclose(values[:, 0], values[:, 2])] = 0.0
    return linearity, np.abs(vectors[:, 2, 2])


def isolated_points(xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """True for returns with almost nothing around them (birds, dust)."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    _, kept = _cloud(xyz).remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    isolated = np.ones(len(xyz), dtype=bool)
    isolated[np.asarray(kept, dtype=np.int64)] = False
    return isolated


def dbscan(xyz: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Cluster label per point, -1 for noise."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(_cloud(xyz).cluster_dbscan(eps=eps, min_points=min_points), dtype=np.int64)
