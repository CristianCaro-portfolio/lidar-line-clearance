# ADR 0002: Open3D for neighbourhood work, laspy for files, SciPy for distances

Status: accepted

## Context

Three kinds of operations dominate the run time: neighbourhood statistics on millions of
points, clustering, and nearest distance queries from vegetation to wires. Surveys arrive as
LAS or LAZ.

## Decision

- **laspy** reads and writes LAS. The classified copy of the survey is written back with
  ASPRS class codes (2 ground, 3 to 5 vegetation, 7 noise, 14 wire conductor, 15 tower), so
  the result opens in any LiDAR viewer.
- **Open3D** does covariance estimation with a hybrid radius and k search, radius outlier
  removal and DBSCAN. All three run in C++ over a KD-tree.
- **SciPy** `cKDTree` answers distance queries against the sampled catenaries and
  `least_squares` with a soft L1 loss fits them. `ndimage` does the raster morphology of the
  ground filter.

## Alternatives considered

- **PDAL** has ready made ground (SMRF) and feature filters and is what I would use to tile
  and reproject a real survey. It is a system dependency with its own pipeline language, and
  the stages here need to exchange arrays with the fitting code, so plain Python was simpler
  to test.
- **scikit-learn** DBSCAN and neighbours would work, but they hand neighbour lists back to
  Python and the covariance would then be computed point by point in numpy. Open3D does the
  search and the covariance in one C++ call.
- **PCL** is the C++ reference. The Python bindings are not maintained.

## Consequences

- Open3D is the heaviest dependency and the one that decides which Python versions work
  (3.10 to 3.13 with Open3D 0.20). On Linux the CPU only wheel is used.
- Distances are measured on raw returns, not on voxel averages. Averaging pulls the crown
  surface inwards and reports more clearance than there is.
- Everything is in memory. A long corridor has to be tiled per span before this scales, see
  the limitations in the README.
