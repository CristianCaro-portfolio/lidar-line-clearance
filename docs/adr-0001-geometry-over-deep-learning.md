# ADR 0001: Geometry and physics instead of a learned point classifier

Status: accepted

## Context

The pipeline has to separate ground, vegetation, towers and wires in a point cloud and then
measure distances that a utility will act on. The usual research answer is a point
segmentation network (RandLA-Net, KPConv, Point Transformer). Those need labelled corridors
from the same sensor, a GPU to train and usually to run, and they return a label per point
with no statement about why.

## Decision

Classification is done with geometry and one physical model:

- ground comes from a progressive morphological filter on the lowest return per cell
- wire candidates are points whose neighbourhood is a horizontal line (covariance
  eigenvalues, linearity above 0.9)
- a tower is a tall filled column that wires run through and kink upwards at
- every conductor is a catenary, fitted with robust least squares, and a point is a wire
  point if it lies on a fitted catenary

## Why

- It runs on one CPU core at 70k to 145k points per second in the benchmark and needs no
  labels.
- The catenary is not an approximation of the wire, it is the wire. Once it is fitted the
  line can be re-sagged for a hotter conductor or swung for wind, which a per point label
  cannot do. The design condition findings exist only because of this choice.
- Every decision has a number an engineer can check: linearity, fit error, coverage. That
  is what makes the fail closed quality gates possible.

## Trade-offs

- Thresholds are tuned for single circuit lines with lattice towers. Bundled conductors,
  double circuits or wooden poles would need new rules, where a network would need new data.
- Vegetation touching a tower is absorbed by the tower mask.
- A learned classifier would likely do better on cluttered scenes (buildings, crossings).
  The clean upgrade path is to swap the candidate step and keep the catenary and clearance
  stages as they are.
