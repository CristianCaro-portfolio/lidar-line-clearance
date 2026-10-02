"""Synthetic drone LiDAR survey of a transmission corridor.

The generator exists so the pipeline can be scored against exact ground truth:
every point carries a class label and every tree and wire is known analytically.
Labels and truth are written next to the LAS file, never inside it, so the
pipeline cannot read them by accident.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .catenary import Catenary, Conductor
from .config import SceneConfig
from .io import write_las

GROUND, LOW_VEG, HIGH_VEG, NOISE, WIRE, TOWER = 2, 3, 5, 7, 14, 15


@dataclass
class Tree:
    tree_id: int
    x: float
    y: float
    base_z: float
    height: float
    crown_radius: float
    crown_base: float

    def crown_surface(self, n: int = 1500) -> np.ndarray:
        """Deterministic dense sample of the crown, used as noise free truth."""
        i = np.arange(n) + 0.5
        polar = np.arccos(1.0 - 2.0 * i / n)
        azimuth = np.pi * (1.0 + 5.0**0.5) * i
        half = (self.height - self.crown_base) / 2.0
        centre_z = self.base_z + self.crown_base + half
        return np.column_stack(
            [
                self.x + self.crown_radius * np.sin(polar) * np.cos(azimuth),
                self.y + self.crown_radius * np.sin(polar) * np.sin(azimuth),
                centre_z + half * np.cos(polar),
            ]
        )


@dataclass
class Scene:
    points: np.ndarray
    labels: np.ndarray
    towers: np.ndarray
    wires: list[dict]
    trees: list[Tree]
    crs: str

    def truth(self) -> dict:
        return {"crs": self.crs, "towers": self.towers, "wires": self.wires, "trees": self.trees}


def _terrain(rng: np.random.Generator):
    """Smooth rolling terrain built from a few long waves."""
    waves = [
        (
            rng.uniform(1.5, 4.0),
            rng.uniform(180.0, 520.0),
            rng.uniform(0, np.pi),
            rng.uniform(0, 6.28),
        )
        for _ in range(4)
    ]
    tilt = rng.uniform(-0.02, 0.02, size=2)

    def height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = 100.0 + tilt[0] * x + tilt[1] * y
        for amp, wavelength, heading, phase in waves:
            z = z + amp * np.sin(
                2 * np.pi * (x * np.cos(heading) + y * np.sin(heading)) / wavelength + phase
            )
        return z

    return height


def _tower_points(rng, base, lateral, cfg: SceneConfig) -> np.ndarray:
    """Lattice tower: four tapered legs, bracing rings, a cross arm and insulators."""
    height = cfg.tower_height_m
    along = np.array([lateral[1], -lateral[0]])
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            t = rng.uniform(0, 1, 600)
            half = 3.0 - 2.2 * t
            xy = base[:2] + np.outer(sx * half, lateral) + np.outer(sy * half, along)
            parts.append(np.column_stack([xy, base[2] + t * height]))
    for level in np.arange(4.0, height, 4.0):
        half = 3.0 - 2.2 * level / height
        ring = rng.uniform(-1, 1, (160, 2)) * half
        edge = rng.integers(0, 4, 160)
        ring[edge == 0, 0], ring[edge == 1, 0] = -half, half
        ring[edge == 2, 1], ring[edge == 3, 1] = -half, half
        xy = base[:2] + np.outer(ring[:, 0], lateral) + np.outer(ring[:, 1], along)
        parts.append(np.column_stack([xy, np.full(160, base[2] + level)]))
    arm_z = base[2] + cfg.attach_height_m + 2.0
    reach = max(abs(o) for o in cfg.phase_offsets_m) + 1.0
    arm = rng.uniform(-reach, reach, 500)
    parts.append(
        np.column_stack([base[:2] + np.outer(arm, lateral), arm_z + rng.uniform(-0.3, 0.3, 500)])
    )
    for offset in cfg.phase_offsets_m:
        drop = rng.uniform(0, 2.0, 60)
        xy = base[:2] + offset * lateral
        parts.append(np.column_stack([np.tile(xy, (60, 1)), arm_z - drop]))
    return np.vstack(parts)


def _wire_points(rng, conductor: Conductor, cfg: SceneConfig) -> np.ndarray:
    n = int(conductor.length * cfg.wire_density_pts_m)
    l = rng.uniform(0, conductor.length, n)
    for _ in range(cfg.wire_dropouts_per_wire):
        start = rng.uniform(0.1, 0.85) * conductor.length
        l = l[(l < start) | (l > start + rng.uniform(1.0, 5.0))]
    ux, uy = conductor.direction_xy
    return np.column_stack(
        [
            conductor.origin_xy[0] + ux * l,
            conductor.origin_xy[1] + uy * l,
            conductor.catenary.z(l),
        ]
    )


def _tree_points(rng, tree: Tree, density: float) -> np.ndarray:
    n = max(int(density * np.pi * tree.crown_radius**2), 40)
    direction = rng.normal(size=(n * 2, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # LiDAR sees the top of the crown much better than its underside
    keep = (direction[:, 2] > 0) | (rng.uniform(size=n * 2) < 0.35)
    direction = direction[keep][:n]
    shell = rng.uniform(0.7, 1.0, len(direction)) ** (1 / 3)
    half = (tree.height - tree.crown_base) / 2.0
    crown = np.column_stack(
        [
            tree.x + tree.crown_radius * shell * direction[:, 0],
            tree.y + tree.crown_radius * shell * direction[:, 1],
            tree.base_z + tree.crown_base + half + half * shell * direction[:, 2],
        ]
    )
    trunk_n = int(4 * tree.crown_base)
    trunk = np.column_stack(
        [
            tree.x + rng.normal(0, 0.12, trunk_n),
            tree.y + rng.normal(0, 0.12, trunk_n),
            tree.base_z + rng.uniform(0.2, tree.crown_base, trunk_n),
        ]
    )
    return np.vstack([crown, trunk])


def build_scene(cfg: SceneConfig) -> Scene:
    """Generate the corridor. The same seed always yields the same survey."""
    rng = np.random.default_rng(cfg.seed)
    terrain = _terrain(rng)

    # --- tower line with gentle bends
    heading = rng.uniform(0, 2 * np.pi)
    xy = [np.zeros(2)]
    headings = []
    for _ in range(cfg.n_spans):
        heading += np.deg2rad(rng.uniform(-cfg.max_bend_deg, cfg.max_bend_deg))
        headings.append(heading)
        length = rng.uniform(*cfg.span_length_m)
        xy.append(xy[-1] + length * np.array([np.cos(heading), np.sin(heading)]))
    xy = np.array(xy)
    tower_z = terrain(xy[:, 0], xy[:, 1])
    towers = np.column_stack([xy, tower_z])

    # the cross arm is perpendicular to the bisector of the two spans it carries
    laterals = []
    for i in range(len(xy)):
        prev_h = headings[max(i - 1, 0)]
        next_h = headings[min(i, len(headings) - 1)]
        mid = (prev_h + next_h) / 2.0
        laterals.append(np.array([-np.sin(mid), np.cos(mid)]))

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    def add(points: np.ndarray, label: int) -> None:
        chunks.append(points)
        labels.append(np.full(len(points), label, dtype=np.uint8))

    for i in range(len(xy)):
        add(_tower_points(rng, towers[i], laterals[i], cfg), TOWER)

    # --- wires
    wires: list[dict] = []
    offsets = [(o, cfg.attach_height_m, "phase") for o in cfg.phase_offsets_m]
    if cfg.shield_wire:
        offsets.append((0.0, cfg.tower_height_m, "shield"))
    for span in range(cfg.n_spans):
        for k, (offset, attach, role) in enumerate(offsets):
            a = np.append(xy[span] + offset * laterals[span], tower_z[span] + attach)
            b = np.append(xy[span + 1] + offset * laterals[span + 1], tower_z[span + 1] + attach)
            length = float(np.linalg.norm(b[:2] - a[:2]))
            ratio = rng.uniform(*cfg.sag_ratio) * (0.6 if role == "shield" else 1.0)
            cat = Catenary.from_endpoints(length, a[2], b[2], length / (8.0 * ratio))
            direction = (b[:2] - a[:2]) / length
            conductor = Conductor((a[0], a[1]), (direction[0], direction[1]), length, cat)
            add(_wire_points(rng, conductor, cfg), WIRE)
            wires.append(
                {
                    "span": span,
                    "index": k,
                    "role": role,
                    "origin_xy": [float(a[0]), float(a[1])],
                    "direction_xy": [float(direction[0]), float(direction[1])],
                    "length": length,
                    "c": cat.c,
                    "l0": cat.l0,
                    "z0": cat.z0,
                    "sag": cat.sag(length),
                }
            )

    # --- trees
    trees: list[Tree] = []
    half_width = cfg.corridor_half_width_m

    def plant(x: float, y: float, height: float) -> None:
        height = float(np.clip(height, 3.0, 27.0))
        trees.append(
            Tree(
                tree_id=len(trees),
                x=float(x),
                y=float(y),
                base_z=float(terrain(x, y)),
                height=height,
                crown_radius=float(np.clip(rng.uniform(0.16, 0.24) * height, 1.2, 4.5)),
                crown_base=float(rng.uniform(0.3, 0.45) * height),
            )
        )

    phases = [w for w in wires if w["role"] == "phase"]
    for span in range(cfg.n_spans):
        a, b = xy[span], xy[span + 1]
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        for _ in range(rng.poisson(cfg.trees_per_100m * length / 100.0)):
            s = rng.uniform(15.0, length - 15.0)
            d = rng.choice([-1, 1]) * rng.uniform(9.0, half_width - 2.0)
            tall = rng.uniform() < 0.2
            pos = a + s * u + d * n
            plant(pos[0], pos[1], rng.uniform(15.0, 26.0) if tall else rng.uniform(4.0, 13.0))
        # trees placed near a phase so their top ends up close to the wire
        span_phases = [w for w in phases if w["span"] == span]
        for _ in range(rng.poisson(cfg.encroaching_trees_per_span)):
            w = span_phases[rng.integers(len(span_phases))]
            l = rng.uniform(0.2, 0.8) * w["length"]
            side = rng.uniform(-5.0, 5.0)
            wu = np.array(w["direction_xy"])
            pos = np.array(w["origin_xy"]) + l * wu + side * np.array([-wu[1], wu[0]])
            wire_z = float(Catenary(w["c"], w["l0"], w["z0"]).z(l))
            gap = rng.uniform(1.2, 7.0)
            plant(pos[0], pos[1], wire_z - gap - float(terrain(pos[0], pos[1])))

    if cfg.trimmed_fraction > 0:
        keep = rng.uniform(size=len(trees)) >= cfg.trimmed_fraction
        trees = [t for t, k in zip(trees, keep, strict=True) if k]
    for t in trees:
        t.height += cfg.growth_m
    for t in trees:
        add(_tree_points(rng, t, cfg.vegetation_density_pts_m2), HIGH_VEG)

    # --- ground and shrubs
    tree_xy = np.array([[t.x, t.y] for t in trees]) if trees else np.zeros((0, 2))
    tree_r = np.array([t.crown_radius for t in trees])
    for span in range(cfg.n_spans):
        a, b = xy[span], xy[span + 1]
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        count = int(cfg.ground_density_pts_m2 * length * 2 * half_width)
        s = rng.uniform(-8.0, length + 8.0, count)
        d = rng.uniform(-half_width, half_width, count)
        pts = a + np.outer(s, u) + np.outer(d, n)
        # canopy hides most of the ground underneath it
        if len(trees):
            near = np.abs(tree_xy - (a + b) / 2).max(axis=1) < length
            shaded = np.zeros(count, dtype=bool)
            for (tx, ty), radius in zip(tree_xy[near], tree_r[near], strict=True):
                shaded |= (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2 < radius**2
            pts = pts[~shaded | (rng.uniform(size=count) < 0.35)]
        z = terrain(pts[:, 0], pts[:, 1]) + rng.normal(0, 0.02, len(pts))
        add(np.column_stack([pts, z]), GROUND)

        for _ in range(rng.poisson(cfg.shrubs_per_100m * length / 100.0)):
            pos = a + rng.uniform(5, length - 5) * u + rng.uniform(-half_width, half_width) * n
            radius, top = rng.uniform(0.5, 1.5), rng.uniform(0.6, 2.0)
            m = int(cfg.vegetation_density_pts_m2 * np.pi * radius**2)
            r = radius * np.sqrt(rng.uniform(size=m))
            ang = rng.uniform(0, 2 * np.pi, m)
            sx, sy = pos[0] + r * np.cos(ang), pos[1] + r * np.sin(ang)
            sz = terrain(sx, sy) + top * np.sqrt(1 - (r / radius) ** 2) * rng.uniform(0.3, 1.0, m)
            add(np.column_stack([sx, sy, sz]), LOW_VEG)

    points = np.vstack(chunks)
    label = np.concatenate(labels)
    points = points + rng.normal(0, cfg.noise_sigma_m, points.shape)

    # --- stray returns: birds, dust, multipath
    stray = int(cfg.outlier_fraction * len(points))
    if stray:
        picks = points[rng.integers(0, len(points), stray)].copy()
        picks[:, :2] += rng.uniform(-10, 10, (stray, 2))
        picks[:, 2] = terrain(picks[:, 0], picks[:, 1]) + rng.uniform(4.0, 45.0, stray)
        points = np.vstack([points, picks])
        label = np.concatenate([label, np.full(stray, NOISE, dtype=np.uint8)])

    order = rng.permutation(len(points))
    points, label = points[order], label[order]
    shift = np.array([cfg.origin_easting, cfg.origin_northing, 0.0])
    points = points + shift
    towers = towers + shift
    for w in wires:
        w["origin_xy"] = [w["origin_xy"][0] + shift[0], w["origin_xy"][1] + shift[1]]
    for t in trees:
        t.x += shift[0]
        t.y += shift[1]
    return Scene(points, label, towers, wires, trees, cfg.crs)


def write_scene(scene: Scene, las_path: str | Path) -> Path:
    """Write ``survey.las`` plus ``survey.truth.npz`` and ``survey.truth.json``."""
    las_path = Path(las_path)
    las_path.parent.mkdir(parents=True, exist_ok=True)
    write_las(las_path, scene.points, crs=scene.crs)
    stem = las_path.with_suffix("")
    np.savez_compressed(f"{stem}.truth.npz", labels=scene.labels)
    truth = scene.truth() | {
        "towers": scene.towers.tolist(),
        "trees": [asdict(t) for t in scene.trees],
    }
    Path(f"{stem}.truth.json").write_text(json.dumps(truth, indent=1))
    return las_path


def load_truth(las_path: str | Path) -> tuple[np.ndarray, dict]:
    stem = Path(las_path).with_suffix("")
    labels = np.load(f"{stem}.truth.npz")["labels"]
    truth = json.loads(Path(f"{stem}.truth.json").read_text())
    truth["trees"] = [Tree(**t) for t in truth["trees"]]
    return labels, truth
