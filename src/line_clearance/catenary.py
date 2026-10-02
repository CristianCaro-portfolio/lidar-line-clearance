"""Catenary geometry: the shape a conductor takes between two supports.

A conductor lives in a vertical plane. Inside that plane we use ``l`` for the
horizontal distance from the first support and ``z`` for elevation:

    z(l) = z0 + c * (cosh((l - l0) / c) - 1)

``c`` is the catenary constant (horizontal tension over weight per metre),
``l0`` the position of the lowest point and ``z0`` its elevation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Catenary:
    c: float
    l0: float
    z0: float

    def z(self, l: np.ndarray | float) -> np.ndarray:
        return self.z0 + self.c * (np.cosh((np.asarray(l, dtype=float) - self.l0) / self.c) - 1.0)

    @classmethod
    def from_endpoints(cls, length: float, z_a: float, z_b: float, c: float) -> Catenary:
        """Catenary with constant ``c`` hanging from (0, z_a) and (length, z_b)."""
        half = length / (2.0 * c)
        l0 = length / 2.0 - c * np.arcsinh((z_b - z_a) / (2.0 * c * np.sinh(half)))
        z0 = z_a - c * (np.cosh(l0 / c) - 1.0)
        return cls(float(c), float(l0), float(z0))

    def sag(self, length: float) -> float:
        """Largest vertical distance between the chord and the curve."""
        z_a, z_b = float(self.z(0.0)), float(self.z(length))
        slope = (z_b - z_a) / length
        l_star = float(np.clip(self.l0 + self.c * np.arcsinh(slope), 0.0, length))
        return float(z_a + slope * l_star - self.z(l_star))


def fit_catenary(l: np.ndarray, z: np.ndarray) -> tuple[Catenary, float]:
    """Robust least squares fit. Raises ValueError when the points do not hang."""
    l = np.asarray(l, dtype=float)
    z = np.asarray(z, dtype=float)
    if l.size < 5 or np.ptp(l) < 1.0:
        raise ValueError("not enough points along the span to fit a catenary")

    # A parabola is the second order expansion of the catenary and gives a
    # closed form starting point for the non linear solve.
    a, b, c0 = np.polyfit(l, z, 2)
    if a <= 1e-7:
        raise ValueError("points are not convex, this is not a hanging wire")
    start = np.array([1.0 / (2.0 * a), -b / (2.0 * a), c0 - b * b / (4.0 * a)])

    def residuals(p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * (np.cosh((l - p[1]) / p[0]) - 1.0) - z

    solution = least_squares(
        residuals,
        start,
        loss="soft_l1",
        f_scale=0.1,
        bounds=([10.0, -np.inf, -np.inf], [1e6, np.inf, np.inf]),
    )
    model = Catenary(*map(float, solution.x))
    rmse = float(np.sqrt(np.mean((model.z(l) - z) ** 2)))
    return model, rmse


@dataclass(frozen=True)
class Conductor:
    """A wire between two supports, placed in world coordinates."""

    origin_xy: tuple[float, float]
    direction_xy: tuple[float, float]
    length: float
    catenary: Catenary

    @property
    def sag(self) -> float:
        return self.catenary.sag(self.length)

    def sample(
        self, step: float = 0.25, sag_factor: float = 1.0, blowout_deg: float = 0.0
    ) -> np.ndarray:
        """Points along the wire as an (n, 3) array.

        ``sag_factor`` models a hotter, longer conductor by scaling the sag while
        keeping the supports fixed. ``blowout_deg`` swings the wire sideways
        around the chord, which is what wind does to it.
        """
        n = max(int(np.ceil(self.length / step)) + 1, 2)
        l = np.linspace(0.0, self.length, n)
        cat = self.catenary
        z_a, z_b = float(cat.z(0.0)), float(cat.z(self.length))
        if sag_factor != 1.0:
            cat = Catenary.from_endpoints(self.length, z_a, z_b, cat.c / sag_factor)
        chord = z_a + (z_b - z_a) * l / self.length
        drop = chord - cat.z(l)

        phi = np.deg2rad(blowout_deg)
        ux, uy = self.direction_xy
        lateral = drop * np.sin(phi)
        x = self.origin_xy[0] + ux * l - uy * lateral
        y = self.origin_xy[1] + uy * l + ux * lateral
        return np.column_stack([x, y, chord - drop * np.cos(phi)])

    def envelope(self, step: float, sag_factor: float, blowout_deg: float) -> np.ndarray:
        """Positions the wire can reach under the design condition."""
        if sag_factor == 1.0 and blowout_deg == 0.0:
            return self.sample(step)
        angles = np.linspace(-blowout_deg, blowout_deg, 5) if blowout_deg else [0.0]
        return np.vstack([self.sample(step, sag_factor, float(a)) for a in angles])
