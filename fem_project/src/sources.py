"""Source-model definitions for monopoles, dipoles, and placeholder distributed currents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCurrentSource:
    x: float
    y: float
    z: float
    current: float


@dataclass(frozen=True)
class MonopoleSource:
    radial_gap: float
    azimuth: float = 0.0
    z: float = 0.0
    current: float = 1e-9

    def radial_position(self, probe_radius: float) -> float:
        return probe_radius + self.radial_gap

    def point_source(self, probe_radius: float) -> PointCurrentSource:
        radius = self.radial_position(probe_radius)
        return PointCurrentSource(
            x=float(radius * np.cos(self.azimuth)),
            y=float(radius * np.sin(self.azimuth)),
            z=float(self.z),
            current=float(self.current),
        )


@dataclass(frozen=True)
class DipoleSource:
    center: np.ndarray
    orientation: np.ndarray
    separation: float
    current: float

    def point_sources(self) -> list[PointCurrentSource]:
        direction = np.asarray(self.orientation, dtype=float)
        direction = direction / np.linalg.norm(direction)
        offset = 0.5 * self.separation * direction
        p_plus = self.center + offset
        p_minus = self.center - offset
        return [
            PointCurrentSource(*p_plus, self.current),
            PointCurrentSource(*p_minus, -self.current),
        ]


def morphology_placeholder(
    center: np.ndarray,
    scale: float,
    current: float,
) -> list[PointCurrentSource]:
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.2],
            [1.4, 0.2, 0.6],
            [1.9, -0.4, 1.0],
            [1.2, 0.8, 1.1],
            [0.4, 0.3, -0.6],
            [1.0, -0.7, -1.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, -0.35, 0.2, -0.15, 0.15, -0.45, -0.4], dtype=float)
    weights -= weights.mean()
    weights /= np.sum(np.abs(weights))
    points = center + scale * offsets
    return [
        PointCurrentSource(float(x), float(y), float(z), float(current * weight))
        for (x, y, z), weight in zip(points, weights)
    ]
