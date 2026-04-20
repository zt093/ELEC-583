"""Structured grid definitions for the 2D axisymmetric and 3D models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AxisymmetricConfig, Cartesian3DConfig


@dataclass(frozen=True)
class AxisymmetricGrid:
    r_edges: np.ndarray
    z_edges: np.ndarray
    r_centers: np.ndarray
    z_centers: np.ndarray
    dr: float
    dz: float
    probe_radius: float
    electrode_height: float
    outer_radius: float
    outer_half_height: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.r_centers.size, self.z_centers.size)

    @property
    def electrode_mask_z(self) -> np.ndarray:
        half_height = 0.5 * self.electrode_height
        return np.abs(self.z_centers) <= half_height


@dataclass(frozen=True)
class CartesianGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    probe_radius: float
    electrode_height: float
    electrode_arc: float
    outer_radius: float
    outer_half_height: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.x.size, self.y.size, self.z.size)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")


def build_axisymmetric_grid(config: AxisymmetricConfig) -> AxisymmetricGrid:
    r_edges = np.linspace(config.probe_radius, config.outer_radius, config.nr + 1)
    z_edges = np.linspace(-config.outer_half_height, config.outer_half_height, config.nz + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dr = r_edges[1] - r_edges[0]
    dz = z_edges[1] - z_edges[0]
    return AxisymmetricGrid(
        r_edges=r_edges,
        z_edges=z_edges,
        r_centers=r_centers,
        z_centers=z_centers,
        dr=dr,
        dz=dz,
        probe_radius=config.probe_radius,
        electrode_height=config.electrode_height,
        outer_radius=config.outer_radius,
        outer_half_height=config.outer_half_height,
    )


def build_cartesian_grid(config: Cartesian3DConfig) -> CartesianGrid:
    x = np.linspace(-config.outer_radius, config.outer_radius, config.nx)
    y = np.linspace(-config.outer_radius, config.outer_radius, config.ny)
    z = np.linspace(-config.outer_half_height, config.outer_half_height, config.nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    return CartesianGrid(
        x=x,
        y=y,
        z=z,
        dx=dx,
        dy=dy,
        dz=dz,
        probe_radius=config.probe_radius,
        electrode_height=config.electrode_height,
        electrode_arc=config.electrode_arc,
        outer_radius=config.outer_radius,
        outer_half_height=config.outer_half_height,
    )
