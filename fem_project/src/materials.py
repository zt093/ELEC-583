"""Conductivity-field builders for brain tissue, probe, and encapsulation regions."""

from __future__ import annotations

import numpy as np

from .geometry import AxisymmetricGrid, CartesianGrid


def build_axisymmetric_conductivity(
    grid: AxisymmetricGrid,
    t_enc: float,
    sigma_enc: float,
    sigma_brain: float,
) -> np.ndarray:
    sigma = np.full(grid.shape, sigma_brain, dtype=float)
    if t_enc <= 0.0:
        return sigma

    r_shell = grid.probe_radius + t_enc
    sigma[grid.r_centers <= r_shell, :] = sigma_enc
    return sigma


def build_cartesian_conductivity(
    grid: CartesianGrid,
    t_enc: float,
    sigma_enc: float,
    sigma_brain: float,
    sigma_probe: float,
) -> np.ndarray:
    xg, yg, _ = grid.mesh()
    rho = np.sqrt(xg**2 + yg**2)
    sigma = np.full(grid.shape, sigma_brain, dtype=float)
    sigma[rho <= grid.probe_radius] = sigma_probe
    if t_enc > 0.0:
        shell_mask = (rho > grid.probe_radius) & (rho <= grid.probe_radius + t_enc)
        sigma[shell_mask] = sigma_enc
    return sigma
