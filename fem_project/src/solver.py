"""Sparse finite-volume solvers for the axisymmetric lead field and 3D potential model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, spsolve

from .config import AxisymmetricConfig, Cartesian3DConfig
from .geometry import AxisymmetricGrid, CartesianGrid, build_axisymmetric_grid, build_cartesian_grid
from .materials import build_axisymmetric_conductivity, build_cartesian_conductivity
from .sources import PointCurrentSource


@dataclass(frozen=True)
class AxisymmetricLeadFieldResult:
    grid: AxisymmetricGrid
    conductivity: np.ndarray
    potential_voltage_drive: np.ndarray
    lead_field: np.ndarray
    total_current: float
    t_enc: float
    sigma_enc: float


@dataclass(frozen=True)
class CartesianPotentialResult:
    grid: CartesianGrid
    conductivity: np.ndarray
    potential: np.ndarray
    sources: tuple[PointCurrentSource, ...]
    t_enc: float
    sigma_enc: float


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _solve_sparse_system(matrix: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    diagonal = matrix.diagonal().copy()
    diagonal[diagonal == 0.0] = 1.0
    inv_diag = 1.0 / diagonal
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x: inv_diag * x)
    solution, info = cg(matrix, rhs, M=preconditioner, atol=0.0, rtol=1e-8, maxiter=4000)
    if info == 0:
        return solution
    return spsolve(matrix, rhs)


def _axisymmetric_index(i: int, j: int, nz: int) -> int:
    return i * nz + j


def _assemble_axisymmetric_matrix(
    grid: AxisymmetricGrid,
    sigma: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    nr, nz = grid.shape
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nr * nz, dtype=float)

    phi_electrode = 1.0
    electrode_mask = grid.electrode_mask_z

    for i in range(nr):
        r_p = grid.r_centers[i]
        r_w = grid.r_edges[i]
        r_e = grid.r_edges[i + 1]
        for j in range(nz):
            idx = _axisymmetric_index(i, j, nz)
            diag = 0.0

            if i < nr - 1:
                s_e = _harmonic_mean(sigma[i, j], sigma[i + 1, j])
                t_e = s_e * r_e * grid.dz / grid.dr
                diag += t_e
                rows.append(idx)
                cols.append(_axisymmetric_index(i + 1, j, nz))
                data.append(-t_e)
            else:
                radius_far = max(np.hypot(r_e, grid.z_centers[j]), 1e-12)
                t_e = sigma[i, j] * r_e * grid.dz / radius_far
                diag += t_e

            if i > 0:
                s_w = _harmonic_mean(sigma[i, j], sigma[i - 1, j])
                t_w = s_w * r_w * grid.dz / grid.dr
                diag += t_w
                rows.append(idx)
                cols.append(_axisymmetric_index(i - 1, j, nz))
                data.append(-t_w)
            elif electrode_mask[j]:
                t_w = sigma[i, j] * r_w * grid.dz / (0.5 * grid.dr)
                diag += t_w
                rhs[idx] += t_w * phi_electrode

            if j < nz - 1:
                s_n = _harmonic_mean(sigma[i, j], sigma[i, j + 1])
                t_n = s_n * r_p * grid.dr / grid.dz
                diag += t_n
                rows.append(idx)
                cols.append(_axisymmetric_index(i, j + 1, nz))
                data.append(-t_n)
            else:
                radius_far = max(np.hypot(r_p, grid.z_edges[-1]), 1e-12)
                t_n = sigma[i, j] * r_p * grid.dr / radius_far
                diag += t_n

            if j > 0:
                s_s = _harmonic_mean(sigma[i, j], sigma[i, j - 1])
                t_s = s_s * r_p * grid.dr / grid.dz
                diag += t_s
                rows.append(idx)
                cols.append(_axisymmetric_index(i, j - 1, nz))
                data.append(-t_s)
            else:
                radius_far = max(np.hypot(r_p, grid.z_edges[0]), 1e-12)
                t_s = sigma[i, j] * r_p * grid.dr / radius_far
                diag += t_s

            rows.append(idx)
            cols.append(idx)
            data.append(diag)

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(nr * nz, nr * nz)).tocsr()
    return matrix, rhs


def _compute_axisymmetric_total_current(
    grid: AxisymmetricGrid,
    sigma: np.ndarray,
    potential: np.ndarray,
) -> float:
    current_reduced = 0.0
    electrode_mask = grid.electrode_mask_z
    i = 0
    r_w = grid.r_edges[0]
    for j in np.where(electrode_mask)[0]:
        face_conductance = sigma[i, j] * r_w * grid.dz / (0.5 * grid.dr)
        current_reduced += face_conductance * (1.0 - potential[i, j])
    return 2.0 * np.pi * current_reduced


def solve_axisymmetric_lead_field(
    config: AxisymmetricConfig,
    t_enc: float,
    sigma_enc: float,
) -> AxisymmetricLeadFieldResult:
    grid = build_axisymmetric_grid(config)
    sigma = build_axisymmetric_conductivity(
        grid=grid,
        t_enc=t_enc,
        sigma_enc=sigma_enc,
        sigma_brain=config.sigma_brain,
    )
    matrix, rhs = _assemble_axisymmetric_matrix(grid, sigma)
    solution = _solve_sparse_system(matrix, rhs).reshape(grid.shape)
    total_current = _compute_axisymmetric_total_current(grid, sigma, solution)
    lead_field = solution / total_current
    return AxisymmetricLeadFieldResult(
        grid=grid,
        conductivity=sigma,
        potential_voltage_drive=solution,
        lead_field=lead_field,
        total_current=total_current,
        t_enc=t_enc,
        sigma_enc=sigma_enc,
    )


def _cartesian_index(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * ny + j) * nz + k


def _add_source_to_rhs(
    rhs: np.ndarray,
    grid: CartesianGrid,
    source: PointCurrentSource,
) -> None:
    x, y, z = source.x, source.y, source.z
    if not (
        grid.x[0] <= x <= grid.x[-1]
        and grid.y[0] <= y <= grid.y[-1]
        and grid.z[0] <= z <= grid.z[-1]
    ):
        return

    ix = np.clip(np.searchsorted(grid.x, x) - 1, 0, grid.x.size - 2)
    iy = np.clip(np.searchsorted(grid.y, y) - 1, 0, grid.y.size - 2)
    iz = np.clip(np.searchsorted(grid.z, z) - 1, 0, grid.z.size - 2)

    tx = (x - grid.x[ix]) / (grid.x[ix + 1] - grid.x[ix])
    ty = (y - grid.y[iy]) / (grid.y[iy + 1] - grid.y[iy])
    tz = (z - grid.z[iz]) / (grid.z[iz + 1] - grid.z[iz])

    for dx, wx in enumerate((1.0 - tx, tx)):
        for dy, wy in enumerate((1.0 - ty, ty)):
            for dz, wz in enumerate((1.0 - tz, tz)):
                weight = wx * wy * wz
                idx = _cartesian_index(ix + dx, iy + dy, iz + dz, grid.y.size, grid.z.size)
                rhs[idx] += source.current * weight


def _assemble_cartesian_matrix(
    grid: CartesianGrid,
    sigma: np.ndarray,
    sources: tuple[PointCurrentSource, ...],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    nx, ny, nz = grid.shape
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nx * ny * nz, dtype=float)

    for source in sources:
        _add_source_to_rhs(rhs, grid, source)

    ax = grid.dy * grid.dz / grid.dx
    ay = grid.dx * grid.dz / grid.dy
    az = grid.dx * grid.dy / grid.dz

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = _cartesian_index(i, j, k, ny, nz)

                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                    rhs[idx] = 0.0
                    continue

                diag = 0.0

                sxp = _harmonic_mean(sigma[i, j, k], sigma[i + 1, j, k])
                sxm = _harmonic_mean(sigma[i, j, k], sigma[i - 1, j, k])
                syp = _harmonic_mean(sigma[i, j, k], sigma[i, j + 1, k])
                sym = _harmonic_mean(sigma[i, j, k], sigma[i, j - 1, k])
                szp = _harmonic_mean(sigma[i, j, k], sigma[i, j, k + 1])
                szm = _harmonic_mean(sigma[i, j, k], sigma[i, j, k - 1])

                txp = sxp * ax
                txm = sxm * ax
                typ = syp * ay
                tym = sym * ay
                tzp = szp * az
                tzm = szm * az

                diag += txp + txm + typ + tym + tzp + tzm

                for neighbour, value in (
                    ((_cartesian_index(i + 1, j, k, ny, nz)), -txp),
                    ((_cartesian_index(i - 1, j, k, ny, nz)), -txm),
                    ((_cartesian_index(i, j + 1, k, ny, nz)), -typ),
                    ((_cartesian_index(i, j - 1, k, ny, nz)), -tym),
                    ((_cartesian_index(i, j, k + 1, ny, nz)), -tzp),
                    ((_cartesian_index(i, j, k - 1, ny, nz)), -tzm),
                ):
                    rows.append(idx)
                    cols.append(neighbour)
                    data.append(value)

                rows.append(idx)
                cols.append(idx)
                data.append(diag)

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(nx * ny * nz, nx * ny * nz)).tocsr()
    return matrix, rhs


def solve_cartesian_potential(
    config: Cartesian3DConfig,
    t_enc: float,
    sigma_enc: float,
    sources: list[PointCurrentSource],
) -> CartesianPotentialResult:
    grid = build_cartesian_grid(config)
    sigma = build_cartesian_conductivity(
        grid=grid,
        t_enc=t_enc,
        sigma_enc=sigma_enc,
        sigma_brain=config.sigma_brain,
        sigma_probe=config.sigma_probe,
    )
    source_tuple = tuple(sources)
    matrix, rhs = _assemble_cartesian_matrix(grid, sigma, source_tuple)
    solution = _solve_sparse_system(matrix, rhs).reshape(grid.shape)
    return CartesianPotentialResult(
        grid=grid,
        conductivity=sigma,
        potential=solution,
        sources=source_tuple,
        t_enc=t_enc,
        sigma_enc=sigma_enc,
    )
