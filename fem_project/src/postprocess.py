"""Recording metrics, stage-aware electrode sampling, and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AxisymmetricConfig, Cartesian3DConfig, MechanismSweepConfig
from .electrodes import ElectrodeLayout, ElectrodeSite
from .solver import AxisymmetricLeadFieldResult, CartesianPotentialResult, solve_axisymmetric_lead_field


@dataclass(frozen=True)
class SweepResults:
    thicknesses: np.ndarray
    sigma_values: np.ndarray
    heatmap: np.ndarray
    mechanism_distance: np.ndarray
    mechanism_conductivity: np.ndarray
    mechanism_combined: np.ndarray


def _bilinear_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    x: float,
    y: float,
) -> float:
    ix = np.clip(np.searchsorted(x_grid, x) - 1, 0, x_grid.size - 2)
    iy = np.clip(np.searchsorted(y_grid, y) - 1, 0, y_grid.size - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    f00 = field[ix, iy]
    f10 = field[ix + 1, iy]
    f01 = field[ix, iy + 1]
    f11 = field[ix + 1, iy + 1]
    return float(
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def evaluate_axisymmetric_recording(
    result: AxisymmetricLeadFieldResult,
    radial_gap: float,
    source_current: float,
    z_source: float = 0.0,
) -> float:
    source_radius = result.grid.probe_radius + radial_gap
    lead_value = _bilinear_sample(
        result.grid.r_centers,
        result.grid.z_centers,
        result.lead_field,
        source_radius,
        z_source,
    )
    return source_current * lead_value


def mechanism_source_gap(
    baseline_gap: float,
    displacement_alpha: float,
    t_enc: float,
) -> float:
    return baseline_gap + displacement_alpha * t_enc


def generate_axisymmetric_sweep(
    config: AxisymmetricConfig,
    sweep_config: MechanismSweepConfig,
) -> SweepResults:
    heatmap = np.zeros((sweep_config.thicknesses.size, sweep_config.sigma_enc_values.size), dtype=float)
    mech_1 = np.zeros(sweep_config.thicknesses.size, dtype=float)
    mech_2 = np.zeros(sweep_config.thicknesses.size, dtype=float)
    combined = np.zeros(sweep_config.thicknesses.size, dtype=float)
    cache: dict[tuple[float, float], object] = {}

    homogeneous_sigma = config.sigma_brain

    def get_result(thickness: float, sigma_enc: float) -> AxisymmetricLeadFieldResult:
        key = (float(thickness), float(sigma_enc))
        if key not in cache:
            cache[key] = solve_axisymmetric_lead_field(
                config=config,
                t_enc=thickness,
                sigma_enc=sigma_enc,
            )
        return cache[key]  # type: ignore[return-value]

    for i, thickness in enumerate(sweep_config.thicknesses):
        distance_only_result = get_result(thickness, homogeneous_sigma)
        mech_1[i] = evaluate_axisymmetric_recording(
            distance_only_result,
            radial_gap=mechanism_source_gap(config.baseline_gap, config.displacement_alpha, thickness),
            source_current=config.source_current,
            z_source=config.z_source,
        )

        conductivity_only_result = get_result(thickness, sweep_config.mechanism_curve_sigma)
        mech_2[i] = evaluate_axisymmetric_recording(
            conductivity_only_result,
            radial_gap=config.baseline_gap,
            source_current=config.source_current,
            z_source=config.z_source,
        )
        combined[i] = evaluate_axisymmetric_recording(
            conductivity_only_result,
            radial_gap=mechanism_source_gap(config.baseline_gap, config.displacement_alpha, thickness),
            source_current=config.source_current,
            z_source=config.z_source,
        )

        for j, sigma_enc in enumerate(sweep_config.sigma_enc_values):
            result = get_result(thickness, sigma_enc)
            heatmap[i, j] = evaluate_axisymmetric_recording(
                result,
                radial_gap=mechanism_source_gap(config.baseline_gap, config.displacement_alpha, thickness),
                source_current=config.source_current,
                z_source=config.z_source,
            )

    return SweepResults(
        thicknesses=sweep_config.thicknesses,
        sigma_values=sweep_config.sigma_enc_values,
        heatmap=heatmap,
        mechanism_distance=mech_1,
        mechanism_conductivity=mech_2,
        mechanism_combined=combined,
    )


def homogeneous_analytic_potential(
    source_current: float,
    conductivity: float,
    distance: np.ndarray,
) -> np.ndarray:
    return source_current / (4.0 * np.pi * conductivity * distance)


def validate_homogeneous_trend(
    config: AxisymmetricConfig,
    radial_gaps: np.ndarray,
) -> dict[str, np.ndarray]:
    result = solve_axisymmetric_lead_field(config=config, t_enc=0.0, sigma_enc=config.sigma_brain)
    numerical = np.array(
        [
            evaluate_axisymmetric_recording(
                result=result,
                radial_gap=gap,
                source_current=config.source_current,
                z_source=config.z_source,
            )
            for gap in radial_gaps
        ]
    )
    analytic = homogeneous_analytic_potential(
        source_current=config.source_current,
        conductivity=config.sigma_brain,
        distance=radial_gaps,
    )
    return {"radial_gaps": radial_gaps, "numerical": numerical, "analytic": analytic}


def validate_domain_size(
    config: AxisymmetricConfig,
    radii: np.ndarray,
) -> dict[str, np.ndarray]:
    values = []
    base_dr = (config.outer_radius - config.probe_radius) / config.nr
    base_dz = 2.0 * config.outer_half_height / config.nz
    aspect = config.outer_half_height / config.outer_radius
    for outer_radius in radii:
        outer_half_height = float(max(config.outer_half_height, aspect * outer_radius))
        nr = max(48, int(np.ceil((outer_radius - config.probe_radius) / base_dr)))
        nz = max(72, int(np.ceil((2.0 * outer_half_height) / base_dz)))
        scaled = AxisymmetricConfig(
            sigma_brain=config.sigma_brain,
            sigma_probe=config.sigma_probe,
            probe_radius=config.probe_radius,
            electrode_height=config.electrode_height,
            outer_radius=float(outer_radius),
            outer_half_height=outer_half_height,
            nr=nr,
            nz=nz,
            baseline_gap=config.baseline_gap,
            displacement_alpha=config.displacement_alpha,
            source_current=config.source_current,
            default_sigma_enc=config.default_sigma_enc,
            default_t_enc=config.default_t_enc,
            z_source=config.z_source,
        )
        result = solve_axisymmetric_lead_field(
            config=scaled,
            t_enc=config.default_t_enc,
            sigma_enc=config.default_sigma_enc,
        )
        values.append(
            evaluate_axisymmetric_recording(
                result=result,
                radial_gap=mechanism_source_gap(
                    config.baseline_gap,
                    config.displacement_alpha,
                    config.default_t_enc,
                ),
                source_current=config.source_current,
                z_source=config.z_source,
            )
        )
    return {"outer_radii": radii, "phi_elec": np.asarray(values)}


def validate_mesh_convergence(
    config: AxisymmetricConfig,
    resolutions: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    values = []
    labels = []
    for nr, nz in resolutions:
        refined = AxisymmetricConfig(
            sigma_brain=config.sigma_brain,
            sigma_probe=config.sigma_probe,
            probe_radius=config.probe_radius,
            electrode_height=config.electrode_height,
            outer_radius=config.outer_radius,
            outer_half_height=config.outer_half_height,
            nr=nr,
            nz=nz,
            baseline_gap=config.baseline_gap,
            displacement_alpha=config.displacement_alpha,
            source_current=config.source_current,
            default_sigma_enc=config.default_sigma_enc,
            default_t_enc=config.default_t_enc,
            z_source=config.z_source,
        )
        result = solve_axisymmetric_lead_field(
            config=refined,
            t_enc=config.default_t_enc,
            sigma_enc=config.default_sigma_enc,
        )
        values.append(
            evaluate_axisymmetric_recording(
                result=result,
                radial_gap=mechanism_source_gap(
                    refined.baseline_gap,
                    refined.displacement_alpha,
                    refined.default_t_enc,
                ),
                source_current=refined.source_current,
                z_source=refined.z_source,
            )
        )
        labels.append(f"{nr}x{nz}")
    return {"labels": np.asarray(labels), "phi_elec": np.asarray(values)}


def _trilinear_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    field: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    ix = np.clip(np.searchsorted(x_grid, x) - 1, 0, x_grid.size - 2)
    iy = np.clip(np.searchsorted(y_grid, y) - 1, 0, y_grid.size - 2)
    iz = np.clip(np.searchsorted(z_grid, z) - 1, 0, z_grid.size - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    z0, z1 = z_grid[iz], z_grid[iz + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    tz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)

    value = 0.0
    for dx, wx in enumerate((1.0 - tx, tx)):
        for dy, wy in enumerate((1.0 - ty, ty)):
            for dz, wz in enumerate((1.0 - tz, tz)):
                value += (
                    wx
                    * wy
                    * wz
                    * field[ix + dx, iy + dy, iz + dz]
                )
    return float(value)


def surface_sample_points(
    config: Cartesian3DConfig,
    n_theta: int = 25,
    n_z: int = 25,
) -> np.ndarray:
    theta = np.linspace(-0.5 * config.electrode_arc, 0.5 * config.electrode_arc, n_theta)
    z_values = np.linspace(-0.5 * config.electrode_height, 0.5 * config.electrode_height, n_z)
    points = []
    for angle in theta:
        x = config.probe_radius * np.cos(angle)
        y = config.probe_radius * np.sin(angle)
        for z in z_values:
            points.append([x, y, z])
    return np.asarray(points, dtype=float)


def surface_sample_points_for_site(
    config: Cartesian3DConfig,
    site: ElectrodeSite,
    n_theta: int = 25,
    n_z: int = 25,
) -> np.ndarray:
    theta = np.linspace(site.azimuth - 0.5 * site.arc, site.azimuth + 0.5 * site.arc, n_theta)
    z_values = np.linspace(site.z_center - 0.5 * site.height, site.z_center + 0.5 * site.height, n_z)
    points = []
    for angle in theta:
        x = config.probe_radius * np.cos(angle)
        y = config.probe_radius * np.sin(angle)
        for z in z_values:
            points.append([x, y, z])
    return np.asarray(points, dtype=float)


def sample_surface_average(
    result: CartesianPotentialResult,
    config: Cartesian3DConfig,
    n_theta: int = 25,
    n_z: int = 25,
) -> float:
    samples = surface_sample_points(config=config, n_theta=n_theta, n_z=n_z)
    values = np.array(
        [
            _trilinear_sample(
                result.grid.x,
                result.grid.y,
                result.grid.z,
                result.potential,
                point[0],
                point[1],
                point[2],
            )
            for point in samples
        ]
    )
    return float(np.mean(values))


def sample_site_average(
    result: CartesianPotentialResult,
    config: Cartesian3DConfig,
    site: ElectrodeSite,
    n_theta: int = 21,
    n_z: int = 21,
) -> float:
    samples = surface_sample_points_for_site(config=config, site=site, n_theta=n_theta, n_z=n_z)
    values = np.array(
        [
            _trilinear_sample(
                result.grid.x,
                result.grid.y,
                result.grid.z,
                result.potential,
                point[0],
                point[1],
                point[2],
            )
            for point in samples
        ]
    )
    return float(np.mean(values))


def sample_layout_recordings(
    result: CartesianPotentialResult,
    config: Cartesian3DConfig,
    layout: ElectrodeLayout,
    n_theta: int = 21,
    n_z: int = 21,
) -> dict[str, float]:
    return {
        site.name: sample_site_average(
            result=result,
            config=config,
            site=site,
            n_theta=n_theta,
            n_z=n_z,
        )
        for site in layout.sites
    }
