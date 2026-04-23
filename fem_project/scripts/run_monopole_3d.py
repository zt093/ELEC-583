"""Run the localized 3D monopole model and export ParaView-ready outputs."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Cartesian3DConfig, OutputConfig
from src.export import export_cartesian_result_to_vti, export_point_sources_to_vtp
from src.postprocess import mechanism_source_gap, sample_surface_average
from src.solver import solve_cartesian_potential
from src.sources import MonopoleSource


def main() -> None:
    output = OutputConfig()
    output.data_dir.mkdir(parents=True, exist_ok=True)
    output.paraview_dir.mkdir(parents=True, exist_ok=True)

    previous_config = replace(Cartesian3DConfig(), nx=61, ny=61, nz=101)
    config = replace(previous_config, nx=71, ny=71, nz=121)
    source = MonopoleSource(
        radial_gap=mechanism_source_gap(
            baseline_gap=config.monopole_baseline_gap,
            displacement_alpha=config.monopole_displacement_alpha,
            t_enc=config.default_t_enc,
        ),
        azimuth=config.monopole_azimuth,
        z=config.monopole_z,
        current=config.source_current,
    )
    point_source = source.point_source(config.probe_radius)
    previous_point_source = source.point_source(previous_config.probe_radius)

    previous_result = solve_cartesian_potential(
        config=previous_config,
        t_enc=previous_config.default_t_enc,
        sigma_enc=previous_config.default_sigma_enc,
        sources=[previous_point_source],
    )
    previous_phi_elec = sample_surface_average(result=previous_result, config=previous_config)

    result = solve_cartesian_potential(
        config=config,
        t_enc=config.default_t_enc,
        sigma_enc=config.default_sigma_enc,
        sources=[point_source],
    )
    phi_elec = sample_surface_average(result=result, config=config)

    np.savez(
        output.data_dir / "monopole_3d_baseline.npz",
        x=result.grid.x,
        y=result.grid.y,
        z=result.grid.z,
        conductivity=result.conductivity,
        potential=result.potential,
        phi_elec=phi_elec,
        previous_phi_elec=previous_phi_elec,
        relative_change_vs_previous=(phi_elec - previous_phi_elec) / previous_phi_elec,
        grid_shape=np.array([config.nx, config.ny, config.nz], dtype=int),
        previous_grid_shape=np.array([previous_config.nx, previous_config.ny, previous_config.nz], dtype=int),
        grid_spacing_um=np.array([result.grid.dx, result.grid.dy, result.grid.dz]) * 1e6,
        previous_grid_spacing_um=np.array(
            [
                2.0 * previous_config.outer_radius / (previous_config.nx - 1),
                2.0 * previous_config.outer_radius / (previous_config.ny - 1),
                2.0 * previous_config.outer_half_height / (previous_config.nz - 1),
            ]
        )
        * 1e6,
        source_xyz=np.array([point_source.x, point_source.y, point_source.z]),
    )
    export_cartesian_result_to_vti(
        output.paraview_dir / "monopole_3d_localized_baseline.vti",
        result=result,
        config=config,
    )
    export_point_sources_to_vtp(
        output.paraview_dir / "monopole_3d_source_points.vtp",
        result=result,
    )

    print(f"Saved 3D monopole outputs to {output.project_root / 'outputs'}")
    print(f"phi_elec = {phi_elec:.6e} V for monopole current {config.source_current:.2e} A")
    print(
        f"previous reference phi_elec = {previous_phi_elec:.6e} V "
        f"on {previous_config.nx}x{previous_config.ny}x{previous_config.nz}"
    )
    print(
        f"refined monopole grid = {config.nx}x{config.ny}x{config.nz} "
        f"with spacing [{result.grid.dx * 1e6:.2f}, {result.grid.dy * 1e6:.2f}, {result.grid.dz * 1e6:.2f}] um"
    )
    print(f"relative change in phi_elec vs previous = {(phi_elec - previous_phi_elec) / previous_phi_elec:+.2%}")
    print(
        "Source position [um] = "
        f"({point_source.x * 1e6:.1f}, {point_source.y * 1e6:.1f}, {point_source.z * 1e6:.1f})"
    )


if __name__ == "__main__":
    main()
