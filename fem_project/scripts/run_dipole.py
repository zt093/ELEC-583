"""Run the Level 2 3D dipole model and export the main visualization set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Cartesian3DConfig, OutputConfig
from src.plotting import plot_3d_clipped_view, plot_3d_geometry, plot_3d_hero, plot_3d_slice_view, save_figure
from src.postprocess import sample_surface_average
from src.solver import solve_cartesian_potential
from src.sources import DipoleSource


def main() -> None:
    output = OutputConfig()
    presentation_dir = output.figures_dir / "presentation"
    validation_dir = output.figures_dir / "validation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output.data_dir.mkdir(parents=True, exist_ok=True)

    config = Cartesian3DConfig()
    dipole = DipoleSource(
        center=np.array([config.probe_radius + config.dipole_center_radius, 0.0, config.dipole_center_z]),
        orientation=np.array([1.0, 0.0, 1.0]),
        separation=config.dipole_separation,
        current=config.source_current,
    )
    result = solve_cartesian_potential(
        config=config,
        t_enc=config.default_t_enc,
        sigma_enc=config.default_sigma_enc,
        sources=dipole.point_sources(),
    )
    phi_elec = sample_surface_average(result=result, config=config)

    np.savez(
        output.data_dir / "dipole_baseline.npz",
        x=result.grid.x,
        y=result.grid.y,
        z=result.grid.z,
        conductivity=result.conductivity,
        potential=result.potential,
        phi_elec=phi_elec,
    )

    save_figure(
        plot_3d_geometry(config=config, t_enc=config.default_t_enc),
        presentation_dir / "dipole_geometry_3d.png",
    )
    save_figure(
        plot_3d_slice_view(result=result, config=config),
        presentation_dir / "dipole_slice_view.png",
    )
    save_figure(
        plot_3d_clipped_view(result=result, config=config),
        presentation_dir / "dipole_clipped_view.png",
    )
    save_figure(
        plot_3d_hero(result=result, config=config),
        presentation_dir / "dipole_hero_figure.png",
    )

    print(f"Saved dipole outputs to {output.project_root / 'outputs'}")
    print(f"phi_elec = {phi_elec:.6e} V for dipole current magnitude {config.source_current:.2e} A")


if __name__ == "__main__":
    main()
