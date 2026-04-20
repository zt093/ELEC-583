"""Run Level 1 parameter sweeps and baseline validation studies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AxisymmetricConfig, MechanismSweepConfig, OutputConfig, um
from src.plotting import (
    plot_heatmap,
    plot_mechanism_curves,
    plot_validation_series,
    save_figure,
)
from src.postprocess import (
    generate_axisymmetric_sweep,
    validate_domain_size,
    validate_mesh_convergence,
)


def main() -> None:
    output = OutputConfig()
    presentation_dir = output.figures_dir / "presentation"
    validation_dir = output.figures_dir / "validation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output.data_dir.mkdir(parents=True, exist_ok=True)

    config = AxisymmetricConfig()
    sweep_config = MechanismSweepConfig()
    sweep = generate_axisymmetric_sweep(config=config, sweep_config=sweep_config)

    domain_validation = validate_domain_size(
        config=config,
        radii=np.array([um(500.0), um(700.0), um(900.0), um(1100.0)]),
    )
    mesh_validation = validate_mesh_convergence(
        config=config,
        resolutions=[(90, 140), (120, 180), (160, 240)],
    )

    np.savez(
        output.data_dir / "parameter_sweep.npz",
        thicknesses=sweep.thicknesses,
        sigma_values=sweep.sigma_values,
        heatmap=sweep.heatmap,
        mechanism_distance=sweep.mechanism_distance,
        mechanism_conductivity=sweep.mechanism_conductivity,
        mechanism_combined=sweep.mechanism_combined,
        outer_radii=domain_validation["outer_radii"],
        phi_domain=domain_validation["phi_elec"],
        mesh_labels=mesh_validation["labels"],
        phi_mesh=mesh_validation["phi_elec"],
    )

    save_figure(plot_heatmap(sweep), presentation_dir / "sweep_heatmap.png")
    save_figure(plot_mechanism_curves(sweep), presentation_dir / "mechanism_curves.png")
    save_figure(
        plot_validation_series(
            x_values=1e6 * domain_validation["outer_radii"],
            y_values=domain_validation["phi_elec"],
            xlabel="Outer domain radius [um]",
            title="Outer-Domain Size Validation",
        ),
        validation_dir / "domain_size_validation.png",
    )
    save_figure(
        plot_validation_series(
            x_values=mesh_validation["labels"],
            y_values=mesh_validation["phi_elec"],
            xlabel="Grid resolution [nr x nz]",
            title="Mesh-Convergence Check",
        ),
        validation_dir / "mesh_convergence_validation.png",
    )

    print(f"Saved sweep outputs to {output.project_root / 'outputs'}")
    print(f"Heatmap shape: {sweep.heatmap.shape}")


if __name__ == "__main__":
    main()
