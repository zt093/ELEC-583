"""Run the Level 1 axisymmetric monopole baseline and export figures/data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AxisymmetricConfig, OutputConfig, um
from src.export import export_axisymmetric_result_to_vtr
from src.plotting import (
    plot_axisymmetric_field_map,
    plot_axisymmetric_mesh,
    plot_geometry_schematic,
    plot_validation_trend,
    save_figure,
)
from src.postprocess import (
    evaluate_axisymmetric_recording,
    mechanism_source_gap,
    validate_homogeneous_trend,
)
from src.solver import solve_axisymmetric_lead_field


def main() -> None:
    output = OutputConfig()
    presentation_dir = output.figures_dir / "presentation"
    validation_dir = output.figures_dir / "validation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output.data_dir.mkdir(parents=True, exist_ok=True)
    output.paraview_dir.mkdir(parents=True, exist_ok=True)

    config = AxisymmetricConfig()
    result = solve_axisymmetric_lead_field(
        config=config,
        t_enc=config.default_t_enc,
        sigma_enc=config.default_sigma_enc,
    )
    phi_elec = evaluate_axisymmetric_recording(
        result=result,
        radial_gap=mechanism_source_gap(
            baseline_gap=config.baseline_gap,
            displacement_alpha=config.displacement_alpha,
            t_enc=config.default_t_enc,
        ),
        source_current=config.source_current,
        z_source=config.z_source,
    )

    validation = validate_homogeneous_trend(
        config=config,
        radial_gaps=np.linspace(um(20.0), um(140.0), 9),
    )

    np.savez(
        output.data_dir / "monopole_baseline.npz",
        r_centers=result.grid.r_centers,
        z_centers=result.grid.z_centers,
        conductivity=result.conductivity,
        voltage_drive=result.potential_voltage_drive,
        lead_field=result.lead_field,
        phi_elec=phi_elec,
        total_current=result.total_current,
        validation_gaps=validation["radial_gaps"],
        validation_numerical=validation["numerical"],
        validation_analytic=validation["analytic"],
    )
    export_axisymmetric_result_to_vtr(
        output.paraview_dir / "level1_axisymmetric_baseline.vtr",
        result=result,
        config=config,
        source_gap=mechanism_source_gap(
            baseline_gap=config.baseline_gap,
            displacement_alpha=config.displacement_alpha,
            t_enc=config.default_t_enc,
        ),
        source_current=config.source_current,
        z_source=config.z_source,
    )

    save_figure(
        plot_geometry_schematic(config=config, t_enc=config.default_t_enc),
        presentation_dir / "geometry_schematic.png",
    )
    save_figure(
        plot_axisymmetric_mesh(result),
        validation_dir / "axisymmetric_mesh_closeup.png",
    )
    save_figure(
        plot_axisymmetric_field_map(
            result,
            radial_gap=mechanism_source_gap(
                baseline_gap=config.baseline_gap,
                displacement_alpha=config.displacement_alpha,
                t_enc=config.default_t_enc,
            ),
            z_source=config.z_source,
            title="Axisymmetric Monopole Recording Lead Field",
        ),
        presentation_dir / "axisymmetric_field_map.png",
    )
    save_figure(
        plot_validation_trend(validation),
        validation_dir / "homogeneous_validation.png",
    )

    print(f"Saved monopole baseline outputs to {output.project_root / 'outputs'}")
    print(f"phi_elec = {phi_elec:.6e} V for I_source = {config.source_current:.2e} A")


if __name__ == "__main__":
    main()
