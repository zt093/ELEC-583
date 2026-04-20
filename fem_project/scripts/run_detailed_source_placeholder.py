"""Run the lightweight Level 3 distributed-source placeholder example."""

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
from src.plotting import plot_3d_slice_view, save_figure
from src.postprocess import sample_surface_average
from src.solver import solve_cartesian_potential
from src.sources import morphology_placeholder


def main() -> None:
    output = OutputConfig()
    presentation_dir = output.figures_dir / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    output.data_dir.mkdir(parents=True, exist_ok=True)

    config = Cartesian3DConfig()
    points = morphology_placeholder(
        center=np.array([config.probe_radius + 55e-6, 0.0, 0.0]),
        scale=15e-6,
        current=config.source_current,
    )
    result = solve_cartesian_potential(
        config=config,
        t_enc=config.default_t_enc,
        sigma_enc=config.default_sigma_enc,
        sources=points,
    )
    phi_elec = sample_surface_average(result=result, config=config)

    np.savez(
        output.data_dir / "morphology_placeholder.npz",
        x=result.grid.x,
        y=result.grid.y,
        z=result.grid.z,
        potential=result.potential,
        phi_elec=phi_elec,
    )
    save_figure(
        plot_3d_slice_view(result=result, config=config),
        presentation_dir / "morphology_placeholder_slice.png",
    )

    print(f"Saved morphology placeholder outputs to {output.project_root / 'outputs'}")
    print(f"phi_elec = {phi_elec:.6e} V")


if __name__ == "__main__":
    main()
