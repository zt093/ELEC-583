"""Public package interface for the FEM neural recording project."""

from .config import (
    AxisymmetricConfig,
    Cartesian3DConfig,
    MechanismSweepConfig,
    OutputConfig,
)
from .export import export_axisymmetric_result_to_vtr, export_cartesian_result_to_vti
from .geometry import build_axisymmetric_grid, build_cartesian_grid
from .materials import build_axisymmetric_conductivity, build_cartesian_conductivity
from .postprocess import evaluate_axisymmetric_recording, sample_surface_average
from .solver import solve_axisymmetric_lead_field, solve_cartesian_potential
from .sources import DipoleSource, MonopoleSource, PointCurrentSource, morphology_placeholder

__all__ = [
    "AxisymmetricConfig",
    "Cartesian3DConfig",
    "export_axisymmetric_result_to_vtr",
    "export_cartesian_result_to_vti",
    "MechanismSweepConfig",
    "OutputConfig",
    "DipoleSource",
    "MonopoleSource",
    "PointCurrentSource",
    "build_axisymmetric_grid",
    "build_axisymmetric_conductivity",
    "build_cartesian_grid",
    "build_cartesian_conductivity",
    "evaluate_axisymmetric_recording",
    "morphology_placeholder",
    "sample_surface_average",
    "solve_axisymmetric_lead_field",
    "solve_cartesian_potential",
]
