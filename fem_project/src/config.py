"""Project-wide configuration dataclasses and unit helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def um(value: float) -> float:
    return value * 1e-6


@dataclass(frozen=True)
class OutputConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    figures_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    paraview_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "figures_dir", self.project_root / "outputs" / "figures")
        object.__setattr__(self, "data_dir", self.project_root / "outputs" / "data")
        object.__setattr__(self, "paraview_dir", self.project_root / "outputs" / "paraview")


@dataclass(frozen=True)
class AxisymmetricConfig:
    sigma_brain: float = 0.3
    sigma_probe: float = 1e-6
    probe_radius: float = um(35.0)
    electrode_height: float = um(15.0)
    outer_radius: float = um(600.0)
    outer_half_height: float = um(800.0)
    nr: int = 180
    nz: int = 280
    baseline_gap: float = um(30.0)
    displacement_alpha: float = 0.6
    source_current: float = 1e-9
    default_sigma_enc: float = 0.1
    default_t_enc: float = um(30.0)
    z_source: float = 0.0


@dataclass(frozen=True)
class Cartesian3DConfig:
    sigma_brain: float = 0.3
    sigma_probe: float = 1e-6
    probe_radius: float = um(35.0)
    electrode_height: float = um(15.0)
    electrode_arc: float = np.deg2rad(45.0)
    outer_radius: float = um(350.0)
    outer_half_height: float = um(420.0)
    nx: int = 41
    ny: int = 41
    nz: int = 69
    source_current: float = 1e-9
    monopole_baseline_gap: float = um(30.0)
    monopole_displacement_alpha: float = 0.6
    monopole_azimuth: float = 0.0
    monopole_z: float = 0.0
    dipole_separation: float = um(20.0)
    dipole_center_radius: float = um(70.0)
    dipole_center_z: float = 0.0
    default_sigma_enc: float = 0.1
    default_t_enc: float = um(25.0)
    source_spread: float = 1.2


@dataclass(frozen=True)
class MechanismSweepConfig:
    thicknesses: np.ndarray = field(
        default_factory=lambda: np.linspace(um(5.0), um(100.0), 11)
    )
    sigma_enc_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.03, 0.6, 10)
    )
    mechanism_curve_sigma: float = 0.08

    @classmethod
    def from_iterables(
        cls,
        thicknesses: Iterable[float],
        sigma_enc_values: Iterable[float],
        mechanism_curve_sigma: float,
    ) -> "MechanismSweepConfig":
        return cls(
            thicknesses=np.asarray(list(thicknesses), dtype=float),
            sigma_enc_values=np.asarray(list(sigma_enc_values), dtype=float),
            mechanism_curve_sigma=mechanism_curve_sigma,
        )
