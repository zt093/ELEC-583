"""Electrode-layout definitions for the staged recording-complexity framework."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Cartesian3DConfig
from .geometry import CartesianGrid


class ElectrodeStage(str, Enum):
    STAGE_A = "A"
    STAGE_B = "B"
    STAGE_C = "C"
    STAGE_D = "D"

    @property
    def label(self) -> str:
        labels = {
            ElectrodeStage.STAGE_A: "Stage A",
            ElectrodeStage.STAGE_B: "Stage B",
            ElectrodeStage.STAGE_C: "Stage C",
            ElectrodeStage.STAGE_D: "Stage D",
        }
        return labels[self]


@dataclass(frozen=True)
class ElectrodeSite:
    name: str
    azimuth: float
    z_center: float
    arc: float
    height: float


@dataclass(frozen=True)
class ElectrodeLayout:
    stage: ElectrodeStage
    sites: tuple[ElectrodeSite, ...]
    includes_probe_body: bool
    description: str

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.sites)


def build_electrode_layout(
    stage: ElectrodeStage,
    config: Cartesian3DConfig,
) -> ElectrodeLayout:
    if stage == ElectrodeStage.STAGE_A:
        return ElectrodeLayout(
            stage=stage,
            sites=(
                ElectrodeSite(
                    name="site_00",
                    azimuth=0.0,
                    z_center=0.0,
                    arc=config.electrode_arc,
                    height=config.electrode_height,
                ),
            ),
            includes_probe_body=False,
            description="Single circular electrode without an electrically explicit probe body.",
        )

    if stage == ElectrodeStage.STAGE_B:
        return ElectrodeLayout(
            stage=stage,
            sites=(
                ElectrodeSite(
                    name="site_00",
                    azimuth=0.0,
                    z_center=0.0,
                    arc=config.electrode_arc,
                    height=config.electrode_height,
                ),
            ),
            includes_probe_body=True,
            description="Single circular electrode with the probe body included in the domain.",
        )

    if stage == ElectrodeStage.STAGE_C:
        azimuths = np.deg2rad([-24.0, 24.0])
        z_values = [-18e-6, 18e-6]
        sites = []
        for row, z_center in enumerate(z_values):
            for col, azimuth in enumerate(azimuths):
                sites.append(
                    ElectrodeSite(
                        name=f"site_{row}{col}",
                        azimuth=float(azimuth),
                        z_center=float(z_center),
                        arc=np.deg2rad(18.0),
                        height=12e-6,
                    )
                )
        return ElectrodeLayout(
            stage=stage,
            sites=tuple(sites),
            includes_probe_body=True,
            description="Simple four-site arrangement on the probe-facing side.",
        )

    if stage == ElectrodeStage.STAGE_D:
        azimuths = np.deg2rad([-10.0, 10.0])
        z_values = np.linspace(-66e-6, 66e-6, 7)
        sites = []
        for row, z_center in enumerate(z_values):
            for col, azimuth in enumerate(azimuths):
                sites.append(
                    ElectrodeSite(
                        name=f"site_{row:02d}_{col}",
                        azimuth=float(azimuth),
                        z_center=float(z_center),
                        arc=np.deg2rad(10.0),
                        height=10e-6,
                    )
                )
        return ElectrodeLayout(
            stage=stage,
            sites=tuple(sites),
            includes_probe_body=True,
            description="Neuropixels-like dense two-column array approximated on the cylindrical probe surface.",
        )

    raise ValueError(f"Unsupported electrode stage: {stage}")


def electrode_patch_mask(
    grid: CartesianGrid,
    site: ElectrodeSite,
) -> np.ndarray:
    xg, yg, zg = grid.mesh()
    rho = np.sqrt(xg**2 + yg**2)
    theta = np.arctan2(yg, xg)
    shell_thickness = max(grid.dx, grid.dy)
    electrode_shell = rho >= max(grid.probe_radius - shell_thickness, 0.0)
    azimuth_delta = np.arctan2(np.sin(theta - site.azimuth), np.cos(theta - site.azimuth))
    return (
        (rho <= grid.probe_radius)
        & electrode_shell
        & (np.abs(azimuth_delta) <= 0.5 * site.arc)
        & (np.abs(zg - site.z_center) <= 0.5 * site.height)
    )
