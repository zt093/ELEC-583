"""Scenario registry and staged runners for source-level and electrode-stage comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .config import Cartesian3DConfig
from .electrodes import ElectrodeLayout, ElectrodeStage, build_electrode_layout
from .postprocess import mechanism_source_gap, sample_layout_recordings
from .solver import CartesianPotentialResult, solve_cartesian_potential
from .sources import DipoleSource, MonopoleSource, morphology_placeholder


class SourceLevel(str, Enum):
    LEVEL_1_MONOPOLE = "monopole"
    LEVEL_2_DIPOLE = "dipole"
    LEVEL_3_DETAILED = "detailed"

    @property
    def label(self) -> str:
        labels = {
            SourceLevel.LEVEL_1_MONOPOLE: "Level 1",
            SourceLevel.LEVEL_2_DIPOLE: "Level 2",
            SourceLevel.LEVEL_3_DETAILED: "Level 3",
        }
        return labels[self]


@dataclass(frozen=True)
class ScenarioDefinition:
    source_level: SourceLevel
    electrode_stage: ElectrodeStage
    solver_family: str
    description: str


@dataclass(frozen=True)
class StageRunResult:
    scenario: ScenarioDefinition
    config: Cartesian3DConfig
    layout: ElectrodeLayout
    result: CartesianPotentialResult
    recordings: dict[str, float]

    @property
    def mean_recording(self) -> float:
        values = np.asarray(list(self.recordings.values()), dtype=float)
        return float(values.mean()) if values.size else 0.0

    @property
    def max_abs_recording(self) -> float:
        values = np.asarray(list(self.recordings.values()), dtype=float)
        return float(np.max(np.abs(values))) if values.size else 0.0


def supported_scenarios() -> list[ScenarioDefinition]:
    definitions = []
    descriptions = {
        SourceLevel.LEVEL_1_MONOPOLE: "Monopole source with glial-thickness-driven source displacement.",
        SourceLevel.LEVEL_2_DIPOLE: "Bipolar source resolved as two opposite point currents.",
        SourceLevel.LEVEL_3_DETAILED: "Lightweight distributed current-source placeholder for future neuron detail.",
    }
    for source_level in SourceLevel:
        for stage in ElectrodeStage:
            definitions.append(
                ScenarioDefinition(
                    source_level=source_level,
                    electrode_stage=stage,
                    solver_family="cartesian_3d_fvm",
                    description=f"{descriptions[source_level]} {build_electrode_layout(stage, cartesian_config_for_stage(stage)).description}",
                )
            )
    return definitions


def cartesian_config_for_stage(stage: ElectrodeStage) -> Cartesian3DConfig:
    base = Cartesian3DConfig()
    if stage in (ElectrodeStage.STAGE_A, ElectrodeStage.STAGE_B):
        config = replace(base, nx=61, ny=61, nz=101)
    elif stage == ElectrodeStage.STAGE_C:
        config = replace(base, nx=61, ny=61, nz=111)
    elif stage == ElectrodeStage.STAGE_D:
        config = replace(base, nx=71, ny=71, nz=141)
    else:
        raise ValueError(f"Unsupported electrode stage: {stage}")

    if stage == ElectrodeStage.STAGE_A:
        config = replace(config, sigma_probe=config.sigma_brain)
    return config


def build_sources(
    source_level: SourceLevel,
    config: Cartesian3DConfig,
    *,
    t_enc: float,
) -> list:
    if source_level == SourceLevel.LEVEL_1_MONOPOLE:
        monopole = MonopoleSource(
            radial_gap=mechanism_source_gap(
                baseline_gap=config.monopole_baseline_gap,
                displacement_alpha=config.monopole_displacement_alpha,
                t_enc=t_enc,
            ),
            azimuth=config.monopole_azimuth,
            z=config.monopole_z,
            current=config.source_current,
        )
        return [monopole.point_source(config.probe_radius)]

    if source_level == SourceLevel.LEVEL_2_DIPOLE:
        dipole = DipoleSource(
            center=np.array([config.probe_radius + config.dipole_center_radius, 0.0, config.dipole_center_z]),
            orientation=np.array([1.0, 0.0, 1.0]),
            separation=config.dipole_separation,
            current=config.source_current,
        )
        return dipole.point_sources()

    if source_level == SourceLevel.LEVEL_3_DETAILED:
        return morphology_placeholder(
            center=np.array([config.probe_radius + 55e-6, 0.0, 0.0]),
            scale=15e-6,
            current=config.source_current,
        )

    raise ValueError(f"Unsupported source level: {source_level}")


def run_stage_case(
    source_level: SourceLevel,
    electrode_stage: ElectrodeStage,
    *,
    t_enc: float | None = None,
    sigma_enc: float | None = None,
) -> StageRunResult:
    config = cartesian_config_for_stage(electrode_stage)
    t_enc_value = config.default_t_enc if t_enc is None else t_enc
    sigma_enc_value = config.default_sigma_enc if sigma_enc is None else sigma_enc
    layout = build_electrode_layout(electrode_stage, config)
    result = solve_cartesian_potential(
        config=config,
        t_enc=t_enc_value,
        sigma_enc=sigma_enc_value,
        sources=build_sources(source_level, config, t_enc=t_enc_value),
    )
    recordings = sample_layout_recordings(result=result, config=config, layout=layout)
    return StageRunResult(
        scenario=ScenarioDefinition(
            source_level=source_level,
            electrode_stage=electrode_stage,
            solver_family="cartesian_3d_fvm",
            description=f"{source_level.label} with {layout.stage.label}.",
        ),
        config=config,
        layout=layout,
        result=result,
        recordings=recordings,
    )


def run_stage_matrix(
    source_levels: list[SourceLevel] | None = None,
    electrode_stages: list[ElectrodeStage] | None = None,
    *,
    t_enc: float | None = None,
    sigma_enc: float | None = None,
) -> list[StageRunResult]:
    levels = list(SourceLevel) if source_levels is None else source_levels
    stages = list(ElectrodeStage) if electrode_stages is None else electrode_stages
    results = []
    for level in levels:
        for stage in stages:
            results.append(
                run_stage_case(
                    source_level=level,
                    electrode_stage=stage,
                    t_enc=t_enc,
                    sigma_enc=sigma_enc,
                )
            )
    return results
