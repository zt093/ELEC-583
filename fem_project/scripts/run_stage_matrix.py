"""Run the staged source-level by electrode-stage comparison framework."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OutputConfig
from src.electrodes import ElectrodeStage
from src.framework import SourceLevel, run_stage_matrix


def parse_source_levels(value: str) -> list[SourceLevel]:
    if value == "all":
        return list(SourceLevel)
    return [SourceLevel(item.strip()) for item in value.split(",")]


def parse_electrode_stages(value: str) -> list[ElectrodeStage]:
    if value == "all":
        return list(ElectrodeStage)
    return [ElectrodeStage(item.strip().upper()) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged source/electrode comparison cases.")
    parser.add_argument("--sources", default="all", help="Comma-separated source levels or 'all'.")
    parser.add_argument("--stages", default="all", help="Comma-separated electrode stages or 'all'.")
    args = parser.parse_args()

    source_levels = parse_source_levels(args.sources)
    electrode_stages = parse_electrode_stages(args.stages)

    output = OutputConfig()
    output.data_dir.mkdir(parents=True, exist_ok=True)

    cases = run_stage_matrix(source_levels=source_levels, electrode_stages=electrode_stages)

    summary_path = output.data_dir / "stage_framework_summary.csv"
    site_path = output.data_dir / "stage_framework_site_recordings.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_level",
                "electrode_stage",
                "solver_family",
                "site_count",
                "includes_probe_body",
                "sigma_probe_S_per_m",
                "t_enc_um",
                "sigma_enc_S_per_m",
                "mean_phi_elec_uV",
                "max_abs_phi_elec_uV",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "source_level": case.scenario.source_level.value,
                    "electrode_stage": case.scenario.electrode_stage.value,
                    "solver_family": case.scenario.solver_family,
                    "site_count": len(case.layout.sites),
                    "includes_probe_body": int(case.layout.includes_probe_body),
                    "sigma_probe_S_per_m": case.config.sigma_probe,
                    "t_enc_um": case.result.t_enc * 1e6,
                    "sigma_enc_S_per_m": case.result.sigma_enc,
                    "mean_phi_elec_uV": case.mean_recording * 1e6,
                    "max_abs_phi_elec_uV": case.max_abs_recording * 1e6,
                }
            )

    with site_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_level",
                "electrode_stage",
                "site_name",
                "phi_elec_uV",
            ],
        )
        writer.writeheader()
        for case in cases:
            for site_name, value in case.recordings.items():
                writer.writerow(
                    {
                        "source_level": case.scenario.source_level.value,
                        "electrode_stage": case.scenario.electrode_stage.value,
                        "site_name": site_name,
                        "phi_elec_uV": value * 1e6,
                    }
                )

    print(f"Saved stage framework summaries to {output.data_dir}")
    for case in cases:
        print(
            f"{case.scenario.source_level.value:>8} | stage {case.scenario.electrode_stage.value} | "
            f"sites={len(case.layout.sites):>2} | mean phi_elec={case.mean_recording * 1e6:>8.3f} uV | "
            f"max |phi_elec|={case.max_abs_recording * 1e6:>8.3f} uV"
        )


if __name__ == "__main__":
    main()
