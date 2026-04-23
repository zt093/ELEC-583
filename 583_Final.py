"""Top-level launcher for the FEM neural recording demos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent / "fem_project"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


def run_script(name: str) -> None:
    script = PROJECT_ROOT / "scripts" / name
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FEM neural recording demos.")
    parser.add_argument(
        "target",
        nargs="?",
        default="monopole",
        choices=["monopole", "monopole3d", "sweep", "dipole", "morphology", "framework", "all"],
    )
    args = parser.parse_args()

    if args.target in ("monopole", "all"):
        run_script("run_monopole.py")
    if args.target in ("monopole3d", "all"):
        run_script("run_monopole_3d.py")
    if args.target in ("sweep", "all"):
        run_script("run_parameter_sweep.py")
    if args.target in ("dipole", "all"):
        run_script("run_dipole.py")
    if args.target in ("morphology", "all"):
        run_script("run_detailed_source_placeholder.py")
    if args.target in ("framework", "all"):
        run_script("run_stage_matrix.py")


if __name__ == "__main__":
    main()
