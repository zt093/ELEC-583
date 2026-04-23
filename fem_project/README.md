# FEM Project for Encapsulation Effects on Neural Recording

This project models extracellular neural recording near an implanted probe with glial encapsulation, with clean separation of two mechanisms:

1. Mechanism 1 changes the neuron-source position as encapsulation thickness grows.
2. Mechanism 2 changes the encapsulation conductivity while keeping the source position fixed.

## Numerical status

This repository currently does **not** use an external FEM package such as `FEniCSx`, `dolfinx`, `gmsh`, or `meshio`.

The present implementation uses:

- `NumPy`
- `SciPy sparse`
- a custom structured-grid finite-volume / finite-difference style solver implemented in `src/solver.py`

So the project should be understood as a clean, modular **FEM-style volume-conduction modeling framework**, but not yet as a true unstructured FEM implementation.

## Numerical design

- Level 1 uses a 2D axisymmetric lead-field solve. The electrode is modeled as an axisymmetric recording band, and reciprocity is used so an off-axis monopole can still be evaluated through the electrode transfer field without forcing an unphysical ring current source.
- The axisymmetric Level 1 baseline remains the fast screening model. A separate localized 3D monopole workflow is also provided for ParaView-ready geometry, slice, clip, and hero views with a true 3D source placement.
- Level 2 uses a full 3D finite-volume solve on a Cartesian grid for dipoles and localized electrode sampling on the probe surface.
- Level 3 reuses the same 3D solver with a list of point-current sources so more realistic morphology-driven source sets can be added later.

In other words:

- physics: quasi-static extracellular volume conduction
- discretization: custom structured-grid sparse solver
- external FEM package: none in the current version

## Staged framework

The current codebase is also organized as two independent complexity axes:

1. Source complexity
   - `monopole`
   - `dipole`
   - `detailed`
2. Electrode complexity
   - `A`: single circular electrode without an electrically explicit probe body
   - `B`: single circular electrode with the probe body included
   - `C`: simple four-site arrangement
   - `D`: Neuropixels-like dense array approximation

The staged runner keeps the same general workflow for every combination:

- geometry and grid
- material assignment
- source definition
- solver
- electrode-surface averaging

The new framework modules are:

- `src/electrodes.py` for stage definitions and electrode layouts
- `src/framework.py` for source/stage scenario definitions and shared runners
- `scripts/run_stage_matrix.py` for direct comparison outputs across source levels and electrode stages

Important approximation note:

- Stage D is Neuropixels-like in channel density and arrangement logic, but it is still represented on the current cylindrical probe surface because the underlying solver geometry has not yet been changed to a planar shank.
- Stage A omits the probe body electrically by setting the probe conductivity equal to the surrounding tissue while keeping the same sampling surface for direct comparison against the later stages.

## Run

From the repository root:

```bash
python3 fem_project/scripts/run_monopole.py
python3 fem_project/scripts/run_monopole_3d.py
python3 fem_project/scripts/run_parameter_sweep.py
python3 fem_project/scripts/run_dipole.py
python3 fem_project/scripts/run_detailed_source_placeholder.py
python3 fem_project/scripts/run_stage_matrix.py
```

Examples:

```bash
python3 fem_project/scripts/run_stage_matrix.py --sources monopole --stages all
python3 fem_project/scripts/run_stage_matrix.py --sources dipole,detailed --stages B,C,D
python3 583_Final.py framework
```

Figures are written to:

- `fem_project/outputs/figures/presentation` for slide/manuscript-style figures
- `fem_project/outputs/figures/validation` for mesh, domain, and trend checks

Arrays are written to `fem_project/outputs/data`.

The staged comparison outputs are:

- `fem_project/outputs/data/stage_framework_summary.csv`
- `fem_project/outputs/data/stage_framework_site_recordings.csv`

ParaView exports are written to `fem_project/outputs/paraview`.

The ParaView workflow guide and automation script live in:

- `fem_project/paraview/PARAVIEW_WORKFLOW.md`
- `fem_project/paraview/paraview_baseline_views.py`
- `fem_project/paraview/paraview_monopole_3d_views.py`
