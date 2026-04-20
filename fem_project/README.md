# FEM Project for Encapsulation Effects on Neural Recording

This project models extracellular neural recording near an implanted probe with glial encapsulation, with clean separation of two mechanisms:

1. Mechanism 1 changes the neuron-source position as encapsulation thickness grows.
2. Mechanism 2 changes the encapsulation conductivity while keeping the source position fixed.

## Numerical design

- Level 1 uses a 2D axisymmetric lead-field solve. The electrode is modeled as an axisymmetric recording band, and reciprocity is used so an off-axis monopole can still be evaluated through the electrode transfer field without forcing an unphysical ring current source.
- Level 2 uses a full 3D finite-volume solve on a Cartesian grid for dipoles and localized electrode sampling on the probe surface.
- Level 3 reuses the same 3D solver with a list of point-current sources so more realistic morphology-driven source sets can be added later.

## Run

From the repository root:

```bash
python3 fem_project/scripts/run_monopole.py
python3 fem_project/scripts/run_parameter_sweep.py
python3 fem_project/scripts/run_dipole.py
python3 fem_project/scripts/run_detailed_source_placeholder.py
```

Figures are written to:

- `fem_project/outputs/figures/presentation` for slide/manuscript-style figures
- `fem_project/outputs/figures/validation` for mesh, domain, and trend checks

Arrays are written to `fem_project/outputs/data`.
