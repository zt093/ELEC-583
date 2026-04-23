# ParaView Workflow Guide

This project now exports ParaView-readable VTK files while keeping the current numerical solver unchanged.

## Exported files

The main ParaView outputs are written to:

- `fem_project/outputs/paraview/level1_axisymmetric_baseline.vtr`
- `fem_project/outputs/paraview/monopole_3d_localized_baseline.vti`
- `fem_project/outputs/paraview/monopole_3d_source_points.vtp`
- `fem_project/outputs/paraview/level2_dipole_baseline.vti`
- `fem_project/outputs/paraview/level3_morphology_placeholder.vti`

## Why these formats were chosen

- Level 1 uses `.vtr` because the axisymmetric baseline is naturally a structured rectilinear meridional cross-section in `(r, z)`, represented as a one-cell-thick rectilinear grid for ParaView.
- The localized 3D monopole, Level 2 dipole, and Level 3 placeholder use `.vti` because the current 3D solver is defined on a uniform Cartesian grid, so `ImageData` is the cleanest and most efficient VTK representation.

These formats are a good fit for:

- slices
- clips
- thresholds
- contours / isosurfaces
- region masking

without changing the solver itself.

## Important interpretation notes

### Level 1

`level1_axisymmetric_baseline.vtr` is **not** a full 3D localized-pad solution.

It is the axisymmetric Level 1 baseline exported as a meridional slice. The main fields are:

- `phi_unit_electrode_drive_V`
  - the reciprocity / adjoint potential for a unit electrode drive
- `lead_field_V_per_A`
  - the recording transfer field
- `recording_response_1nA_uV`
  - the predicted recording proxy for a `1 nA` monopole placed at the sampled location

This is the correct field to visualize for the Level 1 recording baseline, but it should be interpreted as the ring-electrode reciprocity solution, not as a localized-pad extracellular potential field.

### Level 2

`level2_dipole_baseline.vti` contains the actual 3D potential field:

- `phi_V`
- `phi_uV`
- `abs_phi_uV`

plus masks and region labels:

- `material_region_id`
  - `0 = tissue`
  - `1 = encapsulation`
  - `2 = probe`
- `probe_mask`
- `encapsulation_mask`
- `tissue_mask`
- `electrode_patch_mask`
- `source_signed_nA`

This is the file to use for professional 3D ParaView postprocessing.

### Localized 3D monopole

`monopole_3d_localized_baseline.vti` is a true 3D localized monopole solve, not a revolved visualization of the axisymmetric baseline.

Important interpretation details:

- the source is a true 3D monopole placed near the probe and aligned with the localized recording patch
- the source is represented numerically by trilinear deposition onto the eight surrounding Cartesian cells
- `phi_elec` is still the localized electrode-surface average, computed by sampling the field over the electrode patch
- this path is intended for ParaView-quality 3D geometry, slice, clip, and hero views for the monopole case
- the monopole export now uses a refined monopole-only grid relative to the original 3D default, but it is still a uniform structured grid rather than an adaptively refined mesh

The main arrays are:

- `phi_V`
- `phi_uV`
- `abs_phi_uV`
- `log10_abs_phi_uV`
- `material_region_id`
- `probe_mask`
- `encapsulation_mask`
- `tissue_mask`
- `electrode_patch_mask`
- `source_signed_nA`

Use this file when you want the monopole case to look like a real 3D FEM postprocessing result.

`monopole_3d_source_points.vtp` contains the monopole location as a separate ParaView object so the source can be rendered cleanly without looking like a voxelized cube.

## Recommended manual workflow

## 0. Which file to open

- Use `level1_axisymmetric_baseline.vtr` for the fast ring-electrode screening baseline.
- Use `monopole_3d_localized_baseline.vti` for localized 3D monopole figures.
- Use `monopole_3d_source_points.vtp` alongside the monopole `.vti` file when you want the source shown as a distinct highlighted object.
- Use `level2_dipole_baseline.vti` for localized 3D dipole figures.

## 1. Geometry-focused view

Use `monopole_3d_localized_baseline.vti` or `level2_dipole_baseline.vti`.

1. Open the file in ParaView.
2. Click `Apply`.
3. Create `Threshold` on `probe_mask` with range `[1, 1]`.
4. Show it as `Surface`.
5. Color it with a dark blue or graphite tone.
6. Create a second `Threshold` on `encapsulation_mask` with range `[1, 1]`.
7. Show it as `Surface` with lower opacity such as `0.20` to `0.35`.
8. Create a third `Threshold` on `electrode_patch_mask` with range `[1, 1]`.
9. Show it as `Surface` in a strong red or orange.
10. Keep the original dataset as `Outline` only if you want the tissue volume context.

Best result:

- probe opaque
- encapsulation semi-transparent
- electrode patch saturated red
- white background
- source visible as a compact hot feature near the electrode-facing side
- or, for the cleanest result, load `monopole_3d_source_points.vtp` and show the source as a highlighted point object

## 2. Field slice view

Use `monopole_3d_localized_baseline.vti` or `level2_dipole_baseline.vti`.

1. Apply `Cell Data to Point Data`.
2. Create a central `x-z` plane through the source and electrode patch.
3. Color by `abs_phi_uV`.
4. For the monopole case, prefer `log10_abs_phi_uV` for display while keeping `abs_phi_uV` available for quantitative inspection.
5. Use a perceptually ordered map such as `Inferno`.
6. Choose a custom range that keeps the main near-source gradient visible instead of letting a single peak dominate the whole scale.
7. Overlay the probe and encapsulation thresholds from the geometry workflow with low opacity.
8. Keep the electrode patch fully visible.
9. Show the source from `monopole_3d_source_points.vtp` as a separate highlighted object.

Best result:

- slice as the main visual layer
- geometry overlaid lightly
- magnitude colormap on `abs_phi_uV`, preferably displayed through `log10_abs_phi_uV`
- source and electrode both visible in the same composition

## 3. Hero view

Use `monopole_3d_localized_baseline.vti` or `level2_dipole_baseline.vti`.

1. Apply `Cell Data to Point Data`.
2. Create the same central `x-z` plane used in the field slice view.
3. Show the slice colored by `abs_phi_uV`.
4. Add a small number of `Contour` surfaces on `abs_phi_uV`.
5. Overlay the probe, encapsulation shell, electrode patch, and source.
6. Use semi-transparent geometry so the spatial relationship stays readable.

This usually reads better than a generic clipped half-volume because it keeps the source-electrode geometry obvious.

## Useful arrays for thresholding and coloring

### Level 1 `.vtr`

- `recording_response_1nA_uV`
- `abs_recording_response_1nA_uV`
- `conductivity_S_per_m`
- `material_region_id`
- `probe_mask`
- `encapsulation_mask`
- `electrode_band_mask`
- `solution_domain_mask`
- `source_evaluation_mask`

### Level 2 `.vti`

- `phi_uV`
- `abs_phi_uV`
- `log10_abs_phi_uV`
- `conductivity_S_per_m`
- `material_region_id`
- `probe_mask`
- `encapsulation_mask`
- `electrode_patch_mask`
- `source_signed_nA`

### Localized 3D monopole `.vti`

- `phi_uV`
- `abs_phi_uV`
- `conductivity_S_per_m`
- `material_region_id`
- `probe_mask`
- `encapsulation_mask`
- `electrode_patch_mask`
- `source_signed_nA`

## Macro / Python automation

A ParaView Python script is provided at:

- `fem_project/paraview/paraview_baseline_views.py`
- `fem_project/paraview/paraview_monopole_3d_views.py`

Run it with `pvpython`:

```bash
pvpython fem_project/paraview/paraview_baseline_views.py
pvpython fem_project/paraview/paraview_monopole_3d_views.py
```

`paraview_baseline_views.py` creates:

- a geometry-focused view
- a polished slice view
- a clipped 3D field view

for the dipole case.

`paraview_monopole_3d_views.py` creates:

- a geometry-focused view
- a field slice view using `abs_phi_uV`
- a hero view with slice plus iso-contours plus geometry

Both scripts save screenshots into `fem_project/outputs/paraview/screenshots`.

## Practical recommendation

If your goal is presentation-quality 3D figures, ParaView is the right next step for the current solver.

The current solver is already exporting the right structured data layout for good ParaView work. This is a better path than continuing to push matplotlib 3D beyond its natural limit.

At the same time, be realistic about the remaining limit: the 3D solver is still a uniform structured-grid finite-volume model. Even after monopole-grid refinement, it will not look as smooth near the singular source as a genuinely locally refined unstructured FEM mesh. If you eventually want the source neighborhood to look fully like a polished commercial FEM result, the next technical step is adaptive or geometry-aware mesh refinement, not just more ParaView styling.
