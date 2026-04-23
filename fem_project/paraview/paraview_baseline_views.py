"""Create baseline ParaView views from the exported VTK files.

Run with:
    pvpython fem_project/paraview/paraview_baseline_views.py

This script was written for the current export pipeline but was not executed
in this environment because ParaView is not installed here.
"""

from __future__ import annotations

from pathlib import Path

from paraview.simple import (  # type: ignore
    CellDatatoPointData,
    Clip,
    Contour,
    ColorBy,
    CreateLayout,
    CreateView,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    Hide,
    LoadPalette,
    ResetCamera,
    SaveScreenshot,
    SetActiveView,
    Show,
    Threshold,
    XMLImageDataReader,
    _DisableFirstRenderCameraReset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAVIEW_OUTPUT = PROJECT_ROOT / "outputs" / "paraview"
SCREENSHOT_DIR = PARAVIEW_OUTPUT / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL2_FILE = PARAVIEW_OUTPUT / "level2_dipole_baseline.vti"


def configure_view(view):
    view.ViewSize = [1800, 1200]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.AxesGrid.Visibility = 0


def geometry_view(reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    probe = Threshold(Input=reader)
    probe.Scalars = ["CELLS", "probe_mask"]
    probe.LowerThreshold = 1.0
    probe.UpperThreshold = 1.0
    probe_display = Show(probe, render_view)
    probe_display.Representation = "Surface"
    probe_display.AmbientColor = [0.14, 0.22, 0.32]
    probe_display.DiffuseColor = [0.14, 0.22, 0.32]
    probe_display.Opacity = 1.0

    encap = Threshold(Input=reader)
    encap.Scalars = ["CELLS", "encapsulation_mask"]
    encap.LowerThreshold = 1.0
    encap.UpperThreshold = 1.0
    encap_display = Show(encap, render_view)
    encap_display.Representation = "Surface"
    encap_display.AmbientColor = [0.36, 0.75, 0.78]
    encap_display.DiffuseColor = [0.36, 0.75, 0.78]
    encap_display.Opacity = 0.28

    electrode = Threshold(Input=reader)
    electrode.Scalars = ["CELLS", "electrode_patch_mask"]
    electrode.LowerThreshold = 1.0
    electrode.UpperThreshold = 1.0
    electrode_display = Show(electrode, render_view)
    electrode_display.Representation = "Surface"
    electrode_display.AmbientColor = [0.85, 0.29, 0.22]
    electrode_display.DiffuseColor = [0.85, 0.29, 0.22]
    electrode_display.Opacity = 1.0

    outline = Show(reader, render_view)
    outline.Representation = "Outline"
    outline.AmbientColor = [0.78, 0.82, 0.85]
    outline.DiffuseColor = [0.78, 0.82, 0.85]

    ResetCamera(render_view)
    render_view.CameraPosition = [165.0, -225.0, 145.0]
    render_view.CameraFocalPoint = [50.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "geometry_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def slice_view(reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    point_data = CellDatatoPointData(Input=reader)
    slice_filter = Clip(Input=point_data)
    slice_filter.ClipType = "Plane"
    slice_filter.ClipType.Origin = [0.0, 0.0, 0.0]
    slice_filter.ClipType.Normal = [0.0, 1.0, 0.0]
    slice_filter.Invert = 0

    slice_display = Show(slice_filter, render_view)
    slice_display.Representation = "Surface"
    ColorBy(slice_display, ("POINTS", "phi_uV"))
    phi_lut = GetColorTransferFunction("phi_uV")
    phi_lut.ApplyPreset("Cool to Warm", True)
    phi_lut.RescaleTransferFunction(-15.0, 15.0)
    phi_pwf = GetOpacityTransferFunction("phi_uV")
    phi_pwf.RescaleTransferFunction(-15.0, 15.0)

    probe = Threshold(Input=reader)
    probe.Scalars = ["CELLS", "probe_mask"]
    probe.LowerThreshold = 1.0
    probe.UpperThreshold = 1.0
    probe_display = Show(probe, render_view)
    probe_display.Representation = "Surface"
    probe_display.AmbientColor = [0.14, 0.22, 0.32]
    probe_display.DiffuseColor = [0.14, 0.22, 0.32]
    probe_display.Opacity = 0.22

    encap = Threshold(Input=reader)
    encap.Scalars = ["CELLS", "encapsulation_mask"]
    encap.LowerThreshold = 1.0
    encap.UpperThreshold = 1.0
    encap_display = Show(encap, render_view)
    encap_display.Representation = "Surface"
    encap_display.AmbientColor = [0.36, 0.75, 0.78]
    encap_display.DiffuseColor = [0.36, 0.75, 0.78]
    encap_display.Opacity = 0.10

    ResetCamera(render_view)
    render_view.CameraPosition = [180.0, -190.0, 150.0]
    render_view.CameraFocalPoint = [75.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "slice_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def clipped_field_view(reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    point_data = CellDatatoPointData(Input=reader)
    clip_filter = Clip(Input=point_data)
    clip_filter.ClipType = "Plane"
    clip_filter.ClipType.Origin = [0.0, 0.0, 0.0]
    clip_filter.ClipType.Normal = [0.0, 1.0, 0.0]
    clip_filter.Invert = 0

    contour = Contour(Input=clip_filter)
    contour.ContourBy = ["POINTS", "abs_phi_uV"]
    contour.Isosurfaces = [1.0, 3.0, 6.0]
    contour.PointMergeMethod = "Uniform Binning"
    contour_display = Show(contour, render_view)
    contour_display.Representation = "Surface"
    contour_display.Opacity = 0.42
    ColorBy(contour_display, ("POINTS", "phi_uV"))
    phi_lut = GetColorTransferFunction("phi_uV")
    phi_lut.ApplyPreset("Cool to Warm", True)
    phi_lut.RescaleTransferFunction(-15.0, 15.0)

    probe = Threshold(Input=reader)
    probe.Scalars = ["CELLS", "probe_mask"]
    probe.LowerThreshold = 1.0
    probe.UpperThreshold = 1.0
    probe_display = Show(probe, render_view)
    probe_display.Representation = "Surface"
    probe_display.AmbientColor = [0.14, 0.22, 0.32]
    probe_display.DiffuseColor = [0.14, 0.22, 0.32]
    probe_display.Opacity = 0.55

    encap = Threshold(Input=reader)
    encap.Scalars = ["CELLS", "encapsulation_mask"]
    encap.LowerThreshold = 1.0
    encap.UpperThreshold = 1.0
    encap_display = Show(encap, render_view)
    encap_display.Representation = "Surface"
    encap_display.AmbientColor = [0.36, 0.75, 0.78]
    encap_display.DiffuseColor = [0.36, 0.75, 0.78]
    encap_display.Opacity = 0.14

    ResetCamera(render_view)
    render_view.CameraPosition = [185.0, -225.0, 165.0]
    render_view.CameraFocalPoint = [70.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "clipped_field_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def main() -> None:
    _DisableFirstRenderCameraReset()
    LoadPalette("WhiteBackground")

    reader = XMLImageDataReader(FileName=[str(LEVEL2_FILE)])
    reader.CellArrayStatus = [
        "phi_uV",
        "abs_phi_uV",
        "material_region_id",
        "probe_mask",
        "encapsulation_mask",
        "electrode_patch_mask",
    ]

    layout = CreateLayout(name="BaselineViews")
    views = [
        geometry_view(reader),
        slice_view(reader),
        clipped_field_view(reader),
    ]
    for index, view in enumerate(views):
        layout.AssignView(index, view)


if __name__ == "__main__":
    main()
