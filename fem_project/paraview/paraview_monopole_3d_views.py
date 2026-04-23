"""Create intentional ParaView views for the localized 3D monopole export.

Run with:
    pvpython fem_project/paraview/paraview_monopole_3d_views.py

This script targets the true 3D monopole export and writes three composed
screenshots to the ParaView output folder:

- geometry view
- field slice view
- hero view
"""

from __future__ import annotations

from pathlib import Path

from paraview.simple import (  # type: ignore
    CellDatatoPointData,
    ColorBy,
    Contour,
    CreateLayout,
    CreateView,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    LoadPalette,
    ResetCamera,
    SaveScreenshot,
    SetActiveView,
    Show,
    Slice,
    Threshold,
    XMLImageDataReader,
    XMLPolyDataReader,
    _DisableFirstRenderCameraReset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAVIEW_OUTPUT = PROJECT_ROOT / "outputs" / "paraview"
SCREENSHOT_DIR = PARAVIEW_OUTPUT / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

MONOPOLE_FILE = PARAVIEW_OUTPUT / "monopole_3d_localized_baseline.vti"
SOURCE_FILE = PARAVIEW_OUTPUT / "monopole_3d_source_points.vtp"


def configure_view(view):
    view.ViewSize = [1800, 1200]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.AxesGrid.Visibility = 0


def show_region(reader, render_view, array_name, color, opacity=1.0):
    region = Threshold(Input=reader)
    region.Scalars = ["CELLS", array_name]
    region.LowerThreshold = 1.0
    region.UpperThreshold = 1.0
    display = Show(region, render_view)
    display.Representation = "Surface"
    display.AmbientColor = color
    display.DiffuseColor = color
    display.Opacity = opacity
    return region, display


def show_source(reader, render_view):
    display = Show(reader, render_view)
    display.Representation = "Points"
    display.AmbientColor = [0.97, 0.70, 0.24]
    display.DiffuseColor = [0.97, 0.70, 0.24]
    display.Specular = 0.35
    display.PointSize = 18.0
    display.RenderPointsAsSpheres = 1
    return reader, display


def configure_abs_field_coloring(display, render_view, field_name="log10_abs_phi_uV"):
    ColorBy(display, ("POINTS", field_name))
    lut = GetColorTransferFunction(field_name)
    lut.ApplyPreset("Inferno (matplotlib)", True)
    lut.RescaleTransferFunction(-1.0, 1.2)
    pwf = GetOpacityTransferFunction(field_name)
    pwf.RescaleTransferFunction(-1.0, 1.2)
    scalar_bar = GetScalarBar(lut, render_view)
    scalar_bar.Title = "log10 |phi| [uV]"
    scalar_bar.ComponentTitle = ""
    scalar_bar.TitleFontSize = 16
    scalar_bar.LabelFontSize = 13
    scalar_bar.ScalarBarLength = 0.32
    return lut, pwf, scalar_bar


def central_xz_slice(reader):
    point_data = CellDatatoPointData(Input=reader)
    slice_filter = Slice(Input=point_data)
    slice_filter.SliceType = "Plane"
    slice_filter.SliceType.Origin = [0.0, 0.0, 0.0]
    slice_filter.SliceType.Normal = [0.0, 1.0, 0.0]
    return slice_filter


def geometry_view(reader, source_reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    show_region(reader, render_view, "probe_mask", [0.10, 0.18, 0.28], 1.0)
    show_region(reader, render_view, "encapsulation_mask", [0.32, 0.72, 0.78], 0.24)
    show_region(reader, render_view, "electrode_patch_mask", [0.85, 0.29, 0.22], 1.0)
    show_source(source_reader, render_view)
    ResetCamera(render_view)
    render_view.CameraPosition = [175.0, -235.0, 165.0]
    render_view.CameraFocalPoint = [62.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "monopole_3d_geometry_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def field_slice_view(reader, source_reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    slice_filter = central_xz_slice(reader)

    slice_display = Show(slice_filter, render_view)
    slice_display.Representation = "Surface"
    configure_abs_field_coloring(slice_display, render_view)

    show_region(reader, render_view, "probe_mask", [0.10, 0.18, 0.28], 0.18)
    show_region(reader, render_view, "encapsulation_mask", [0.32, 0.72, 0.78], 0.09)
    show_region(reader, render_view, "electrode_patch_mask", [0.85, 0.29, 0.22], 0.95)
    show_source(source_reader, render_view)

    ResetCamera(render_view)
    render_view.CameraPosition = [168.0, -205.0, 150.0]
    render_view.CameraFocalPoint = [68.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "monopole_3d_field_slice_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def hero_view(reader, source_reader):
    render_view = CreateView("RenderView")
    configure_view(render_view)
    SetActiveView(render_view)

    clip_filter = central_xz_slice(reader)

    slice_display = Show(clip_filter, render_view)
    slice_display.Representation = "Surface"
    configure_abs_field_coloring(slice_display, render_view)

    contour = Contour(Input=clip_filter)
    contour.ContourBy = ["POINTS", "log10_abs_phi_uV"]
    contour.Isosurfaces = [-0.7, 0.0, 0.7]
    contour.PointMergeMethod = "Uniform Binning"
    contour_display = Show(contour, render_view)
    contour_display.Representation = "Surface"
    contour_display.Opacity = 0.34
    contour_display.LineWidth = 2.0
    configure_abs_field_coloring(contour_display, render_view)

    show_region(reader, render_view, "probe_mask", [0.10, 0.18, 0.28], 0.48)
    show_region(reader, render_view, "encapsulation_mask", [0.32, 0.72, 0.78], 0.12)
    show_region(reader, render_view, "electrode_patch_mask", [0.85, 0.29, 0.22], 1.0)
    show_source(source_reader, render_view)

    ResetCamera(render_view)
    render_view.CameraPosition = [182.0, -220.0, 160.0]
    render_view.CameraFocalPoint = [70.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    SaveScreenshot(str(SCREENSHOT_DIR / "monopole_3d_hero_view.png"), render_view, ImageResolution=[1800, 1200])
    return render_view


def main() -> None:
    _DisableFirstRenderCameraReset()
    LoadPalette("WhiteBackground")

    reader = XMLImageDataReader(FileName=[str(MONOPOLE_FILE)])
    reader.CellArrayStatus = [
        "phi_uV",
        "abs_phi_uV",
        "log10_abs_phi_uV",
        "material_region_id",
        "probe_mask",
        "encapsulation_mask",
        "electrode_patch_mask",
        "source_signed_nA",
    ]
    source_reader = XMLPolyDataReader(FileName=[str(SOURCE_FILE)])
    source_reader.PointArrayStatus = ["source_current_nA", "source_index"]

    layout = CreateLayout(name="Monopole3DViews")
    views = [
        geometry_view(reader, source_reader),
        field_slice_view(reader, source_reader),
        hero_view(reader, source_reader),
    ]
    for index, view in enumerate(views):
        layout.AssignView(index, view)


if __name__ == "__main__":
    main()
