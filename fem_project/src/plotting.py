"""Reusable figure-generation helpers for presentation and validation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import zoom

from .config import AxisymmetricConfig, Cartesian3DConfig
from .postprocess import SweepResults
from .solver import AxisymmetricLeadFieldResult, CartesianPotentialResult


BACKGROUND = "#fbfbf8"
INK = "#14213d"
MUTED = "#667085"
GRID = "#d9dde3"
PROBE = "#25364a"
ENCAP = "#5eb6c0"
ELECTRODE = "#d94841"
TISSUE = "#edf1ec"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field_cmap",
    ["#081c2d", "#23395d", "#4d648d", "#bc5090", "#ff8c61", "#ffe6a7"],
)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "heatmap_cmap",
    ["#12263a", "#274c77", "#3d7ea6", "#34a0a4", "#80ed99", "#f4d35e"],
)
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "signed_cmap",
    ["#16324f", "#2d6a8f", "#9ecae1", "#f7f7f5", "#f4a261", "#e76f51", "#9d0208"],
)


def apply_publication_style(purpose: str = "presentation") -> None:
    title_size = 17 if purpose == "presentation" else 14
    label_size = 13 if purpose == "presentation" else 11.5
    tick_size = 11.5 if purpose == "presentation" else 10
    line_width = 2.6 if purpose == "presentation" else 2.0
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": tick_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.2,
            "axes.grid": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": line_width,
            "legend.frameon": False,
            "legend.fontsize": 11.5,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _um(values: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(values) * 1e6


def _style_2d_axes(ax: plt.Axes, purpose: str = "presentation") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if purpose == "validation":
        ax.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    ax.tick_params(axis="both", which="major", pad=6)


def _add_side_colorbar(fig: plt.Figure, mappable, ax, label: str) -> mpl.colorbar.Colorbar:
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(length=4, width=0.8)
    colorbar.set_label(label, rotation=90, labelpad=12)
    return colorbar


def _style_3d_axes(
    ax,
    *,
    box_aspect: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hide_ticks: bool = False,
) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(False)
    ax.set_box_aspect(box_aspect)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
        ax.tick_params(colors=MUTED, pad=2)


def _add_figure_note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.28", "fc": "#ffffff", "ec": "#e6e8ec", "alpha": 0.92},
    )


def _cylinder_surface(
    radius: float,
    z_half: float,
    theta_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    theta_count: int = 180,
    z_count: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(theta_range[0], theta_range[1], theta_count)
    z = np.linspace(-z_half, z_half, z_count)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    return _um(x), _um(y), _um(z_grid)


def _electrode_patch_surface(
    config: Cartesian3DConfig,
    theta_count: int = 100,
    z_count: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(-0.5 * config.electrode_arc, 0.5 * config.electrode_arc, theta_count)
    z = np.linspace(-0.5 * config.electrode_height, 0.5 * config.electrode_height, z_count)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = config.probe_radius * np.cos(theta_grid)
    y = config.probe_radius * np.sin(theta_grid)
    return _um(x), _um(y), _um(z_grid)


def _symmetric_norm(field: np.ndarray):
    vmax = float(np.nanmax(np.abs(field)))
    if vmax <= 0.0:
        return colors.Normalize(vmin=-1.0, vmax=1.0)
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _upsample_plane(
    x_values: np.ndarray,
    y_values: np.ndarray,
    field: np.ndarray,
    factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field_zoom = zoom(field, factor, order=3)
    x_zoom = np.linspace(x_values[0], x_values[-1], field_zoom.shape[0])
    y_zoom = np.linspace(y_values[0], y_values[-1], field_zoom.shape[1])
    xx, yy = np.meshgrid(x_zoom, y_zoom, indexing="ij")
    return xx, yy, field_zoom


def _crop_vector(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (values >= lower) & (values <= upper)


def plot_geometry_schematic(config: AxisymmetricConfig, t_enc: float) -> plt.Figure:
    apply_publication_style("presentation")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    probe_radius = float(_um(config.probe_radius))
    outer_radius = float(_um(config.outer_radius))
    outer_half_height = float(_um(config.outer_half_height))
    t_enc_um = float(_um(t_enc))
    electrode_height = float(_um(config.electrode_height))
    y_limit = 175.0

    radial_gradient = np.linspace(0.0, 1.0, 600)[None, :]
    ax.imshow(
        radial_gradient,
        extent=(0.0, outer_radius, -y_limit, y_limit),
        cmap=LinearSegmentedColormap.from_list("tissue_bg", ["#fdfdfb", "#eef3ef"]),
        aspect="auto",
        alpha=1.0,
        zorder=0,
    )

    probe = patches.FancyBboxPatch(
        (0.0, -y_limit),
        probe_radius,
        2.0 * y_limit,
        boxstyle="round,pad=0,rounding_size=5",
        facecolor=PROBE,
        edgecolor="none",
        zorder=3,
    )
    shell = patches.FancyBboxPatch(
        (probe_radius, -y_limit),
        t_enc_um,
        2.0 * y_limit,
        boxstyle="round,pad=0,rounding_size=5",
        facecolor=ENCAP,
        edgecolor="none",
        alpha=0.75,
        zorder=2,
    )
    electrode = patches.FancyBboxPatch(
        (probe_radius - 3.0, -0.5 * electrode_height),
        6.0,
        electrode_height,
        boxstyle="round,pad=0.0,rounding_size=2.2",
        facecolor=ELECTRODE,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    for patch in (shell, probe, electrode):
        ax.add_patch(patch)

    ax.axvline(probe_radius + t_enc_um, color="#96cfd6", lw=1.6, ls=(0, (4, 4)), zorder=4)
    ax.annotate(
        "Localized tissue domain",
        xy=(0.78 * outer_radius, 110.0),
        xytext=(0.60 * outer_radius, 146.0),
        color=INK,
        arrowprops={"arrowstyle": "-", "lw": 1.2, "color": INK},
    )
    ax.annotate(
        "Encapsulation sheath",
        xy=(probe_radius + 0.55 * max(t_enc_um, 4.0), -95.0),
        xytext=(probe_radius + 45.0, -135.0),
        color="#0d5661",
        arrowprops={"arrowstyle": "-", "lw": 1.2, "color": "#0d5661"},
    )
    ax.annotate(
        "Recording band\n(Level 1 ring electrode)",
        xy=(probe_radius, 0.0),
        xytext=(probe_radius + 90.0, 22.0),
        color=ELECTRODE,
        arrowprops={"arrowstyle": "-", "lw": 1.3, "color": ELECTRODE},
    )
    ax.text(8.0, 118.0, "Probe shaft", color="white", fontsize=12.7, fontweight="bold")

    ax.set_xlim(0.0, min(outer_radius, probe_radius + max(t_enc_um, 80.0) + 260.0))
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Radial position [µm]")
    ax.set_ylabel("Axial position [µm]")
    ax.set_title("Axisymmetric Baseline Geometry", loc="left", pad=12)
    _style_2d_axes(ax, "presentation")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16)
    return fig


def plot_axisymmetric_mesh(result: AxisymmetricLeadFieldResult, stride: int = 6) -> plt.Figure:
    apply_publication_style("validation")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    r = _um(result.grid.r_edges)
    z = _um(result.grid.z_edges)
    for value in r[::stride]:
        ax.plot([value, value], [z[0], z[-1]], color="#b6bec9", lw=0.5, alpha=0.95)
    for value in z[::stride]:
        ax.plot([r[0], r[-1]], [value, value], color="#b6bec9", lw=0.5, alpha=0.95)
    ax.axvline(_um(result.grid.probe_radius), color=PROBE, lw=1.8, label="Probe surface")
    ax.axvline(_um(result.grid.probe_radius + result.t_enc), color=ENCAP, lw=1.8, ls="--", label="Encapsulation edge")
    ax.set_xlim(_um(result.grid.probe_radius), _um(result.grid.probe_radius) + max(_um(result.t_enc), 60.0) + 140.0)
    ax.set_ylim(-120.0, 120.0)
    ax.set_xlabel("Radial position [µm]")
    ax.set_ylabel("Axial position [µm]")
    ax.set_title("Mesh Close-Up Near the Probe", loc="left")
    ax.legend(loc="upper right")
    _style_2d_axes(ax, "validation")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.15)
    return fig


def plot_axisymmetric_field_map(
    result: AxisymmetricLeadFieldResult,
    radial_gap: float | None = None,
    z_source: float = 0.0,
    title: str = "Axisymmetric Monopole Lead Field",
) -> plt.Figure:
    apply_publication_style("presentation")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    rr, zz = np.meshgrid(_um(result.grid.r_centers), _um(result.grid.z_centers), indexing="ij")
    magnitude = np.abs(result.lead_field)
    vmin = np.quantile(magnitude[magnitude > 0.0], 0.06)
    vmax = np.quantile(magnitude, 0.995)
    mesh = ax.pcolormesh(
        rr,
        zz,
        magnitude,
        shading="gouraud",
        cmap=FIELD_CMAP,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    contour_levels = np.geomspace(vmin * 1.4, vmax * 0.92, 7)
    ax.contour(rr, zz, magnitude, levels=contour_levels, colors="white", linewidths=0.5, alpha=0.18)

    probe_r = float(_um(result.grid.probe_radius))
    shell_r = float(_um(result.grid.probe_radius + result.t_enc))
    ax.axvline(probe_r, color="white", lw=2.2)
    if result.t_enc > 0.0:
        ax.axvline(shell_r, color="#ccecf1", lw=1.6, ls=(0, (4, 4)))

    if radial_gap is not None:
        source_r = float(_um(result.grid.probe_radius + radial_gap))
        ax.scatter([source_r], [float(_um(z_source))], s=90, color="#f7f7f5", edgecolor=INK, linewidth=1.1, zorder=6)
        ax.annotate(
            "Evaluation point",
            xy=(source_r, float(_um(z_source))),
            xytext=(source_r + 24.0, 70.0),
            color="white",
            fontsize=11.2,
            arrowprops={"arrowstyle": "-", "lw": 1.2, "color": "white"},
        )

    ax.text(probe_r + 5.0, 148.0, "Probe", color="white", fontsize=12.5, fontweight="bold")
    if result.t_enc > 0.0:
        ax.text(shell_r + 3.0, 148.0, "Encapsulation edge", color="#d9f3f6", fontsize=11.2)

    ax.set_xlim(probe_r, probe_r + 230.0)
    ax.set_ylim(-170.0, 170.0)
    ax.set_xlabel("Radial position [µm]")
    ax.set_ylabel("Axial position [µm]")
    ax.set_title(title, loc="left", pad=12)
    _add_figure_note(ax, "Log-scaled transfer field for the Level 1 ring-electrode baseline")
    _style_2d_axes(ax, "presentation")
    _add_side_colorbar(fig, mesh, ax, r"$|G(\mathbf{r})|$ [V A$^{-1}$]")
    fig.subplots_adjust(left=0.10, right=0.91, top=0.90, bottom=0.16)
    return fig


def plot_heatmap(sweep: SweepResults) -> plt.Figure:
    apply_publication_style("presentation")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    image = ax.imshow(
        1e6 * sweep.heatmap,
        origin="lower",
        cmap=HEATMAP_CMAP,
        interpolation="bicubic",
        extent=[
            sweep.sigma_values[0],
            sweep.sigma_values[-1],
            _um(sweep.thicknesses[0]),
            _um(sweep.thicknesses[-1]),
        ],
        aspect="auto",
    )
    sigma_grid, thickness_grid = np.meshgrid(sweep.sigma_values, _um(sweep.thicknesses))
    contours = ax.contour(
        sigma_grid,
        thickness_grid,
        1e6 * sweep.heatmap,
        levels=5,
        colors="white",
        linewidths=0.75,
        alpha=0.55,
    )
    ax.clabel(contours, fmt="%.1f", fontsize=9.8, inline=True)

    ax.set_xlabel("Encapsulation conductivity [S/m]")
    ax.set_ylabel("Encapsulation thickness [µm]")
    ax.set_title(r"Recording Proxy $\phi_{\mathrm{elec}}$ Across Encapsulation Space", loc="left", pad=12)
    _add_figure_note(ax, "Brighter regions indicate stronger recorded potential for the 1 nA baseline source")
    _style_2d_axes(ax, "presentation")
    _add_side_colorbar(fig, image, ax, r"$\phi_{\mathrm{elec}}$ [µV] for 1 nA")
    fig.subplots_adjust(left=0.11, right=0.91, top=0.90, bottom=0.16)
    return fig


def plot_mechanism_curves(sweep: SweepResults) -> plt.Figure:
    apply_publication_style("presentation")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = _um(sweep.thicknesses)
    y_distance = 1e6 * sweep.mechanism_distance
    y_conductivity = 1e6 * sweep.mechanism_conductivity
    y_combined = 1e6 * sweep.mechanism_combined

    ax.plot(x, y_distance, color="#1b7f5a", lw=2.7)
    ax.plot(x, y_conductivity, color="#c0392b", lw=2.7)
    ax.plot(x, y_combined, color=INK, lw=3.6)
    ax.fill_between(x, y_distance, y_combined, color="#1b7f5a", alpha=0.08)
    ax.fill_between(x, y_combined, y_conductivity, color="#c0392b", alpha=0.08)

    ax.text(x[-1] - 18.0, y_distance[-1] - 0.28, "Mechanism 1\nDistance only", color="#1b7f5a", ha="right", va="center")
    ax.text(x[-1] - 18.0, y_conductivity[-1] + 0.06, "Mechanism 2\nConductivity only", color="#c0392b", ha="right", va="center")
    ax.text(x[-1] - 18.0, y_combined[-1] + 0.10, "Combined", color=INK, ha="right", va="center", fontweight="bold")

    ax.set_xlabel("Encapsulation thickness [µm]")
    ax.set_ylabel(r"$\phi_{\mathrm{elec}}$ [µV]")
    ax.set_title("Separated Encapsulation Mechanisms", loc="left", pad=12)
    _style_2d_axes(ax, "presentation")
    ax.set_xlim(x[0], x[-1] + 5.0)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.16)
    return fig


def plot_validation_trend(validation: dict[str, np.ndarray]) -> plt.Figure:
    apply_publication_style("validation")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = _um(validation["radial_gaps"])
    ax.plot(x, 1e6 * validation["numerical"], color=INK, label="Numerical baseline")
    ax.plot(x, 1e6 * validation["analytic"], color="#d1495b", ls="--", label=r"Analytic $1/r$")
    ax.set_xlabel("Neuron-electrode distance [µm]")
    ax.set_ylabel(r"$\phi_{\mathrm{elec}}$ [µV]")
    ax.set_title("Homogeneous-Medium Validation", loc="left")
    ax.legend(loc="upper right")
    _style_2d_axes(ax, "validation")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.16)
    return fig


def plot_validation_series(
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlabel: str,
    title: str,
) -> plt.Figure:
    apply_publication_style("validation")
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    ax.plot(x_values, 1e6 * y_values, marker="o", ms=7.5, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\phi_{\mathrm{elec}}$ [µV]")
    ax.set_title(title, loc="left")
    _style_2d_axes(ax, "validation")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.17)
    return fig


def plot_3d_geometry(config: Cartesian3DConfig, t_enc: float) -> plt.Figure:
    apply_publication_style("presentation")
    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    z_half = 180e-6
    probe = _cylinder_surface(config.probe_radius, z_half, theta_count=220, z_count=140)
    shell = _cylinder_surface(config.probe_radius + t_enc, z_half, theta_count=220, z_count=140)
    electrode = _electrode_patch_surface(config)

    ax.plot_surface(*shell, color=ENCAP, alpha=0.10, linewidth=0, antialiased=True, shade=False)
    ax.plot_surface(*probe, color=PROBE, alpha=0.97, linewidth=0, antialiased=True, shade=True)
    ax.plot_surface(*electrode, color=ELECTRODE, alpha=1.0, linewidth=0, shade=False)

    ax.set_title("Localized Patch Electrode in the Full 3D Model", loc="left", pad=12)
    ax.set_xlabel("x [µm]", color=MUTED)
    ax.set_ylabel("y [µm]", color=MUTED)
    ax.set_zlabel("z [µm]", color=MUTED)
    ax.view_init(elev=16, azim=-62)
    ax.set_xlim(-95.0, 115.0)
    ax.set_ylim(-95.0, 95.0)
    ax.set_zlim(-170.0, 170.0)
    _style_3d_axes(ax, box_aspect=(1.0, 1.0, 1.45))
    fig.subplots_adjust(left=0.00, right=0.97, top=0.92, bottom=0.02)
    return fig


def plot_3d_slice_view(result: CartesianPotentialResult, config: Cartesian3DConfig) -> plt.Figure:
    apply_publication_style("presentation")
    fig = plt.figure(figsize=(9.6, 6.9))
    ax = fig.add_subplot(111, projection="3d")

    x_um = _um(result.grid.x)
    y_um = _um(result.grid.y)
    z_um = _um(result.grid.z)
    field = 1e6 * result.potential
    norm = _symmetric_norm(field)

    iy = np.argmin(np.abs(result.grid.y))
    iz = np.argmin(np.abs(result.grid.z))
    x_mask = _crop_vector(x_um, 15.0, 210.0)
    y_mask = _crop_vector(y_um, -110.0, 110.0)
    z_mask = _crop_vector(z_um, -120.0, 120.0)
    xx_y, zz_y, plane_y = _upsample_plane(x_um[x_mask], z_um[z_mask], field[np.ix_(x_mask, [iy], z_mask)][:, 0, :], factor=5)
    xx_z, yy_z, plane_z = _upsample_plane(x_um[x_mask], y_um[y_mask], field[np.ix_(x_mask, y_mask, [iz])][:, :, 0], factor=5)

    y_plane = np.full_like(xx_y, y_um[iy] - 8.0)
    z_plane = np.full_like(xx_z, z_um[iz] - 40.0)
    ax.plot_surface(
        xx_y,
        y_plane,
        zz_y,
        facecolors=SIGNED_CMAP(norm(plane_y)),
        linewidth=0,
        shade=False,
        alpha=0.98,
    )
    ax.plot_surface(
        xx_z,
        yy_z,
        z_plane,
        facecolors=SIGNED_CMAP(norm(plane_z)),
        linewidth=0,
        shade=False,
        alpha=0.18,
    )

    z_half = 180e-6
    shell = _cylinder_surface(config.probe_radius + result.t_enc, z_half, theta_count=220, z_count=130)
    probe = _cylinder_surface(config.probe_radius, z_half, theta_count=220, z_count=130)
    electrode = _electrode_patch_surface(config)
    ax.plot_wireframe(*shell, color=ENCAP, linewidth=0.4, alpha=0.16, rcount=26, ccount=24)
    ax.plot_surface(*probe, color=PROBE, alpha=0.96, linewidth=0, shade=False)
    ax.plot_surface(*electrode, color=ELECTRODE, alpha=1.0, linewidth=0, shade=False)

    for source in result.sources:
        color = "#f4a261" if source.current > 0 else "#355070"
        ax.scatter(_um(source.x), _um(source.y), _um(source.z), s=72, color=color, edgecolor="white", linewidth=0.8, depthshade=False)

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=SIGNED_CMAP)
    mapper.set_array(field)
    _add_side_colorbar(fig, mapper, ax, "Potential [µV]")

    ax.set_title("3D Dipole Potential with Orthogonal Slice Planes", loc="left", pad=10)
    ax.view_init(elev=16, azim=-62)
    ax.set_xlim(-30.0, 210.0)
    ax.set_ylim(-150.0, 150.0)
    ax.set_zlim(-170.0, 170.0)
    _style_3d_axes(ax, box_aspect=(1.1, 1.0, 1.15), hide_ticks=True)
    fig.subplots_adjust(left=0.00, right=0.92, top=0.91, bottom=0.02)
    return fig


def plot_3d_clipped_view(result: CartesianPotentialResult, config: Cartesian3DConfig) -> plt.Figure:
    apply_publication_style("presentation")
    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    field = 1e6 * result.potential
    field_abs = np.abs(field)
    norm = _symmetric_norm(field)
    xg, yg, zg = np.meshgrid(_um(result.grid.x), _um(result.grid.y), _um(result.grid.z), indexing="ij")

    for quantile, size, alpha in ((0.988, 24.0, 0.13), (0.994, 34.0, 0.25), (0.9975, 46.0, 0.48)):
        threshold = np.quantile(field_abs, quantile)
        mask = (field_abs >= threshold) & (xg >= 0.0) & (np.abs(zg) <= 150.0)
        if np.any(mask):
            ax.scatter(
                xg[mask],
                yg[mask],
                zg[mask],
                c=SIGNED_CMAP(norm(field[mask])),
                s=size,
                alpha=alpha,
                depthshade=False,
                linewidths=0,
            )

    iy = np.argmin(np.abs(result.grid.y))
    x_um = _um(result.grid.x)
    z_um = _um(result.grid.z)
    x_mask = _crop_vector(x_um, 15.0, 210.0)
    z_mask = _crop_vector(z_um, -120.0, 120.0)
    xx_y, zz_y, plane_y = _upsample_plane(x_um[x_mask], z_um[z_mask], field[np.ix_(x_mask, [iy], z_mask)][:, 0, :], factor=4)
    y_plane = np.full_like(xx_y, -40.0)
    ax.plot_surface(
        xx_y,
        y_plane,
        zz_y,
        facecolors=SIGNED_CMAP(norm(plane_y)),
        linewidth=0,
        shade=False,
        alpha=0.30,
    )

    probe = _cylinder_surface(config.probe_radius, 185e-6, theta_range=(-0.4 * np.pi, 0.6 * np.pi), theta_count=200, z_count=130)
    shell = _cylinder_surface(config.probe_radius + result.t_enc, 185e-6, theta_range=(-0.4 * np.pi, 0.6 * np.pi), theta_count=140, z_count=90)
    electrode = _electrode_patch_surface(config)
    ax.plot_wireframe(*shell, color=ENCAP, linewidth=0.45, alpha=0.18, rcount=22, ccount=20)
    ax.plot_surface(*probe, color=PROBE, alpha=0.94, linewidth=0, shade=False)
    ax.plot_surface(*electrode, color=ELECTRODE, alpha=1.0, linewidth=0, shade=False)

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=SIGNED_CMAP)
    mapper.set_array(field)
    _add_side_colorbar(fig, mapper, ax, "Potential [µV]")

    ax.set_title("Clipped View of the Strong-Field Region", loc="left", pad=10)
    ax.view_init(elev=18, azim=-62)
    ax.set_xlim(-10.0, 210.0)
    ax.set_ylim(-150.0, 150.0)
    ax.set_zlim(-180.0, 180.0)
    _style_3d_axes(ax, box_aspect=(1.08, 1.0, 1.12), hide_ticks=True)
    fig.subplots_adjust(left=0.00, right=0.92, top=0.91, bottom=0.02)
    return fig


def plot_3d_hero(result: CartesianPotentialResult, config: Cartesian3DConfig) -> plt.Figure:
    apply_publication_style("presentation")
    fig = plt.figure(figsize=(9.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    field = 1e6 * result.potential
    field_abs = np.abs(field)
    norm = _symmetric_norm(field)
    xg, yg, zg = np.meshgrid(_um(result.grid.x), _um(result.grid.y), _um(result.grid.z), indexing="ij")

    iy = np.argmin(np.abs(result.grid.y))
    x_um = _um(result.grid.x)
    z_um = _um(result.grid.z)
    x_mask = _crop_vector(x_um, 15.0, 210.0)
    z_mask = _crop_vector(z_um, -120.0, 120.0)
    xx_y, zz_y, plane_y = _upsample_plane(x_um[x_mask], z_um[z_mask], field[np.ix_(x_mask, [iy], z_mask)][:, 0, :], factor=5)
    ax.plot_surface(
        xx_y,
        np.full_like(xx_y, 0.0),
        zz_y,
        facecolors=SIGNED_CMAP(norm(plane_y)),
        linewidth=0,
        shade=False,
        alpha=0.98,
    )

    for quantile, size, alpha in ((0.992, 28.0, 0.12), (0.996, 40.0, 0.26), (0.9985, 52.0, 0.54)):
        threshold = np.quantile(field_abs, quantile)
        mask = (field_abs >= threshold) & (xg >= 0.0) & (np.abs(yg) <= 125.0) & (np.abs(zg) <= 150.0)
        if np.any(mask):
            ax.scatter(
                xg[mask],
                yg[mask],
                zg[mask],
                c=SIGNED_CMAP(norm(field[mask])),
                s=size,
                alpha=alpha,
                depthshade=False,
                linewidths=0,
            )

    probe = _cylinder_surface(config.probe_radius, 185e-6, theta_range=(-0.55 * np.pi, 0.55 * np.pi), theta_count=220, z_count=140)
    shell = _cylinder_surface(config.probe_radius + result.t_enc, 185e-6, theta_range=(-0.55 * np.pi, 0.55 * np.pi), theta_count=180, z_count=110)
    electrode = _electrode_patch_surface(config)
    ax.plot_wireframe(*shell, color=ENCAP, linewidth=0.4, alpha=0.16, rcount=20, ccount=22)
    ax.plot_surface(*probe, color=PROBE, alpha=0.94, linewidth=0, shade=False)
    ax.plot_surface(*electrode, color=ELECTRODE, alpha=1.0, linewidth=0, shade=False)

    for source in result.sources:
        color = "#f4a261" if source.current > 0 else "#264653"
        ax.scatter(_um(source.x), _um(source.y), _um(source.z), s=90, color=color, edgecolor="white", linewidth=0.9, depthshade=False)

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=SIGNED_CMAP)
    mapper.set_array(field)
    _add_side_colorbar(fig, mapper, ax, "Potential [µV]")

    ax.set_title("Full 3D Dipole Baseline", loc="left", pad=10)
    ax.text2D(
        0.02,
        0.94,
        "Hero figure: localized patch electrode, encapsulation shell, dipole slice, and clipped strong-field halo",
        transform=ax.transAxes,
        color=INK,
        fontsize=10.6,
    )
    ax.view_init(elev=17, azim=-62)
    ax.set_xlim(-10.0, 210.0)
    ax.set_ylim(-140.0, 140.0)
    ax.set_zlim(-175.0, 175.0)
    _style_3d_axes(ax, box_aspect=(1.14, 1.0, 1.08), hide_ticks=True)
    fig.subplots_adjust(left=0.00, right=0.92, top=0.91, bottom=0.02)
    return fig
