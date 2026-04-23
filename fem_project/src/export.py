"""Structured VTK export helpers for ParaView postprocessing."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .config import AxisymmetricConfig, Cartesian3DConfig
from .solver import AxisymmetricLeadFieldResult, CartesianPotentialResult


REGION_TISSUE = 0
REGION_ENCAPSULATION = 1
REGION_PROBE = 2


def _vtk_type(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.floating):
        return "Float64"
    if np.issubdtype(array.dtype, np.signedinteger):
        return "Int32"
    if np.issubdtype(array.dtype, np.unsignedinteger) or array.dtype == np.bool_:
        return "UInt8"
    raise TypeError(f"Unsupported dtype for VTK export: {array.dtype}")


def _format_array(array: np.ndarray) -> str:
    values = np.asarray(array)
    if values.dtype == np.bool_:
        values = values.astype(np.uint8)
    flat = np.ravel(values, order="F")
    if np.issubdtype(values.dtype, np.floating):
        return " ".join(f"{value:.12e}" for value in flat)
    return " ".join(str(int(value)) for value in flat)


def _write_data_block(parent: ET.Element, data: dict[str, np.ndarray]) -> None:
    for name, values in data.items():
        array = np.asarray(values)
        data_array = ET.SubElement(
            parent,
            "DataArray",
            type=_vtk_type(array),
            Name=name,
            format="ascii",
        )
        data_array.text = _format_array(array)


def write_rectilinear_grid(
    path: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)

    nx = x_coords.size - 1
    ny = y_coords.size - 1
    nz = z_coords.size - 1
    expected_shape = (nx, ny, nz)
    for name, values in cell_data.items():
        if np.asarray(values).shape != expected_shape:
            raise ValueError(f"Cell data '{name}' has shape {np.asarray(values).shape}, expected {expected_shape}")

    vtk = ET.Element(
        "VTKFile",
        type="RectilinearGrid",
        version="0.1",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(
        vtk,
        "RectilinearGrid",
        WholeExtent=f"0 {nx} 0 {ny} 0 {nz}",
    )
    piece = ET.SubElement(
        grid,
        "Piece",
        Extent=f"0 {nx} 0 {ny} 0 {nz}",
    )

    if field_data:
        field_node = ET.SubElement(piece, "FieldData")
        _write_data_block(field_node, field_data)

    cell_node = ET.SubElement(piece, "CellData")
    _write_data_block(cell_node, cell_data)

    coords = ET.SubElement(piece, "Coordinates")
    for axis_name, values in (("x_coordinates", x_coords), ("y_coordinates", y_coords), ("z_coordinates", z_coords)):
        data_array = ET.SubElement(
            coords,
            "DataArray",
            type="Float64",
            Name=axis_name,
            NumberOfComponents="1",
            format="ascii",
        )
        data_array.text = _format_array(values)

    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def write_image_data(
    path: Path,
    *,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = next(iter(cell_data.values())).shape
    nx, ny, nz = shape
    for name, values in cell_data.items():
        if np.asarray(values).shape != shape:
            raise ValueError(f"Cell data '{name}' has shape {np.asarray(values).shape}, expected {shape}")

    vtk = ET.Element(
        "VTKFile",
        type="ImageData",
        version="0.1",
        byte_order="LittleEndian",
    )
    image = ET.SubElement(
        vtk,
        "ImageData",
        WholeExtent=f"0 {nx} 0 {ny} 0 {nz}",
        Origin=f"{origin[0]:.12e} {origin[1]:.12e} {origin[2]:.12e}",
        Spacing=f"{spacing[0]:.12e} {spacing[1]:.12e} {spacing[2]:.12e}",
    )
    piece = ET.SubElement(image, "Piece", Extent=f"0 {nx} 0 {ny} 0 {nz}")

    if field_data:
        field_node = ET.SubElement(piece, "FieldData")
        _write_data_block(field_node, field_data)

    cell_node = ET.SubElement(piece, "CellData")
    _write_data_block(cell_node, cell_data)
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def write_polydata_points(
    path: Path,
    points: np.ndarray,
    point_data: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Point array must have shape (n, 3), got {pts.shape}")

    n_points = pts.shape[0]
    connectivity = np.arange(n_points, dtype=np.int32)
    offsets = np.arange(1, n_points + 1, dtype=np.int32)

    vtk = ET.Element(
        "VTKFile",
        type="PolyData",
        version="0.1",
        byte_order="LittleEndian",
    )
    poly = ET.SubElement(vtk, "PolyData")
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(n_points),
        NumberOfVerts=str(n_points),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )

    if point_data:
        for name, values in point_data.items():
            if np.asarray(values).shape[0] != n_points:
                raise ValueError(f"Point data '{name}' length does not match number of points")
        point_node = ET.SubElement(piece, "PointData")
        _write_data_block(point_node, point_data)

    points_node = ET.SubElement(piece, "Points")
    point_array = ET.SubElement(
        points_node,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    point_array.text = _format_array(pts)

    verts = ET.SubElement(piece, "Verts")
    conn_array = ET.SubElement(verts, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn_array.text = _format_array(connectivity)
    offset_array = ET.SubElement(verts, "DataArray", type="Int32", Name="offsets", format="ascii")
    offset_array.text = _format_array(offsets)

    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def _deposit_sources_to_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sources,
) -> np.ndarray:
    field = np.zeros((x.size, y.size, z.size), dtype=float)
    for source in sources:
        sx, sy, sz, current = source.x, source.y, source.z, source.current
        if not (x[0] <= sx <= x[-1] and y[0] <= sy <= y[-1] and z[0] <= sz <= z[-1]):
            continue

        ix = int(np.clip(np.searchsorted(x, sx) - 1, 0, x.size - 2))
        iy = int(np.clip(np.searchsorted(y, sy) - 1, 0, y.size - 2))
        iz = int(np.clip(np.searchsorted(z, sz) - 1, 0, z.size - 2))
        tx = 0.0 if x[ix + 1] == x[ix] else (sx - x[ix]) / (x[ix + 1] - x[ix])
        ty = 0.0 if y[iy + 1] == y[iy] else (sy - y[iy]) / (y[iy + 1] - y[iy])
        tz = 0.0 if z[iz + 1] == z[iz] else (sz - z[iz]) / (z[iz + 1] - z[iz])

        for dx, wx in enumerate((1.0 - tx, tx)):
            for dy, wy in enumerate((1.0 - ty, ty)):
                for dz, wz in enumerate((1.0 - tz, tz)):
                    field[ix + dx, iy + dy, iz + dz] += current * wx * wy * wz
    return field


def export_axisymmetric_result_to_vtr(
    path: Path,
    result: AxisymmetricLeadFieldResult,
    config: AxisymmetricConfig,
    *,
    source_gap: float,
    source_current: float,
    z_source: float,
) -> Path:
    dr = result.grid.dr
    n_probe = max(1, int(np.ceil(config.probe_radius / dr)))
    probe_edges = np.linspace(0.0, config.probe_radius, n_probe + 1)
    x_edges = np.concatenate([probe_edges, result.grid.r_edges[1:]]) * 1e6
    y_edges = np.array([-0.5, 0.5], dtype=float)
    z_edges = result.grid.z_edges * 1e6

    nr_full = x_edges.size - 1
    nz = result.grid.z_centers.size
    shape = (nr_full, 1, nz)

    phi_unit_drive = np.zeros(shape, dtype=float)
    lead_field = np.zeros(shape, dtype=float)
    response_1na_uv = np.zeros(shape, dtype=float)
    conductivity = np.full(shape, config.sigma_probe, dtype=float)
    region_label = np.full(shape, REGION_PROBE, dtype=np.int32)
    probe_mask = np.ones(shape, dtype=np.uint8)
    encapsulation_mask = np.zeros(shape, dtype=np.uint8)
    tissue_mask = np.zeros(shape, dtype=np.uint8)
    solution_domain_mask = np.zeros(shape, dtype=np.uint8)
    electrode_band_mask = np.zeros(shape, dtype=np.uint8)
    source_evaluation_mask = np.zeros(shape, dtype=np.uint8)

    exterior = slice(n_probe, None)
    phi_unit_drive[exterior, 0, :] = result.potential_voltage_drive
    lead_field[exterior, 0, :] = result.lead_field
    response_1na_uv[exterior, 0, :] = result.lead_field * source_current * 1e6
    conductivity[exterior, 0, :] = result.conductivity
    solution_domain_mask[exterior, 0, :] = 1
    probe_mask[exterior, 0, :] = 0

    shell_rows = result.grid.r_centers <= (config.probe_radius + result.t_enc + 1e-18)
    encap_full = np.zeros((nr_full, nz), dtype=bool)
    encap_full[n_probe:, :] = shell_rows[:, None]
    tissue_full = np.zeros((nr_full, nz), dtype=bool)
    tissue_full[n_probe:, :] = ~shell_rows[:, None]
    encapsulation_mask[:, 0, :] = encap_full.astype(np.uint8)
    tissue_mask[:, 0, :] = tissue_full.astype(np.uint8)

    region_full = np.full((nr_full, nz), REGION_PROBE, dtype=np.int32)
    region_full[n_probe:, :] = np.where(shell_rows[:, None], REGION_ENCAPSULATION, REGION_TISSUE)
    region_label[:, 0, :] = region_full

    electrode_z = result.grid.electrode_mask_z
    electrode_band_mask[n_probe - 1, 0, electrode_z] = 1
    electrode_band_mask[n_probe, 0, electrode_z] = 1

    source_r = config.probe_radius + source_gap
    i_src = n_probe + int(np.argmin(np.abs(result.grid.r_centers - source_r)))
    j_src = int(np.argmin(np.abs(result.grid.z_centers - z_source)))
    source_evaluation_mask[i_src, 0, j_src] = 1

    field_data = {
        "source_radius_um": np.array([source_r * 1e6]),
        "source_z_um": np.array([z_source * 1e6]),
        "source_current_nA": np.array([source_current * 1e9]),
        "probe_radius_um": np.array([config.probe_radius * 1e6]),
        "encapsulation_thickness_um": np.array([result.t_enc * 1e6]),
    }
    cell_data = {
        "phi_unit_electrode_drive_V": phi_unit_drive,
        "lead_field_V_per_A": lead_field,
        "recording_response_1nA_uV": response_1na_uv,
        "abs_recording_response_1nA_uV": np.abs(response_1na_uv),
        "conductivity_S_per_m": conductivity,
        "material_region_id": region_label,
        "probe_mask": probe_mask,
        "encapsulation_mask": encapsulation_mask,
        "tissue_mask": tissue_mask,
        "solution_domain_mask": solution_domain_mask,
        "electrode_band_mask": electrode_band_mask,
        "source_evaluation_mask": source_evaluation_mask,
    }
    write_rectilinear_grid(path, x_edges, y_edges, z_edges, cell_data, field_data)
    return path


def export_cartesian_result_to_vti(
    path: Path,
    result: CartesianPotentialResult,
    config: Cartesian3DConfig,
) -> Path:
    x = result.grid.x
    y = result.grid.y
    z = result.grid.z
    dx_um = result.grid.dx * 1e6
    dy_um = result.grid.dy * 1e6
    dz_um = result.grid.dz * 1e6
    origin = (
        (x[0] - 0.5 * result.grid.dx) * 1e6,
        (y[0] - 0.5 * result.grid.dy) * 1e6,
        (z[0] - 0.5 * result.grid.dz) * 1e6,
    )

    xg, yg, zg = result.grid.mesh()
    rho = np.sqrt(xg**2 + yg**2)
    theta = np.arctan2(yg, xg)
    probe_mask = rho <= config.probe_radius
    encapsulation_mask = (rho > config.probe_radius) & (rho <= config.probe_radius + result.t_enc)
    tissue_mask = ~(probe_mask | encapsulation_mask)
    electrode_shell = rho >= max(config.probe_radius - max(result.grid.dx, result.grid.dy), 0.0)
    electrode_patch_mask = probe_mask & electrode_shell & (np.abs(theta) <= 0.5 * config.electrode_arc) & (
        np.abs(zg) <= 0.5 * config.electrode_height
    )
    region_label = np.where(probe_mask, REGION_PROBE, np.where(encapsulation_mask, REGION_ENCAPSULATION, REGION_TISSUE)).astype(np.int32)
    source_signed_nA = _deposit_sources_to_cells(x, y, z, result.sources) * 1e9
    source_positions = (
        np.array([[source.x, source.y, source.z] for source in result.sources], dtype=float)
        if result.sources
        else np.zeros((0, 3), dtype=float)
    )
    source_currents = (
        np.array([source.current for source in result.sources], dtype=float)
        if result.sources
        else np.zeros(0, dtype=float)
    )
    source_centroid = source_positions.mean(axis=0) if source_positions.size else np.zeros(3, dtype=float)

    field_data = {
        "probe_radius_um": np.array([config.probe_radius * 1e6]),
        "encapsulation_thickness_um": np.array([result.t_enc * 1e6]),
        "electrode_height_um": np.array([config.electrode_height * 1e6]),
        "electrode_arc_deg": np.array([np.rad2deg(config.electrode_arc)]),
        "source_count": np.array([len(result.sources)], dtype=np.int32),
        "net_source_current_nA": np.array([source_currents.sum() * 1e9]),
        "source_centroid_x_um": np.array([source_centroid[0] * 1e6]),
        "source_centroid_y_um": np.array([source_centroid[1] * 1e6]),
        "source_centroid_z_um": np.array([source_centroid[2] * 1e6]),
    }
    cell_data = {
        "phi_V": result.potential,
        "phi_uV": result.potential * 1e6,
        "abs_phi_uV": np.abs(result.potential) * 1e6,
        "log10_abs_phi_uV": np.log10(np.maximum(np.abs(result.potential) * 1e6, 1e-6)),
        "conductivity_S_per_m": result.conductivity,
        "material_region_id": region_label,
        "probe_mask": probe_mask.astype(np.uint8),
        "encapsulation_mask": encapsulation_mask.astype(np.uint8),
        "tissue_mask": tissue_mask.astype(np.uint8),
        "electrode_patch_mask": electrode_patch_mask.astype(np.uint8),
        "source_signed_nA": source_signed_nA,
    }
    write_image_data(
        path,
        origin=origin,
        spacing=(dx_um, dy_um, dz_um),
        cell_data=cell_data,
        field_data=field_data,
    )
    return path


def export_point_sources_to_vtp(
    path: Path,
    result: CartesianPotentialResult,
) -> Path:
    points_um = np.array([[source.x, source.y, source.z] for source in result.sources], dtype=float) * 1e6
    point_data = {
        "source_current_nA": np.array([source.current for source in result.sources], dtype=float) * 1e9,
        "source_index": np.arange(len(result.sources), dtype=np.int32),
    }
    write_polydata_points(path, points_um, point_data=point_data)
    return path
