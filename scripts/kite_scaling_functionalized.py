#!/usr/bin/env python3

import numpy as np
import math as m
import yaml
from pathlib import Path
from scipy.special import comb
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from geometrical_kite_optimization.kite_definition import KiteDefinition


def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
    )
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals


def get_flat_span(y, z):
    length = 0
    for i in range(len(y) - 1):
        dy = y[i] - y[i + 1]
        dz = z[i] - z[i + 1]
        dist = np.sqrt(dy**2 + dz**2)
        length += dist
    return length


def get_le_arc_curve(phi, gamma, delta, old_kite):
    tip_height = -m.tan(phi)
    points = [[1, tip_height], [1, tip_height + gamma], [delta, 0], [0, 0]]
    bez_y, bez_z = bezier(points, nTimes=1000)

    LE_norm_y = []
    LE_norm_z = []
    for ind in old_kite.ind_lst:
        LE_norm_y.append(bez_y[ind])
        LE_norm_z.append(bez_z[ind])

    return LE_norm_y, LE_norm_z


def scale_arc_to_span(phi, gamma, delta, old_kite, new_ar, area):
    LE_y_norm, LE_z_norm = get_le_arc_curve(phi, gamma, delta, old_kite)
    normalized_span = get_flat_span(LE_y_norm, LE_z_norm)
    target_span = m.sqrt(new_ar * area)
    span_scaling_factor = target_span / (2 * normalized_span)

    LE_y_scaled, LE_z_scaled = [0], [0]
    for i in range(len(LE_y_norm) - 1):
        j = len(LE_y_norm) - 1 - i
        dz = LE_z_norm[j - 1] - LE_z_norm[j]
        dy = LE_y_norm[j - 1] - LE_y_norm[j]
        LE_y_scaled.append(LE_y_scaled[-1] + span_scaling_factor * dy)
        LE_z_scaled.append(LE_z_scaled[-1] + span_scaling_factor * dz)

    old_tip_height = old_kite.height()
    LE_z_scaled_trans = [z - min(LE_z_scaled) + old_tip_height for z in LE_z_scaled]

    return LE_y_scaled, LE_z_scaled_trans


def get_area_preservation_scaling_factor(old_kite, new_ar):
    original_area = old_kite.get_old_area()
    original_span = old_kite.get_old_span()[0]
    original_ar = old_kite.old_aspect_ratio()

    span_scaling_for_ar_only = m.sqrt(new_ar / original_ar)
    new_span_ar_only = original_span * span_scaling_for_ar_only
    new_area_without_chord_adjustment = (new_span_ar_only**2) / new_ar
    area_scaling_factor = m.sqrt(original_area / new_area_without_chord_adjustment)

    return area_scaling_factor


def calculate_chord_scaling(old_kite, new_ar, area):
    theoretical_new_span = m.sqrt(new_ar * area)
    old_span = old_kite.get_old_span()[0]
    old_ar = old_kite.old_aspect_ratio()

    basic_chord_scaling = theoretical_new_span / old_span * old_ar / new_ar
    area_scaling_factor = get_area_preservation_scaling_factor(old_kite, new_ar)

    return basic_chord_scaling * area_scaling_factor


def align_center_crossing(full_arc_LE, full_arc_TE, old_kite):
    orig_LE, orig_TE = old_kite.get_arc(halve=False)
    orig_y_coords = orig_LE[:, 1]
    abs_y_orig = np.abs(orig_y_coords)
    center_indices_orig = np.argsort(abs_y_orig)[:2]

    center_point_orig_LE = 0.5 * (
        orig_LE[center_indices_orig[0]] + orig_LE[center_indices_orig[1]]
    )
    center_point_orig_TE = 0.5 * (
        orig_TE[center_indices_orig[0]] + orig_TE[center_indices_orig[1]]
    )

    new_y_coords = full_arc_LE[:, 1]
    abs_y_new = np.abs(new_y_coords)
    center_indices_new = np.argsort(abs_y_new)[:2]

    center_point_new_LE = 0.5 * (
        full_arc_LE[center_indices_new[0]] + full_arc_LE[center_indices_new[1]]
    )
    center_point_new_TE = 0.5 * (
        full_arc_TE[center_indices_new[0]] + full_arc_TE[center_indices_new[1]]
    )

    translation_LE = center_point_orig_LE - center_point_new_LE
    translation_TE = center_point_orig_TE - center_point_new_TE

    return full_arc_LE + translation_LE, full_arc_TE + translation_TE


def generate_chord_vectors(old_kite, phi, gamma, delta, new_ar, area):
    chord_scaling = calculate_chord_scaling(old_kite, new_ar, area)
    LE_arc_y, LE_arc_z = scale_arc_to_span(phi, gamma, delta, old_kite, new_ar, area)
    LE_coords, TE_coords = old_kite.get_arc(halve=True)

    new_LE_coords = []
    new_TE_coords = []

    for i, LE_data in enumerate(LE_coords):
        j = len(LE_coords) - 1 - i
        LE_vector = LE_data
        TE_vector = TE_coords[i]
        old_chord_vector = TE_vector - LE_vector
        qchord = LE_vector + 0.25 * old_chord_vector
        hchord_yz_offset = np.array([qchord[1], qchord[2]]) - np.array(
            [LE_vector[1], LE_vector[2]]
        )

        new_qchord = np.array(
            [
                qchord[0],
                LE_arc_y[j] + hchord_yz_offset[0],
                LE_arc_z[j] + hchord_yz_offset[1],
            ]
        )
        new_LE = new_qchord - 0.25 * old_chord_vector * chord_scaling
        new_TE = new_qchord + 0.75 * old_chord_vector * chord_scaling

        new_LE_coords.append(new_LE)
        new_TE_coords.append(new_TE)

    full_arc_LE = np.vstack(
        [new_LE_coords[:-1], np.flip(new_LE_coords[:-1], 0) * np.array([1, -1, 1])]
    )
    full_arc_TE = np.vstack(
        [new_TE_coords[:-1], np.flip(new_TE_coords[:-1], 0) * np.array([1, -1, 1])]
    )

    return align_center_crossing(full_arc_LE, full_arc_TE, old_kite)


def generate_wing_sections_data(new_wing_sections):
    headers = [
        "airfoil_id",
        "LE_x",
        "LE_y",
        "LE_z",
        "TE_x",
        "TE_y",
        "TE_z",
        "VUP_x",
        "VUP_y",
        "VUP_z",
    ]
    data = []

    for rib in new_wing_sections:
        row = [rib["airfoil_id"]] + list(rib["LE"]) + list(rib["TE"]) + list(rib["VUP"])
        data.append(row)

    return {"headers": headers, "data": data}


def create_new_wing_sections(full_arc_LE, full_arc_TE, old_kite):
    new_wing_sections = []

    for i, LE_coord in enumerate(full_arc_LE):
        rib = {
            "LE": LE_coord,
            "TE": full_arc_TE[i],
            "VUP": old_kite.config["wing_sections"]["data"][i][
                old_kite.header_map["VUP_x"] : old_kite.header_map["VUP_z"] + 1
            ],
            "airfoil_id": old_kite.config["wing_sections"]["data"][i][
                old_kite.header_map["airfoil_id"]
            ],
        }
        new_wing_sections.append(rib)

    return generate_wing_sections_data(new_wing_sections)


def export_yaml(old_kite, wing_sections, new_ar, PROJECT_DIR):
    old_file = old_kite.config
    new_yaml_file_path = (
        Path(PROJECT_DIR)
        / "processed_data"
        / f"{old_kite.kite_name}_aspect_ratio_{new_ar:.2f}.yaml"
    )

    yaml_data = {
        "wing_sections": {
            "headers": wing_sections["headers"],
            "data": wing_sections["data"],
        },
        "wing_airfoils": old_file["wing_airfoils"],
        "bridle_nodes": old_file.get("bridle_nodes", {}),
        "bridle_lines": old_file.get("bridle_lines", {}),
        "bridle_connections": old_file.get("bridle_connections", {}),
    }

    new_yaml_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(new_yaml_file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    return new_yaml_file_path


def main(
    base_kite,
    new_ar=None,
    new_phi=None,
    new_delta=None,
    new_gamma=None,
    PROJECT_DIR=".",
    is_gamma_a_percentage=True,
):
    # Get base parameters
    _, _, _, _, _, base_phi, base_delta, base_gamma = base_kite.get_bezier_curve(
        normalized=True, plot=False
    )

    # Set parameters
    new_ar = new_ar or base_kite.old_aspect_ratio()
    phi = np.deg2rad(new_phi) if new_phi is not None else base_phi
    delta = new_delta if new_delta is not None else base_delta

    if new_gamma is not None:
        gamma = new_gamma
    else:
        gamma = base_gamma
        if is_gamma_a_percentage:
            base_tip_height = m.tan(base_phi)
            gamma_fraction = base_gamma / base_tip_height
            new_tip_height = m.tan(phi)
            gamma = new_tip_height * gamma_fraction

    area = base_kite.get_old_area()

    # Generate scaled geometry
    full_arc_LE, full_arc_TE = generate_chord_vectors(
        base_kite, phi, gamma, delta, new_ar, area
    )
    wing_sections = create_new_wing_sections(full_arc_LE, full_arc_TE, base_kite)

    # Export YAML
    yaml_path = export_yaml(base_kite, wing_sections, new_ar, PROJECT_DIR)

    # Create new KiteDefinition object from the generated YAML
    new_kite_name = yaml_path.stem  # Get filename without extension
    new_kite = KiteDefinition(new_kite_name)
    new_kite.process()

    return yaml_path, new_kite
