#!/usr/bin/env python3

import numpy as np
import math as m
import yaml
import re
import os
from pathlib import Path
from scipy.special import comb
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

    # Helper function to convert NumPy types to Python types
    def convert_numpy_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.integer, np.floating)):
            return item.item()
        else:
            return item

    for rib in new_wing_sections:
        # Convert each coordinate component individually to ensure proper conversion
        le_coords = [convert_numpy_item(x) for x in rib["LE"]]
        te_coords = [convert_numpy_item(x) for x in rib["TE"]]
        vup_coords = [convert_numpy_item(x) for x in rib["VUP"]]

        row = [rib["airfoil_id"]] + le_coords + te_coords + vup_coords
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


def export_yaml(old_kite, wing_sections, new_ar):
    old_file_config = old_kite.config

    yaml_data = {
        "wing_sections": {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - airfoil_id: integer, unique identifier for the airfoil (matches wing_airfoils)": None,
            "#   - LE_x: x-coordinate of leading edge": None,
            "#   - LE_y: y-coordinate of leading edge": None,
            "#   - LE_z: z-coordinate of leading edge": None,
            "#   - TE_x: x-coordinate of trailing edge": None,
            "#   - TE_y: y-coordinate of trailing edge": None,
            "#   - TE_z: z-coordinate of trailing edge": None,
            "# ---------------------------------------------------------------": None,
            **wing_sections,
        },
        "": None,  # Empty line before wing_airfoils
        "wing_airfoils": {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - airfoil_id: integer, unique identifier for the airfoil": None,
            "#   - type: one of [neuralfoil, breukels_regression, masure_regression, polars]": None,
            "#   - info_dict: dictionary with parameters depending on 'type'": None,
            "#": None,
            "# info_dict fields by type:": None,
            "#   - breukels_regression:": None,
            "#       t: Tube diameter non-dimensionalized by chord (required)": None,
            "#       kappa: Maximum camber height/magnitude, non-dimensionalized by chord (required)": None,
            "#   - neuralfoil:": None,
            "#       dat_file_path: Path to airfoil .dat file (x, y columns)": None,
            '#       model_size: NeuralFoil model size (e.g., "xxxlarge")': None,
            "#       xtr_lower: Lower transition location (0=forced, 1=free)": None,
            "#       xtr_upper: Upper transition location": None,
            "#       n_crit: Critical amplification factor (see guidelines below)": None,
            "#         n_crit guidelines:": None,
            "#           Sailplane:           12–14": None,
            "#           Motorglider:         11–13": None,
            "#           Clean wind tunnel:   10–12": None,
            '#           Average wind tunnel: 9   (standard "e^9 method")': None,
            "#           Dirty wind tunnel:   4–8": None,
            "#   - polars:": None,
            "#       csv_file_path: Path to polar CSV file (columns: alpha [rad], cl, cd, cm)": None,
            "#   - masure_regression:": None,
            "#       t, eta, kappa, delta, lamba, phi: Regression parameters": None,
            "#   - inviscid:": None,
            "#       no further data is required": None,
            "# ---------------------------------------------------------------": None,
            **old_file_config["wing_airfoils"],
        },
    }

    bridle_lines_yaml = old_file_config.get("bridle_lines", {})
    bridle_nodes = old_file_config.get("bridle_nodes", {})
    bridle_connections = old_file_config.get("bridle_connections", {})

    yaml_data[" "] = None  # Empty line before bridle_nodes
    yaml_data["bridle_nodes"] = {
        "# ---------------------------------------------------------------": None,
        "# headers:": None,
        "#   - id: integer, unique identifier for the node": None,
        "#   - x: x-coordinate [m]": None,
        "#   - y: y-coordinate [m]": None,
        "#   - z: z-coordinate [m]": None,
        "#   - type: node type, either 'knot' or 'pulley'": None,
        "# ---------------------------------------------------------------": None,
        **bridle_nodes,
    }

    yaml_data["  "] = None  # Empty line before bridle_lines
    yaml_data["bridle_lines"] = {
        "# ---------------------------------------------------------------": None,
        "# headers:": None,
        "#   - name: string, line name": None,
        "#   - rest_length: measured rest length [m]": None,
        "#   - diameter: line diameter [m]": None,
        "#   - material: string, material type (e.g., dyneema)": None,
        "#   - density: material density [kg/m^3]": None,
        "# ---------------------------------------------------------------": None,
        **bridle_lines_yaml,
    }

    yaml_data["   "] = None  # Empty line before bridle_connections
    yaml_data["bridle_connections"] = {
        "# ---------------------------------------------------------------": None,
        "# headers:": None,
        "#   - name: string, line name": None,
        "#   - ci: integer, node id (start)": None,
        "#   - cj: integer, node id (end)": None,
        "#   - ck: integer, third node id (only for pulleys, else omitted or 0)": None,
        "# ---------------------------------------------------------------": None,
        **bridle_connections,
    }

    old_yaml_path = old_kite.yaml_path
    kite_name = old_yaml_path.parent.name  # Get the kite name from parent directory

    # Create new path in processed_data directory with kite subdirectory
    # Go up to project root and then into processed_data/kite_name/
    processed_data_dir = (
        old_yaml_path.parent.parent.parent / "processed_data" / kite_name
    )
    new_yaml_path = processed_data_dir / f"config_kite_aspect_ratio_{new_ar:.2f}.yaml"

    # Ensure the processed_data directory exists
    new_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom YAML representers to handle comments and formatting
    def represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "")

    def represent_list(self, data):
        # Force inline (flow) style for data arrays
        # Check if this is a data row (list with mixed types including dicts)
        if data and len(data) <= 10:  # Typical data row length
            # Special handling for rows that contain dictionaries
            if any(isinstance(item, dict) for item in data):
                # For wing_airfoils data rows with info_dict
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )
            # For simple numeric/string data rows
            elif all(isinstance(item, (int, float, str)) for item in data):
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    yaml.add_representer(type(None), represent_none)
    yaml.add_representer(list, represent_list)

    with open(new_yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # Post-process to clean up comments (remove quotes and None values)
    with open(new_yaml_path, "r") as f:
        content = f.read()

    # Clean up comment formatting
    content = content.replace("'#", "#")
    content = content.replace("': null", "")
    content = content.replace("': ''", "")

    # Convert empty line keys to actual blank lines - handle all patterns
    import re

    # Remove lines that are just empty keys with various patterns
    content = re.sub(r"^\? ''\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? ' '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '  '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '   '\n:\n", "\n", content, flags=re.MULTILINE)

    # Also handle simple key patterns
    content = re.sub(r"^'': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^' ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'  ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'   ': *$", "", content, flags=re.MULTILINE)

    with open(new_yaml_path, "w") as f:
        f.write(content)

    return new_yaml_path


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
    yaml_path = export_yaml(base_kite, wing_sections, new_ar)
    print(f"✅ New kite YAML saved to: {yaml_path}")

    # Create new KiteDefinition object from the generated YAML
    new_kite = KiteDefinition(yaml_path)
    new_kite.process()

    return yaml_path, new_kite
