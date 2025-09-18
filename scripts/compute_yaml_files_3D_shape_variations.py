import yaml
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from scipy.special import comb
from pathlib import Path
import math as m
import matplotlib.pyplot as plt


def load_yaml(filename):
    """
    Loads a YAML file containing 'wing_sections' with 'headers' and 'data',
    and returns a pandas DataFrame.
    """
    with open(filename, "r") as f:
        yml = yaml.safe_load(f)
    wing_sections = yml["wing_sections"]
    headers = wing_sections["headers"]
    data = wing_sections["data"]
    df = pd.DataFrame(data, columns=headers)
    return df


def export_to_yaml(full_arc_LE, full_arc_TE, output_filename, original_df=None):
    """
    Export scaled kite coordinates to YAML format matching the original config structure.

    Args:
        full_arc_LE: numpy array of leading edge coordinates [x, y, z]
        full_arc_TE: numpy array of trailing edge coordinates [x, y, z]
        output_filename: string, path for the output YAML file
        original_df: pandas DataFrame, original kite data for airfoil_id mapping (optional)
    """

    # If original_df is provided, use the original airfoil_ids
    if original_df is not None:
        airfoil_ids = original_df["airfoil_id"].values
    else:
        # Generate airfoil_ids based on the pattern in the original file
        # From tip to root, then root to tip (symmetric)
        n_sections = len(full_arc_LE)
        half_sections = n_sections // 2

        # Create symmetric airfoil ID pattern
        airfoil_ids = []

        # First half: from tip (18) to root (1)
        for i in range(half_sections):
            airfoil_id = 18 - i  # Starts at 18, decreases to 1
            if airfoil_id < 1:
                airfoil_id = 1
            airfoil_ids.append(airfoil_id)

        # Second half: mirror the first half
        for i in range(half_sections):
            airfoil_id = airfoil_ids[half_sections - 1 - i]
            airfoil_ids.append(airfoil_id)

        # Handle odd number of sections
        if len(airfoil_ids) < n_sections:
            airfoil_ids.append(1)  # Center section gets airfoil_id = 1

    # Create data rows
    data_rows = []
    for i in range(len(full_arc_LE)):
        le_coord = full_arc_LE[i]
        te_coord = full_arc_TE[i]

        # Ensure we have enough airfoil IDs
        if i < len(airfoil_ids):
            airfoil_id = int(airfoil_ids[i])
        else:
            airfoil_id = 1  # Default fallback

        row = [
            airfoil_id,
            float(le_coord[0]),  # LE_x
            float(le_coord[1]),  # LE_y
            float(le_coord[2]),  # LE_z
            float(te_coord[0]),  # TE_x
            float(te_coord[1]),  # TE_y
            float(te_coord[2]),  # TE_z
        ]
        data_rows.append(row)

    # Write custom YAML format to match the original
    with open(output_filename, "w") as f:
        f.write("wing_sections:\n")
        f.write(
            "  # ---------------------------------------------------------------':\n"
        )
        f.write("  # headers:':\n")
        f.write(
            "  #   - airfoil_id: integer, unique identifier for the airfoil (matches wing_airfoils)':\n"
        )
        f.write("  #   - LE_x: x-coordinate of leading edge':\n")
        f.write("  #   - LE_y: y-coordinate of leading edge':\n")
        f.write("  #   - LE_z: z-coordinate of leading edge':\n")
        f.write("  #   - TE_x: x-coordinate of trailing edge':\n")
        f.write("  #   - TE_y: y-coordinate of trailing edge':\n")
        f.write("  #   - TE_z: z-coordinate of trailing edge':\n")
        f.write("  headers: [airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z]\n")
        f.write("  data:\n")

        for row in data_rows:
            # Format each row as an inline list (flow style)
            row_str = "  - [" + ", ".join([f"{x}" for x in row]) + "]\n"
            f.write(row_str)

    return data_rows


def flat_span(y, z):
    """Calculate 3D arc length in the yz-plane"""
    y = np.asarray(y)
    z = np.asarray(z)
    dy = np.diff(y)
    dz = np.diff(z)
    return float(np.sum(np.sqrt(dy * dy + dz * dz)))


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


def get_halve_le_te(le, te):
    n = len(le) // 2
    le_h = le[:n].copy()
    te_h = te[:n].copy()

    # If last of half isn't at y≈0, optionally append a center section from the last half-row
    if abs(le_h[-1, 1]) > 1e-9:
        le_h = np.vstack([le_h, [le_h[-1, 0], 0.0, le_h[-1, 2]]])
        te_h = np.vstack([te_h, [te_h[-1, 0], 0.0, te_h[-1, 2]]])  # use TE z here!

    return le_h, te_h


def get_bezier_curve(le_coords, normalized=True, plot=False):
    """
    Generate a Bezier curve based on the old kite geometry.
    Automatically fit delta and gamma to best match the baseline arc.
    Returns bez_x, bez_y, control points, le_arc_proj_y, le_arc_proj_z, phi, delta, gamma.
    """

    arc_y, arc_z = le_coords[:, 1], le_coords[:, 2]
    lenarc = len(arc_y)

    # Objective function to minimize error between Bezier curve and arc
    def fit_error(params):
        delta, gamma = params
        points = [
            [arc_y[0], arc_z[0]],
            [arc_y[0], arc_z[0] + gamma],
            [arc_y[-1] + delta, arc_z[-1]],
            [arc_y[-1], arc_z[-1]],
        ]
        bez_x, bez_y = bezier(points, nTimes=1000)
        bez_points = np.column_stack((bez_x, bez_y))
        arc_points = np.column_stack((arc_y, arc_z))
        error = 0.0
        for pt in arc_points:
            dists = np.linalg.norm(bez_points - pt, axis=1)
            error += np.min(dists)
        return error

    # Initial guess for delta and gamma
    delta0 = (max(arc_y) - min(arc_y)) / 2 if lenarc > 1 else 3.8
    gamma0 = (max(arc_z) - min(arc_z)) / 2 if lenarc > 1 else 4.0

    # Use physically meaningful offset bounds (not absolute positions)
    delta_max = float(arc_y[0] - arc_y[-1])  # half-span in y for the half wing
    gamma_max = float(max(arc_z) - min(arc_z))  # vertical extent

    delta_bounds = (0.0, 2.0 * delta_max)  # generous but sane
    gamma_bounds = (0.0, 2.0 * gamma_max)

    res = minimize(
        fit_error,
        x0=[delta_max * 0.5, gamma_max * 0.5],
        bounds=[delta_bounds, gamma_bounds],
    )

    delta_best, gamma_best = res.x

    # Use best-fit delta and gamma to generate Bezier curve
    points = [
        [arc_y[0], arc_z[0]],
        [arc_y[0], arc_z[0] + gamma_best],
        [arc_y[-1] + delta_best, arc_z[-1]],
        [arc_y[-1], arc_z[-1]],
    ]
    bez_x, bez_y = bezier(points, nTimes=1000)

    # Project arc points onto Bezier curve
    bez_points = np.column_stack((bez_x, bez_y))
    arc_points = np.column_stack((arc_y, arc_z))
    le_arc_proj_y = []
    le_arc_proj_z = []
    for pt in arc_points:
        dists = np.linalg.norm(bez_points - pt, axis=1)
        min_idx = np.argmin(dists)
        le_arc_proj_y.append(bez_x[min_idx])
        le_arc_proj_z.append(bez_y[min_idx])
    le_arc_proj_y = np.array(le_arc_proj_y)
    le_arc_proj_z = np.array(le_arc_proj_z)

    # Use proper slope between start and end of half LE in yz-plane
    dy = arc_y[-1] - arc_y[0]  # typically negative for tip->root half
    dz = arc_z[-1] - arc_z[0]
    phi = np.arctan2(dz, abs(dy))  # Use abs(dy) to get proper anhedral angle magnitude

    projected_span = max(bez_x)
    if normalized:
        height = bez_y[0]
        bez_x = bez_x / projected_span
        bez_y = (bez_y - height) / projected_span
        points = [
            np.array(point) - np.array([0, height]) for point in points
        ] / projected_span
        le_arc_proj_y = le_arc_proj_y / projected_span
        le_arc_proj_z = (le_arc_proj_z - height) / projected_span
        delta_best = delta_best / projected_span
        gamma_best = gamma_best / projected_span
        ind_lst = []
        for i in range(len(le_arc_proj_y)):
            distances = np.abs(bez_y - le_arc_proj_z[i])
            closest_idx = np.argmin(distances)
            ind_lst.append(closest_idx)

    if plot:
        plt.plot(bez_x, bez_y, "r-", label="Bezier Curve")
        plt.plot(
            [points[i][0] for i in range(len(points))],
            [points[i][1] for i in range(len(points))],
            "ro",
            label="Control Points",
        )
        plt.scatter(
            le_arc_proj_y, le_arc_proj_z, color="blue", label="Projected le Arc"
        )
        plt.axis("equal")
        plt.xlabel("Y-axis")
        plt.ylabel("Z-axis")
        plt.title("Bezier Curve Fit of the kite le")
        plt.legend()
        plt.grid()
        plt.show()

    return (
        bez_x,
        bez_y,
        np.array(points),
        le_arc_proj_y,
        le_arc_proj_z,
        phi,
        delta_best,
        gamma_best,
        ind_lst,
        projected_span,
    )


def calculate_area_span_ar_avgchord(full_arc_le, full_arc_TE):
    """
    Calculate consistent 3D surface area, span, aspect ratio (AR), and average chord length using path integration.

    Args:
        full_arc_le: numpy array of leading edge coordinates [x, y, z]
        full_arc_TE: numpy array of trailing edge coordinates [x, y, z]

    Returns:
        tuple: (area, span, ar, avg_chord) where:
            - area: 3D surface area using trapezoidal rule
            - span: cumulative 3D arc length along leading edge
            - ar: aspect ratio (span^2 / area)
            - avg_chord: average chord length along the arc
    """
    area = 0.0
    span = 0.0
    chord_sum = 0.0
    chord_count = 0

    for i in range(len(full_arc_le) - 1):
        curr_le = full_arc_le[i]
        next_le = full_arc_le[i + 1]
        curr_TE = full_arc_TE[i]
        next_TE = full_arc_TE[i + 1]

        chord_curr = np.linalg.norm(curr_TE - curr_le)
        chord_next = np.linalg.norm(next_TE - next_le)
        span_segment = np.linalg.norm(next_le - curr_le)

        span += span_segment
        area += 0.5 * (chord_curr + chord_next) * span_segment

        chord_sum += chord_curr + chord_next
        chord_count += 2

    ar = span**2 / area if area != 0 else np.nan
    avg_chord = chord_sum / chord_count if chord_count > 0 else np.nan
    return area, span, ar, avg_chord


def get_le_curve_in_yz_plane(phi, gamma, delta, ind_lst):
    tip_height = -m.tan(phi)
    points = [[1, tip_height], [1, tip_height + gamma], [delta, 0], [0, 0]]
    bez_y, bez_z = bezier(points, nTimes=1000)

    le_norm_y = []
    le_norm_z = []
    for ind in ind_lst:
        le_norm_y.append(bez_y[ind])
        le_norm_z.append(bez_z[ind])

    return le_norm_y, le_norm_z


def scale_arc_to_span(le_y_norm, le_z_norm, span_scaling_factor):
    le_y_scaled, le_z_scaled = [0], [0]
    for i in range(len(le_y_norm) - 1):
        j = len(le_y_norm) - 1 - i
        dz = le_z_norm[j - 1] - le_z_norm[j]
        dy = le_y_norm[j - 1] - le_y_norm[j]
        le_y_scaled.append(le_y_scaled[-1] + span_scaling_factor * dy)
        le_z_scaled.append(le_z_scaled[-1] + span_scaling_factor * dz)
    return le_y_scaled, le_z_scaled


def generate_chord_vectors(
    chord_scaling, LE_arc_y, LE_arc_z, le_coords_halve, te_coords_halve
):
    new_LE_coords = []
    new_TE_coords = []

    for i, LE_data in enumerate(le_coords_halve):
        j = len(le_coords_halve) - 1 - i
        LE_vector = LE_data
        TE_vector = te_coords_halve[i]
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

    return full_arc_LE, full_arc_TE


def generate_chord_vectors_LE_anchored(
    chord_scaling, LE_arc_y, LE_arc_z, le_half, te_half
):
    """
    LE-anchored chord scaling to prevent LE path changes during chord scaling.
    This ensures area scales exactly as span × chord.
    """
    new_LE, new_TE = [], []
    n = len(le_half)
    for i in range(n):
        j = n - 1 - i
        LE0 = le_half[i]
        TE0 = te_half[i]
        chord_vec0 = TE0 - LE0
        # Keep LE on the target arc (same x as original LE)
        LE_new = np.array([LE0[0], LE_arc_y[j], LE_arc_z[j]])
        TE_new = LE_new + chord_vec0 * chord_scaling
        new_LE.append(LE_new)
        new_TE.append(TE_new)

    full_LE = np.vstack([new_LE[:-1], np.flip(new_LE[:-1], 0) * np.array([1, -1, 1])])
    full_TE = np.vstack([new_TE[:-1], np.flip(new_TE[:-1], 0) * np.array([1, -1, 1])])
    return full_LE, full_TE


def plot_kites(le_coords_1, te_coords_1, le_coords_2, te_coords_2):
    kites = [
        (le_coords_1, te_coords_1, "Original Kite", "blue"),
        (le_coords_2, te_coords_2, "Scaled Kite", "orange"),
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Kite Configurations")

    all_coords = []

    for le_coords, te_coords, label, color in kites:
        full_arc_LE = le_coords
        full_arc_TE = te_coords
        full_arc_qchord = full_arc_LE + 0.25 * (full_arc_TE - full_arc_LE)

        all_coords.append(full_arc_LE)
        all_coords.append(full_arc_TE)
        all_coords.append(full_arc_qchord)

        # Plot leading edge
        ax.plot3D(
            full_arc_LE[:, 0],
            full_arc_LE[:, 1],
            full_arc_LE[:, 2],
            color=color,
            linestyle="-",
            label=f"{label} - LE",
            linewidth=3,
            alpha=0.8,
        )
        # Plot trailing edge
        ax.plot3D(
            full_arc_TE[:, 0],
            full_arc_TE[:, 1],
            full_arc_TE[:, 2],
            color=color,
            linestyle="-",
            label=f"{label} - TE",
            linewidth=3,
            alpha=0.8,
        )

    # Set equal axis
    all_coords = np.vstack(all_coords)
    x_limits = [np.min(all_coords[:, 0]), np.max(all_coords[:, 0])]
    y_limits = [np.min(all_coords[:, 1]), np.max(all_coords[:, 1])]
    z_limits = [np.min(all_coords[:, 2]), np.max(all_coords[:, 2])]
    max_range = (
        max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
        / 2.0
    )

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.show()


def scale_to_target_ar(
    target_ar,
    ar,
    phi,
    delta,
    gamma,
    le_coords,
    te_coords,
    ind_lst,
):

    ar_scaling_factor = target_ar / ar  # Scale AR to target value

    # Use algebraic scaling: AR = b²/A, so to scale AR by x while keeping A constant:
    # span_scale = √x, chord_scale = 1/√x
    span_scaling_factor = m.sqrt(ar_scaling_factor)  # √1.5 ≈ 1.225
    chord_scaling_factor = 1 / m.sqrt(ar_scaling_factor)  # 1/√1.5 ≈ 0.816

    ### Improved scaling using arc length and LE-anchored approach

    # Get normalized half-arc
    le_norm_y, le_norm_z = get_le_curve_in_yz_plane(phi, gamma, delta, ind_lst)
    norm_half_span = flat_span(le_norm_y, le_norm_z)

    # # Baseline half-span from original geometry (what area/span calculation uses)
    # A0, b0, AR0, cbar0 = calculate_area_span_ar_avgchord(le_coords, te_coords)
    # base_half_span = 0.5 * b0

    # Debug: Check what the actual half-span of the original halved geometry is
    le_coords_halve, te_coords_halve = get_halve_le_te(le_coords, te_coords)
    actual_half_span = flat_span(le_coords_halve[:, 1], le_coords_halve[:, 2])

    # Use the actual measured half-span instead of calculated half-span
    target_half_span = span_scaling_factor * actual_half_span
    span_scale = target_half_span / norm_half_span

    # Apply improved scaling
    le_arc_y, le_arc_z = scale_arc_to_span(le_norm_y, le_norm_z, span_scale)
    le_coords_scaled, te_coords_scaled = generate_chord_vectors_LE_anchored(
        chord_scaling_factor, le_arc_y, le_arc_z, le_coords_halve, te_coords_halve
    )

    # Final calculations after alignment
    area_new, span_new, ar_new, avg_chord_new = calculate_area_span_ar_avgchord(
        le_coords_scaled, te_coords_scaled
    )
    return (
        le_coords_scaled,
        te_coords_scaled,
        area_new,
        span_new,
        ar_new,
        avg_chord_new,
    )


def iterative_scaling_to_target(
    yaml_path,
    ar_scaling=1.0,
    phi_scaling=1.0,
    max_iter=10,
    is_plot=False,
    is_with_gamma_fraction=False,
):
    df = load_yaml(yaml_path)
    le_coords = np.column_stack((df["LE_x"], df["LE_y"], df["LE_z"])).astype(float)
    te_coords = np.column_stack((df["TE_x"], df["TE_y"], df["TE_z"])).astype(float)

    # Calculating area, span, ar
    area, span, ar, avg_chord = calculate_area_span_ar_avgchord(le_coords, te_coords)

    # Generating the bezier curve
    le_coords_halve, te_coords_halve = get_halve_le_te(le_coords, te_coords)
    (
        bez_x,
        bez_y,
        points,
        le_arc_proj_y,
        le_arc_proj_z,
        phi_base,
        delta_base,
        gamma_base,
        ind_lst,
        projected_span_for_normalisation,
    ) = get_bezier_curve(le_coords_halve, normalized=True, plot=False)

    target_ar = ar_scaling * ar
    phi = phi_base * phi_scaling
    delta = delta_base
    if is_with_gamma_fraction:
        gamma = gamma_base
        min_z = np.min(le_coords[:, 2])
        max_z = np.max(te_coords[:, 2])
        height = max_z - min_z
        gamma_fraction = gamma / height

    le_coords_for_iteration = le_coords.copy()
    te_coords_for_iteration = te_coords.copy()
    ar_for_iteration = ar
    for i in range(max_iter):
        if is_with_gamma_fraction:
            gamma = gamma_base
            min_z = np.min(le_coords_for_iteration[:, 2])
            max_z = np.max(le_coords_for_iteration[:, 2])
            height = max_z - min_z
            gamma = gamma_fraction * height

        else:
            gamma = gamma_base
        (
            le_coords_for_iteration,
            te_coords_for_iteration,
            area_new,
            span_new,
            ar_for_iteration,
            avg_chord_new,
        ) = scale_to_target_ar(
            target_ar,
            ar_for_iteration,
            phi,
            delta,
            gamma,
            le_coords_for_iteration,
            te_coords_for_iteration,
            ind_lst,
        )

    # compute maximum z of le_coords
    max_z_le = np.max(le_coords[:, 2])
    max_z_le_new = np.max(le_coords_for_iteration[:, 2])
    z_offset = max_z_le - max_z_le_new
    le_coords_for_iteration[:, 2] += z_offset
    te_coords_for_iteration[:, 2] += z_offset

    # calculating final properties
    area_new, span_new, ar_for_iteration, avg_chord_new = (
        calculate_area_span_ar_avgchord(
            le_coords_for_iteration, te_coords_for_iteration
        )
    )
    le_coords_halve, te_coords_halve = get_halve_le_te(
        le_coords_for_iteration, te_coords_for_iteration
    )
    (
        bez_x,
        bez_y,
        points,
        le_arc_proj_y,
        le_arc_proj_z,
        phi_final,
        delta_final,
        gamma_final,
        ind_lst,
        projected_span_for_normalisation,
    ) = get_bezier_curve(le_coords_halve, normalized=True, plot=False)
    print(
        f"Original: Area: {area:.3f} m², Span: {span:.3f} m, AR: {ar:.3f}, phi: {np.rad2deg(phi_base):.3f}°"
    )
    print(
        f"while keeping area equal ----->  target AR: {target_ar:.2f}, target phi: {np.rad2deg(phi_base*phi_scaling):.3f}° "
    )
    print(
        f"Final:    Area: {area_new:.3f} m², Span: {span_new:.3f} m, AR: {ar_for_iteration:.3f}, phi: {np.rad2deg(phi_final):.3f}°"
    )
    if is_plot:
        plot_kites(
            le_coords, te_coords, le_coords_for_iteration, te_coords_for_iteration
        )
    return le_coords_for_iteration, te_coords_for_iteration


def main(
    yaml_path,
    ar_scaling,
    phi_scaling,
    max_iter=10,
    is_plot=True,
    is_with_gamma_fraction=False,
):
    print(
        f"\nProcessing: AR scaling {ar_scaling}, phi scaling {phi_scaling}, gamma_fraction: {is_with_gamma_fraction}"
    )
    le_coords, te_coords = iterative_scaling_to_target(
        yaml_path,
        ar_scaling,
        phi_scaling,
        max_iter=max_iter,
        is_plot=is_plot,
        is_with_gamma_fraction=is_with_gamma_fraction,
    )
    output_yaml_path = (
        Path(yaml_path).parent
        / f"config_kite_ar{ar_scaling:.1f}_phi{phi_scaling:.1f}.yaml"
    )

    original_df = load_yaml(yaml_path)
    export_to_yaml(le_coords, te_coords, output_yaml_path, original_df=original_df)
    return


if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent
    yaml_path = (
        Path(PROJECT_DIR) / "processed_data" / "TUDELFT_V3_KITE" / "config_kite.yaml"
    )
    main(yaml_path, ar_scaling=1.0, phi_scaling=1.0, is_plot=True)

    main(yaml_path, ar_scaling=1.5, phi_scaling=1.0, is_plot=True)

    main(yaml_path, ar_scaling=1.0, phi_scaling=1.5, is_plot=True)

    # TODO: at extreme phi's with gamma scaling to height, the area does not remain constant
    main(
        yaml_path,
        ar_scaling=1.0,
        phi_scaling=0.25,
        is_plot=True,
        is_with_gamma_fraction=True,
    )
