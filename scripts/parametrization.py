from pathlib import Path

# from geometrical_kite_optimization.utils import PROJECT_DIR
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math as m
import csv
from scipy.special import comb
from SurfplanAdapter.process_surfplan import generate_wing_yaml


class KiteDefinition:
    """Class to handle kite definition."""

    def __init__(self, kite_name):
        self.kite_name = kite_name

    def process(self, plot=False):
        self.get_old_kite()
        self.find_old_chords()
        self.get_old_span()
        self.get_old_area()
        self.get_arc()
        self.get_bezier_curve(plot=plot)

    def get_old_kite(self):
        """Retrieve the old kite geometry."""
        self.source_dir = Path(PROJECT_DIR) / "data" / f"{self.kite_name}"
        self.old_kite_path = Path(self.source_dir) / "config_kite.yaml"

        if not self.old_kite_path.exists():
            raise FileNotFoundError(
                f"\nSurfplan file {self.old_kite_path} does not exist. "
                "Please check the .csv file name and ensure it matches the data_dir name."
                "It is essential that the kite_name matches the name of the surfplan file."
            )

        yaml_file_path = Path(self.old_kite_path)

        # Load YAML configuration
        with open(yaml_file_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract wing sections data
        self.wing_sections = self.config["wing_sections"]
        self.wing_sections_data = self.wing_sections["data"]
        self.headers = self.wing_sections["headers"]

        # Create mapping from headers to indices
        self.header_map = {header: idx for idx, header in enumerate(self.headers)}

        return self.wing_sections_data

    def find_old_chords(self):
        """Calculate chord lengths for each wing section."""

        wing_sections = self.get_old_kite()

        chord_lst = []

        # Process each wing section
        for i, section_data in enumerate(wing_sections):
            # Extract coordinates from section data
            airfoil_id = section_data[self.header_map["airfoil_id"]]
            le_x = section_data[self.header_map["LE_x"]]
            le_y = section_data[self.header_map["LE_y"]]
            le_z = section_data[self.header_map["LE_z"]]
            te_x = section_data[self.header_map["TE_x"]]
            te_y = section_data[self.header_map["TE_y"]]
            te_z = section_data[self.header_map["TE_z"]]
            vup_x = section_data[self.header_map["VUP_x"]]
            vup_y = section_data[self.header_map["VUP_y"]]
            vup_z = section_data[self.header_map["VUP_z"]]

            # Create vectors
            le_point = np.array([le_x, le_y, le_z])
            te_point = np.array([te_x, te_y, te_z])
            vup_vector = np.array([vup_x, vup_y, vup_z])

            # Calculate chord vector and length
            chord_vector = te_point - le_point
            chord_length = np.linalg.norm(chord_vector)
            chord_unit = chord_vector / chord_length

            chord_lst.append(chord_length)

            # Normalize VUP vector
            vup_unit = vup_vector / np.linalg.norm(vup_vector)

            # Create coordinate system for the airfoil
            # x_local: along chord (LE to TE)
            # y_local: perpendicular to chord in the VUP direction
            # z_local: perpendicular to both (right-hand rule)
            x_local = chord_unit
            y_local = vup_unit
            z_local = np.cross(x_local, y_local)
            z_local = z_local / np.linalg.norm(z_local)

            # Find corresponding .dat file
            dat_file_path = self.source_dir / "profiles" / f"prof_{airfoil_id}.dat"

            if not dat_file_path.exists():
                print(
                    f"Warning: Profile file {dat_file_path} not found, skipping airfoil {airfoil_id}"
                )
                continue

            try:
                # Read airfoil coordinates from .dat file
                airfoil_coords = []
                with open(dat_file_path, "r") as f:
                    lines = f.readlines()

                # Skip the first line (profile name) and read coordinates
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            airfoil_coords.append([x, y])

                if not airfoil_coords:
                    print(f"Warning: No coordinates found in {dat_file_path}")
                    continue

                airfoil_coords = np.array(airfoil_coords)

                # Scale airfoil coordinates by chord length
                # .dat file coordinates are normalized (0 to 1 in x-direction)
                airfoil_x = airfoil_coords[:, 0] * chord_length
                airfoil_y = airfoil_coords[:, 1] * chord_length
                airfoil_z = np.zeros_like(
                    airfoil_x
                )  # Start with z=0 in local coordinates

                # Transform airfoil coordinates to 3D world coordinates
                world_coords = []
                for j in range(len(airfoil_x)):
                    # Local airfoil coordinate
                    local_coord = np.array([airfoil_x[j], airfoil_y[j], airfoil_z[j]])

                    # Transform to world coordinates using the local coordinate system
                    world_coord = (
                        le_point
                        + local_coord[0] * x_local
                        + local_coord[1] * y_local
                        + local_coord[2] * z_local
                    )
                    world_coords.append(world_coord)

                world_coords = np.array(world_coords)

            except Exception as e:
                print(f"Error processing airfoil {airfoil_id}: {e}")
                continue

        return chord_lst

    def get_old_span(self):
        """Calculate the span of the kite as the sum of distances between sections in the z-y plane."""
        wing_sections = self.get_old_kite()
        span = 0.0
        span_offset_lst = []
        prev_y = None
        prev_z = None

        for section_data in wing_sections:
            le_y = section_data[self.header_map["LE_y"]]
            le_z = section_data[self.header_map["LE_z"]]
            if prev_y is not None and prev_z is not None:
                dy = le_y - prev_y
                dz = le_z - prev_z
                dist = np.sqrt(dy**2 + dz**2)
                span += dist
                span_offset_lst.append(dist)
            prev_y = le_y
            prev_z = le_z

        self.span = span
        self.span_offsets = span_offset_lst

        return span, span_offset_lst

    def get_old_average_chord(self):
        """Calculate the averarge chord length"""
        return np.average(self.find_old_chords())

    def get_old_area(self):
        """Calculate the flat surface area of the full kite from the chords and span"""

        try:
            span_offset_lst = self.span_offsets
        except:
            self.get_old_span()
            span_offset_lst = self.span_offsets

        span_offset_lst = self.span_offsets
        chords = self.find_old_chords()
        area = 0.0

        for i, offset in enumerate(span_offset_lst):
            area_rectangle = 0.5 * (chords[i] + chords[i + 1]) * offset
            area += area_rectangle

        return area

    def old_aspect_ratio(self):
        """Calculate the aspect ratio with the span and SMC"""
        return self.get_old_span()[0] ** 2 / self.get_old_area()

    def get_arc(self, halve=False, plot=False):
        """Retrieve the leading edge (LE) and trailing edge (TE) arcs of the kite."""
        LE_arc_x = np.array(self.get_old_kite())[:, self.header_map["LE_x"]]
        LE_arc_y = np.array(self.get_old_kite())[:, self.header_map["LE_y"]]
        LE_arc_z = np.array(self.get_old_kite())[:, self.header_map["LE_z"]]
        TE_arc_x = np.array(self.get_old_kite())[:, self.header_map["TE_x"]]
        TE_arc_y = np.array(self.get_old_kite())[:, self.header_map["TE_y"]]
        TE_arc_z = np.array(self.get_old_kite())[:, self.header_map["TE_z"]]

        LE_coords = np.column_stack((LE_arc_x, LE_arc_y, LE_arc_z))
        TE_coords = np.column_stack((TE_arc_x, TE_arc_y, TE_arc_z))

        if halve:
            LE_coords = LE_coords[: len(LE_coords) // 2]
            TE_coords = TE_coords[: len(TE_coords) // 2]
            if len(LE_arc_y) // 2 != 0:
                LE_coords = np.vstack(
                    [LE_coords, [LE_coords[-1, 0], 0, LE_coords[-1, 2]]]
                )
                TE_coords = np.vstack(
                    [TE_coords, [TE_coords[-1, 0], 0, LE_coords[-1, 2]]]
                )

        if plot:
            self.plot_arc(
                coords=[LE_arc_y, LE_arc_z, TE_arc_y, TE_arc_z],
                lables=["LE Arc", "TE Arc"],
                style=["red", "blue"],
                title="LE and TE Arcs of the Kite",
            )

        return LE_coords, TE_coords

    def plot_arc(self, coords, lables, style, title, grid=False):
        for i in range(len(coords) // 2):
            plt.plot(coords[2 * i], coords[2 * i + 1], style[i], label=lables[i])
        plt.axis("equal")
        plt.xlabel("Y-axis")
        plt.ylabel("Z-axis")
        plt.title(title)
        plt.legend()
        if grid:
            plt.grid()
        plt.show()

    def chord_vectors(self):
        """Calculate chord vectors for each wing section."""
        """Calculate half-chord points for each wing section."""
        _, _, _, LE_arc_proj_y, LE_arc_proj_z, _, _, _ = self.get_bezier_curve()
        LE_coords, TE_coords = self.get_arc(halve=True)
        old_chord_vectors = []
        old_qchord_points = []
        new_chord_vectors = []
        new_qchord_points = []
        new_LE_coords = []
        new_TE_coords = []

        for i, LE_data in enumerate(LE_coords):
            LE_vector = LE_data
            TE_vector = TE_coords[i]
            old_chord_vector = TE_vector - LE_vector
            old_chord_vectors.append(old_chord_vector)
            qchord = LE_vector + 0.25 * old_chord_vector
            old_qchord_points.append(qchord)
            hchord_yz_offset = np.array([qchord[1], qchord[2]]) - np.array(
                [LE_data[1], LE_data[2]]
            )
            new_qchord = np.array(
                [qchord[0], LE_arc_proj_y[i] + hchord_yz_offset[0], LE_arc_proj_z[i]]
                + hchord_yz_offset[1]
            )
            new_qchord_points.append(new_qchord)
            new_LE = new_qchord - 0.25 * old_chord_vector
            new_LE_coords.append(new_LE)
            new_TE = new_qchord + 0.75 * old_chord_vector
            new_TE_coords.append(new_TE)
            new_chord_vector = new_TE - new_LE
            new_chord_vectors.append(new_chord_vector)

        full_arc_LE = np.vstack(
            [new_LE_coords[:-1], np.flip(new_LE_coords[:-1], 0) * np.array([1, -1, 1])]
        )
        full_arc_TE = np.vstack(
            [new_TE_coords[:-1], np.flip(new_TE_coords[:-1], 0) * np.array([1, -1, 1])]
        )
        full_arc_qchord = np.vstack(
            [
                new_qchord_points[:-1],
                np.flip(new_qchord_points[:-1], 0) * np.array([1, -1, 1]),
            ]
        )

        return (
            new_chord_vectors,
            np.array(new_qchord_points),
            new_LE_coords,
            new_TE_coords,
            full_arc_LE,
            full_arc_TE,
            full_arc_qchord,
        )

    def get_bezier_curve(self, normalized=False, plot=False):
        """
        Generate a Bezier curve based on the old kite geometry.
        Automatically fit delta and gamma to best match the baseline arc.
        Returns bez_x, bez_y, control points, LE_arc_proj_y, LE_arc_proj_z, phi, delta, gamma.
        """

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

        LE_coords, TE_coords = self.get_arc(halve=True)
        arc_y, arc_z = LE_coords[:, 1], LE_coords[:, 2]
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
        from scipy.optimize import minimize

        res = minimize(
            fit_error,
            x0=[delta0, gamma0],
            bounds=[(arc_y[-1], arc_y[0]), (arc_z[0], arc_z[-1])],
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
        LE_arc_proj_y = []
        LE_arc_proj_z = []
        for pt in arc_points:
            dists = np.linalg.norm(bez_points - pt, axis=1)
            min_idx = np.argmin(dists)
            LE_arc_proj_y.append(bez_x[min_idx])
            LE_arc_proj_z.append(bez_y[min_idx])
        LE_arc_proj_y = np.array(LE_arc_proj_y)
        LE_arc_proj_z = np.array(LE_arc_proj_z)

        phi = np.arctan((points[3][1] - points[0][1]) / (points[0][0]))

        self.projected_span = max(bez_x)

        if normalized:
            height = bez_y[0]
            bez_x = bez_x / self.projected_span
            bez_y = (bez_y - height) / self.projected_span
            points = [
                np.array(point) - np.array([0, height]) for point in points
            ] / self.projected_span
            LE_arc_proj_y = LE_arc_proj_y / self.projected_span
            LE_arc_proj_z = (LE_arc_proj_z - height) / self.projected_span
            delta_best = delta_best / self.projected_span
            gamma_best = gamma_best / self.projected_span
            ind_lst = []
            for i in range(len(LE_arc_proj_y)):
                ind_lst.append(list(bez_y).index(LE_arc_proj_z[i]))
            self.ind_lst = ind_lst

        if plot:
            plt.plot(bez_x, bez_y, "r-", label="Bezier Curve")
            plt.plot(
                [points[i][0] for i in range(len(points))],
                [points[i][1] for i in range(len(points))],
                "ro",
                label="Control Points",
            )
            plt.scatter(
                LE_arc_proj_y, LE_arc_proj_z, color="blue", label="Projected LE Arc"
            )
            plt.axis("equal")
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.title("Bezier Curve Fit of the kite LE")
            plt.legend()
            plt.grid()
            plt.show()

        return (
            bez_x,
            bez_y,
            np.array(points),
            LE_arc_proj_y,
            LE_arc_proj_z,
            phi,
            delta_best,
            gamma_best,
        )

    def height(self):
        "Returns the height of the original kite"
        _, _, _, _, LE_arc_proj_z, _, _, _ = self.get_bezier_curve()
        return LE_arc_proj_z[0]


class KiteScaling:
    """Class to handle kite scaling."""

    def __init__(
        self,
        kitedefinition: KiteDefinition,
        new_ar,
        arc_parameters=[0.5398012, 0.42633213, 0.51579476],
    ):
        if not isinstance(kitedefinition, KiteDefinition):
            raise TypeError(
                "The kitedefinition input must be an instance of KiteDefinition"
            )
        self.kite_name = kitedefinition.kite_name + "_scaled"
        self.old_kite = kitedefinition
        self.new_ar = new_ar
        self.delta = arc_parameters[0]
        self.gamma = arc_parameters[1]
        self.phi = arc_parameters[2]
        self.ar_scaling_factor = self.new_ar / self.old_kite.old_aspect_ratio()
        self.area = self.old_kite.get_old_area()

    def get_le_arc_curve(self):
        """
        Generate a Bezier curve based on the new kite parameters.
        Project arc points onto the Bezier curve and export as new LE_arc.
        Returns bez_x, bez_y, control points, LE_arc_proj_y, LE_arc_proj_z.
        """

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

        tip_height = -m.tan(self.phi)

        points = [
            [1, tip_height],
            [1, tip_height + self.gamma],
            [self.delta, 0],
            [0, 0],
        ]
        bez_y, bez_z = bezier(points, nTimes=1000)

        LE_norm_y = []
        LE_norm_z = []

        for ind in self.old_kite.ind_lst:
            LE_norm_y.append(bez_y[ind])
            LE_norm_z.append(bez_z[ind])

        return LE_norm_y, LE_norm_z, bez_y, bez_z, points

    def get_flat_span(self, y, z):
        """
        Calculate the arc length of a curve defined by points (y, z).
        Uses the trapezoidal rule for numerical integration.
        """
        length = 0
        for i in range(len(y) - 1):
            dy = y[i] - y[i + 1]
            dz = z[i] - z[i + 1]
            dist = np.sqrt(dy**2 + dz**2)
            length += dist
        return length

    def get_new_span(self):
        """Calculate the new span offsets based on the new aspect ratio"""
        return m.sqrt(self.new_ar * self.area)

    def scale_arc_to_span(self):
        """Rescales the LE arc curve to the correct span and height"""
        LE_y_norm, LE_z_norm, _, _, _ = self.get_le_arc_curve()
        normalized_span = self.get_flat_span(LE_y_norm, LE_z_norm)
        span_scaling_factor = self.get_new_span() / (2 * normalized_span)

        LE_y_scaled, LE_z_scaled = [0], [0]
        for i, LE_point in enumerate(zip(LE_y_norm[::-1], LE_z_norm[::-1])):
            if i == len(LE_y_norm) - 1:
                continue
            j = len(LE_y_norm) - 1 - i
            dz = LE_z_norm[j - 1] - LE_z_norm[j]
            dy = LE_y_norm[j - 1] - LE_y_norm[j]
            LE_y_scaled.append(LE_y_scaled[-1] + span_scaling_factor * dy)
            LE_z_scaled.append(LE_z_scaled[-1] + span_scaling_factor * dz)

        old_tip_height = self.old_kite.height()
        LE_z_scaled_trans = [z - min(LE_z_scaled) + old_tip_height for z in LE_z_scaled]

        return LE_y_scaled, LE_z_scaled_trans

    def plot_span_experiment(self, plot=False):
        """Plot the effect of span scaling on the leading edge arc."""
        old_span, _ = self.old_kite.get_old_span()
        self.new_span = self.ar_scaling_factor * old_span

        _, _, old_points, LE_arc_proj_y, LE_arc_proj_z, _, _, _ = (
            self.old_kite.get_bezier_curve(plot=False)
        )
        new_y, new_z = [LE_arc_proj_y[-1]], [LE_arc_proj_z[-1]]
        for i in range(len(LE_arc_proj_y)):
            if i == len(LE_arc_proj_y) - 1:
                continue
            else:
                j = len(LE_arc_proj_y) - 1 - i
            dz = LE_arc_proj_z[j - 1] - LE_arc_proj_z[j]
            dy = LE_arc_proj_y[j - 1] - LE_arc_proj_y[j]
            new_y.append(new_y[-1] + self.ar_scaling_factor * dy)
            new_z.append(new_z[-1] + self.ar_scaling_factor * dz)

        height_offset = LE_arc_proj_z[0] - new_z[-1]
        new_z_height = [z + height_offset for z in new_z]

        if plot:
            plt.plot(LE_arc_proj_y, LE_arc_proj_z, "k--", label="Original LE Arc")
            plt.plot(new_y, new_z, "b-", label="Scaled LE Arc")
            normalized_y, normalized_z, control_norm = self.normalized_arc(
                LE_arc_proj_y, LE_arc_proj_z, control_points=old_points
            )
            bez_x, bez_y, _, _, _, _ = self.bezier_curve(points_ex=control_norm)
            plt.plot(normalized_y, normalized_z, "g--", label="Normalized Arc Data")
            plt.scatter(
                [float(control[0]) for control in control_norm],
                [float(control[1]) for control in control_norm],
                color="blue",
                label="Normalized control points",
            )
            print(control_norm)
            norm_y_new, norm_z_new = self.normalized_arc(
                np.array(new_y[::-1]), np.array(new_z[::-1])
            )
            plt.plot(
                norm_y_new,
                norm_z_new + (normalized_z[-1] - norm_z_new[-1]),
                "m--",
                label="New Normalized Arc Data",
            )
            plt.plot(bez_x, bez_y, "y-", label="New Bezier Curve")
            # Add a straight line through both arc end points, extended from ymin to ymax
            ymin = min(LE_arc_proj_y)
            ymax = max(new_y)
            # Fit a line through the first and last arc points
            x0, y0 = LE_arc_proj_y[0], LE_arc_proj_z[0]
            x1, y1 = LE_arc_proj_y[-1], LE_arc_proj_z[-1]
            # Calculate slope and intercept
            if x1 != x0:
                slope = (y1 - y0) / (x1 - x0)
                intercept = y0 - slope * x0
                line_y = np.array([ymin, ymax])
                line_z = slope * line_y + intercept
                plt.plot(line_y, line_z, "g--", label="End-to-End Line")
            else:
                # Vertical line case
                plt.plot([x0, x1], [y0, y1], "g--", label="End-to-End Line")
            plt.axis("equal")
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.title(
                f"Effect of Span Scaling by {self.ar_scaling_factor:.2f} on the kite LE Arc"
            )
            plt.legend()
            plt.grid()
            plt.show()

        return new_y, new_z

    def plot_arc(self, coords, lables, style, title, grid=False):
        for i in range(len(coords) // 2):
            plt.plot(coords[2 * i], coords[2 * i + 1], style[i], label=lables[i])
        plt.axis("equal")
        plt.xlabel("Y-axis")
        plt.ylabel("Z-axis")
        plt.title(title)
        plt.legend()
        if grid:
            plt.grid()
        plt.show()

    def chord_scaling(self):
        new_span = self.get_new_span()
        old_span = self.old_kite.get_old_span()
        old_ar = self.old_kite.old_aspect_ratio()
        return new_span / old_span[0] * old_ar / self.new_ar

    def chord_vectors(self, plot=False):
        """Calculate chord vectors for each wing section."""
        """Calculate half-chord points for each wing section."""
        chord_scaling = self.chord_scaling()
        LE_arc_y, LE_arc_z = self.scale_arc_to_span()
        LE_coords, TE_coords = self.old_kite.get_arc(halve=True)
        new_chord_vectors = []
        new_qchord_points = []
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
            new_qchord_points.append(new_qchord)
            new_LE = new_qchord - 0.25 * old_chord_vector * chord_scaling
            new_LE_coords.append(new_LE)
            new_TE = new_qchord + 0.75 * old_chord_vector * chord_scaling
            new_TE_coords.append(new_TE)
            new_chord_vector = new_TE - new_LE
            new_chord_vectors.append(new_chord_vector)

        full_arc_LE = np.vstack(
            [new_LE_coords[:-1], np.flip(new_LE_coords[:-1], 0) * np.array([1, -1, 1])]
        )
        full_arc_TE = np.vstack(
            [new_TE_coords[:-1], np.flip(new_TE_coords[:-1], 0) * np.array([1, -1, 1])]
        )
        full_arc_qchord = np.vstack(
            [
                new_qchord_points[:-1],
                np.flip(new_qchord_points[:-1], 0) * np.array([1, -1, 1]),
            ]
        )

        if plot:
            plt.plot(
                np.array(full_arc_LE)[:, 1],
                np.array(full_arc_LE)[:, 2],
                "g-",
                label="LE",
            )
            plt.plot(np.array(full_arc_LE)[:, 1], np.array(full_arc_LE)[:, 2], "gx")
            plt.plot(
                np.array(full_arc_TE)[:, 1],
                np.array(full_arc_TE)[:, 2],
                "r-",
                label="TE",
            )
            plt.plot(np.array(full_arc_TE)[:, 1], np.array(full_arc_TE)[:, 2], "rx")
            plt.plot(
                np.array(full_arc_qchord)[:, 1],
                np.array(full_arc_qchord)[:, 2],
                "b--",
                label="QC",
            )
            plt.plot(
                np.array(full_arc_qchord)[:, 1], np.array(full_arc_qchord)[:, 2], "bx"
            )

            plt.axis("equal")
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.title(f"New arc location debugging")
            plt.legend()
            plt.grid()
            plt.show()

        return (
            new_chord_vectors,
            np.array(new_qchord_points),
            new_LE_coords,
            new_TE_coords,
            full_arc_LE,
            full_arc_TE,
            full_arc_qchord,
        )

    def get_new_wing_sections(self):
        """Generate new wing sections with scaled span and adjusted LE arc."""

        _, _, _, _, full_arc_LE, full_arc_TE, _ = self.chord_vectors()
        new_wing_sections = []

        for i, LE_coord in enumerate(full_arc_LE):
            rib = {}
            rib["LE"] = LE_coord
            rib["TE"] = full_arc_TE[i]
<<<<<<< HEAD
            rib["VUP"] = self.config["wing_sections"]["data"][i][
                self.header_map["VUP_x"] : self.header_map["VUP_z"] + 1
            ]
            rib["airfoil_id"] = self.config["wing_sections"]["data"][i][
                self.header_map["airfoil_id"]
            ]
=======
            rib["VUP"] = self.old_kite.config["wing_sections"]["data"][i][self.old_kite.header_map["VUP_x"]:self.old_kite.header_map["VUP_z"]+1]
            rib["airfoil_id"] = self.old_kite.config["wing_sections"]["data"][i][self.old_kite.header_map["airfoil_id"]]
>>>>>>> refs/remotes/origin/main
            new_wing_sections.append(rib)

        return generate_wing_yaml.generate_wing_sections_data(new_wing_sections)
<<<<<<< HEAD

    def export_data(self, filename="kite_data.txt"):
        """Export kite data to a text file."""
        new_yaml_file_path = (
            Path(PROJECT_DIR) / "processed_data" / f"{self.kite_name}_scaled"
        )
        old_file = self.config
=======
    
    
    def export_data(self, filename=None):
        """Export kite data to a text file."""
        if not filename:
            filename = self.kite_name
        new_yaml_file_path = Path(PROJECT_DIR) / "processed_data" / f"{filename}_scaled"
        old_file = self.old_kite.config
>>>>>>> refs/remotes/origin/main

        wing_sections = self.get_new_wing_sections()
        wing_airfoils = old_file["wing_airfoils"]

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
                **wing_airfoils,
            },
        }

        bridle_lines_yaml = old_file.get("bridle_lines", {})
        bridle_nodes = old_file.get("bridle_nodes", {})
        bridle_connections = old_file.get("bridle_connections", {})

        # if not (bridle_lines_yaml and bridle_nodes and bridle_connections):
        #     print(f"Warning: Could not include bridle data. Taking data from default kite.")

        #     default_kite_path = self.old_kite_path

        #     if not default_kite_path.exists():
        #         raise FileNotFoundError(
        #         f"\nSurfplan file {default_kite_path} does not exist. "
        #         "Please check the .yaml file name and ensure it matches the data_dir name."
        #         "It is essential that the kite_name matches the name of the surfplan file."
        #         )

        #     yaml_file_path = Path(default_kite_path)

        #     # Load YAML configuration
        #     with open(yaml_file_path, "r") as f:
        #         default_config = yaml.safe_load(f)

        #     bridle_lines_yaml = default_config.get("bridle_lines", {})
        #     bridle_nodes = default_config.get("bridle_nodes", {})
        #     bridle_connections = default_config.get("bridle_connections", {})

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

        # Save YAML file
        os.makedirs(new_yaml_file_path, exist_ok=True)
        yaml_file_path = Path(new_yaml_file_path) / "config_kite.yaml"

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
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=False
            )

        yaml.add_representer(type(None), represent_none)
        yaml.add_representer(list, represent_list)

        with open(yaml_file_path, "w") as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        # Post-process to clean up comments (remove quotes and None values)
        with open(yaml_file_path, "r") as f:
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

        with open(yaml_file_path, "w") as f:
            f.write(content)

        print(f'Generated YAML file and saved at "{yaml_file_path}"')
        print(f'Wing sections: {len(wing_sections["data"])}')
        print(f'Wing airfoils: {len(wing_airfoils["data"])}')
        try:
            print(f"Bridle lines: {len(bridle_lines)}")
        except:
            pass

        return yaml_file_path, yaml_data
<<<<<<< HEAD

    def vsm_csv_sheets(self):
        """Generate VSM data sheets for the kite."""
        _, yaml_data = self.export_data()
        new_csv_file_path = (
            Path(PROJECT_DIR) / "processed_data" / f"{self.kite_name}_scaled"
        )
=======
    

    def vsm_csv_sheets(self, filename=None):
        """Generate VSM data sheets for the kite."""
        if not filename:
            filename = self.kite_name
        _, yaml_data = self.export_data(filename=filename)
        new_csv_file_path = Path(PROJECT_DIR) / "processed_data" / f"{filename}_scaled"
>>>>>>> refs/remotes/origin/main
        bridle_csv_path = Path(new_csv_file_path) / "bridle_lines.csv"
        wing_csv_path = Path(new_csv_file_path) / "wing_geometry.csv"

        bridle_nodes = yaml_data["bridle_nodes"]["data"]
        bridle_lines = yaml_data["bridle_lines"]["data"]
        bridle_connections = yaml_data["bridle_connections"]["data"]

        bridle_data = [None] * len(bridle_connections)

        for i, connection in enumerate(bridle_connections):
            name, start, end = connection[0], connection[1], connection[2]
            line_diameter = [line[2] for line in bridle_lines if line[0] == name]
            start_node = [node for node in bridle_nodes if node[0] == start][0]
            end_node = [node for node in bridle_nodes if node[0] == end][0]
            if line_diameter and start_node and end_node:
                entry = start_node[1:-1] + end_node[1:-1] + line_diameter
                bridle_data[i] = entry
                entry = None
            else:
                print(f"Warning: Missing data for connection {name}, skipping.")

        # Write to CSV
        with open(bridle_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z", "diameter"]
            )
            for row in bridle_data:
                writer.writerow(row)

        print(f"Bridle data successfully saved to '{bridle_csv_path}'.")

        # Wing csv file creation
        wing_sections = yaml_data["wing_sections"]["data"]
        airfoil_data = yaml_data["wing_airfoils"]["data"]
        wing_data = [None] * len(wing_sections)

        for i, section in enumerate(wing_sections):
            airfoil_id = section[0]
            le_coord = section[1:4]
            te_coord = section[4:7]
            d_tube = [
                airfoil_info[2]["meta_parameters"]["d_tube_from_surfplan_txt"]
                for airfoil_info in airfoil_data
                if airfoil_info[0] == airfoil_id
            ]
            camber = [
                airfoil_info[2]["meta_parameters"]["eta"]
                for airfoil_info in airfoil_data
                if airfoil_info[0] == airfoil_id
            ]
            wing_data[i] = le_coord + te_coord + d_tube + camber

        # Write to CSV
        with open(wing_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["LE_x", "LE_y", "LE_z", "TE_x", "TE_y", "TE_z", "d_tube", "camber"]
            )
            for row in wing_data:
                writer.writerow(row)

        return bridle_data, wing_data


<<<<<<< HEAD
def plot_kite(kite_lst):
    """Plot the kite geometry in 3d."""
    ax = plt.axes(projection="3d")
    for kite in kite_lst:
        _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()
        ax.plot3D(
            full_arc_LE[:, 0],
            full_arc_LE[:, 1],
            full_arc_LE[:, 2],
            "r-",
            label="LE Arc",
        )
        ax.plot3D(
            full_arc_TE[:, 0],
            full_arc_TE[:, 1],
            full_arc_TE[:, 2],
            "b-",
            label="TE Arc",
        )
        ax.scatter3D(
            full_arc_qchord[:, 0],
            full_arc_qchord[:, 1],
            full_arc_qchord[:, 2],
            color="m",
            label="Quarter Chord",
        )
        for i in range(len(full_arc_LE)):
            ax.plot3D(
                [full_arc_LE[i, 0], full_arc_TE[i, 0]],
                [full_arc_LE[i, 1], full_arc_TE[i, 1]],
                [full_arc_LE[i, 2], full_arc_TE[i, 2]],
                "g--",
                alpha=0.5,
            )
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid()
    ax.set_aspect("equal")
    ax.set_title("Kite Geometry with Chord Vectors and Quarter Chord Points")
    ax.legend()
=======
def plot_kite(kite_lst, style_lst=None, use_subplots=False):
    """Plot the kite geometry in 3D or as 4 subplots from different angles."""
    if not style_lst:
        style_lst = ['base'] * len(kite_lst)
    if not use_subplots:
        ax = plt.axes(projection='3d')
        for idx, kite in enumerate(kite_lst):
            _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()
            style = style_lst[idx]
            for i in range(len(full_arc_LE)):
                ax.plot3D([full_arc_LE[i,0], full_arc_TE[i,0]], 
                          [full_arc_LE[i,1], full_arc_TE[i,1]], 
                          [full_arc_LE[i,2], full_arc_TE[i,2]], 'g--' if style == 'base' else 'g-', alpha=0.5)
            ax.scatter3D(full_arc_qchord[:,0], full_arc_qchord[:,1], full_arc_qchord[:,2], color='m', marker='x' if style == 'base' else 'o')
            ax.plot3D(full_arc_LE[:,0], full_arc_LE[:,1], full_arc_LE[:,2], 'r--' if style=='base' else 'r-')
            ax.plot3D(full_arc_TE[:,0], full_arc_TE[:,1], full_arc_TE[:,2], 'b--' if style=='base' else 'b-')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.grid()
        ax.set_aspect('equal')
        ax.set_title('Kite Geometry')
        ax.view_init(elev=15, azim=200)
    else:
        fig = plt.figure(figsize=(14, 10))
        angles = [(0, 180), (0, -90), (270, 180), (15, 200)]
        titles = ['Front View', 'Side View', 'Top View', 'Isometric View']
        for id, (angle, title) in enumerate(zip(angles, titles)):
            ax = fig.add_subplot(2, 2, id+1, projection='3d')
            for idx, kite in enumerate(kite_lst):
                _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()
                style = style_lst[idx]
                for i in range(len(full_arc_LE)):
                    ax.plot3D([full_arc_LE[i,0], full_arc_TE[i,0]], 
                              [full_arc_LE[i,1], full_arc_TE[i,1]], 
                              [full_arc_LE[i,2], full_arc_TE[i,2]], 'g--' if style == 'base' else 'g-', alpha=0.5)
                ax.scatter3D(full_arc_qchord[:,0], full_arc_qchord[:,1], full_arc_qchord[:,2], color='m', marker='x' if style == 'base' else 'o')
                ax.plot3D(full_arc_LE[:,0], full_arc_LE[:,1], full_arc_LE[:,2], 'r--' if style=='base' else 'r-')
                ax.plot3D(full_arc_TE[:,0], full_arc_TE[:,1], full_arc_TE[:,2], 'b--' if style=='base' else 'b-')
            ax.set_title(title)
            ax.grid()
            ax.set_aspect('equal')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.view_init(elev=angle[0], azim=angle[1])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
>>>>>>> refs/remotes/origin/main
    plt.show()


if __name__ == "__main__":
<<<<<<< HEAD

    print("V9.60")
    kite_name = "strawman55"
    base_kite = KiteDefinition(kite_name)
    base_kite.process(plot=False)
    print(f"Old span: {base_kite.get_old_span()[0]}")
    print(f"Old Area: {base_kite.get_old_area()}")
    print(f"Old AR: {base_kite.old_aspect_ratio()}")
    LE_base, TE_base = base_kite.get_arc()
    bez_x, bez_y, points, LE_arc_proj_y, LE_arc_proj_z, phi, delta_best, gamma_best = (
        base_kite.get_bezier_curve()
    )
    base_kite.plot_arc(
        [
            LE_base[:, 1],
            LE_base[:, 2],
            bez_x,
            bez_y,
            points[:, 0],
            points[:, 1],
            LE_base[:, 1],
            LE_base[:, 2],
        ],
        ["Old LE", "Bezier Arc", "Control Points", ""],
        ["k-", "r-", "rx", "kx"],
        "Base kite LE plot",
        grid=True,
    )
    _, _, _, LE_full_y, LE_full_z, _, _, _ = base_kite.get_bezier_curve(
        normalized=False, plot=False
    )
    _, _, points, LE_y, LE_z, phi, delta, gamma = base_kite.get_bezier_curve(
        normalized=True, plot=False
    )
    print(f"delta: {delta}")
    print(f"gamma: {gamma}")
    print(f"phi:   {phi}")
=======
    
    save_data=False
    plot=False
    print("V9.60")
    kite_name = "strawman55"
    base_kite = KiteDefinition(kite_name)
    base_kite.process(plot=plot)
    print(f'Old span: {round(base_kite.get_old_span()[0], 3)} [m]')
    print(f'Old Area: {round(base_kite.get_old_area(), 3)} [m^2]')
    print(f'Old AR: {round(base_kite.old_aspect_ratio(), 3)} [-]')
    LE_base, TE_base = base_kite.get_arc()
    bez_x, bez_y, points, LE_arc_proj_y, LE_arc_proj_z, phi, delta_best, gamma_best = base_kite.get_bezier_curve()
    if plot:
        base_kite.plot_arc([LE_base[:,1], LE_base[:,2], bez_x, bez_y, points[:,0], points[:,1], LE_base[:,1], LE_base[:,2]],
                            ["Old LE", "Bezier Arc", "Control Points", ""],
                            ["k-", "r-", "rx", "kx"],
                            "Base kite LE plot",
                            grid=True)
    _, _, _, LE_full_y, LE_full_z, _, _, _ = base_kite.get_bezier_curve(normalized=False, plot=False)
    _, _, points, LE_y, LE_z, phi, delta, gamma = base_kite.get_bezier_curve(normalized=True, plot=False)
    print(f'delta: {round(delta, 6)}')
    print(f'gamma: {round(gamma, 6)}')
    print(f'phi:   {round(phi, 6)}')
>>>>>>> refs/remotes/origin/main

    scaled_kite = KiteScaling(base_kite, new_ar=6.5, arc_parameters=[delta, gamma, phi])
    LE_norm_y, LE_norm_z, bez_y, bez_z, points = scaled_kite.get_le_arc_curve()
    LE_y_scaled, LE_z_scaled_trans = scaled_kite.scale_arc_to_span()
<<<<<<< HEAD
    scaled_kite.plot_arc(
        [LE_full_y, LE_full_z, LE_y_scaled, LE_z_scaled_trans],
        ["Old LE", "Scaled Kite"],
        ["kx", "rx"],
        "Scaled LE plot",
        grid=True,
    )
    (
        new_chord_vectors,
        new_qchord_points,
        new_LE_coords,
        new_TE_coords,
        full_arc_LE,
        full_arc_TE,
        full_arc_qchord,
    ) = scaled_kite.chord_vectors(plot=False)
    plot_kite([base_kite, scaled_kite])
=======
    if plot:
        scaled_kite.plot_arc([LE_full_y, LE_full_z, LE_full_y, LE_full_z, LE_y_scaled, LE_z_scaled_trans, LE_y_scaled, LE_z_scaled_trans],
                             ["", "Old LE", "", "Scaled Kite"],
                             ["kx", "k-", "rx", "r-"],
                             "Scaled LE plot",
                             grid=True)
    new_chord_vectors, new_qchord_points, new_LE_coords, new_TE_coords, full_arc_LE, full_arc_TE, full_arc_qchord = scaled_kite.chord_vectors(plot=False)
    if plot:
        # plot_kite([base_kite], style_lst=['base'])
        plot_kite([scaled_kite], style_lst=['scaled'], use_subplots=True)
        # plot_kite([base_kite, scaled_kite], style_lst=['base', 'scaled'])
>>>>>>> refs/remotes/origin/main

    if save_data:
        output_filename = "test"
        scaled_kite.vsm_csv_sheets(filename=output_filename)
