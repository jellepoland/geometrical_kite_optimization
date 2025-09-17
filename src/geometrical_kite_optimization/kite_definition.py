from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from geometrical_kite_optimization.utils import PROJECT_DIR


class KiteDefinition:
    """Class to handle kite definition."""

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path

    def process(self, plot=False):
        self.get_old_kite()
        self.find_old_chords()
        self.get_old_span()
        self.area = self.get_old_area()
        self.get_arc()
        self.get_bezier_curve(plot=plot)

    def get_old_kite(self):
        """Retrieve the old kite geometry."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"\nSurfplan file {self.yaml_path} does not exist. "
                "Please check the .csv file name and ensure it matches the data_dir name."
                "It is essential that the kite_name matches the name of the surfplan file."
            )

        # Load YAML configuration
        with open(self.yaml_path, "r") as f:
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

            # Create vectors
            le_point = np.array([le_x, le_y, le_z])
            te_point = np.array([te_x, te_y, te_z])

            # Calculate chord vector and length
            chord_vector = te_point - le_point
            chord_length = np.linalg.norm(chord_vector)
            chord_lst.append(chord_length)

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
        """Calculate the average chord length"""
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

        # Ensure proper bounds (min, max) for optimization
        y_bounds = (min(arc_y[0], arc_y[-1]), max(arc_y[0], arc_y[-1]))
        z_bounds = (min(arc_z[0], arc_z[-1]), max(arc_z[0], arc_z[-1]))

        res = minimize(
            fit_error,
            x0=[delta0, gamma0],
            bounds=[y_bounds, z_bounds],
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


def load_kite_from_yaml(yaml_path):
    """
    Load a kite configuration from a YAML file path.

    Args:
        yaml_path (str or Path): Path to the YAML file

    Returns:
        KiteDefinition: Loaded kite configuration
    """
    yaml_path = Path(yaml_path)

    # Extract kite name from directory structure
    # Assume structure: .../kite_name/config_kite.yaml
    kite_name = yaml_path.parent.name

    print(f"📋 Loading kite: {kite_name}")
    print(f"   From: {yaml_path}")

    try:
        kite = KiteDefinition(kite_name)
        print(f"✅ Successfully loaded {kite_name}")
        return kite
    except Exception as e:
        print(f"❌ Error loading {kite_name}: {e}")
        return None


def get_current_anhedral_angle(kite_definition):
    """Helper function to calculate current anhedral angle."""
    LE_coords, _ = kite_definition.get_arc(halve=True)
    root_z = LE_coords[0, 2]  # Center (Y=0) Z-coordinate
    tip_z = LE_coords[-1, 2]  # Wing tip Z-coordinate
    span_y = abs(LE_coords[-1, 1])  # Wing tip Y-coordinate

    if span_y > 0:
        anhedral_angle = np.arctan2(root_z - tip_z, span_y)
    else:
        anhedral_angle = 0.0

    return anhedral_angle
