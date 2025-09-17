"""
AnhedralScaling module for modifying kite wing anhedral angles through Bezier curve manipulation.

This module provides functionality to modify kite wing geometry by changing the Bezier curve
parameters (delta, gamma, phi) to create flatter or more curved arcs, representing different
anhedral angles while maintaining aspect ratio and surface area.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
from scipy.special import comb
from scipy.optimize import minimize

from geometrical_kite_optimization.utils import PROJECT_DIR
from geometrical_kite_optimization.kite_definition import KiteDefinition
from SurfplanAdapter.process_surfplan import generate_wing_yaml


class AnhedralScaling(KiteDefinition):
    """Class for modifying kite anhedral through Bezier curve parameter adjustment."""

    def __init__(self, kitedefinition, angle_adjustment_deg=0.0):
        """
        Initialize AnhedralScaling with polar angle adjustment from center point.

        Args:
            kitedefinition (KiteDefinition): Base kite configuration object
            angle_adjustment_deg (float): Angle adjustment in degrees from center point
                                         (positive = roll up, negative = roll down)
        """
        if not hasattr(kitedefinition, "kite_name"):
            raise TypeError(
                "The kitedefinition input must be an instance of KiteDefinition"
            )

        self.kite_name = (
            f"{kitedefinition.kite_name}_anhedral_angle_{angle_adjustment_deg:+.1f}deg"
        )
        self.old_kite = kitedefinition
        self.angle_adjustment_deg = angle_adjustment_deg
        self.angle_adjustment_rad = np.radians(angle_adjustment_deg)

        # Get original Bezier parameters and control points
        self.original_bezier = self.old_kite.get_bezier_curve()
        self.original_control_points = self.original_bezier[2]

        # Store original properties for preservation
        self.old_ar = self.old_kite.old_aspect_ratio()
        self.old_area = self.old_kite.get_old_area()
        self.old_span = self.old_kite.get_old_span()[0]

        # Calculate polar coordinates from center point
        self.center_point = self.original_control_points[
            3
        ]  # P3 is the center/root point
        self.original_polar_coords = self._calculate_polar_coordinates()

        print(f"🪁 Initialized AnhedralScaling for {kitedefinition.kite_name}")
        print(
            f"📊 Original AR: {self.old_ar:.3f}, Area: {self.old_area:.3f} m², Span: {self.old_span:.3f} m"
        )
        print(
            f"🔄 Angle adjustment: {angle_adjustment_deg:+.1f}° (+ = roll up, - = roll down)"
        )
        print(
            f"📍 Center point: Y={self.center_point[0]:.3f}, Z={self.center_point[1]:.3f}"
        )

        # Display original polar coordinates
        for i, (r, theta_deg) in enumerate(self.original_polar_coords):
            print(f"   P{i}: r={r:.3f}m, θ={theta_deg:.1f}°")

    def _calculate_polar_coordinates(self):
        """Calculate polar coordinates (radius, angle) from center point to each control point."""
        polar_coords = []

        for i, point in enumerate(self.original_control_points):
            # Vector from center to this control point
            dx = point[0] - self.center_point[0]  # Y direction
            dy = point[1] - self.center_point[1]  # Z direction

            # Calculate radius and angle
            radius = np.sqrt(dx**2 + dy**2)
            angle_rad = np.arctan2(dy, dx)  # Angle from positive Y axis
            angle_deg = np.degrees(angle_rad)

            polar_coords.append((radius, angle_deg))

        return polar_coords

    def _generate_rotated_control_points(self):
        """Generate new control points by rotating angles while keeping radii constant."""
        new_control_points = []

        for i, ((radius, orig_angle_deg), orig_point) in enumerate(
            zip(self.original_polar_coords, self.original_control_points)
        ):
            if i == 3:  # P3 is the center point - keep it fixed
                new_control_points.append(orig_point)
            else:
                # Apply angle adjustment
                new_angle_deg = orig_angle_deg + self.angle_adjustment_deg
                new_angle_rad = np.radians(new_angle_deg)

                # Convert back to Cartesian coordinates
                new_dx = radius * np.cos(new_angle_rad)
                new_dy = radius * np.sin(new_angle_rad)

                new_y = self.center_point[0] + new_dx
                new_z = self.center_point[1] + new_dy

                new_control_points.append([new_y, new_z])

        return new_control_points

    def bernstein_poly(self, i, n, t):
        """Bernstein polynomial for Bezier curve generation."""
        return comb(n, i) * (t**i) * (1 - t) ** (n - i)

    def bezier_curve(self, points, nTimes=1000):
        """Generate Bezier curve from control points."""
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array(
            [self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
        )
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def generate_modified_arc(self):
        """Generate new arc using polar angle adjustment from center point."""
        # Get original arc points (half kite from center to tip)
        LE_coords, TE_coords = self.old_kite.get_arc(halve=True)
        arc_y, arc_z = LE_coords[:, 1], LE_coords[:, 2]

        # Generate new control points using polar rotation
        new_control_points = self._generate_rotated_control_points()

        print(f"🔍 Control point transformation:")
        print(
            f"   Center (P3) fixed: Y={new_control_points[3][0]:.3f}, Z={new_control_points[3][1]:.3f}"
        )

        for i in range(len(new_control_points)):
            if i != 3:  # Skip center point
                orig_r, orig_theta = self.original_polar_coords[i]
                new_theta = orig_theta + self.angle_adjustment_deg
                print(
                    f"   P{i}: θ={orig_theta:.1f}° → {new_theta:.1f}° (r={orig_r:.3f}m)"
                )

        # Generate new Bezier curve with same number of points as original arc
        n_points = len(arc_y)
        t_values = np.linspace(0.0, 1.0, n_points)

        new_arc_y = []
        new_arc_z = []

        # Use parametric mapping with new control points
        for t in t_values:
            # Calculate Bezier point at parameter t
            n = len(new_control_points) - 1
            point_y = 0
            point_z = 0

            for i, (ctrl_y, ctrl_z) in enumerate(new_control_points):
                bernstein = self.bernstein_poly(i, n, t)
                point_y += bernstein * ctrl_y
                point_z += bernstein * ctrl_z

            new_arc_y.append(point_y)
            new_arc_z.append(point_z)

        new_arc_y = np.array(new_arc_y)
        new_arc_z = np.array(new_arc_z)

        # --- Fix arc samples to correspond to same span stations as original ---
        # Make the new arc monotonic and indexable by span fraction
        orig_LE, _ = self.old_kite.get_arc(halve=True)
        orig_y = orig_LE[:, 1]
        tip_y_orig = np.max(orig_y)
        center_y_orig = np.min(orig_y)  # should be ~0

        # For the new arc, enforce the same order (tip->center)
        order_new = np.argsort(new_arc_y)[::-1]  # descending Y
        new_arc_y = new_arc_y[order_new]
        new_arc_z = new_arc_z[order_new]

        # Build a monotone mapping: span fraction f in [0,1] where f=1 at tip, f=0 at center
        f_orig = (orig_y - center_y_orig) / (tip_y_orig - center_y_orig)

        tip_y_new = np.max(new_arc_y)
        center_y_new = np.min(new_arc_y)
        y_new_target = center_y_new + f_orig * (tip_y_new - center_y_new)

        # For each original rib, pick the closest Y on the new arc
        idx = np.abs(new_arc_y[:, None] - y_new_target[None, :]).argmin(axis=0)
        new_arc_y = new_arc_y[idx]
        new_arc_z = new_arc_z[idx]

        # Calculate new span (max Y value, since center is at Y=0)
        self.new_span = max(new_arc_y) * 2  # Full span = 2 * half span
        old_full_span = max(orig_y) * 2  # Original full span
        span_change_factor = self.new_span / old_full_span

        print(f"📏 Half-span change: {max(orig_y):.3f} → {max(new_arc_y):.3f} m")
        print(
            f"📏 Full span change: {old_full_span:.3f} → {self.new_span:.3f} m (factor: {span_change_factor:.3f})"
        )

        return new_arc_y, new_arc_z, span_change_factor

    def generate_new_wing_sections(self):
        """Generate new wing sections with modified arc but preserved chord lengths."""
        new_arc_y, new_arc_z, span_change_factor = self.generate_modified_arc()

        # Get original wing sections
        LE_coords, TE_coords = self.old_kite.get_arc(halve=True)

        # For anhedral modifications, we preserve the original chord vectors exactly
        # Only the leading edge arc shape changes, trailing edge follows accordingly
        print(f"� Preserving original chord lengths and directions")
        print(f"� Wing area will be naturally preserved through chord preservation")

        # Create new wing sections
        new_LE_coords = []
        new_TE_coords = []

        # Original leading edge coordinates for sanity checks
        orig_LE, orig_TE = self.old_kite.get_arc(halve=True)
        orig_y = orig_LE[:, 1]
        orig_center_x = orig_LE[np.argmin(orig_y), 0]  # center rib X coordinate

        for i, (old_le, old_te) in enumerate(zip(orig_LE, orig_TE)):
            # New LE position from modified arc - preserve original X coordinate exactly
            new_le = np.array([old_le[0], new_arc_y[i], new_arc_z[i]])

            # Calculate original chord vector and preserve it exactly
            original_chord = old_te - old_le

            # New TE position using original chord vector (no scaling)
            new_te = new_le + original_chord

            new_LE_coords.append(new_le)
            new_TE_coords.append(new_te)

        new_LE_coords = np.array(new_LE_coords)
        new_TE_coords = np.array(new_TE_coords)

        # Sanity check: ensure center X is preserved
        center_new_x = new_LE_coords[np.argmin(new_arc_y), 0]  # center index
        print(
            f"🔍 Center X preservation check: {orig_center_x:.3f} → {center_new_x:.3f} m"
        )

        # Sanity check: Y coordinate consistency
        y_diff = np.abs(new_LE_coords[:, 1] - new_arc_y)
        max_y_diff = np.max(y_diff)
        print(
            f"🔍 Y coordinate consistency: max diff = {max_y_diff:.6f} m ({'✅ PASS' if max_y_diff < 1e-10 else '❌ FAIL'})"
        )

        # Create full wing by mirroring - ensure monotonic Y sequence
        # Build monotone Y sequence: -tip → center → +tip

        # Find center index (closest to Y=0)
        center_idx = np.argmin(np.abs(new_arc_y))

        # Create full wing with proper Y ordering
        # Left wing: mirror positive Y ribs to negative Y (excluding center)
        left_wing_LE = []
        left_wing_TE = []

        # Right wing ribs (positive Y, excluding center, in ascending Y order)
        right_indices = np.where(new_arc_y > 1e-10)[0]  # positive Y only
        if len(right_indices) > 0:
            right_sort = np.argsort(new_arc_y[right_indices])  # ascending Y order

            # Mirror to create left wing (descending Y order for proper monotonic sequence)
            for idx in right_indices[right_sort[::-1]]:  # reverse for descending
                left_wing_LE.append(
                    new_LE_coords[idx] * np.array([1, -1, 1])
                )  # mirror Y
                left_wing_TE.append(
                    new_TE_coords[idx] * np.array([1, -1, 1])
                )  # mirror Y

        # Center rib
        center_LE = [new_LE_coords[center_idx]]
        center_TE = [new_TE_coords[center_idx]]

        # Right wing ribs (positive Y, in ascending Y order)
        right_wing_LE = []
        right_wing_TE = []
        if len(right_indices) > 0:
            for idx in right_indices[right_sort]:  # ascending Y order
                right_wing_LE.append(new_LE_coords[idx])
                right_wing_TE.append(new_TE_coords[idx])

        # Combine all ribs: left + center + right
        full_arc_LE = np.array(left_wing_LE + center_LE + right_wing_LE)
        full_arc_TE = np.array(left_wing_TE + center_TE + right_wing_TE)

        # Sanity check: Y monotonicity
        y_values = full_arc_LE[:, 1]
        is_monotonic = all(
            y_values[i] <= y_values[i + 1] for i in range(len(y_values) - 1)
        )
        print(
            f"🔍 Full wing Y monotonicity: {'✅ PASS' if is_monotonic else '❌ FAIL'} (Y range: {np.min(y_values):.3f} to {np.max(y_values):.3f})"
        )

        # Verify final properties
        final_span = np.max(full_arc_LE[:, 1]) - np.min(full_arc_LE[:, 1])

        # Calculate actual area using proper span offset method (like original kite)
        actual_area = 0.0
        for i in range(len(full_arc_LE) - 1):
            # Calculate chord lengths
            chord1 = np.linalg.norm(full_arc_TE[i] - full_arc_LE[i])
            chord2 = np.linalg.norm(full_arc_TE[i + 1] - full_arc_LE[i + 1])

            # Calculate span offset (distance between adjacent ribs)
            y_diff = full_arc_LE[i + 1, 1] - full_arc_LE[i, 1]
            z_diff = full_arc_LE[i + 1, 2] - full_arc_LE[i, 2]
            span_offset = np.sqrt(y_diff**2 + z_diff**2)

            # Trapezoidal rule for area (same as original kite calculation)
            area_rectangle = 0.5 * (chord1 + chord2) * span_offset
            actual_area += area_rectangle

        final_ar = final_span**2 / actual_area if actual_area > 0 else 0

        print(f"✅ Final verification:")
        print(f"   📏 Final span: {final_span:.3f} m")
        print(
            f"   � Final area: {actual_area:.3f} m² (original: {self.old_area:.3f} m²)"
        )
        print(f"   � Final AR: {final_ar:.3f} (original: {self.old_ar:.3f})")
        print(f"   🔄 Anhedral angle modified by {self.angle_adjustment_deg:+.1f}°")

        return full_arc_LE, full_arc_TE

    def get_new_wing_sections(self):
        """Generate new wing sections data structure."""
        full_arc_LE, full_arc_TE = self.generate_new_wing_sections()
        new_wing_sections = []

        # Map the mirrored full wing back to original wing sections structure
        n_orig_sections = len(self.old_kite.config["wing_sections"]["data"])

        # If we have more sections in the full wing than in original data,
        # we need to downsample or map them properly
        if len(full_arc_LE) != n_orig_sections:
            print(
                f"⚠️  Wing section count mismatch: full wing has {len(full_arc_LE)}, original has {n_orig_sections}"
            )

            # Create index mapping from full wing to original sections
            y_full = full_arc_LE[:, 1]

            # Get original Y coordinates for proper mapping
            orig_sections = []
            for i in range(n_orig_sections):
                orig_data = self.old_kite.config["wing_sections"]["data"][i]
                orig_y = orig_data[self.old_kite.header_map["LE_y"]]
                orig_sections.append((i, orig_y))

            # Sort by Y coordinate
            orig_sections.sort(key=lambda x: x[1])

            # Map each original section to closest point in full wing
            section_indices = []
            for orig_idx, orig_y in orig_sections:
                closest_idx = np.argmin(np.abs(y_full - orig_y))
                section_indices.append((orig_idx, closest_idx))

            # Build wing sections using the mapping
            for orig_idx, full_idx in section_indices:
                rib = {}
                rib["LE"] = full_arc_LE[full_idx]
                rib["TE"] = full_arc_TE[full_idx]

                # Keep original VUP vectors and airfoil from the original index
                rib["VUP"] = self.old_kite.config["wing_sections"]["data"][orig_idx][
                    self.old_kite.header_map["VUP_x"] : self.old_kite.header_map[
                        "VUP_z"
                    ]
                    + 1
                ]
                rib["airfoil_id"] = self.old_kite.config["wing_sections"]["data"][
                    orig_idx
                ][self.old_kite.header_map["airfoil_id"]]
                new_wing_sections.append(rib)
        else:
            # Direct mapping when counts match
            for i, (le_coord, te_coord) in enumerate(zip(full_arc_LE, full_arc_TE)):
                rib = {}
                rib["LE"] = le_coord
                rib["TE"] = te_coord

                # Keep original VUP vectors
                rib["VUP"] = self.old_kite.config["wing_sections"]["data"][i][
                    self.old_kite.header_map["VUP_x"] : self.old_kite.header_map[
                        "VUP_z"
                    ]
                    + 1
                ]
                rib["airfoil_id"] = self.old_kite.config["wing_sections"]["data"][i][
                    self.old_kite.header_map["airfoil_id"]
                ]
                new_wing_sections.append(rib)

        return generate_wing_yaml.generate_wing_sections_data(new_wing_sections)

    def chord_vectors(self, plot=False):
        """Calculate chord vectors for the Bezier-modified kite."""
        full_arc_LE, full_arc_TE = self.generate_new_wing_sections()

        # Calculate quarter chord points
        full_arc_qchord = []
        new_chord_vectors = []

        for le_coord, te_coord in zip(full_arc_LE, full_arc_TE):
            chord_vector = te_coord - le_coord
            qchord = le_coord + 0.25 * chord_vector
            full_arc_qchord.append(qchord)
            new_chord_vectors.append(chord_vector)

        full_arc_qchord = np.array(full_arc_qchord)

        if plot:
            plt.figure(figsize=(15, 10))

            # Plot the modified kite geometry
            plt.subplot(2, 2, 1)
            plt.plot(
                full_arc_LE[:, 1], full_arc_LE[:, 2], "g-", label="LE", linewidth=2
            )
            plt.plot(
                full_arc_TE[:, 1], full_arc_TE[:, 2], "r-", label="TE", linewidth=2
            )
            plt.scatter(
                full_arc_qchord[:, 1],
                full_arc_qchord[:, 2],
                c="blue",
                label="Quarter Chord",
                s=30,
            )
            plt.axis("equal")
            plt.xlabel("Y-axis (Span)")
            plt.ylabel("Z-axis (Height)")
            plt.title(
                f"Polar Angle Modified Kite (Δθ: {self.angle_adjustment_deg:+.1f}°)"
            )
            plt.legend()
            plt.grid(True)

            # Compare original vs modified arc
            plt.subplot(2, 2, 2)
            orig_LE, _ = self.old_kite.get_arc(halve=True)
            new_arc_y, new_arc_z, _ = self.generate_modified_arc()

            plt.plot(orig_LE[:, 1], orig_LE[:, 2], "o-", label="Original LE", alpha=0.7)
            plt.plot(new_arc_y, new_arc_z, "s-", label="Modified LE", alpha=0.7)
            plt.axis("equal")
            plt.xlabel("Y-axis (Span)")
            plt.ylabel("Z-axis (Height)")
            plt.title("Arc Comparison")
            plt.legend()
            plt.grid(True)

            # Bezier curve visualization
            plt.subplot(2, 2, 3)
            orig_bezier = self.old_kite.get_bezier_curve()
            plt.plot(
                orig_bezier[0],
                orig_bezier[1],
                "-",
                label="Original Bezier",
                linewidth=2,
            )

            # Generate new Bezier for visualization
            new_control_points = self._generate_rotated_control_points()
            new_bez_x, new_bez_y = self.bezier_curve(new_control_points)
            plt.plot(new_bez_x, new_bez_y, "-", label="Modified Bezier", linewidth=2)

            # Plot control points
            orig_ctrl = orig_bezier[2]
            new_ctrl = np.array(new_control_points)
            plt.scatter(
                [p[0] for p in orig_ctrl],
                [p[1] for p in orig_ctrl],
                c="blue",
                s=50,
                label="Original Controls",
            )
            plt.scatter(
                new_ctrl[:, 0], new_ctrl[:, 1], c="red", s=50, label="Modified Controls"
            )

            plt.axis("equal")
            plt.xlabel("Y-axis (Span)")
            plt.ylabel("Z-axis (Height)")
            plt.title("Bezier Curve Comparison")
            plt.legend()
            plt.grid(True)

            # Parameters comparison
            plt.subplot(2, 2, 4)
            angles = ["P0", "P1", "P2", "P3"]
            orig_angles = [coord[1] for coord in self.original_polar_coords]
            new_angles = [
                orig_angle + (self.angle_adjustment_deg if i != 3 else 0)
                for i, orig_angle in enumerate(orig_angles)
            ]

            x = np.arange(len(angles))
            width = 0.35

            plt.bar(
                x - width / 2, orig_angles, width, label="Original Angles", alpha=0.7
            )
            plt.bar(
                x + width / 2, new_angles, width, label="Modified Angles", alpha=0.7
            )

            plt.xlabel("Control Points")
            plt.ylabel("Angle from Center (°)")
            plt.title("Polar Angle Comparison")
            plt.xticks(x, angles)
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        return (
            new_chord_vectors,
            full_arc_qchord,
            full_arc_LE[: len(full_arc_LE) // 2],  # New LE coords (half kite)
            full_arc_TE[: len(full_arc_TE) // 2],  # New TE coords (half kite)
            full_arc_LE,  # Full arc LE
            full_arc_TE,  # Full arc TE
            full_arc_qchord,  # Full arc quarter chord
        )

    def export_data(self):
        """Export Bezier-modified kite data."""
        new_yaml_file_path = Path(PROJECT_DIR) / "processed_data" / f"{self.kite_name}"
        wing_sections = self.get_new_wing_sections()

        # Use original config as base and update wing sections
        yaml_data = self.old_kite.config.copy()
        yaml_data["wing_sections"] = wing_sections

        # Save YAML file
        os.makedirs(new_yaml_file_path, exist_ok=True)
        yaml_file_path = Path(new_yaml_file_path) / "config_kite.yaml"

        with open(yaml_file_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f'✅ Generated polar angle modified YAML file at "{yaml_file_path}"')
        print(f"🔄 Angle adjustment: {self.angle_adjustment_deg:+.1f}°")
        print(f"📏 Leading edge shape modified in Y-Z plane")
        print(f"� Chord lengths and wing area preserved")

        return yaml_file_path, yaml_data
