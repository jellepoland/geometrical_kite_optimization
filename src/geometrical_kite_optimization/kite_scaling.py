from pathlib import Path
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math as m
import csv
from scipy.special import comb
from SurfplanAdapter.process_surfplan import generate_wing_yaml
from geometrical_kite_optimization.utils import PROJECT_DIR


class KiteScaling:
    """Class to scaling."""

    def __init__(
        self,
        base_kite,
        new_ar=None,
        new_phi=None,
        new_delta=None,
        new_gamma=None,
        is_gamma_a_percentage=True,
    ):
        if not hasattr(base_kite, "kite_name"):
            raise TypeError("The base_kite input must be an instance of base_kite")

        # Get Bezier parameters of base kite
        (
            bez_x,
            bez_y,
            points_array,
            LE_arc_proj_y,
            LE_arc_proj_z,
            base_phi,
            base_delta,
            base_gamma,
        ) = base_kite.get_bezier_curve(normalized=True, plot=False)

        self.kite_name = base_kite.kite_name
        self.old_kite = base_kite
        self.is_gamma_a_percentage = is_gamma_a_percentage
        if new_ar == None:
            self.new_ar = base_kite.old_aspect_ratio()
        else:
            print(
                f"--> adjusting aspect ratio from {base_kite.old_aspect_ratio():.2f} to {new_ar:.2f}"
            )
            self.new_ar = new_ar
        if new_phi == None:
            self.phi = base_phi
        else:
            print(
                f"--> adjusting phi (anhedral angle) from {np.rad2deg(base_phi):.2f}° to {new_phi:.2f}°"
            )
            self.phi = new_phi = np.deg2rad(new_phi)
        if new_delta == None:
            self.delta = base_delta
        else:
            print(
                f"-->adjusting delta (width of control point 1) from {base_delta:.2f} to {new_delta:.2f}"
            )
            self.delta = new_delta
        if new_gamma == None:
            self.gamma = base_gamma
            if self.is_gamma_a_percentage:
                base_tip_height = m.tan(base_phi)
                gamma_fraction = base_gamma / base_tip_height
                new_tip_height = m.tan(self.phi)
                self.gamma = new_tip_height * gamma_fraction
        else:
            print(
                f"--> adjusting gamma (height of control point 2) from {base_gamma:.2f} to {new_gamma:.2f}"
            )
            self.gamma = new_gamma

        self.ar_scaling_factor = self.new_ar / self.old_kite.old_aspect_ratio()
        self.area = (
            self.old_kite.get_old_area()
        )  # Original area, preserved during scaling

    def get_current_area(self):
        """
        Get the current effective area of the scaled kite.
        For area preservation scaling, this should equal the original area.
        """
        return self.area  # Area is preserved, so it remains the original area

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

    def get_new_span(self, full_arc_LE):
        """
        Calculate the actual flat span directly from geometry using arc length.

        Args:
            full_arc_LE: Leading edge coordinates array

        Returns:
            float: Actual flat span (arc length) in meters
        """
        # Extract Y and Z coordinates for flat span calculation
        y_coords = full_arc_LE[:, 1]
        z_coords = full_arc_LE[:, 2]

        # Calculate the actual arc length (flat span)
        return self.get_flat_span(y_coords, z_coords)

    def get_theoretical_span(self):
        """
        Calculate theoretical span using sqrt(AR * area) formula.
        This is useful for initial estimates and avoiding circular dependencies.

        Returns:
            float: Theoretical span in meters
        """
        return m.sqrt(self.new_ar * self.area)

    def scale_arc_to_span(self):
        """Rescales the LE arc curve to the correct span and height"""
        LE_y_norm, LE_z_norm, _, _, _ = self.get_le_arc_curve()
        normalized_span = self.get_flat_span(LE_y_norm, LE_z_norm)

        # Calculate target span using theoretical relationship for scaling
        target_span = m.sqrt(self.new_ar * self.area)
        span_scaling_factor = target_span / (2 * normalized_span)

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
        """
        Calculate chord scaling factor using two-step approach:
        1. Scale to achieve target aspect ratio
        2. Account for area preservation scaling
        """
        # Step 1: Calculate basic chord scaling for aspect ratio change
        theoretical_new_span = m.sqrt(self.new_ar * self.area)
        old_span = self.old_kite.get_old_span()
        old_ar = self.old_kite.old_aspect_ratio()

        # Basic chord scaling to achieve target AR (without area preservation)
        basic_chord_scaling = theoretical_new_span / old_span[0] * old_ar / self.new_ar

        # Step 2: Calculate area preservation scaling factor
        area_scaling_factor = self.get_area_preservation_scaling_factor()

        # Combined scaling: first for AR, then for area preservation
        final_chord_scaling = basic_chord_scaling * area_scaling_factor

        return final_chord_scaling

    def two_step_scaling_process(self):
        """
        Perform the complete two-step scaling process:
        1. Scale to achieve target aspect ratio
        2. Scale to preserve original area
        3. Apply center alignment

        Returns:
            dict: Scaling information and factors
        """
        original_area = self.old_kite.get_old_area()
        original_span = self.old_kite.get_old_span()[0]
        original_ar = self.old_kite.old_aspect_ratio()

        # Step 1: Calculate target span for new aspect ratio
        target_span = m.sqrt(self.new_ar * self.area)

        # Step 2: Calculate area preservation scaling
        area_scaling_factor = self.get_area_preservation_scaling_factor()

        # Step 3: Calculate final dimensions
        final_span = target_span  # Span stays at target
        final_chord_scaling = self.chord_scaling()  # Includes both AR and area scaling

        # Calculate final area (should match original)
        final_area = final_span**2 / self.new_ar * (area_scaling_factor**2)

        scaling_info = {
            "original_area": original_area,
            "original_span": original_span,
            "original_ar": original_ar,
            "target_ar": self.new_ar,
            "target_span": target_span,
            "area_scaling_factor": area_scaling_factor,
            "final_chord_scaling": final_chord_scaling,
            "final_area": final_area,
            "area_preservation_error": abs(final_area - original_area)
            / original_area
            * 100,
        }

        return scaling_info

    def get_area_preservation_scaling_factor(self):
        """
        Calculate the scaling factor needed to preserve the original wing area.
        This is applied after the aspect ratio scaling.

        Returns:
            float: Scaling factor to preserve area (applied to chord lengths)
        """
        original_area = self.old_kite.get_old_area()
        original_span = self.old_kite.get_old_span()[0]
        original_ar = self.old_kite.old_aspect_ratio()

        # Calculate span scaling factor needed to achieve target AR (if we kept the same area)
        # For constant area: span² = AR * area, so span_new = sqrt(new_ar * original_area)
        target_span_for_constant_area = m.sqrt(self.new_ar * original_area)
        span_scaling_factor = target_span_for_constant_area / original_span

        # However, we want to scale ONLY the span to achieve the AR, and then adjust chords to preserve area
        # If we only scale span by the AR ratio: new_span = old_span * sqrt(new_ar / old_ar)
        span_scaling_for_ar_only = m.sqrt(self.new_ar / original_ar)
        new_span_ar_only = original_span * span_scaling_for_ar_only

        # With this span and no chord adjustment, the new area would be:
        # new_area = new_span² / new_ar
        new_area_without_chord_adjustment = (new_span_ar_only**2) / self.new_ar

        # To preserve the original area, we need to scale chords by:
        # chord_scaling = sqrt(original_area / new_area_without_chord_adjustment)
        area_scaling_factor = m.sqrt(original_area / new_area_without_chord_adjustment)

        return area_scaling_factor

    def align_mid_span_leading(self, full_arc_LE, full_arc_TE):
        """
        Align the center crossing point (Y≈0) of the new kite to match the original kite.

        Args:
            full_arc_LE: Leading edge coordinates array
            full_arc_TE: Trailing edge coordinates array

        Returns:
            tuple: (aligned_LE, aligned_TE, translation_info)
        """
        # Get original kite's center crossing point (Y≈0)
        orig_LE, orig_TE = self.old_kite.get_arc(halve=False)  # Get full arc
        orig_y_coords = orig_LE[:, 1]

        # Find the two points closest to Y=0 in original kite
        abs_y_orig = np.abs(orig_y_coords)
        center_indices_orig = np.argsort(abs_y_orig)[:2]  # Two closest points to Y=0

        # Calculate the midpoint of these two closest points in original kite
        center_point_orig_LE = 0.5 * (
            orig_LE[center_indices_orig[0]] + orig_LE[center_indices_orig[1]]
        )
        center_point_orig_TE = 0.5 * (
            orig_TE[center_indices_orig[0]] + orig_TE[center_indices_orig[1]]
        )

        # Find the two points closest to Y=0 in new kite
        new_y_coords = full_arc_LE[:, 1]
        abs_y_new = np.abs(new_y_coords)
        center_indices_new = np.argsort(abs_y_new)[:2]  # Two closest points to Y=0

        # Calculate the midpoint of these two closest points in new kite
        center_point_new_LE = 0.5 * (
            full_arc_LE[center_indices_new[0]] + full_arc_LE[center_indices_new[1]]
        )
        center_point_new_TE = 0.5 * (
            full_arc_TE[center_indices_new[0]] + full_arc_TE[center_indices_new[1]]
        )

        # Calculate translation needed to align center points
        translation_LE = center_point_orig_LE - center_point_new_LE
        translation_TE = center_point_orig_TE - center_point_new_TE

        # Apply translation to align center points
        full_arc_LE_aligned = full_arc_LE + translation_LE
        full_arc_TE_aligned = full_arc_TE + translation_TE

        # Prepare translation info for debugging
        translation_info = {
            "orig_center_LE": center_point_orig_LE,
            "new_center_LE": center_point_new_LE,
            "translation_LE": translation_LE,
            "translation_TE": translation_TE,
            "center_after_LE": 0.5
            * (
                full_arc_LE_aligned[center_indices_new[0]]
                + full_arc_LE_aligned[center_indices_new[1]]
            ),
        }

        return full_arc_LE_aligned, full_arc_TE_aligned, translation_info

    def chord_vectors(self, plot=False):
        """Calculate chord vectors for each wing section."""
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

        # Apply center alignment to ensure Y=0 crossing point matches original kite
        full_arc_LE_aligned, full_arc_TE_aligned, translation_info = (
            self.align_mid_span_leading(full_arc_LE, full_arc_TE)
        )

        # Also align quarter chord points with the same translation
        full_arc_qchord_aligned = full_arc_qchord + translation_info["translation_LE"]

        if plot:
            plt.plot(
                np.array(full_arc_LE_aligned)[:, 1],
                np.array(full_arc_LE_aligned)[:, 2],
                "g-",
                label="LE",
            )
            plt.plot(
                np.array(full_arc_LE_aligned)[:, 1],
                np.array(full_arc_LE_aligned)[:, 2],
                "gx",
            )
            plt.plot(
                np.array(full_arc_TE_aligned)[:, 1],
                np.array(full_arc_TE_aligned)[:, 2],
                "r-",
                label="TE",
            )
            plt.plot(
                np.array(full_arc_TE_aligned)[:, 1],
                np.array(full_arc_TE_aligned)[:, 2],
                "rx",
            )
            plt.plot(
                np.array(full_arc_qchord_aligned)[:, 1],
                np.array(full_arc_qchord_aligned)[:, 2],
                "b--",
                label="QC",
            )
            plt.plot(
                np.array(full_arc_qchord_aligned)[:, 1],
                np.array(full_arc_qchord_aligned)[:, 2],
                "bx",
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
            full_arc_LE_aligned,
            full_arc_TE_aligned,
            full_arc_qchord_aligned,
        )

    def calculate_wing_area(self, full_arc_LE, full_arc_TE):
        """
        Calculate the total wing area as a sum of quadrilateral panels.
        Each panel is formed by two adjacent wing sections (ribs).

        Args:
            full_arc_LE: Leading edge coordinates array
            full_arc_TE: Trailing edge coordinates array

        Returns:
            float: Total wing area in m²
        """
        total_area = 0.0

        for i in range(len(full_arc_LE) - 1):
            # Get the four corners of the quadrilateral panel
            # Panel between rib i and rib i+1
            LE_i = full_arc_LE[i]  # Leading edge of rib i
            TE_i = full_arc_TE[i]  # Trailing edge of rib i
            LE_i1 = full_arc_LE[i + 1]  # Leading edge of rib i+1
            TE_i1 = full_arc_TE[i + 1]  # Trailing edge of rib i+1

            # Calculate area of quadrilateral using the shoelace formula
            # Split quadrilateral into two triangles: (LE_i, TE_i, LE_i1) and (TE_i, TE_i1, LE_i1)

            # Triangle 1: LE_i -> TE_i -> LE_i1
            v1 = TE_i - LE_i
            v2 = LE_i1 - LE_i
            area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))

            # Triangle 2: TE_i -> TE_i1 -> LE_i1
            v1 = TE_i1 - TE_i
            v2 = LE_i1 - TE_i
            area2 = 0.5 * np.linalg.norm(np.cross(v1, v2))

            panel_area = area1 + area2
            total_area += panel_area

        return total_area

    def get_new_wing_sections(self):
        """Generate new wing sections with scaled span and adjusted LE arc."""
        _, _, _, _, full_arc_LE, full_arc_TE, _ = self.chord_vectors()

        # Apply center alignment (already done in chord_vectors, but get debug info)
        _, _, translation_info = self.align_mid_span_leading(full_arc_LE, full_arc_TE)

        # Get two-step scaling information
        scaling_info = self.two_step_scaling_process()

        print(f"🎯 Center alignment:")
        print(
            f"   Original center (LE): X={translation_info['orig_center_LE'][0]:.3f}, Y={translation_info['orig_center_LE'][1]:.3f}, Z={translation_info['orig_center_LE'][2]:.3f}"
        )
        print(
            f"   New center (LE):      X={translation_info['new_center_LE'][0]:.3f}, Y={translation_info['new_center_LE'][1]:.3f}, Z={translation_info['new_center_LE'][2]:.3f}"
        )
        print(
            f"   Translation (LE):     ΔX={translation_info['translation_LE'][0]:.3f}, ΔY={translation_info['translation_LE'][1]:.3f}, ΔZ={translation_info['translation_LE'][2]:.3f}"
        )
        print(
            f"   After translation:    X={translation_info['center_after_LE'][0]:.3f}, Y={translation_info['center_after_LE'][1]:.3f}, Z={translation_info['center_after_LE'][2]:.3f}"
        )

        print(f"\n🔧 Two-Step Scaling Process:")
        print(f"   Step 1 - AR Scaling:")
        print(f"     Original AR:        {scaling_info['original_ar']:.3f}")
        print(f"     Target AR:          {scaling_info['target_ar']:.3f}")
        print(f"     Original span:      {scaling_info['original_span']:.3f} m")
        print(f"     Target span:        {scaling_info['target_span']:.3f} m")
        print(f"   Step 2 - Area Preservation:")
        print(f"     Original area:      {scaling_info['original_area']:.3f} m²")
        print(f"     Area scaling factor: {scaling_info['area_scaling_factor']:.4f}")
        print(f"     Final chord scaling: {scaling_info['final_chord_scaling']:.4f}")
        print(f"     Final area:         {scaling_info['final_area']:.3f} m²")
        print(
            f"     Area preservation:  {scaling_info['area_preservation_error']:.4f}% error"
        )

        # Calculate and verify wing area
        calculated_area = self.calculate_wing_area(full_arc_LE, full_arc_TE)
        original_area = self.old_kite.get_old_area()
        area_change_percent = ((calculated_area - original_area) / original_area) * 100

        print(f"\n📐 Wing Area Verification (Panel Summation):")
        print(f"   Original area:   {original_area:.3f} m²")
        print(f"   Calculated area: {calculated_area:.3f} m²")
        print(f"   Area change:     {area_change_percent:+.2f}%")

        # Calculate span verification - both theoretical and actual
        span_new_actual = self.get_new_span(full_arc_LE)  # actual geometric span
        span_new_theoretical = m.sqrt(self.new_ar * self.area)  # theoretical target
        span_original = self.old_kite.get_old_span()[0]
        span_change_actual = ((span_new_actual - span_original) / span_original) * 100
        span_change_theoretical = (
            (span_new_theoretical - span_original) / span_original
        ) * 100

        print(f"📏 Span Verification:")
        print(f"   Original span:       {span_original:.3f} m")
        print(
            f"   Theoretical target:  {span_new_theoretical:.3f} m ({span_change_theoretical:+.2f}%)"
        )
        print(
            f"   Actual geometry:     {span_new_actual:.3f} m ({span_change_actual:+.2f}%)"
        )
        print(
            f"   Target vs Actual:    {((span_new_actual - span_new_theoretical) / span_new_theoretical * 100):+.3f}% difference"
        )  # Calculate aspect ratio verification
        ar_new = span_new_actual**2 / calculated_area
        ar_original = self.old_kite.old_aspect_ratio()
        ar_target = self.new_ar

        print(f"🪁 Aspect Ratio Verification:")
        print(f"   Original AR:     {ar_original:.3f}")
        print(f"   Target AR:       {ar_target:.3f}")
        print(f"   Calculated AR:   {ar_new:.3f}")
        print(f"   AR accuracy:     {((ar_new - ar_target) / ar_target * 100):+.2f}%")

        new_wing_sections = []

        for i, LE_coord in enumerate(full_arc_LE):
            rib = {}
            rib["LE"] = LE_coord
            rib["TE"] = full_arc_TE[i]
            rib["VUP"] = self.old_kite.config["wing_sections"]["data"][i][
                self.old_kite.header_map["VUP_x"] : self.old_kite.header_map["VUP_z"]
                + 1
            ]
            rib["airfoil_id"] = self.old_kite.config["wing_sections"]["data"][i][
                self.old_kite.header_map["airfoil_id"]
            ]
            new_wing_sections.append(rib)

        return generate_wing_yaml.generate_wing_sections_data(new_wing_sections)

    def export_data(self):
        """Export kite data to a text file."""
        new_yaml_file_path = (
            Path(PROJECT_DIR)
            / "processed_data"
            / f"{self.kite_name}_aspect_ratio_{self.new_ar:.2f}.yaml"
        )
        old_file = self.old_kite.config

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
            print(f"Bridle lines: {len(bridle_lines_yaml['data'])}")
        except:
            pass

        return yaml_file_path, yaml_data
