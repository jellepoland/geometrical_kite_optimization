"""
AnhedralScaling module for modifying kite wing anhedral angles through panel angle percentage changes.

This module provides functionality to modify kite wing geometry by changing the angles
between consecutive wing panels by a specified percentage, preserving the non-linear
character of the original anhedral distribution while maintaining aspect ratio and surface area.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path

from utils import PROJECT_DIR
from scripts.new.kite_definition import KiteDefinition
from SurfplanAdapter.process_surfplan import generate_wing_yaml


class AnhedralScaling(KiteDefinition):
    """Class for modifying kite anhedral through panel angle percentage changes."""

    def __init__(self, kitedefinition, panel_angle_change_percent=0.0):
        """
        Initialize AnhedralScaling with panel angle modification.

        Args:
            kitedefinition (KiteDefinition): Base kite configuration object
            panel_angle_change_percent (float): Percentage change to apply to panel angles
                                               (positive = more anhedral, negative = less anhedral)
        """
        if not hasattr(kitedefinition, "kite_name"):
            raise TypeError(
                "The kitedefinition input must be an instance of KiteDefinition"
            )

        self.kite_name = (
            f"{kitedefinition.kite_name}_anhedral_{panel_angle_change_percent:+.1f}pct"
        )
        self.old_kite = kitedefinition
        self.panel_angle_change_percent = panel_angle_change_percent

        # Store original properties for preservation
        self.old_ar = self.old_kite.old_aspect_ratio()
        self.area = self.old_kite.get_old_area()

        print(f"🪁 Initialized AnhedralScaling for {kitedefinition.kite_name}")
        print(f"📊 Original AR: {self.old_ar:.3f}, Area: {self.area:.3f} m²")
        print(f"🔄 Panel angle change: {panel_angle_change_percent:+.1f}%")

    def calculate_panel_angles(self, coords):
        """Calculate angles between consecutive panels."""
        panel_angles = []

        for i in range(len(coords) - 1):
            # Vector from current panel to next panel
            panel_vector = coords[i + 1] - coords[i]

            # Project onto Y-Z plane (remove X component for pure span-height analysis)
            panel_vector_yz = np.array([0, panel_vector[1], panel_vector[2]])

            # Reference horizontal vector in Y direction
            horizontal_ref = np.array([0, 1, 0])

            # Calculate angle between panel vector and horizontal
            if np.linalg.norm(panel_vector_yz) > 1e-10:  # Avoid division by zero
                cos_angle = np.dot(panel_vector_yz, horizontal_ref) / (
                    np.linalg.norm(panel_vector_yz) * np.linalg.norm(horizontal_ref)
                )
                # Clamp to valid range for arccos
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Determine sign based on Z component (negative Z means downward angle)
                if panel_vector[2] < 0:
                    angle = -angle

                panel_angles.append(angle)
            else:
                panel_angles.append(0.0)

        return panel_angles

    def apply_panel_angle_modification(self):
        """Apply panel angle modification to wing sections."""
        LE_coords, TE_coords = self.old_kite.get_arc(halve=True)

        # Calculate current panel angles
        original_panel_angles = self.calculate_panel_angles(LE_coords)

        # Modify angles by percentage
        modified_angles = [
            angle * (1 + self.panel_angle_change_percent / 100)
            for angle in original_panel_angles
        ]

        # Reconstruct wing sections with modified angles
        new_LE_coords = [LE_coords[0].copy()]  # Start with root position
        new_TE_coords = [TE_coords[0].copy()]

        for i, modified_angle in enumerate(modified_angles):
            # Calculate the original panel segment length
            original_segment = LE_coords[i + 1] - LE_coords[i]

            # Create new panel vector with modified angle
            # Keep X component unchanged, modify Y and Z based on new angle
            x_component = original_segment[0]

            # For Y-Z plane, use the modified angle
            yz_length = np.sqrt(original_segment[1] ** 2 + original_segment[2] ** 2)
            new_y = yz_length * np.cos(modified_angle)
            new_z = yz_length * np.sin(modified_angle)

            # Construct new segment vector
            new_segment = np.array([x_component, new_y, new_z])

            # Add to previous position to get new coordinates
            new_le_coord = new_LE_coords[-1] + new_segment
            new_LE_coords.append(new_le_coord)

            # Apply same transformation to TE coordinates
            te_original_segment = TE_coords[i + 1] - TE_coords[i]
            te_x_component = te_original_segment[0]
            te_yz_length = np.sqrt(
                te_original_segment[1] ** 2 + te_original_segment[2] ** 2
            )
            te_new_y = te_yz_length * np.cos(modified_angle)
            te_new_z = te_yz_length * np.sin(modified_angle)

            new_te_segment = np.array([te_x_component, te_new_y, te_new_z])
            new_te_coord = new_TE_coords[-1] + new_te_segment
            new_TE_coords.append(new_te_coord)

        # Convert to numpy arrays
        new_LE_coords = np.array(new_LE_coords)
        new_TE_coords = np.array(new_TE_coords)

        # Create full wing by mirroring
        full_arc_LE = np.vstack(
            [new_LE_coords[:-1], np.flip(new_LE_coords[:-1], 0) * np.array([1, -1, 1])]
        )
        full_arc_TE = np.vstack(
            [new_TE_coords[:-1], np.flip(new_TE_coords[:-1], 0) * np.array([1, -1, 1])]
        )

        return full_arc_LE, full_arc_TE, original_panel_angles, modified_angles

    def get_new_wing_sections(self):
        """Generate new wing sections with modified panel angles."""
        full_arc_LE, full_arc_TE, _, _ = self.apply_panel_angle_modification()
        new_wing_sections = []

        for i, (le_coord, te_coord) in enumerate(zip(full_arc_LE, full_arc_TE)):
            rib = {}
            rib["LE"] = le_coord
            rib["TE"] = te_coord

            # Keep original VUP vectors
            rib["VUP"] = self.old_kite.config["wing_sections"]["data"][i][
                self.old_kite.header_map["VUP_x"] : self.old_kite.header_map["VUP_z"]
                + 1
            ]
            rib["airfoil_id"] = self.old_kite.config["wing_sections"]["data"][i][
                self.old_kite.header_map["airfoil_id"]
            ]
            new_wing_sections.append(rib)

        return generate_wing_yaml.generate_wing_sections_data(new_wing_sections)

    def chord_vectors(self, plot=False):
        """Calculate chord vectors for the panel-angle-modified kite."""
        full_arc_LE, full_arc_TE, original_angles, modified_angles = (
            self.apply_panel_angle_modification()
        )

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
            plt.figure(figsize=(12, 8))

            # Plot the modified kite
            plt.subplot(1, 2, 1)
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
                f"Panel Angles Modified by {self.panel_angle_change_percent:.1f}%"
            )
            plt.legend()
            plt.grid(True)

            # Plot angle comparison
            plt.subplot(1, 2, 2)
            x_pos = range(len(original_angles))
            plt.plot(
                x_pos,
                np.degrees(original_angles),
                "o-",
                label="Original Angles",
                linewidth=2,
            )
            plt.plot(
                x_pos,
                np.degrees(modified_angles),
                "s-",
                label="Modified Angles",
                linewidth=2,
            )
            plt.xlabel("Panel Index")
            plt.ylabel("Panel Angle (degrees)")
            plt.title("Panel Angle Comparison")
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
        """Export panel-angle-modified kite data."""
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

        # Get angle statistics for reporting
        _, _, original_angles, modified_angles = self.apply_panel_angle_modification()
        avg_original = np.degrees(np.mean(np.abs(original_angles)))
        avg_modified = np.degrees(np.mean(np.abs(modified_angles)))

        print(f'✅ Generated panel-angle-modified YAML file at "{yaml_file_path}"')
        print(f"🔄 Panel angle change: {self.panel_angle_change_percent:.1f}%")
        print(f"📐 Average original panel angle: {avg_original:.2f}°")
        print(f"📐 Average modified panel angle: {avg_modified:.2f}°")
        print(f"📏 Preserved AR: {self.old_ar:.3f}")
        print(f"📦 Preserved Area: {self.area:.3f} m²")

        return yaml_file_path, yaml_data
