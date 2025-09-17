#!/usr/bin/env python3
"""
Script to plot and compare two kite configurations from YAML files.

This script loads two kite configurations from YAML files and creates
a 3D visualization comparing their geometries using equal axis scales.
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from KiteDefinition import KiteDefinition
from utils import plot_kite


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


def plot_two_kites_comparison(kite1, kite2, save_plot=False, output_path=None):
    """
    Create a 3D comparison plot of two kites with equal axis scales.

    Args:
        kite1 (KiteDefinition): First kite configuration
        kite2 (KiteDefinition): Second kite configuration
        save_plot (bool): Whether to save the plot to file
        output_path (str): Path to save the plot (if save_plot=True)
    """
    print(f"🎨 Creating comparison plot...")

    # Create the plot using the utils function
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    kite_list = [kite1, kite2]
    colors = ["red", "blue"]
    labels = [kite1.kite_name, kite2.kite_name]

    # Collect all coordinates for equal axis scaling
    all_coords = []

    for i, (kite, color, label) in enumerate(zip(kite_list, colors, labels)):
        try:
            _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()

            # Add coordinates for axis scaling
            all_coords.extend([full_arc_LE, full_arc_TE, full_arc_qchord])

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
                linestyle="--",
                label=f"{label} - TE",
                linewidth=2,
                alpha=0.8,
            )

            # Plot quarter chord points
            ax.scatter3D(
                full_arc_qchord[:, 0],
                full_arc_qchord[:, 1],
                full_arc_qchord[:, 2],
                color=color,
                label=f"{label} - 1/4 Chord",
                alpha=0.6,
                s=30,
            )

            # Draw chord lines (lighter)
            for j in range(
                0, len(full_arc_LE), 2
            ):  # Every other chord to reduce clutter
                ax.plot3D(
                    [full_arc_LE[j, 0], full_arc_TE[j, 0]],
                    [full_arc_LE[j, 1], full_arc_TE[j, 1]],
                    [full_arc_LE[j, 2], full_arc_TE[j, 2]],
                    color=color,
                    linestyle=":",
                    alpha=0.2,
                    linewidth=1,
                )

        except Exception as e:
            print(f"❌ Error plotting {label}: {e}")
            continue

    # Set equal axis scales
    if all_coords:
        all_coords = np.vstack(all_coords)

        # Get the ranges for each axis
        x_range = [np.min(all_coords[:, 0]), np.max(all_coords[:, 0])]
        y_range = [np.min(all_coords[:, 1]), np.max(all_coords[:, 1])]
        z_range = [np.min(all_coords[:, 2]), np.max(all_coords[:, 2])]

        # Calculate the maximum range
        max_range = (
            max(
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            )
            / 2.0
        )

        # Set equal axis limits centered on the data
        x_center = (x_range[1] + x_range[0]) / 2.0
        y_center = (y_range[1] + y_range[0]) / 2.0
        z_center = (z_range[1] + z_range[0]) / 2.0

        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)

    # Formatting
    ax.set_xlabel("X-axis (Chord direction)")
    ax.set_ylabel("Y-axis (Span direction)")
    ax.set_zlabel("Z-axis (Height direction)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Kite Configuration Comparison", fontsize=16, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add text with kite properties
    props1 = f"AR: {kite1.old_aspect_ratio():.3f}, Area: {kite1.get_old_area():.1f} m²"
    props2 = f"AR: {kite2.old_aspect_ratio():.3f}, Area: {kite2.get_old_area():.1f} m²"

    ax.text2D(
        0.02,
        0.98,
        f"{labels[0]}: {props1}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color=colors[0],
        weight="bold",
    )
    ax.text2D(
        0.02,
        0.93,
        f"{labels[1]}: {props2}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color=colors[1],
        weight="bold",
    )

    plt.tight_layout()

    # Save plot if requested
    if save_plot and output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"💾 Plot saved to: {output_path}")

    plt.show()
    print(f"✅ Comparison plot complete!")


def main():
    """Main function to handle command line arguments and create the comparison plot."""
    parser = argparse.ArgumentParser(
        description="Compare two kite configurations from YAML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_config_kite_yamls.py kite1/config_kite.yaml kite2/config_kite.yaml
  python plot_config_kite_yamls.py ../processed_data/KITE1/config_kite.yaml ../processed_data/KITE2/config_kite.yaml --save comparison.png
        """,
    )

    parser.add_argument(
        "yaml_path1", type=str, help="Path to the first kite YAML configuration file"
    )

    parser.add_argument(
        "yaml_path2", type=str, help="Path to the second kite YAML configuration file"
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save the plot to the specified file path",
        default=None,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KITE CONFIGURATION COMPARISON")
    print("=" * 60)

    # Load the two kite configurations
    kite1 = load_kite_from_yaml(args.yaml_path1)
    kite2 = load_kite_from_yaml(args.yaml_path2)

    if kite1 is None or kite2 is None:
        print("❌ Failed to load one or both kite configurations")
        sys.exit(1)

    # Display kite properties
    print(f"\n📊 Kite Properties Comparison:")
    print(f"   {kite1.kite_name}:")
    print(f"     AR: {kite1.old_aspect_ratio():.3f}")
    print(f"     Area: {kite1.get_old_area():.3f} m²")
    print(f"     Span: {kite1.get_old_span()[0]:.3f} m")

    print(f"   {kite2.kite_name}:")
    print(f"     AR: {kite2.old_aspect_ratio():.3f}")
    print(f"     Area: {kite2.get_old_area():.3f} m²")
    print(f"     Span: {kite2.get_old_span()[0]:.3f} m")

    # Create the comparison plot
    save_plot = args.save is not None
    plot_two_kites_comparison(kite1, kite2, save_plot, args.save)

    print(f"\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
