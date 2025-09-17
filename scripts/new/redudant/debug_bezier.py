#!/usr/bin/env python3
"""
Debug script to understand the Bezier curve coordinate system.
"""

from scripts.new.kite_definition import KiteDefinition
import numpy as np
import matplotlib.pyplot as plt


def debug_bezier_coordinates():
    """Debug the Bezier curve coordinate system and control points."""

    print("=" * 60)
    print("BEZIER COORDINATE SYSTEM DEBUG")
    print("=" * 60)

    # Load base kite
    base_kite = KiteDefinition("TUDELFT_V3_KITE")

    # Get original arc and Bezier data
    LE_coords, _ = base_kite.get_arc(halve=True)
    arc_y, arc_z = LE_coords[:, 1], LE_coords[:, 2]

    # Get original Bezier curve
    bezier_data = base_kite.get_bezier_curve()
    bez_x, bez_y = bezier_data[0], bezier_data[1]
    control_points = bezier_data[2]
    delta = bezier_data[6]
    gamma = bezier_data[7]
    phi = bezier_data[5]

    print(f"📋 Arc coordinates:")
    print(f"   Root (y, z): ({arc_y[0]:.3f}, {arc_z[0]:.3f})")
    print(f"   Tip (y, z): ({arc_y[-1]:.3f}, {arc_z[-1]:.3f})")
    print(f"   Span: {np.max(arc_y) - np.min(arc_y):.3f} m")

    print(f"\n🔧 Bezier parameters:")
    print(f"   Delta: {delta:.3f}")
    print(f"   Gamma: {gamma:.3f}")
    print(f"   Phi: {phi:.3f}")

    print(f"\n🎯 Control points:")
    for i, cp in enumerate(control_points):
        print(f"   Point {i}: ({cp[0]:.3f}, {cp[1]:.3f})")

    print(f"\n📏 Bezier curve:")
    print(f"   Min Y: {np.min(bez_x):.3f}, Max Y: {np.max(bez_x):.3f}")
    print(f"   Span: {np.max(bez_x) - np.min(bez_x):.3f} m")
    print(f"   Min Z: {np.min(bez_y):.3f}, Max Z: {np.max(bez_y):.3f}")

    # Create a visualization
    plt.figure(figsize=(12, 8))

    # Plot original arc
    plt.plot(arc_y, arc_z, "o-", label="Original Arc", linewidth=2, markersize=6)

    # Plot Bezier curve
    plt.plot(bez_x, bez_y, "-", label="Fitted Bezier", linewidth=2, alpha=0.7)

    # Plot control points
    ctrl_y = [cp[0] for cp in control_points]
    ctrl_z = [cp[1] for cp in control_points]
    plt.plot(ctrl_y, ctrl_z, "s--", label="Control Points", markersize=8, alpha=0.7)

    # Annotate control points
    for i, (y, z) in enumerate(zip(ctrl_y, ctrl_z)):
        plt.annotate(f"C{i}", (y, z), xytext=(5, 5), textcoords="offset points")

    plt.axis("equal")
    plt.xlabel("Y-axis (Span)")
    plt.ylabel("Z-axis (Height)")
    plt.title("Original Kite Arc vs Fitted Bezier Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test manual Bezier construction with different gamma
    print(f"\n🧪 Testing manual Bezier construction:")

    # Test with reduced gamma
    test_gamma = gamma * 0.5
    test_control_points = [
        [arc_y[0], arc_z[0]],  # Root point (fixed)
        [arc_y[0], arc_z[0] + test_gamma],  # Root control (modified gamma)
        [arc_y[-1] + delta, arc_z[-1]],  # Tip control
        [arc_y[-1], arc_z[-1]],  # Tip point
    ]

    print(f"   Original gamma: {gamma:.3f}")
    print(f"   Test gamma: {test_gamma:.3f}")
    print(f"   Test control points:")
    for i, cp in enumerate(test_control_points):
        print(f"     Point {i}: ({cp[0]:.3f}, {cp[1]:.3f})")


if __name__ == "__main__":
    debug_bezier_coordinates()
