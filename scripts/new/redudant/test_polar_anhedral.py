#!/usr/bin/env python3
"""
Test script for polar angle-based anhedral modification.
"""

from scripts.new.kite_definition import KiteDefinition
from scripts.new.anhedral_scaling_bezier import AnhedralScaling
import numpy as np


def test_polar_angle_modification():
    """Test polar angle-based anhedral modification with various angle adjustments."""

    print("=" * 70)
    print("POLAR ANGLE-BASED ANHEDRAL MODIFICATION TEST")
    print("=" * 70)

    # Load base kite
    base_kite = KiteDefinition("TUDELFT_V3_KITE")
    print(f"📋 Base kite loaded: {base_kite.kite_name}")

    # Test different angle adjustments
    test_angles = [-15, -10, -5, 0, 5, 10, 15]  # degrees

    for angle_adj in test_angles:
        print(f"\n🔄 Testing angle adjustment: {angle_adj:+d}°")
        print(f"   (+ = roll up, - = roll down)")

        try:
            # Create polar angle-modified kite
            polar_kite = AnhedralScaling(base_kite, angle_adj)

            # Generate the modified geometry
            new_arc_y, new_arc_z, span_change = polar_kite.generate_modified_arc()

            # Calculate some statistics
            span_ratio = span_change
            tip_height_change = new_arc_z[0] - 0.921  # Outboard point height change

            print(f"   📊 Span change factor: {span_ratio:.3f}")
            print(f"   📏 Tip height change: {tip_height_change:+.3f} m")
            print(f"   ✅ Success!")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_polar_angle_modification()
