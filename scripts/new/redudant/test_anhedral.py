#!/usr/bin/env python3
"""
Test script for panel angle modification with different percentage values.
"""

from scripts.new.kite_definition import KiteDefinition
from scripts.new.redudant.AnhedralScaling import AnhedralScaling
import numpy as np


def test_panel_angle_modification():
    """Test panel angle modification with various percentage changes."""

    print("=" * 60)
    print("PANEL ANGLE MODIFICATION TEST")
    print("=" * 60)

    # Load base kite
    base_kite = KiteDefinition("TUDELFT_V3_KITE")
    print(f"📋 Base kite loaded: {base_kite.kite_name}")

    # Test different panel angle changes
    test_percentages = [-20, -10, 0, 10, 20, 30]

    for percent_change in test_percentages:
        print(f"\n🔄 Testing {percent_change:+d}% panel angle change...")

        try:
            # Create anhedral-modified kite
            anh_kite = AnhedralScaling(base_kite, percent_change)

            # Generate the modified geometry
            full_arc_LE, full_arc_TE, orig_angles, mod_angles = (
                anh_kite.apply_panel_angle_modification()
            )

            # Calculate statistics
            avg_orig = np.degrees(np.mean(np.abs(orig_angles)))
            avg_mod = np.degrees(np.mean(np.abs(mod_angles)))
            angle_change = avg_mod - avg_orig

            print(f"   📐 Original avg angle: {avg_orig:.2f}°")
            print(f"   📐 Modified avg angle: {avg_mod:.2f}°")
            print(f"   📊 Actual change: {angle_change:+.2f}°")
            print(f"   ✅ Success!")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_panel_angle_modification()
