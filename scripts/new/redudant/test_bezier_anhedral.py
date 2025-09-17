#!/usr/bin/env python3
"""
Test script for Bezier-based anhedral modification.
"""

from kite_definition import KiteDefinition
from anhedral_scaling_bezier import AnhedralScaling
import numpy as np


def test_bezier_anhedral_modification():
    """Test Bezier-based anhedral modification with various gamma scale factors."""

    print("=" * 70)
    print("BEZIER-BASED ANHEDRAL MODIFICATION TEST")
    print("=" * 70)

    # Load base kite
    base_kite = KiteDefinition("TUDELFT_V3_KITE")
    print(f"📋 Base kite loaded: {base_kite.kite_name}")

    # Get original properties
    original_bezier = base_kite.get_bezier_curve()
    orig_delta = original_bezier[6]
    orig_gamma = original_bezier[7]
    orig_phi = original_bezier[5]

    print(f"\n🔧 Original Bezier parameters:")
    print(f"   Delta: {orig_delta:.4f}")
    print(f"   Gamma: {orig_gamma:.4f}")
    print(f"   Phi: {orig_phi:.4f}")

    # Test different gamma scale factors
    test_gamma_scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]

    for gamma_scale in test_gamma_scales:
        print(f"\n🔄 Testing gamma scale factor: {gamma_scale:.1f}")
        print(f"   (< 1.0 = flatter/less anhedral, > 1.0 = more curved/more anhedral)")

        try:
            # Create Bezier-modified kite
            bezier_kite = AnhedralScaling(base_kite, gamma_scale)

            # Generate the modified geometry (this will print span and scaling info)
            new_arc_y, new_arc_z, span_change = bezier_kite.generate_modified_arc()

            # Calculate some statistics
            orig_arc_height = np.max(original_bezier[1]) - np.min(original_bezier[1])
            new_arc_height = np.max(new_arc_z) - np.min(new_arc_z)
            height_ratio = new_arc_height / orig_arc_height

            print(
                f"   📊 Arc height change: {orig_arc_height:.3f} → {new_arc_height:.3f} m (ratio: {height_ratio:.3f})"
            )
            print(f"   ✅ Success!")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_bezier_anhedral_modification()
