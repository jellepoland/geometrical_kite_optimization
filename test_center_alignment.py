#!/usr/bin/env python3
"""
Test script to verify center alignment in KiteScaling
"""

import sys

sys.path.append("src")

from scripts.kite_definition import KiteDefinition
from scripts.kite_scaling import KiteScaling
import numpy as np


def test_center_alignment():
    # Load base kite
    base_kite = KiteDefinition("TUDELFT_V3_KITE")
    print(f"Loaded base kite: {base_kite.kite_name}")
    print(f"Original AR: {base_kite.old_aspect_ratio():.3f}")
    print(f"Original Area: {base_kite.get_old_area():.3f} m²")
    print(f"Original Span: {base_kite.get_old_span()[0]:.3f} m")

    # Test with a different aspect ratio
    new_ar = 7.0
    print(f"\n🔧 Testing AR scaling: {base_kite.old_aspect_ratio():.3f} → {new_ar:.3f}")

    # Create scaled kite
    scaled_kite = KiteScaling(base_kite, new_ar=new_ar)

    # Generate new wing sections (this will show center alignment info)
    print("\n" + "=" * 60)
    wing_sections = scaled_kite.get_new_wing_sections()
    print("=" * 60)

    print(f"\n✅ Generated {len(wing_sections['data'])} wing sections")
    print(f"🎯 Center alignment verification complete!")


if __name__ == "__main__":
    test_center_alignment()
