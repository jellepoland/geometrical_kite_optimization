#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scripts.kite_scaling import Kite_scaling
from geometrical_kite_optimization.main import Kite_shape
import math as m

# Create a test case
kite = Kite_shape(gamma_percentage=False, gamma=1.4, delta=0.7, phi=26)

# Test scaling to AR=10.0 (big change from ~5.1)
new_ar = 10.0
scaler = Kite_scaling(kite, new_ar=new_ar)

# Get area preservation factor
area_factor = scaler.get_area_preservation_scaling_factor()

# Debug the calculation
original_area = kite.get_old_area()
original_span = kite.get_old_span()[0]
original_ar = kite.old_aspect_ratio()

print("\n=== DEBUGGING AREA PRESERVATION CALCULATION ===")
print(f"Original area: {original_area:.3f} m²")
print(f"Original span: {original_span:.3f} m")
print(f"Original AR: {original_ar:.3f}")
print(f"Target AR: {new_ar:.3f}")

# Step 1: Calculate span scaling for AR only
span_scaling_for_ar_only = m.sqrt(new_ar / original_ar)
new_span_ar_only = original_span * span_scaling_for_ar_only
print(f"\nStep 1 - AR-only span scaling:")
print(f"  Span scaling factor: {span_scaling_for_ar_only:.4f}")
print(f"  New span (AR only): {new_span_ar_only:.3f} m")

# Step 2: Calculate what area would be without chord adjustment
new_area_without_chord_adjustment = (new_span_ar_only**2) / new_ar
print(f"\nStep 2 - Area without chord adjustment:")
print(
    f"  New area = span²/AR = {new_span_ar_only:.3f}²/{new_ar:.1f} = {new_area_without_chord_adjustment:.3f} m²"
)

# Step 3: Calculate chord scaling needed
area_scaling_factor = m.sqrt(original_area / new_area_without_chord_adjustment)
print(f"\nStep 3 - Chord scaling for area preservation:")
print(
    f"  Need to scale area from {new_area_without_chord_adjustment:.3f} to {original_area:.3f}"
)
print(
    f"  Area scaling factor: sqrt({original_area:.3f}/{new_area_without_chord_adjustment:.3f}) = {area_scaling_factor:.6f}"
)

print(f"\n=== RESULT ===")
print(f"Area preservation scaling factor: {area_factor:.6f}")
print(f"Expected value: < 1.0 (reducing chord to preserve area)")
print(f"Is working correctly: {area_factor < 1.0}")
