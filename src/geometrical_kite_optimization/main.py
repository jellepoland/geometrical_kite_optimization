#!/usr/bin/env python3
"""
Main script for kite parametrization studies.

This script demonstrates:
1. Loading a base kite geometry from YAML
2. AR scaling while preserving surface area and arc shape
3. Anhedral angle scaling while preserving AR and surface area
"""

import numpy as np

# import sys
from pathlib import Path

# Add the scripts directory to Python path for relative imports
# sys.path.insert(0, str(Path(__file__).parent))

from geometrical_kite_optimization.kite_definition import KiteDefinition
from geometrical_kite_optimization.kite_scaling import KiteScaling
from geometrical_kite_optimization.utils import plot_kite, PROJECT_DIR
from geometrical_kite_optimization import kite_scaling_functionalized


def main():
    """Main function for kite parametrization study."""

    # Configuration
    kite_name = "TUDELFT_V3_KITE"
    new_ar = 10
    new_phi = 20  # degrees

    # Load base kite
    yaml_path = Path(
        PROJECT_DIR / "processed_data" / f"{kite_name}" / "config_kite.yaml"
    )
    base_kite = KiteDefinition(yaml_path)
    base_kite.process(plot=False)

    # Get original properties
    base_span = base_kite.get_old_span()[0]
    base_area = base_kite.get_old_area()
    base_ar = base_kite.old_aspect_ratio()
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

    # # AR scaling study
    # new_ar_kite = KiteScaling(
    #     base_kite,
    #     new_ar=new_ar,
    # )
    # new_ar_kite.export_data()

    # # Phi scaling
    # new_phi_kite = KiteScaling(
    #     base_kite,
    #     new_phi=new_phi,
    # )
    # new_phi_kite.export_data()

    # Functionalized approach
    yaml_path_new_ar, new_ar_kite = kite_scaling_functionalized.main(
        base_kite, new_ar=new_ar
    )
    yaml_path_new_phi, new_phi_kite = kite_scaling_functionalized.main(
        base_kite, new_phi=np.deg2rad(new_phi)
    )

    ### printing
    print(
        f"\nbase    (ar:{base_ar:.3f}, phi:{np.rad2deg(base_phi):.3f}°) -- Area: {base_area:.3f}m²"
    )
    print(
        f"new_ar  (ar:{new_ar:.3f}, phi:{np.rad2deg(base_phi):.3f}°) -- Area: {new_ar_kite.area:.3f}m²"
    )
    print(
        f"new_phi (ar:{base_ar:.3f}, phi:{new_phi:.3f}°) -- Area: {new_phi_kite.area:.3f}m²"
    )

    # Visualization
    plot_kite([base_kite, new_ar_kite, new_phi_kite])
    # plot_kite([base_kite, new_phi_kite])


if __name__ == "__main__":
    main()
