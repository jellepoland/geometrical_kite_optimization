from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Project directory definition
PROJECT_DIR = Path(__file__).parent.parent.parent  # Points to the project root


def plot_kite(kite_lst):
    """Plot the kite geometry in 3d."""
    ax = plt.axes(projection="3d")
    colors = ["red", "blue", "green", "orange", "purple"]

    for i, kite in enumerate(kite_lst):
        _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()
        color = colors[i % len(colors)]

        ax.plot3D(
            full_arc_LE[:, 0],
            full_arc_LE[:, 1],
            full_arc_LE[:, 2],
            color=color,
            linestyle="-",
            label=f"LE Arc {i+1}",
            linewidth=2,
        )
        ax.plot3D(
            full_arc_TE[:, 0],
            full_arc_TE[:, 1],
            full_arc_TE[:, 2],
            color=color,
            linestyle="--",
            label=f"TE Arc {i+1}",
            linewidth=2,
        )
        ax.scatter3D(
            full_arc_qchord[:, 0],
            full_arc_qchord[:, 1],
            full_arc_qchord[:, 2],
            color=color,
            label=f"Quarter Chord {i+1}",
            alpha=0.7,
        )

        # Draw chord lines
        for j in range(len(full_arc_LE)):
            ax.plot3D(
                [full_arc_LE[j, 0], full_arc_TE[j, 0]],
                [full_arc_LE[j, 1], full_arc_TE[j, 1]],
                [full_arc_LE[j, 2], full_arc_TE[j, 2]],
                color=color,
                linestyle=":",
                alpha=0.3,
            )

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.grid()
    ax.set_title("Kite Geometry with Chord Vectors and Quarter Chord Points")
    ax.legend()

    # Ensure equal axis scaling
    # Collect all points from all kites
    all_points = []
    for kite in kite_lst:
        _, _, _, _, full_arc_LE, full_arc_TE, full_arc_qchord = kite.chord_vectors()
        all_points.append(full_arc_LE)
        all_points.append(full_arc_TE)
        all_points.append(full_arc_qchord)
    all_points = np.concatenate(all_points, axis=0)
    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]

    # Find the max range for all axes
    max_range = (
        max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
        / 2.0
    )

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
