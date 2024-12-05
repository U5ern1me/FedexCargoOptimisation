import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from .io import read_allocation, read_packages, read_ulds


# typing
from typing import List, Tuple


class plot_3D:
    """
    Class to plot the ULD and packages in matplotlib.
    """

    def __init__(
        self, V: Tuple[int, int, int], alpha: float = 0.5, style: str = "default"
    ):
        """
        Args:
            V: Dimensions of the ULD.
            alpha: Alpha value for the boxes.
            style: Style of the plot.
        """

        if style:
            plt.style.use(style)

        # Create figure and axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.alpha = alpha
        self.V = V

        # Set axis labels
        self.ax.set_xlabel("Length")
        self.ax.set_ylabel("Width")
        self.ax.set_zlabel("Height")

        # Set axis limits to container dimensions
        self.ax.set_xlim(0, V[0])
        self.ax.set_ylim(0, V[1])
        self.ax.set_zlim(0, V[2])

        max_range = np.array([V[0], V[1], V[2]]).max()
        self.ax.set_box_aspect((V[0] / max_range, V[1] / max_range, V[2] / max_range))

        # Add these lines to adjust the view angle
        self.ax.view_init(elev=20, azim=-45)

    def add_box(
        self,
        min_corner: Tuple[int, int, int],
        max_corner: Tuple[int, int, int],
        mode: str = "EMS",
        label: str = "box",
    ):
        """
        Add a box to the plot using diagonal corners.
        """
        x = [min_corner[0], max_corner[0]]
        y = [min_corner[1], max_corner[1]]
        z = [min_corner[2], max_corner[2]]
        self.ax.bar3d(
            x[0], y[0], z[0], x[1] - x[0], y[1] - y[0], z[1] - z[0], alpha=self.alpha
        )

        if label:
            self.ax.text(
                (x[0] + x[1]) / 2,
                (y[0] + y[1]) / 2,
                (z[0] + z[1]) / 2,
                label,
                fontsize=12,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor="black", edgecolor="none", alpha=1),
            )

    def add_title(self, title: str):
        """
        Add a title to the plot.
        """
        plt.title(title, fontsize=16, pad=10, fontweight="bold")

    def show(self):
        """
        Show the plot.
        """
        plt.show()


def draw(
    bin_dimensions: Tuple[int, int, int],
    boxes: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
    title: str,
):
    """
    Draw the ULD and packages in matplotlib.

    Args:
        bin_dimensions: Dimensions of the ULD.
        boxes: List of packages in the ULD.
        title: Title of the plot.
    """
    container = plot_3D(V=bin_dimensions)
    for box in boxes:
        container.add_box(box[0], box[1], mode="EMS", label=box[2])
    container.add_title(title)
    container.show()


def visualize(output_path: str, input_folder: str):
    """
    Visualize the solution in matplotlib 3D.

    Args:
        output_path: Path to the output directory.
        input_folder: Path to the input directory.
    """
    packages = read_packages(input_folder)
    ulds = read_ulds(input_folder)
    allocation = read_allocation(output_path)

    for uld in ulds:
        uld_packages = [
            (
                allocation[package.id]["Point 1"],
                allocation[package.id]["Point 2"],
                package.id,
            )
            for package in packages
            if allocation[package.id]["ULD Identifier"] == uld.id
        ]
        print(f"Visualizing ULD {uld.id}...")
        draw((uld.length, uld.width, uld.height), uld_packages, uld.id)
