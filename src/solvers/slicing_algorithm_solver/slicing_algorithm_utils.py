from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

# type hinting
from typing import List, Tuple


class PackageSelector:
    def __init__(self, epsilon=2):
        """
        Initialize the PackageSelector with a fixed epsilon value.

        Parameters:
            epsilon (int): The range tolerance for selecting packages.
        """
        self.epsilon = epsilon
        self.mp = {i: set() for i in range(1, 200)}

    def build_dimension_map(self, pdf):
        """
        Build the dimension-to-box mapping.

        Parameters:
            pdf (DataFrame): DataFrame containing 'length', 'width', 'height', and 'box no'.
        """
        for _, row in pdf.iterrows():
            self.mp[row["length"]].add(row["box no"])
            self.mp[row["width"]].add(row["box no"])
            self.mp[row["height"]].add(row["box no"])

    def find_max_index(self):
        """
        Find the dimension range with the most packages.

        Returns:
            tuple: (max_index, max_value)
        """
        max_index, max_value = None, -1
        for i in range(4, 200):
            curr_value = sum(
                len(self.mp.get(i - j, set())) for j in range(self.epsilon + 1)
            )
            if curr_value > max_value:
                max_value = curr_value
                max_index = i
        return max_index, max_value

    def select_and_adjust_packages(self, pdf, max_index):
        """
        Select and adjust packages to fit within the dimension range.

        Parameters:
            pdf (DataFrame): DataFrame containing package details.
            max_index (int): The selected maximum dimension index.

        Returns:
            list: Selected packages with their dimensions.
        """
        selected_packages = set()
        for j in range(self.epsilon + 1):
            selected_packages.update(self.mp.get(max_index - j, set()))

        select_package_list = []
        for box_no in selected_packages:
            package = pdf[pdf["box no"] == box_no].iloc[0]
            if (
                package["height"] < max_index - self.epsilon
                or package["height"] > max_index
            ):
                if (
                    package["length"] >= max_index - self.epsilon
                    and package["length"] <= max_index
                ):
                    pdf.loc[pdf["box no"] == box_no, ["length", "height"]] = (
                        package["height"],
                        package["length"],
                    )
                else:
                    pdf.loc[pdf["box no"] == box_no, ["width", "height"]] = (
                        package["height"],
                        package["width"],
                    )
            select_package_list.append((box_no, package["length"], package["width"]))
        return select_package_list

    def choose_packages(self, pdf):
        """
        Main method to select and adjust packages based on their dimensions.

        Parameters:
            pdf (DataFrame): DataFrame containing 'length', 'width', 'height', and 'box no'.

        Returns:
            tuple: (max_index, selected_packages)
        """
        self.build_dimension_map(pdf)
        max_index, _ = self.find_max_index()
        selected_packages = self.select_and_adjust_packages(pdf, max_index)
        return max_index, selected_packages


class NFDHPacker:
    def __init__(self, container_width: int, container_height: int):
        """
        Initialize the NFDH (Next Fit Decreasing Height) packer.

        Args:
            container_width: The width of the container.
            container_height: The height of the container.
        """
        self.container_width = container_width
        self.container_height = container_height

    def pack_bins(
        self, uld_bins: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Pack bins into a container using the NFDH algorithm.

        The algorithm places the bins row by row, ensuring that each row is filled
        before moving to the next one. Within a row, bins are packed from left to right.

        Args:
            uld_bins: A list of tuples where each tuple contains:
                - ULD identifier (str): Unique identifier for the bin.
                - bin width (int): Width of the bin.
                - bin height (int): Height of the bin.

        Returns:
            A list of tuples where each tuple contains:
                - ULD identifier (str)
                - x-coordinate of the bin's top-left corner (int)
                - y-coordinate of the bin's top-left corner (int)
                - bin width (int)
                - bin height (int)

        Note:
            If any bin cannot fit into the container, the method stops packing and
            returns the bins packed up to that point.
        """
        # Sort bins by height in descending order (taller bins are placed first)
        uld_bins = sorted(uld_bins, key=lambda x: x[2], reverse=True)

        packed_bins = []  # List to store the packed bins and their positions
        current_y = 0  # Vertical position of the current row
        current_x = 0  # Horizontal position within the current row
        current_row_height = 0  # Maximum height of the current row

        for bin_id, bin_width, bin_height in uld_bins:
            # Skip bins that are larger than the container dimensions
            if bin_width > self.container_width or bin_height > self.container_height:
                # If a bin cannot fit, return the packed bins so far
                return packed_bins

            # Check if the bin fits horizontally in the current row
            if current_x + bin_width > self.container_width:
                # Move to the next row since there is no horizontal space left
                current_y += (
                    current_row_height  # Advance vertical position by row height
                )
                if current_y + bin_height > self.container_height:
                    # If there is no vertical space for the next row, return packed bins
                    return packed_bins

                # Reset horizontal position and row height for the new row
                current_x = 0
                current_row_height = 0

            # Place the bin in the current row
            packed_bins.append((bin_id, current_x, current_y, bin_width, bin_height))

            # Update the current row's horizontal position and maximum height
            current_x += bin_width  # Advance horizontal position
            current_row_height = max(
                current_row_height, bin_height
            )  # Update row height

        return packed_bins


class FFDHPacker:
    def __init__(self, container_width: int, container_height: int):
        """
        Initialize the FFDH (First Fit Decreasing Height) packer.

        Args:
            container_width: The width of the container.
            container_height: The height of the container.
        """
        self.container_width = container_width
        self.container_height = container_height

    def pack_bins(
        self, uld_bins: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Pack bins into a container using the FFDH algorithm.

        The algorithm places bins into the first row that can accommodate them,
        row by row. Rows are created as needed.

        Args:
            uld_bins: A list of tuples where each tuple contains:
                - ULD identifier (str): Unique identifier for the bin.
                - bin width (int): Width of the bin.
                - bin height (int): Height of the bin.

        Returns:
            A list of tuples where each tuple contains:
                - ULD identifier (str)
                - x-coordinate of the bin's top-left corner (int)
                - y-coordinate of the bin's top-left corner (int)
                - bin width (int)
                - bin height (int)

        Note:
            If any bin cannot fit into the container, the method stops packing and
            returns the bins packed up to that point.
        """
        # Sort bins by height in descending order (taller bins are placed first)
        uld_bins = sorted(uld_bins, key=lambda x: x[2], reverse=True)

        packed_bins = []  # List to store the packed bins and their positions
        rows = (
            []
        )  # List of rows; each row is a tuple (current_y_position, current_width_of_row)

        for bin_id, bin_width, bin_height in uld_bins:
            # Skip bins that are larger than the container dimensions
            if bin_width > self.container_width or bin_height > self.container_height:
                # If a bin cannot fit, return the packed bins so far
                return packed_bins

            placed = False  # Flag to track if the bin has been placed

            # Try to place the bin in an existing row
            for i, (row_height, row_width) in enumerate(rows):
                # Check if the bin fits horizontally in the current row
                if row_width + bin_width <= self.container_width:
                    # Place the bin in this row
                    x_pos = row_width
                    y_pos = row_height
                    packed_bins.append((bin_id, x_pos, y_pos, bin_width, bin_height))

                    # Update the row's width and ensure the row height is sufficient
                    rows[i] = (row_height, row_width + bin_width)
                    placed = True
                    break

            # If the bin couldn't be placed in any existing row, create a new row
            if not placed:
                # Check if there's enough vertical space for a new row
                total_row_height = sum(r[0] for r in rows)
                if total_row_height + bin_height > self.container_height:
                    return packed_bins  # No space left for additional rows

                # Add the bin to a new row
                x_pos = 0  # New row starts at the leftmost position
                y_pos = total_row_height
                packed_bins.append((bin_id, x_pos, y_pos, bin_width, bin_height))

                # Add the new row to the list of rows
                rows.append((y_pos + bin_height, bin_width))

        return packed_bins


class MRA_Packer:
    def __init__(self, container_width: int, container_height: int):
        """
        Initialize the Maximal Rectangle Algorithm (MRA) packer.

        Args:
            container_width: The width of the container.
            container_height: The height of the container.
        """
        self.container_width = container_width
        self.container_height = container_height

    def pack_bins(
        self, uld_bins: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Pack bins into a container using the Maximal Rectangle Algorithm (MRA).

        The algorithm uses a 2D grid to represent the container and places bins in the first
        available maximal rectangle that can fit the bin, row by row.

        Args:
            uld_bins: A list of tuples where each tuple contains:
                - ULD identifier (str): Unique identifier for the bin.
                - bin width (int): Width of the bin.
                - bin height (int): Height of the bin.

        Returns:
            A list of tuples where each tuple contains:
                - ULD identifier (str)
                - x-coordinate of the bin's top-left corner (int)
                - y-coordinate of the bin's top-left corner (int)
                - bin width (int)
                - bin height (int)

        Note:
            If a bin cannot fit into the container, it is skipped.
        """
        # Sort bins by height in descending order to prioritize taller bins
        uld_bins = sorted(uld_bins, key=lambda x: x[2], reverse=True)

        packed_bins = []  # List to store packed bin positions
        # Initialize a 2D grid to represent the container; 0 means unoccupied, 1 means occupied
        grid = [[0] * self.container_width for _ in range(self.container_height)]

        for bin_id, bin_width, bin_height in uld_bins:
            # Skip bins that exceed the container dimensions
            if bin_width > self.container_width or bin_height > self.container_height:
                continue  # Skip to the next bin

            placed = False  # Flag to check if the bin has been placed

            # Iterate through the grid to find a suitable position for the bin
            for y in range(self.container_height - bin_height + 1):
                for x in range(self.container_width - bin_width + 1):
                    # Check if the bin can be placed at (x, y)
                    can_place = True
                    for dy in range(bin_height):
                        for dx in range(bin_width):
                            if grid[y + dy][x + dx] != 0:  # Cell is already occupied
                                can_place = False
                                break
                        if not can_place:
                            break

                    if can_place:
                        # Place the bin at position (x, y)
                        for dy in range(bin_height):
                            for dx in range(bin_width):
                                grid[y + dy][x + dx] = 1  # Mark cells as occupied

                        # Record the bin's position and dimensions
                        packed_bins.append((bin_id, x, y, bin_width, bin_height))
                        placed = True
                        break

                if placed:
                    break

        return packed_bins


def generate_packing_figure(bin_size, boxes, file_path):
    """
    Generate a visualization of rectangle packing within a bin.

    Args:
        bin_size (tuple): A tuple (width, height) representing the size of the bin.
        boxes (list): A list of tuples, where each tuple represents a box in the format
                      (x, y, width, height).
        file_path (str): The path to save the generated figure. If directories in the path
                         do not exist, they will be created.

    Raises:
        ValueError: If the provided file path is invalid or cannot be used.
    """
    # Validate and create directories if necessary
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise ValueError(f"Failed to create directories for the file path: {e}")

    # Generate the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw the bin boundary
    bin_width, bin_height = bin_size
    ax.plot(
        [0, bin_width, bin_width, 0, 0],  # X coordinates of bin boundary
        [0, 0, bin_height, bin_height, 0],  # Y coordinates of bin boundary
        "k-",
        linewidth=2,
        label="Bin",  # Black border line with a label
    )

    # Assign random colors to boxes for visual distinction
    colors = [(random.random(), random.random(), random.random()) for _ in boxes]

    # Draw each box within the bin
    for i, (x, y, width, height) in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle(
                (x, y), width, height, facecolor=colors[i], edgecolor="black", alpha=0.8
            )
        )

    # Configure plot properties
    ax.set_xlim(0, bin_width + 5)  # Extra space for better visualization
    ax.set_ylim(0, bin_height + 5)
    ax.set_aspect("equal")  # Maintain aspect ratio
    ax.set_title("Rectangle Packing Visualization")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend(loc="upper right")  # Legend for the bin boundary
    plt.grid(visible=True)

    # Save the figure to the specified file path
    try:
        plt.savefig(file_path)
    except Exception as e:
        raise ValueError(f"Failed to save the figure: {e}")

    plt.close(fig)  # Close the plot to free memory


class PackagePackerNFDH:
    def __init__(self, udf: pd.DataFrame, pdf: pd.DataFrame):
        """
        Initializes the package packing class using the Next Fit Decreasing Height (NFDH) algorithm.

        Args:
            udf (pd.DataFrame): ULD data frame containing container dimensions and height.
                                Columns expected: ['length', 'width', 'height'].
            pdf (pd.DataFrame): Package data frame containing box dimensions and identifiers.
                                Columns expected: ['box no', 'length', 'width', 'height'].
        """
        self.udf = udf
        self.pdf = pdf
        self.total_packed_bins = []
        self.total_boxes_packed = 0

    def choose_packages(self, pdf: pd.DataFrame):
        """
        Select packages for packing based on height compatibility.

        Args:
            pdf (pd.DataFrame): Package data frame to select from.

        Returns:
            select_height (int): The total height of the selected packages.
            select_package_list (List[Tuple[str, int, int]]): Selected packages with dimensions.
        """
        select_height = pdf["height"].min()
        selected_packages = pdf[pdf["height"] == select_height]
        select_package_list = [
            (row["box no"], row["length"], row["width"])
            for _, row in selected_packages.iterrows()
        ]
        return select_height, select_package_list

    def generate_diagram(
        self, index: int, current_layer: int, row: Dict, packed_bins: List
    ):
        """
        Generate a visual diagram of the packing arrangement.

        Args:
            index (int): Index of the current ULD.
            current_layer (int): Current layer number in the ULD.
            row (Dict): Current row of ULD data being processed.
            packed_bins (List): List of packed bin details.
        """
        uld_size = (row["length"], row["width"])
        bin_coordinates = {bin_id: (x, y, w, h) for bin_id, x, y, w, h in packed_bins}
        file_path = f"../fig/nfdh_diagram_{index}_{current_layer}.pdf"

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Generate the diagram
        generate_packing_figure(uld_size, list(bin_coordinates.values()), file_path)

    def pack(self) -> List[Tuple[str, int, int, int, int, int, int]]:
        """
        Pack the packages into ULD containers using NFDH.

        Returns:
            List of tuples, where each tuple contains:
                (box no, x, y, z, x+width, y+height, z+height).
        """
        for index, row in self.udf.iterrows():
            curr_base = 0  # Tracks the current z-coordinate base for stacking
            current_layer = 0  # Tracks the number of layers used for the current ULD

            while row["height"] >= self.pdf["height"].min():
                current_layer += 1
                # Initialize the packer for the current ULD dimensions
                packer2d = NFDHPacker(row["length"], row["width"])

                # Choose packages to fit in the current height layer
                select_height, select_package_list = self.choose_packages(self.pdf)
                if select_height > row["height"]:
                    break  # No more packages can fit in this ULD

                selected_box_nos = [box[0] for box in select_package_list]
                # Filter the DataFrame where 'box no' is in the selected_box_nos
                filtered_select_df = self.pdf[self.pdf["box no"].isin(selected_box_nos)]

                while row["weight_limit"] < filtered_select_df["weight"].sum():
                    max_weight_index = filtered_select_df["weight"].idxmax()
                    # Drop the row with the maximum 'Weight' using its index
                    filtered_select_df = filtered_select_df.drop(index=max_weight_index)
                if filtered_select_df.empty:
                    break
                select_package_list = list(
                    filtered_select_df[["box no", "length", "width"]].itertuples(
                        index=False, name=None
                    )
                )

                # Pack the selected packages
                packed_bins = packer2d.pack_bins(select_package_list)

                # Generate the diagram for the current packing layer
                self.generate_diagram(index, current_layer, row, packed_bins)

                # Add packed bins to the total packed bins list
                for bin in packed_bins:
                    box_row = self.pdf[self.pdf["box no"] == bin[0]]
                    if not box_row.empty:
                        height = box_row.iloc[0]["height"]
                    self.total_packed_bins.append(
                        (
                            bin[0],  # Box identifier
                            bin[1],  # x-coordinate
                            bin[2],  # y-coordinate
                            curr_base,  # z-coordinate base
                            bin[1] + bin[3],  # x+width
                            bin[2] + bin[4],  # y+height
                            curr_base + height,  # z+height
                        )
                    )

                # Remove packed boxes from the package data frame
                packed_box_nos = [box_no for box_no, _, _, _, _ in packed_bins]
                self.pdf = self.pdf[~self.pdf["box no"].isin(packed_box_nos)]

                # Update the height of the ULD and the base for stacking
                row["height"] -= select_height
                row["weight_limit"] -= filtered_select_df["weight"].sum()
                curr_base += select_height

                # Increment the total packed box count
                self.total_boxes_packed += len(packed_box_nos)

        return self.total_packed_bins


class PackagePackerFFDH:
    def __init__(self, udf: pd.DataFrame, pdf: pd.DataFrame):
        """
        Initializes the package packing class using the First Fit Decreasing Height (FFDH) algorithm.

        Args:
            udf (pd.DataFrame): ULD data frame containing container dimensions and height.
                                Columns expected: ['length', 'width', 'height'].
            pdf (pd.DataFrame): Package data frame containing box dimensions and identifiers.
                                Columns expected: ['box no', 'length', 'width', 'height'].
        """
        self.udf = udf
        self.pdf = pdf
        self.total_packed_bins = []
        self.total_boxes_packed = 0

    def choose_packages(self, pdf: pd.DataFrame):
        """
        Select packages for packing based on height compatibility.

        Args:
            pdf (pd.DataFrame): Package data frame to select from.

        Returns:
            select_height (int): The total height of the selected packages.
            select_package_list (List[Tuple[str, int, int]]): Selected packages with dimensions.
        """
        select_height = pdf["height"].min()
        selected_packages = pdf[pdf["height"] == select_height]
        select_package_list = [
            (row["box no"], row["length"], row["width"])
            for _, row in selected_packages.iterrows()
        ]
        return select_height, select_package_list

    def generate_diagram(
        self, index: int, current_layer: int, row: Dict, packed_bins: List
    ):
        """
        Generate a visual diagram of the packing arrangement.

        Args:
            index (int): Index of the current ULD.
            current_layer (int): Current layer number in the ULD.
            row (Dict): Current row of ULD data being processed.
            packed_bins (List): List of packed bin details.
        """
        uld_size = (row["length"], row["width"])
        bin_coordinates = {bin_id: (x, y, w, h) for bin_id, x, y, w, h in packed_bins}
        file_path = f"../fig/ffdh_diagram_{index}_{current_layer}.pdf"

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Generate the diagram
        generate_packing_figure(uld_size, list(bin_coordinates.values()), file_path)

    def pack(self) -> List[Tuple[str, int, int, int, int, int, int]]:
        """
        Pack the packages into ULD containers using FFDH.

        Returns:
            List of tuples, where each tuple contains:
                (box no, x, y, z, x+width, y+height, z+height).
        """
        for index, row in self.udf.iterrows():
            curr_base = 0  # Tracks the current z-coordinate base for stacking
            current_layer = 0  # Tracks the number of layers used for the current ULD

            while row["height"] >= self.pdf["height"].min():
                current_layer += 1
                # Initialize the packer for the current ULD dimensions
                packer2d = FFDHPacker(row["length"], row["width"])

                # Choose packages to fit in the current height layer
                select_height, select_package_list = self.choose_packages(self.pdf)
                if select_height > row["height"]:
                    break  # No more packages can fit in this ULD

                selected_box_nos = [box[0] for box in select_package_list]
                # Filter the DataFrame where 'box no' is in the selected_box_nos
                filtered_select_df = self.pdf[self.pdf["box no"].isin(selected_box_nos)]

                while row["weight_limit"] < filtered_select_df["weight"].sum():
                    max_weight_index = filtered_select_df["weight"].idxmax()
                    # Drop the row with the maximum 'Weight' using its index
                    filtered_select_df = filtered_select_df.drop(index=max_weight_index)
                if filtered_select_df.empty:
                    break
                select_package_list = list(
                    filtered_select_df[["box no", "length", "width"]].itertuples(
                        index=False, name=None
                    )
                )
                # Pack the selected packages
                packed_bins = packer2d.pack_bins(select_package_list)

                # Generate the diagram for the current packing layer
                self.generate_diagram(index, current_layer, row, packed_bins)

                # Add packed bins to the total packed bins list
                for bin in packed_bins:
                    box_row = self.pdf[self.pdf["box no"] == bin[0]]
                    if not box_row.empty:
                        height = box_row.iloc[0]["height"]
                    self.total_packed_bins.append(
                        (
                            bin[0],  # Box identifier
                            bin[1],  # x-coordinate
                            bin[2],  # y-coordinate
                            curr_base,  # z-coordinate base
                            bin[1] + bin[3],  # x+width
                            bin[2] + bin[4],  # y+height
                            curr_base + height,  # z+height
                        )
                    )

                # Remove packed boxes from the package data frame
                packed_box_nos = [box_no for box_no, _, _, _, _ in packed_bins]
                self.pdf = self.pdf[~self.pdf["box no"].isin(packed_box_nos)]

                # Update the height of the ULD and the base for stacking
                row["height"] -= select_height
                row["weight_limit"] -= filtered_select_df["weight"].sum()
                curr_base += select_height

                # Increment the total packed box count
                self.total_boxes_packed += len(packed_box_nos)

        return self.total_packed_bins


class PackagePackerMRA:
    def __init__(self, udf: pd.DataFrame, pdf: pd.DataFrame):
        """
        Initializes the package packing class using the Maximal Rectangle Algorithm (MRA).

        Args:
            udf (pd.DataFrame): ULD data frame containing container dimensions and height.
                                Columns expected: ['length', 'width', 'height'].
            pdf (pd.DataFrame): Package data frame containing box dimensions and identifiers.
                                Columns expected: ['box no', 'length', 'width', 'height'].
        """
        self.udf = udf
        self.pdf = pdf
        self.total_packed_bins = []
        self.total_boxes_packed = 0

    def choose_packages(self, pdf: pd.DataFrame):
        """
        Select packages for packing based on height compatibility.

        Args:
            pdf (pd.DataFrame): Package data frame to select from.

        Returns:
            select_height (int): The total height of the selected packages.
            select_package_list (List[Tuple[str, int, int]]): Selected packages with dimensions.
        """
        select_height = pdf["height"].min()
        selected_packages = pdf[pdf["height"] == select_height]
        select_package_list = [
            (row["box no"], row["length"], row["width"])
            for _, row in selected_packages.iterrows()
        ]
        return select_height, select_package_list

    def generate_diagram(self, index: int, current: int, row: Dict, packed_bins: List):
        """
        Generate a visual diagram of the packing arrangement.

        Args:
            index (int): Index of the current ULD.
            current (int): Current iteration for the ULD.
            row (Dict): Current row of ULD data being processed.
            packed_bins (List): List of packed bin details.
        """
        uld_size = (row["length"], row["width"])
        bin_coordinates = {bin_id: (x, y, w, h) for bin_id, x, y, w, h in packed_bins}
        file_path = f"../fig/mra_diagram_{index}_{current}.pdf"

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Generate the diagram
        generate_packing_figure(uld_size, list(bin_coordinates.values()), file_path)

    def pack(self) -> List[Tuple[str, int, int, int, int, int, int]]:
        """
        Pack the packages into ULD containers using MRA.

        Returns:
            List of tuples, where each tuple contains:
                (box no, x, y, z, x+width, y+height, z+height).
        """
        for index, row in self.udf.iterrows():
            current = 0  # Tracks the number of layers used for the current ULD
            curr_base = 0  # Tracks the current z-coordinate base for stacking

            while row["height"] >= self.pdf["height"].min():
                current += 1
                # Initialize the packer for the current ULD dimensions
                packer2d = MRA_Packer(row["length"], row["width"])

                # Choose packages to fit in the current height layer
                select_height, select_package_list = self.choose_packages(self.pdf)
                if select_height > row["height"]:
                    break  # No more packages can fit in this ULD

                selected_box_nos = [box[0] for box in select_package_list]
                # Filter the DataFrame where 'box no' is in the selected_box_nos
                filtered_select_df = self.pdf[self.pdf["box no"].isin(selected_box_nos)]

                while row["weight_limit"] < filtered_select_df["weight"].sum():
                    max_weight_index = filtered_select_df["weight"].idxmax()
                    # Drop the row with the maximum 'Weight' using its index
                    filtered_select_df = filtered_select_df.drop(index=max_weight_index)
                if filtered_select_df.empty:
                    break
                select_package_list = list(
                    filtered_select_df[["box no", "length", "width"]].itertuples(
                        index=False, name=None
                    )
                )

                # Pack the selected packages
                packed_bins = packer2d.pack_bins(select_package_list)

                # Generate the diagram for the current packing layer
                self.generate_diagram(index, current, row, packed_bins)

                # Add packed bins to the total packed bins list
                for bin in packed_bins:
                    box_row = self.pdf[self.pdf["box no"] == bin[0]]
                    if not box_row.empty:
                        height = box_row.iloc[0]["height"]
                    self.total_packed_bins.append(
                        (
                            bin[0],  # Box identifier
                            bin[1],  # x-coordinate
                            bin[2],  # y-coordinate
                            curr_base,  # z-coordinate base
                            bin[1] + bin[3],  # x+width
                            bin[2] + bin[4],  # y+height
                            curr_base + height,  # z+height
                        )
                    )

                # Remove packed boxes from the package data frame
                packed_box_nos = [box_no for box_no, _, _, _, _ in packed_bins]
                self.pdf = self.pdf[~self.pdf["box no"].isin(packed_box_nos)]

                # Update the height of the ULD and the base for stacking
                row["height"] -= select_height
                row["weight_limit"] -= filtered_select_df["weight"].sum()
                curr_base += select_height

                # Increment the total packed box count
                self.total_boxes_packed += len(packed_box_nos)

        return self.total_packed_bins


def pack_package(
    udf: pd.DataFrame, pdf: pd.DataFrame
) -> List[Tuple[str, int, int, int, int, int, int]]:
    """
    Pack the packages using multiple strategies and select the best of them.

    Args:
        udf (pd.DataFrame): ULD data frame containing container dimensions and height.
        pdf (pd.DataFrame): Package data frame containing box dimensions and identifiers.

    Returns:
        List of tuples, where each tuple contains:
            (box no, x, y, z, x+width, y+height, z+height).
    """
    # Initialize packers for each algorithm
    packer_nfdh = PackagePackerNFDH(udf, pdf)
    packer_ffdh = PackagePackerFFDH(udf, pdf)
    packer_mra = PackagePackerMRA(udf, pdf)

    # Pack using each algorithm
    packed_bins_nfdh = packer_nfdh.pack()
    packed_bins_ffdh = packer_ffdh.pack()
    packed_bins_mra = packer_mra.pack()

    # Compare the results and choose the best one
    packed_bins = max(
        [packed_bins_nfdh, packed_bins_ffdh, packed_bins_mra],
        key=lambda bins: len(bins),
    )
    return packed_bins
