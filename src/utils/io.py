import os
import pandas as pd
import json

from models.package import Package
from models.uld import ULD


def read_packages(file_path):
    """
    Reads package details from a CSV file.
    """
    df = pd.read_csv(os.path.join(file_path, "package.csv"))
    packages = []
    for _, row in df.iterrows():
        priority = 0
        if row["Type"] == "Priority":
            priority = 1
        packages.append(
            Package(
                id=row["Package Identifier"],
                length=row["Length (cm)"],
                width=row["Width (cm)"],
                height=row["Height (cm)"],
                weight=row["Weight (kg)"],
                priority=priority,
                delay_cost=int(
                    row["Cost of Delay"] if row["Cost of Delay"] != "-" else 0
                ),
            )
        )
    return packages


def read_ulds(file_path):
    """
    Reads ULD details from a CSV file.
    """
    df = pd.read_csv(os.path.join(file_path, "uld.csv"))
    ulds = []
    for _, row in df.iterrows():
        ulds.append(
            ULD(
                id=row["ULD Identifier"],
                length=row["Length (cm)"],
                width=row["Width (cm)"],
                height=row["Height (cm)"],
                weight_limit=row["Weight Limit (kg)"],
            )
        )
    return ulds


def write_output(allocation, total_cost, num_packed, num_priority_uld, file_path):
    """
    Write the output to a text file in format specified in problem statement.
    Additionally, creates a csv file with the allocation of the packages to the ULDs.
    """
    rows = []
    with open(os.path.join(file_path, "output.txt"), "w") as f:
        # Iterate over the allocation and build rows
        f.write(f"{total_cost},{num_packed},{num_priority_uld}\n")

        for row in allocation:
            rows.append(
                {
                    "Package Identifier": row[0],
                    "ULD Identifier": row[1] if row[1] is not None else "NONE",
                    "x1": row[2][0],
                    "y1": row[2][1],
                    "z1": row[2][2],
                    "x2": row[3][0],
                    "y2": row[3][1],
                    "z2": row[3][2],
                }
            )
            f.write(
                f"{row[0]},{row[1]},{row[2][0]},{row[2][1]},{row[2][2]},{row[3][0]},{row[3][1]},{row[3][2]}\n"
            )

    # Convert the list of rows into a DataFrame
    allocation_df = pd.DataFrame(rows)

    allocation_df.to_csv(os.path.join(file_path, "allocation.csv"), index=False)


def read_k(file_path):
    """
    Read k value from a text file.
    """
    with open(os.path.join(file_path, "k.txt"), "r") as f:
        try:
            k = int(f.read().strip())
        except ValueError:
            raise ValueError("k value is not an integer")
    return k


def read_allocation(file_path):
    """
    Read allocation data from a csv file.
    """
    df = pd.read_csv(os.path.join(file_path, "allocation.csv"))

    allocation_dict = {}
    for i, row in df.iterrows():
        allocation_dict[row["Package Identifier"]] = {}
        allocation_dict[row["Package Identifier"]]["Point 1"] = (
            row["x1"],
            row["y1"],
            row["z1"],
        )
        allocation_dict[row["Package Identifier"]]["Point 2"] = (
            row["x2"],
            row["y2"],
            row["z2"],
        )
        allocation_dict[row["Package Identifier"]]["ULD Identifier"] = row[
            "ULD Identifier"
        ]

    return allocation_dict


def load_config(config_path):
    """
    Load the configuration from a json file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
