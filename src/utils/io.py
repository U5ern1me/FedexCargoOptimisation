import os
import pandas as pd
import json
import pickle

from models.package import Package
from models.uld import ULD

# type hinting
from typing import Dict, Any, Optional, List, Tuple

COST_MAX = 1000000


def read_packages(file_path: str) -> List[Package]:
    """
    Reads package details from a CSV file.

    Args:
        file_path: Path to the folder containing the input data

    Returns:
        List of packages
    """
    df = pd.read_csv(os.path.join(file_path, "package.csv"))
    packages = []

    if df["Package Identifier"].duplicated().any():
        duplicate_ids = df[df["Package Identifier"].duplicated()][
            "Package Identifier"
        ].tolist()
        raise ValueError(
            f"Duplicate package identifiers found: {', '.join(duplicate_ids)}"
        )

    for _, row in df.iterrows():
        priority = 0
        if row["Type (P/E)"] == "Priority":
            priority = 1

        if row["Length (cm)"] == 0 or row["Width (cm)"] == 0 or row["Height (cm)"] == 0:
            raise ValueError(
                f"Package {row['Package Identifier']} has invalid dimensions"
            )

        if row["Weight (kg)"] == 0:
            raise ValueError(f"Package {row['Package Identifier']} has invalid weight")

        packages.append(
            Package(
                id=row["Package Identifier"],
                length=row["Length (cm)"],
                width=row["Width (cm)"],
                height=row["Height (cm)"],
                weight=row["Weight (kg)"],
                priority=priority,
                delay_cost=int(
                    row["Cost of Delay"] if row["Cost of Delay"] != "-" else COST_MAX
                ),
            )
        )
    return packages


def read_ulds(file_path: str) -> List[ULD]:
    """
    Reads ULD details from a CSV file.

    Args:
        file_path: Path to the folder containing the input data

    Returns:
        List of ULDs
    """
    df = pd.read_csv(os.path.join(file_path, "uld.csv"))
    ulds = []

    if df["ULD Identifier"].duplicated().any():
        duplicate_ids = df[df["ULD Identifier"].duplicated()]["ULD Identifier"].tolist()
        raise ValueError(f"Duplicate ULD identifiers found: {', '.join(duplicate_ids)}")

    for _, row in df.iterrows():
        if row["Length (cm)"] == 0 or row["Width (cm)"] == 0 or row["Height (cm)"] == 0:
            raise ValueError(f"ULD {row['ULD Identifier']} has invalid dimensions")

        if row["Weight Limit (kg)"] == 0:
            raise ValueError(f"ULD {row['ULD Identifier']} has invalid weight limit")

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


def write_output(
    allocation: List[
        Tuple[
            str, Optional[str], Tuple[float, float, float], Tuple[float, float, float]
        ]
    ],
    total_cost: float,
    num_packed: int,
    num_priority_uld: int,
    package_map: Dict[Tuple[int, int, int, int, int, int], List[str]],
    uld_map: Dict[Tuple[int, int, int, int], List[str]],
    k: int,
    file_path: str,
):
    """
    Write the output to a text file in format specified in problem statement.
    Additionally, creates a csv file with the allocation of the packages to the ULDs.

    Args:
        allocation: Allocation of the packages to the ULDs
        total_cost: Total cost of the solution
        num_packed: Number of packed packages
        num_priority_uld: Number of priority ULDs
        packages: List of packages
        ulds: List of ULDs
        file_path: Path to the folder to save the output
    """
    rows = []
    sols = {}

    with open(os.path.join(file_path, "output.txt"), "w") as f:
        # Iterate over the allocation and build rows
        f.write(f"{total_cost},{num_packed},{num_priority_uld}\n")

        for row in allocation:
            rows.append(
                {
                    "Package Identifier": row[0],
                    "ULD Identifier": row[1] if row[1] is not None else "None",
                    "x1": row[2][0],
                    "y1": row[2][1],
                    "z1": row[2][2],
                    "x2": row[3][0],
                    "y2": row[3][1],
                    "z2": row[3][2],
                }
            )
            sols[row[0]] = {
                "ULD Identifier": row[1],
                "point1": row[2],
                "point2": row[3],
            }
            f.write(
                f"{row[0]},{row[1] if row[1] is not None else 'NONE'},{row[2][0]},{row[2][1]},{row[2][2]},{row[3][0]},{row[3][1]},{row[3][2]}\n"
            )
    allocation_df = pd.DataFrame(rows)
    allocation_df.to_csv(os.path.join(file_path, "allocation.csv"), index=False)
    with open(os.path.join(file_path, "solution.pkl"), "wb") as f:
        pickle.dump(
            {
                "packages": package_map,
                "ulds": uld_map,
                "k": k,
                "cost": total_cost,
                "solution": sols,
            },
            f,
        )


def read_k(file_path: str) -> int:
    """
    Read k value from a text file.

    Args:
        file_path: Path to the folder containing the input data

    Returns:
        k
    """
    with open(os.path.join(file_path, "k.txt"), "r") as f:
        try:
            k = int(f.read().strip())
        except ValueError:
            raise ValueError("k value is not an integer")
    return k


def read_allocation(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read allocation data from a csv file.

    Args:
        file_path: Path to the folder containing the input data

    Returns:
        Allocation dictionary
    """
    df = pd.read_csv(os.path.join(file_path, "allocation.csv"))

    df["ULD Identifier"] = df["ULD Identifier"].where(
        df["ULD Identifier"].notnull(), None
    )

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


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from a json file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
