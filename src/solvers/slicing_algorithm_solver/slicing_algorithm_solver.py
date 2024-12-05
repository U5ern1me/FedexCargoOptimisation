import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.solver import Solver
from .slicing_algorithm_utils import pack_package
import pandas as pd

# typing
from models.uld import ULD
from models.package import Package
from typing import List, Dict, Any


class SlicingAlgorithmSolver(Solver):
    def __init__(self, ulds: List[ULD], packages: List[Package]):
        super().__init__(ulds, packages)
        self.package_map = {}
        self.uld_map = {}
        package_data = [
            {
                "box no": pkg.id,
                "length": pkg.length,
                "width": pkg.width,
                "height": pkg.height,
                "weight": pkg.weight,
            }
            for pkg in self.packages
        ]

        # Convert the list of dictionaries into a pandas DataFrame
        self.package_specific_df = pd.DataFrame(package_data)
        self.response = {}

    def remove_packed_bins(self, package_specific_df, packed_bins):
        # Extract the 'box no' values from packed_bins (first element of each tuple)
        packed_box_nos = [box[0] for box in packed_bins]

        # Filter out rows from the DataFrame where 'box no' is in packed_box_nos
        updated_df = package_specific_df[
            ~package_specific_df["box no"].isin(packed_box_nos)
        ]

        return updated_df

    async def _solve(self):
        for uld in self.ulds:
            uld_data = {
                "ULD no": [uld.id],
                "length": [uld.length],
                "width": [uld.width],
                "height": [uld.height],
                "weight_limit": [uld.weight_limit],
            }
            # Convert the dictionary into a pandas DataFrame
            uld_specific_df = pd.DataFrame(uld_data)

            packed_bins = pack_package(uld_specific_df, self.package_specific_df)

            self.response[uld.id] = packed_bins

            self.package_specific_df = self.remove_packed_bins(
                self.package_specific_df, packed_bins
            )

    async def _get_result(self):
        try:
            if self.response is None:
                raise Exception("No response from slicing algorithm can solver")

            return self.response
        except Exception as e:
            raise Exception(f"Error getting result from slicing algorithm solver: {e}")

    async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        return self.package_specific_df.empty

    async def _parse_result(self, result: Dict[str, Any]):
        for uld_id, packed_bins in result.items():
            for packed_bin in packed_bins:
                for package in self.packages:
                    if packed_bin[0] == package.id:
                        package.uld_id = uld_id
                        package.point1 = (packed_bin[1], packed_bin[2], packed_bin[3])
                        package.point2 = (packed_bin[4], packed_bin[5], packed_bin[6])
                        break
