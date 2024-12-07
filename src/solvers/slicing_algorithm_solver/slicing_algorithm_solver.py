import os
import sys
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.solver import Solver
from .slicing_algorithm_utils import pack_package
import pandas as pd

# type hinting
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

    async def _solve(self,session):
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
        num_packages = len(self.packages)

        for _uld in result.keys():
            for _package in result[_uld]:
                num_packages -= 1

        if num_packages != 0:
            return False

        uld_package_map = {}

        for _uld in result.keys():
            for package in result[_uld]:
                uld_id = _uld
                if uld_id not in uld_package_map:
                    uld_package_map[uld_id] = []
                    package = {
                        "id": package[0],
                        "coordinates": {
                            "x1": package[1],
                            "y1": package[2],
                            "z1": package[3],
                            "x2": package[4],
                            "y2": package[5],
                            "z2": package[6],
                        },
                    }
                uld_package_map[uld_id].append(package)

        for uld_id, packages in uld_package_map.items():
            uld = next((uld for uld in self.ulds if uld.id == uld_id), None)

            for i in range(len(packages)):

                if (
                    packages[i]["coordinates"]["x2"] > uld.length
                    or packages[i]["coordinates"]["y2"] > uld.width
                    or packages[i]["coordinates"]["z2"] > uld.height
                ):
                    self.error = (
                        f"Package {packages[i]['id']} exceeds ULD {uld_id} dimensions"
                    )
                    return False

                for j in range(i + 1, len(packages)):
                    x1_min, x1_max = min(
                        packages[i]["coordinates"]["x1"],
                        packages[i]["coordinates"]["x2"],
                    ), max(
                        packages[i]["coordinates"]["x1"],
                        packages[i]["coordinates"]["x2"],
                    )
                    y1_min, y1_max = min(
                        packages[i]["coordinates"]["y1"],
                        packages[i]["coordinates"]["y2"],
                    ), max(
                        packages[i]["coordinates"]["y1"],
                        packages[i]["coordinates"]["y2"],
                    )
                    z1_min, z1_max = min(
                        packages[i]["coordinates"]["z1"],
                        packages[i]["coordinates"]["z2"],
                    ), max(
                        packages[i]["coordinates"]["z1"],
                        packages[i]["coordinates"]["z2"],
                    )

                    x2_min, x2_max = min(
                        packages[j]["coordinates"]["x1"],
                        packages[j]["coordinates"]["x2"],
                    ), max(
                        packages[j]["coordinates"]["x1"],
                        packages[j]["coordinates"]["x2"],
                    )
                    y2_min, y2_max = min(
                        packages[j]["coordinates"]["y1"],
                        packages[j]["coordinates"]["y2"],
                    ), max(
                        packages[j]["coordinates"]["y1"],
                        packages[j]["coordinates"]["y2"],
                    )
                    z2_min, z2_max = min(
                        packages[j]["coordinates"]["z1"],
                        packages[j]["coordinates"]["z2"],
                    ), max(
                        packages[j]["coordinates"]["z1"],
                        packages[j]["coordinates"]["z2"],
                    )

                    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

                    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                        self.error = f"Packages {packages[i]['id']} and {packages[j]['id']} overlap"
                        return False

        return True

    async def _parse_result(self, result: Dict[str, Any]):
        for uld in result.keys():
            uld_id = uld
            for package in result[uld]:
                package.uld_id = uld_id
                package.point1 = (
                    package[1],
                    package[2],
                    package[3],
                )
                package.point2 = (
                    package[4],
                    package[5],
                    package[6],
                )

    def check_all_fit(self) -> bool:
        """
        Check if all packages fit in the ULDs
        """
        for package in self.packages:
            if package.uld_id == None:
                return False

        return True

    async def get_fit(self,session) -> bool:
        """
        Get the result of the solving process
        """

        result = await self._get_result()

        if self.only_check_fits:
            valid = await self._only_check_fits(result)
        else:
            await self._parse_result(result)
            valid = self.check_all_fit()

        return valid
