import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from solvers.solver import Solver
from utils.io import load_config

# typing
from typing import Dict, Any, List
from models.package import Package
from models.uld import ULD

config = load_config(
    os.path.join(os.path.dirname(__file__), "threeD_bin_packing.config")
)


class ThreeDBinPackingSolver(Solver):

    def __init__(self, ulds: List[ULD], packages: List[Package]):
        super().__init__(ulds, packages)
        self.package_map = {}

    def get_package_data(self, package: Package) -> Dict[str, Any]:
        """
        convert package into json format required by 3D bin packing solver

        Args:
            package (Package): package to convert

        Returns:
            Dict[str, Any]: package in json format
        """
        self.package_map[package.id] = package
        return {
            "id": package.id,
            "w": package.length,
            "h": package.width,
            "d": package.height,
            "wg": package.weight,
            "vr": 1,
            "q": 1,
        }

    def get_uld_data(self, uld: ULD) -> Dict[str, Any]:
        """
        convert uld into json format required by 3D bin packing solver

        Args:
            uld (ULD): uld to convert

        Returns:
            Dict[str, Any]: uld in json format
        """
        return {
            "id": uld.id,
            "w": uld.length,
            "h": uld.width,
            "d": uld.height,
            "wg": 0,
            "max_wg": uld.weight_limit,
            "q": 1,
        }

    async def _solve(self):
        headers = {
            "Accept": "application/json",
        }

        request_url = config["base url"] + "packIntoMany"

        packages = [self.get_package_data(package) for package in self.packages]
        ulds = [self.get_uld_data(uld) for uld in self.ulds]
        params = {"item_coordinates": 1}

        request_body = {
            "username": config["username"],
            "api_key": config["api_key"],
            "items": packages,
            "bins": ulds,
            "params": params,
        }

        self.response = requests.post(request_url, headers=headers, json=request_body)

    async def _get_result(self):
        try:
            if self.response is None:
                raise Exception("No response from 3D bin packing solver")

            return self.response.json()["response"]
        except Exception as e:
            raise Exception(f"Error getting result from 3D bin packing solver: {e}")

    async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        num_packages = len(self.packages)

        for _uld in result["bins_packed"]:
            for _package in _uld["items"]:
                num_packages -= 1

        if num_packages != 0:
            return False

        uld_package_map = {}

        for _uld in result["bins_packed"]:
            for package in _uld["items"]:
                uld_id= _uld["bin_data"]["id"]
                if uld_id not in uld_package_map:
                    uld_package_map[uld_id] = []

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
                        packages[i]["coordinates"]["x1"], packages[i]["coordinates"]["x2"]
                    ), max(packages[i]["coordinates"]["x1"], packages[i]["coordinates"]["x2"])
                    y1_min, y1_max = min(
                        packages[i]["coordinates"]["y1"], packages[i]["coordinates"]["y2"]
                    ), max(packages[i]["coordinates"]["y1"], packages[i]["coordinates"]["y2"])
                    z1_min, z1_max = min(
                        packages[i]["coordinates"]["z1"], packages[i]["coordinates"]["z2"]
                    ), max(packages[i]["coordinates"]["z1"], packages[i]["coordinates"]["z2"])

                    x2_min, x2_max = min(
                        packages[j]["coordinates"]["x1"], packages[j]["coordinates"]["x2"]
                    ), max(packages[j]["coordinates"]["x1"], packages[j]["coordinates"]["x2"])
                    y2_min, y2_max = min(
                        packages[j]["coordinates"]["y1"], packages[j]["coordinates"]["y2"]
                    ), max(packages[j]["coordinates"]["y1"], packages[j]["coordinates"]["y2"])
                    z2_min, z2_max = min(
                        packages[j]["coordinates"]["z1"], packages[j]["coordinates"]["z2"]
                    ), max(packages[j]["coordinates"]["z1"], packages[j]["coordinates"]["z2"])

                    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

                    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                        self.error = (
                            f"Packages {packages[i]['id']} and {packages[j]['id']} overlap"
                        )
                        return False

        return True

    async def _parse_result(self, result: Dict[str, Any]):
        for _uld in result["bins_packed"]:
            for _package in _uld["items"]:
                # get package
                package_id = _package["id"]
                package = self.package_map[package_id]

                # set uld id and coordinates
                uld_id = _uld["bin_data"]["id"]
                package.uld_id = uld_id
                x = _package["coordinates"]["x1"]
                y = _package["coordinates"]["y1"]
                z = _package["coordinates"]["z1"]
                package.point1 = (x, y, z)
                x = _package["coordinates"]["x2"]
                y = _package["coordinates"]["y2"]
                z = _package["coordinates"]["z2"]
                package.point2 = (x, y, z)
