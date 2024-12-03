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

    def _solve(self):
        headers = {
            "Accept": "application/json",
        }

        request_url = "https://global-api.3dbinpacking.com/packer/packIntoMany"

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

    def _get_result(self):
        try:
            if self.response is None:
                raise Exception("No response from 3D bin packing solver")

            return self.response.json()
        except Exception as e:
            raise Exception(f"Error getting result from 3D bin packing solver: {e}")
