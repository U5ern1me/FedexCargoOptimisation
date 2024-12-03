import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time

from solvers.solver import Solver
from utils.io import load_config

# typing
from typing import Dict, Any, List
from models.package import Package
from models.uld import ULD

config = load_config(os.path.join(os.path.dirname(__file__), "sardine_can.config"))


class SardineCanSolver(Solver):

    def __init__(self, ulds: List[ULD], packages: List[Package]):
        super().__init__(ulds, packages)
        self.package_map = {}
        self.uld_map = {}

    def get_package_data(self, package: Package) -> Dict[str, Any]:
        """
        convert package into json format required by sardine can solver

        Args:
            package (Package): package to convert

        Returns:
            Dict[str, Any]: package in json format
        """
        package_id = int(package.id.split("-")[1])
        self.package_map[package_id] = package
        return {
            "id": package_id,
            "weight": package.weight,
            "cubes": [
                {
                    "length": package.length,
                    "width": package.width,
                    "height": package.height,
                    "x": 0,
                    "y": 0,
                    "z": 0,
                }
            ],
        }

    def get_uld_data(self, uld: ULD) -> Dict[str, Any]:
        """
        convert uld into json format required by sardine can solver

        Args:
            uld (ULD): uld to convert

        Returns:
            Dict[str, Any]: uld in json format
        """
        uld_id = int(uld.id[1:])
        self.uld_map[uld_id] = uld
        return {
            "id": uld_id,
            "length": uld.length,
            "width": uld.width,
            "height": uld.height,
            "maxWeight": uld.weight_limit,
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
            "configuration": {
                "goal": config["goal"],
                "timeLimitInSeconds": config["time limit"],
                "iterationsLimit": config["iteration limit"],
                "type": config["packing type"],
            },
            "priority": 1,
            "instance": {
                "name": f"box_packing_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                "containers": ulds,
                "pieces": packages,
            },
        }

        self.response = requests.post(request_url, headers=headers, json=request_body)

    def _get_result(self):
        try:
            if self.response is None:
                raise Exception("No response from 3D bin packing solver")

            return self.response.json()
        except Exception as e:
            raise Exception(f"Error getting result from 3D bin packing solver: {e}")
