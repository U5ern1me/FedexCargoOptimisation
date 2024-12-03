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
        request_url = config["base url"] + "calculations"

        packages = [self.get_package_data(package) for package in self.packages]
        ulds = [self.get_uld_data(uld) for uld in self.ulds]

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

        self.response = requests.post(request_url, json=request_body)

    def _get_result(self):
        try:
            if self.response is None:
                raise Exception("No response from sardine can solver")

            res = self.response.json()
            status_url = config["base url"] + res["statusUrl"]
            result_url = config["base url"] + res["resultUrl"]

            # polling until the result is ready
            while True:
                response = requests.get(status_url)
                response = response.json()
                if response["status"] == "DONE":
                    break
                time.sleep(config["polling interval"])

            return requests.get(result_url).json()
        except Exception as e:
            raise Exception(f"Error getting result from sardine can solver: {e}")

    def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        num_packages = len(self.packages)

        for _uld in result["containers"]:
            for _package in _uld["assignments"]:
                num_packages -= 1

        return num_packages == 0

    def _parse_result(self, result: Dict[str, Any]):
        for _uld in result["containers"]:
            for _package in _uld["assignments"]:
                # get package
                package_id = _package["piece"]
                package = self.package_map[package_id]

                # get uld
                uld_id = _uld["id"]
                uld = self.uld_map[uld_id]

                # set uld id and coordinates
                package.uld_id = uld.id
                x = _package["position"]["x"]
                y = _package["position"]["y"]
                z = _package["position"]["z"]
                package.point1 = (x, y, z)
                package.point2 = (
                    x + _package["cubes"][0]["length"],
                    y + _package["cubes"][0]["width"],
                    z + _package["cubes"][0]["height"],
                )
