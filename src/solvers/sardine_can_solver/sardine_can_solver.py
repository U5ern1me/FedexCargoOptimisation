import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
import asyncio
import time

from solvers.solver import Solver
from utils.io import load_config

# typing
from typing import Dict, Any, List
from models.package import Package
from models.uld import ULD
from utils.api_error import APIError

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

    async def _solve(self, session: aiohttp.ClientSession = None):
        request_url = config["base url"] + "calculations"

        packages = [self.get_package_data(package) for package in self.packages]
        ulds = [self.get_uld_data(uld) for uld in self.ulds]

        if len(packages) == 0 or len(ulds) == 0:
            return

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

        async with session.post(request_url, json=request_body) as response:
            if response.status != 200:
                res_text = await response.text()
                raise APIError(res_text)

            self.response = await response.json()

    async def _get_result(self, session: aiohttp.ClientSession = None):
        if len(self.packages) == 0 or len(self.ulds) == 0:
            return {
                "containers": [],
            }

        try:
            if self.response is None:
                raise Exception("No response from sardine can solver")

            res = self.response
            status_url = config["base url"] + res["statusUrl"]
            result_url = config["base url"] + res["resultUrl"]

            # polling until the result is ready
            while True:
                async with session.get(status_url) as response:
                    res = await response.json()
                    if res["status"] == "DONE":
                        break

                await asyncio.sleep(config["polling interval"])

            async with session.get(result_url) as response:
                return await response.json()

        # to be checked
        except APIError as e:
            raise APIError(f"API error with sardine can solver: {e}")
        except aiohttp.ClientConnectionResetError as e:
            raise APIError(f"Connection reset error with sardine can solver: {e}")
        except aiohttp.ClientConnectionError as e:
            raise APIError(f"Connection error with sardine can solver: {e}")
        except aiohttp.ClientResponseError as e:
            raise APIError(f"Response error with sardine can solver: {e}")
        except aiohttp.ClientTimeout as e:
            raise APIError(f"Timeout error with sardine can solver: {e}")
        except aiohttp.ClientError as e:
            raise APIError(f"API error with sardine can solver: {e}")
        except Exception as e:
            raise Exception(f"Error getting result from sardine can solver: {e}")

    async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        num_packages = len(self.packages)

        for _uld in result["containers"]:
            for _package in _uld["assignments"]:
                num_packages -= 1

        return num_packages == 0

    async def _parse_result(self, result: Dict[str, Any]):
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

    async def get_packing_json(self, session: aiohttp.ClientSession = None):
        """
        get the packing json from sardine can solver

        Returns:
            Dict[str, Any]: packing json
        """

        response = await self._get_result(session=session)

        response_json = {}
        response_json["ulds"] = []

        for _uld in response["containers"]:
            uld_json = {}
            _uld_id = _uld["id"]
            uld_json["id"] = self.uld_map[_uld_id].id
            uld_json["packages"] = []
            for _package in _uld["assignments"]:
                package = self.package_map[_package["piece"]]
                uld_json["packages"].append(package.id)
            response_json["ulds"].append(uld_json)

        return response_json
