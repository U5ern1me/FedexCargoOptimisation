import os
import sys
import numpy as np
import aiohttp
import asyncio
import random
from typing import Dict, Any, Optional

# Adjust the import paths as needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.solver import Solver
from utils.io import load_config
from models.package import Package
from models.uld import ULD
from utils.api_error import APIError

# Load configuration
config = load_config(
    os.path.join(os.path.dirname(__file__), "threeD_bin_packing.config")
)


class ThreeDBinPackingSolver(Solver):
    def __init__(self, ulds: np.ndarray, packages: np.ndarray):
        super().__init__(ulds, packages)
        self.package_map: Dict[str, Package] = {}
        self.session: Optional[aiohttp.ClientSession] = None  # To reuse the session
        self.response = None  # Initialize response attribute

    def get_package_data(self, package: Package) -> Dict[str, Any]:
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
        return {
            "id": uld.id,
            "w": uld.length,
            "h": uld.width,
            "d": uld.height,
            "wg": 0,
            "max_wg": uld.weight_limit,
            "q": 1,
        }

    async def _solve_with_retry(
        self, session: aiohttp.ClientSession = None, max_retries: int = 3
    ):
        retries = 0
        while retries < max_retries:
            try:
                await self._send_request(session=session)
                return
            except Exception as e:
                if retries < max_retries - 1:
                    backoff_time = random.uniform(
                        2**retries, 2 ** (retries + 1)
                    )  # Exponential backoff
                    await asyncio.sleep(backoff_time)
                    retries += 1
                else:
                    raise e

    async def _send_request(self, session: aiohttp.ClientSession = None):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        request_url = config.get("base url", "")
        packages = [self.get_package_data(package) for package in self.packages]
        ulds = [self.get_uld_data(uld) for uld in self.ulds]
        params = {"item_coordinates": 1}

        if len(packages) == 0 or len(ulds) == 0:
            self.response = {
                "bins_packed": [],
            }
            return

        request_body = {
            "packages": packages,
            "ulds": ulds,
            "params": params,
        }

        try:
            async with session.post(
                request_url, headers=headers, json=request_body
            ) as response:
                if response.status != 200:
                    res_text = await response.text()
                    raise APIError(res_text)

                res_json = await response.json()

                if res_json["response"]["status"] == -1:
                    raise APIError(res_json["response"]["errors"][0]["message"])

                self.response = res_json["response"]

        except APIError as e:
            raise APIError(f"API error with 3D bin packing solver: {e}")
        except aiohttp.ClientConnectionResetError as e:
            raise APIError(f"Connection reset error with 3D bin packing solver: {e}")
        except aiohttp.ClientConnectionError as e:
            raise APIError(f"Connection error with 3D bin packing solver: {e}")
        except aiohttp.ClientResponseError as e:
            raise APIError(f"Response error with 3D bin packing solver: {e}")
        except aiohttp.ClientTimeout as e:
            raise APIError(f"Timeout error with 3D bin packing solver: {e}")
        except aiohttp.ClientError as e:
            raise APIError(f"API error with 3D bin packing solver: {e}")
        except Exception as e:
            raise Exception(f"Error getting result from 3D bin packing solver: {e}")

    async def _solve(self, session: aiohttp.ClientSession = None):
        await self._solve_with_retry(session=session, max_retries=3)

    async def _get_result(self, session: aiohttp.ClientSession = None):
        try:
            if self.response is None:
                raise APIError("No response from 3D bin packing solver")

            response = self.response

            return response

        except APIError as e:
            raise APIError(f"API error with 3D bin packing solver: {e}")
        except Exception as e:
            raise Exception(f"Error with 3D bin packing solver: {e}")

    async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        num_packages = len(self.packages)
        for _uld in result.get("bins_packed", []):
            for _package in _uld.get("items", []):
                num_packages -= 1

        if num_packages != 0:
            return False

        uld_package_map = {}

        for _uld in result["bins_packed"]:
            for package in _uld["items"]:
                uld_id = _uld["bin_data"]["id"]
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
                        return False

        return True

    async def _parse_result(self, result: Dict[str, Any]) -> None:
        for _uld in result.get("bins_packed", []):
            bin_data = _uld.get("bin_data", {})
            uld_id = bin_data.get("id", "")
            for _package in _uld.get("items", []):
                package_id = _package.get("id", "")
                package = self.package_map.get(package_id)
                if not package:
                    continue  # Skip if package is not found

                # Set ULD ID and coordinates
                package.uld_id = uld_id
                coordinates = _package.get("coordinates", {})
                package.point1 = (
                    coordinates.get("x1", 0),
                    coordinates.get("y1", 0),
                    coordinates.get("z1", 0),
                )
                package.point2 = (
                    coordinates.get("x2", 0),
                    coordinates.get("y2", 0),
                    coordinates.get("z2", 0),
                )

    async def get_packing_json(self, session: aiohttp.ClientSession = None):
        result = await self._get_result(session=session)

        response_json = {}
        response_json["ulds"] = []

        for _uld in result["bins_packed"]:
            uld_json = {}
            uld_json["id"] = _uld["bin_data"]["id"]
            uld_json["packages"] = []
            for _package in _uld["items"]:
                package = self.package_map[_package["id"]]
                uld_json["packages"].append(package.id)
            response_json["ulds"].append(uld_json)

        return response_json
