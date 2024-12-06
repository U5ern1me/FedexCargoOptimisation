# import os
# import sys
# import numpy as np
# import aiohttp # TODO: Add to requirements.txt
# import asyncio
# from typing import Dict, Any, List, Tuple, Optional
# from time import sleep
# import functools

# # Adjust the import paths as needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from solvers.solver import Solver
# from utils.io import load_config
# from models.package import Package
# from models.uld import ULD

# # Load configuration
# config = load_config(
#     os.path.join(os.path.dirname(__file__), "threeD_bin_packing.config")
# )


# class ThreeDBinPackingSolver(Solver):
#     def __init__(self, ulds: np.ndarray, packages: np.ndarray):
#         super().__init__(ulds, packages)
#         self.package_map: Dict[str, Package] = {}
#         self.session: Optional[aiohttp.ClientSession] = None  # To reuse the session

#     def get_package_data(self, package: Package) -> Dict[str, Any]:
#         """
#         Convert a Package instance into the JSON format required by the 3D bin packing solver.

#         Args:
#             package (Package): The package to convert.

#         Returns:
#             Dict[str, Any]: The package data in JSON format.
#         """
#         self.package_map[package.id] = package
#         return {
#             "id": package.id,
#             "w": package.length,
#             "h": package.width,
#             "d": package.height,
#             "wg": package.weight,
#             "vr": 1,
#             "q": 1,
#         }

#     def get_uld_data(self, uld: ULD) -> Dict[str, Any]:
#         """
#         Convert a ULD instance into the JSON format required by the 3D bin packing solver.

#         Args:
#             uld (ULD): The ULD to convert.

#         Returns:
#             Dict[str, Any]: The ULD data in JSON format.
#         """
#         return {
#             "id": uld.id,
#             "w": uld.length,
#             "h": uld.width,
#             "d": uld.height,
#             "wg": 0,
#             "max_wg": uld.weight_limit,
#             "q": 1,
#         }

#     async def _initialize_session(self):
#         """
#         Initialize the aiohttp ClientSession if it hasn't been initialized yet.
#         """
#         if self.session is None:
#             self.session = aiohttp.ClientSession()

#     async def close_session(self):
#         """
#         Close the aiohttp ClientSession if it's open.
#         """
#         if self.session is not None:
#             await self.session.close()
#             self.session = None

#     async def _solve(self) -> None:
#         """
#         Send a POST request to the 3D bin packing solver API to perform the packing.

#         Raises:
#             Exception: If the request fails or the response is invalid.
#         """
#         await self._initialize_session()

#         headers = {
#             "Accept": "application/json",
#             "Content-Type": "application/json",
#         }

#         request_url = config.get("base url", "") + "packIntoMany"

#         packages = [self.get_package_data(package) for package in self.packages]
#         ulds = [self.get_uld_data(uld) for uld in self.ulds]
#         params = {"item_coordinates": 1}

#         request_body = {
#             "username": config.get("username", ""),
#             "api_key": config.get("api_key", ""),
#             "items": packages,
#             "bins": ulds,
#             "params": params,
#         }

#         try:
#             async with self.session.post(
#                 request_url, headers=headers, json=request_body
#             ) as response:
#                 if response.status != 200:
#                     text = await response.text()
#                     raise Exception(f"Solver API returned status {response.status}: {text}")
#                 self.response = await response.json()
#                 sleep(0.1)  # Sleep to avoid rate limiting
#         except Exception as e:
#             raise Exception(f"Error during solving: {e}")

#     async def _get_result(self) -> Dict[str, Any]:
#         """
#         Retrieve and validate the result from the solver.

#         Returns:
#             Dict[str, Any]: The solver's response.

#         Raises:
#             Exception: If no response is available or if parsing fails.
#         """
#         try:
#             if not hasattr(self, 'response') or self.response is None:
#                 raise Exception("No response from 3D bin packing solver")

#             return self.response.get("response", {})
#         except Exception as e:
#             raise Exception(f"Error getting result from 3D bin packing solver: {e}")

#     async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
#         """
#         Check if all packages fit into the ULDs based on the solver's result.

#         Args:
#             result (Dict[str, Any]): The solver's result.

#         Returns:
#             bool: True if all packages fit, False otherwise.
#         """
#         num_packages = len(self.packages)

#         for _uld in result.get("bins_packed", []):
#             for _package in _uld.get("items", []):
#                 num_packages -= 1

#         return num_packages == 0

#     async def _parse_result(self, result: Dict[str, Any]) -> None:
#         """
#         Parse the solver's result and update package placement details.

#         Args:
#             result (Dict[str, Any]): The solver's result.
#         """
#         for _uld in result.get("bins_packed", []):
#             bin_data = _uld.get("bin_data", {})
#             uld_id = bin_data.get("id", "")
#             for _package in _uld.get("items", []):
#                 package_id = _package.get("id", "")
#                 package = self.package_map.get(package_id)
#                 if not package:
#                     continue  # Skip if package is not found

#                 # Set ULD ID and coordinates
#                 package.uld_id = uld_id
#                 coordinates = _package.get("coordinates", {})
#                 package.point1 = (
#                     coordinates.get("x1", 0),
#                     coordinates.get("y1", 0),
#                     coordinates.get("z1", 0),
#                 )
#                 package.point2 = (
#                     coordinates.get("x2", 0),
#                     coordinates.get("y2", 0),
#                     coordinates.get("z2", 0),
#                 )

#     async def solve(self) -> None:
#         """
#         Public method to solve the bin packing problem asynchronously.
#         """
#         await self._solve()

#     async def get_result(self) -> Dict[str, Any]:
#         """
#         Public method to retrieve the result after solving.

#         Returns:
#             Dict[str, Any]: The solver's response.

#         Raises:
#             Exception: If the result is invalid.
#         """
#         result = await self._get_result()
#         await self._parse_result(result)
#         return result

#     async def __aenter__(self):
#         """
#         Enable usage of the solver with async context managers.
#         """
#         await self._initialize_session()
#         return self

#     async def __aexit__(self, exc_type, exc, tb):
#         """
#         Ensure the session is closed when exiting the async context manager.
#         """
#         await self.close_session()


import os
import sys
import numpy as np
import aiohttp
import asyncio
import random
from typing import Dict, Any, List, Optional
from time import sleep

# Adjust the import paths as needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
import random
from solvers.solver import Solver
from utils.io import load_config
from models.package import Package
from models.uld import ULD

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

        request_url = config.get("base url", "") + "packIntoMany"
        packages = [self.get_package_data(package) for package in self.packages]
        ulds = [self.get_uld_data(uld) for uld in self.ulds]
        params = {"item_coordinates": 1}

        request_body = {
            "username": config.get("username", ""),
            "api_key": config.get("api_key", ""),
            "items": packages,
            "bins": ulds,
            "params": params,
        }

        try:
            async with session.post(
                request_url, headers=headers, json=request_body
            ) as response:
                if response.status != 200:
                    res_text = await response.text()
                    raise Exception(res_text)

                res_json = await response.json()

                if res_json["response"]["status"] == -1:
                    raise Exception(res_json["response"]["errors"][0]["message"])

                self.response = res_json["response"]

        except Exception as e:
            raise Exception(f"API error with 3D bin packing solver: {e}")

    async def _solve(self, session: aiohttp.ClientSession = None):
        await self._solve_with_retry(session=session, max_retries=3)

    async def _get_result(self, session: aiohttp.ClientSession = None):
        try:
            if self.response is None:
                raise Exception("No response from 3D bin packing solver")

            response = self.response

            if "errors" in response:
                raise Exception("API access was locked out.")

            return response

        except Exception as e:
            raise Exception(f"API error with 3D bin packing solver: {e}")

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
