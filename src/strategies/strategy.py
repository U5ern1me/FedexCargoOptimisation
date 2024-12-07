import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
import time
import pickle

# typing
from typing import Dict, Any, Optional, List, Tuple
from models.package import Package
from models.uld import ULD


class Strategy(ABC):
    """
    Abstract class for a ULD packing strategy.
    If running an executable to solve the problem, overload the run function and ensure output handling.
    If running a function implemented in Python, simply overload the solve function.
    """

    def __init__(
        self,
        k_cost: Optional[float] = 0,
        ulds: List[ULD] = [],
        packages: List[Package] = [],
        output_path: Optional[str] = None,
    ):
        """
        Args:
            k_cost: Cost of priority ULDs
            ulds: List of ULDs
            packages: List of packages
            output_path: Path to the output directory
        """
        self.k_cost = k_cost
        self.ulds = ulds
        self.packages = packages
        self.output_path = output_path
        # check if packages and ulds are not empty
        if self.packages is None or len(self.packages) == 0:
            raise ValueError("Packages not provided")
        if self.ulds is None or len(self.ulds) == 0:
            raise ValueError("ULD not provided")

        self.solution_found = False
        self.time_start = time.time()
        self.time_end = -1

        self._package_map = {}
        self._uld_map = {}

        # to check logging
        self.debug = int(os.environ.get("DEBUG", "0"))

        self.initialize_maps()

    async def get_allocation(
        self,
    ) -> List[Tuple[str, Optional[str], Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Get the allocation of the packages to the ULDs.

        Returns:
            List of tuples containing the package ID, ULD ID, and the start and end points of the package
        """
        allocation = []
        for package in self.packages:
            allocation.append(
                (package.id, package.uld_id, package.point1, package.point2)
            )

        return allocation

    def get_num_packed(self) -> int:
        """
        Get the number of packages that are packed.

        Returns:
            Number of packed packages
        """
        num_packed = 0
        for package in self.packages:
            if package.uld_id is not None:
                num_packed += 1
        return num_packed

    def initialize_maps(self):
        """
        Create maps for the packages and ULDs to speed up the lookup.
        """
        for i in range(len(self.packages)):
            hash_tuple = (
                self.packages[i].length,
                self.packages[i].width,
                self.packages[i].height,
                self.packages[i].weight,
                self.packages[i].delay_cost,
                1 if self.packages[i].priority else 0,
            )
            if hash_tuple in self._package_map:
                self._package_map[hash_tuple].append(self.packages[i].id)
            else:
                self._package_map[hash_tuple] = [self.packages[i].id]

        for i in range(len(self.ulds)):
            hash_tuple = (
                self.ulds[i].length,
                self.ulds[i].width,
                self.ulds[i].height,
                self.ulds[i].weight_limit,
            )
            if hash_tuple in self._uld_map:
                self._uld_map[hash_tuple].append(self.ulds[i].id)
            else:
                self._uld_map[hash_tuple] = [self.ulds[i].id]

    def get_num_priority_uld(self) -> int:
        """
        Get the number of ULDs that have priority packages.

        Returns:
            Number of priority ULDs
        """
        priority_ulds = set()

        for package in self.packages:
            if package.uld_id is not None and package.priority:
                priority_ulds.add(package.uld_id)
        return len(priority_ulds)

    def calculate_cost(self) -> float:
        """
        Calculate the total cost of the solution.

        Returns:
            Total cost of the solution
        """
        total_cost = 0

        total_cost += self.get_num_priority_uld() * self.k_cost
        for package in self.packages:
            if package.uld_id is None:
                total_cost += package.delay_cost

        return total_cost

    async def post_process(self):
        uld_map = {}
        package_map = {}
        final_found = False
        curr_best_cost = self.calculate_cost()

        for f in os.listdir(self.output_path):
            if not os.path.exists(os.path.join(self.output_path, f, "solution.pkl")):
                continue

            with open(os.path.join(self.output_path, f, "solution.pkl"), "rb") as file:
                data = pickle.load(file)

                not_found = False

                if data["k"] != self.k_cost:
                    continue

                _uld_map = {}

                for uld_data in self._uld_map:
                    if uld_data not in data["ulds"] or len(
                        self._uld_map[uld_data]
                    ) != len(data["ulds"][uld_data]):
                        not_found = True
                        break

                    for i in range(len(self._uld_map[uld_data])):
                        _uld_map[data["ulds"][uld_data][i]] = self._uld_map[uld_data][i]

                if not_found:
                    continue

                _package_map = {}

                for package_data in self._package_map:
                    if package_data not in data["packages"] or len(
                        self._package_map[package_data]
                    ) != len(data["packages"][package_data]):
                        not_found = True
                        break

                    for i in range(len(self._package_map[package_data])):
                        _package_map[self._package_map[package_data][i]] = data[
                            "packages"
                        ][package_data][i]

            if not not_found:
                if data["cost"] < curr_best_cost:
                    final_found = True
                    curr_best_cost = data["cost"]
                    uld_map = _uld_map
                    package_map = _package_map
                break

        if not final_found:
            return False

        for package in self.packages:
            if package.id not in package_map:
                return False

            s = data["solution"][package_map[package.id]]

            package.uld_id = (
                None if s["ULD Identifier"] is None else uld_map[s["ULD Identifier"]]
            )
            package.point1 = s["point1"]
            package.point2 = s["point2"]

        return True

    async def get_outputs(self) -> Tuple[
        List[Tuple[str, Optional[str], Tuple[int, int, int], Tuple[int, int, int]]],
        float,
        int,
        int,
        Dict[Tuple[int, int, int, int, int, int], List[str]],
        Dict[Tuple[int, int, int, int], List[str]],
        float,
    ]:
        """
        Get the allocation, total cost, number of packed packages, and number of priority ULDs.

        Returns:
            Allocation, total cost, number of packed packages, number of priority ULDs, package map, ULD map, and k cost
        """
        total_cost = self.calculate_cost()
        allocation = await self.get_allocation()
        num_packed = self.get_num_packed()
        num_priority_uld = self.get_num_priority_uld()

        return (
            allocation,
            total_cost,
            num_packed,
            num_priority_uld,
            self._package_map,
            self._uld_map,
            self.k_cost,
        )

    @abstractmethod
    async def solve(self):
        """
        Solve the ULD packing problem. Update the packages with the ULD ID and the end points.
        """

        pass

    async def gravity_stabilization(self):
        """
        Perform gravity stabilization on the packages which might be floating in the air.
        """

        package_uld_map = {}

        for package in self.packages:
            if package.uld_id is None:
                continue

            if package.uld_id not in package_uld_map:
                package_uld_map[package.uld_id] = []
            package_uld_map[package.uld_id].append(package)

        for uld_id, packages in package_uld_map.items():
            sorted_packages = sorted(packages, key=lambda x: int(x.point1[2]))

            for package in sorted_packages:
                level = 0
                for other_package in sorted_packages:
                    if package.id == other_package.id:
                        continue

                    x_overlap = max(
                        0,
                        min(package.point2[0], other_package.point2[0])
                        - max(package.point1[0], other_package.point1[0]),
                    )
                    y_overlap = max(
                        0,
                        min(package.point2[1], other_package.point2[1])
                        - max(package.point1[1], other_package.point1[1]),
                    )

                    if (
                        x_overlap > 0
                        and y_overlap > 0
                        and package.point1[2] >= other_package.point2[2]
                    ):
                        level = max(level, int(other_package.point2[2]))

                short_by = int(package.point1[2]) - level
                new_point1 = (
                    package.point1[0],
                    package.point1[1],
                    int(package.point1[2] - short_by),
                )
                new_point2 = (
                    package.point2[0],
                    package.point2[1],
                    int(package.point2[2] - short_by),
                )

                package.point1 = new_point1
                package.point2 = new_point2

    async def run(self):
        """
        Run the strategy and write the output to a file.
        """
        if self.packages is None:
            raise ValueError("Packages not provided")
        if self.ulds is None:
            raise ValueError("ULD not provided")

        self.start_time = time.time()

        await self.solve()

        await self.gravity_stabilization()

        self.time_end = time.time()
        self.solution_found = True

    async def validate(self) -> bool:
        """
        Validate the solution. If invalid, set the error message and return False.
        """
        for package in self.packages:
            if package.priority and package.uld_id is None:
                self.error = "All priority packages cannot be allocated in given ULDs"
                return False

        uld_package_map = {}

        for package in self.packages:
            if package.uld_id is None:
                continue

            if package.uld_id not in uld_package_map:
                uld_package_map[package.uld_id] = []

            uld_package_map[package.uld_id].append(package)

        for uld_id, packages in uld_package_map.items():
            uld = next((uld for uld in self.ulds if uld.id == uld_id), None)

            for i in range(len(packages)):

                if (
                    packages[i].point2[0] > uld.length
                    or packages[i].point2[1] > uld.width
                    or packages[i].point2[2] > uld.height
                ):
                    self.error = (
                        f"Package {packages[i].id} exceeds ULD {uld.id} dimensions"
                    )
                    return False

                for j in range(i + 1, len(packages)):
                    x1_min, x1_max = min(
                        packages[i].point1[0], packages[i].point2[0]
                    ), max(packages[i].point1[0], packages[i].point2[0])
                    y1_min, y1_max = min(
                        packages[i].point1[1], packages[i].point2[1]
                    ), max(packages[i].point1[1], packages[i].point2[1])
                    z1_min, z1_max = min(
                        packages[i].point1[2], packages[i].point2[2]
                    ), max(packages[i].point1[2], packages[i].point2[2])

                    x2_min, x2_max = min(
                        packages[j].point1[0], packages[j].point2[0]
                    ), max(packages[j].point1[0], packages[j].point2[0])
                    y2_min, y2_max = min(
                        packages[j].point1[1], packages[j].point2[1]
                    ), max(packages[j].point1[1], packages[j].point2[1])
                    z2_min, z2_max = min(
                        packages[j].point1[2], packages[j].point2[2]
                    ), max(packages[j].point1[2], packages[j].point2[2])

                    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

                    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                        self.error = (
                            f"Packages {packages[i].id} and {packages[j].id} overlap"
                        )
                        return False

        return True

    async def get_solution_json(self) -> Dict[str, Any]:
        """
        Get the solution in the json format for the API.

        Returns:
            Solution in the json format
        """
        allocation = []
        uld_package_map = {}

        # Initialize the ULD package map to calculate the efficiency of the ULDs.
        for uld in self.ulds:
            uld_package_map[uld.id] = {
                "volume_used": 0,
                "weight_used": 0,
                "num_economy_packages": 0,
                "num_priority_packages": 0,
            }

        # Allocation of the packages to the ULDs.
        packed_volume = 0
        packed_weight = 0
        total_priority_packages = 0
        for package in self.packages:
            allocation.append(
                {
                    "id": package.id,
                    "uld_id": package.uld_id,
                    "x1": float(package.point1[0]),
                    "y1": float(package.point1[1]),
                    "z1": float(package.point1[2]),
                    "x2": float(package.point2[0]),
                    "y2": float(package.point2[1]),
                    "z2": float(package.point2[2]),
                    "height": float(package.height),
                    "width": float(package.width),
                    "length": float(package.length),
                    "priority": package.priority,
                    "delay_cost": float(package.delay_cost),
                    "weight": float(package.weight),
                }
            )

            # Update the packing data of the ULDs in the map.
            if package.uld_id:
                packed_volume += package.height * package.width * package.length
                packed_weight += package.weight

                uld_package_map[package.uld_id]["volume_used"] += packed_volume
                uld_package_map[package.uld_id]["weight_used"] += packed_weight

                if package.priority:
                    uld_package_map[package.uld_id]["num_priority_packages"] += 1
                    total_priority_packages += 1
                else:
                    uld_package_map[package.uld_id]["num_economy_packages"] += 1

        # ULDs data and efficiency.
        ulds = []
        uld_volume = 0
        uld_weight = 0
        for uld in self.ulds:
            _uld_weight = uld.weight_limit
            _uld_volume = uld.height * uld.width * uld.length

            ulds.append(
                {
                    "uld_id": uld.id,
                    "height": uld.height,
                    "width": uld.width,
                    "length": uld.length,
                    "weight_limit": uld.weight_limit,
                    "volume_used": uld_package_map[uld.id]["volume_used"],
                    "weight_used": uld_package_map[uld.id]["weight_used"],
                    "num_economy_packages": uld_package_map[uld.id][
                        "num_economy_packages"
                    ],
                    "num_priority_packages": uld_package_map[uld.id][
                        "num_priority_packages"
                    ],
                    "total_packages": (
                        uld_package_map[uld.id]["num_economy_packages"]
                        + uld_package_map[uld.id]["num_priority_packages"]
                    ),
                    "has_priority_packages": (
                        1 if uld_package_map[uld.id]["num_priority_packages"] > 0 else 0
                    ),
                    "volume_efficiency": (
                        uld_package_map[uld.id]["volume_used"] / _uld_volume
                    ),
                    "weight_efficiency": (
                        uld_package_map[uld.id]["weight_used"] / _uld_weight
                    ),
                }
            )

            uld_volume += _uld_volume
            uld_weight += _uld_weight

        # Other metrics
        cost = self.calculate_cost()
        total_packed_packages = self.get_num_packed()
        num_priority_ulds = self.get_num_priority_uld()

        return {
            "allocation": allocation,
            "total_cost": float(cost),
            "delay_cost": float(cost - num_priority_ulds * self.k_cost),
            "priority_cost": float(num_priority_ulds * self.k_cost),
            "total_packed_packages": total_packed_packages,
            "total_packed_priority_packages": total_priority_packages,
            "total_packed_economy_packages": total_packed_packages
            - total_priority_packages,
            "num_priority_ulds": num_priority_ulds,
            "total_unpacked_packages": len(self.packages) - total_packed_packages,
            "ulds": ulds,
            "packed_volume": float(packed_volume),
            "packed_weight": float(packed_weight),
            "volume_efficiency": float(packed_volume / uld_volume),
            "weight_efficiency": float(packed_weight / uld_weight),
        }

    def end(self):
        self.time_end = time.time()

    def reset(self):
        self.time_start = time.time()
        self.time_end = -1
        self.solution_found = False
        for package in self.packages:
            package.uld_id = None
            package.point1 = (-1, -1, -1)
            package.point2 = (-1, -1, -1)
