import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
import time


class Strategy(ABC):
    """
    Abstract class for a ULD packing strategy.
    If running an executable to solve the problem, overload the run function and ensure output handling.
    If running a function implemented in Python, simply overload the solve function.
    """

    def __init__(
        self,
        k_cost=None,
        ulds=None,
        packages=None,
    ):
        self.k_cost = k_cost
        self.ulds = ulds
        self.packages = packages

        self.solution_found = False
        self.time_start = time.time()

    def get_allocation(self):
        """
        Get the allocation of the packages to the ULDs.
        """
        allocation = []
        for package in self.packages:
            allocation.append(
                (package.id, package.uld_id, package.point1, package.point2)
            )

        return allocation

    def get_num_packed(self):
        """
        Get the number of packages that are packed.
        """
        num_packed = 0
        for package in self.packages:
            if package.uld_id is not None:
                num_packed += 1
        return num_packed

    def get_num_priority_uld(self):
        """
        Get the number of ULDs that have priority packages.
        """
        priority_ulds = set()

        for package in self.packages:
            if package.uld_id is not None and package.priority:
                priority_ulds.add(package.uld_id)
        return len(priority_ulds)

    def calculate_cost(self):
        """
        Calculate the total cost of the solution.
        """
        total_cost = 0

        total_cost += self.get_num_priority_uld() * self.k_cost
        for package in self.packages:
            if package.uld_id is None:
                total_cost += package.delay_cost

        return total_cost

    def get_outputs(self):
        """
        Get the allocation, total cost, number of packed packages, and number of priority ULDs.
        """
        total_cost = self.calculate_cost()
        allocation = self.get_allocation()
        num_packed = self.get_num_packed()
        num_priority_uld = self.get_num_priority_uld()

        return allocation, total_cost, num_packed, num_priority_uld

    @abstractmethod
    def solve(self):
        """
        Solve the ULD packing problem. Update the packages with the ULD ID and the end points.
        """

        pass

    def run(self):
        """
        Run the strategy and write the output to a file.
        """
        if self.packages is None:
            raise ValueError("Packages not provided")
        if self.ulds is None:
            raise ValueError("ULD not provided")

        self.start_time = time.time()

        self.solve()

        self.time_end = time.time()
        self.solution_found = True

    def validate(self):
        """
        Validate the solution. If invalid, set the error message and return False.
        """
        for package in self.packages:
            if package.priority and package.uld_id is None:
                self.error = f"Priority Package {package.id} not allocated"
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

    def get_solution_json(self):
        """
        Get the solution in the json format for the API.
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
                    "x1": package.point1[0],
                    "y1": package.point1[1],
                    "z1": package.point1[2],
                    "x2": package.point2[0],
                    "y2": package.point2[1],
                    "z2": package.point2[2],
                    "height": package.height,
                    "width": package.width,
                    "length": package.length,
                    "priority": package.priority,
                    "delay_cost": package.delay_cost,
                    "weight": package.weight,
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
            "total_cost": cost,
            "delay_cost": cost - num_priority_ulds * self.k_cost,
            "priority_cost": num_priority_ulds * self.k_cost,
            "total_packed_packages": total_packed_packages,
            "total_packed_priority_packages": total_priority_packages,
            "total_packed_economy_packages": total_packed_packages
            - total_priority_packages,
            "num_priority_ulds": num_priority_ulds,
            "total_unpacked_packages": len(self.packages) - total_packed_packages,
            "ulds": ulds,
            "packed_volume": packed_volume,
            "packed_weight": packed_weight,
            "volume_efficiency": (packed_volume / uld_volume),
            "weight_efficiency": (packed_weight / uld_weight),
        }
