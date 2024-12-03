import copy
from models.uld import ULD

# typing
from typing import List, Any
from models.package import Package
from hexaly.optimizer import HxModel


class HexalySolver:
    def __init__(
        self,
        Package_data: List[Package],
        container_data: List[ULD],
        k: float,
        model: HxModel,
    ):
        """
        Args:
            Package_data: List of packages
            container_data: List of ULDs
            k: Cost of priority ULDs
            model: Hexaly model
        """
        self.Package_data = copy.deepcopy(Package_data)
        self.container_data = copy.deepcopy(container_data)
        self.K = k
        self.M = 1000000

        self.max_package_length = max([package.length for package in self.Package_data])
        self.max_package_width = max([package.width for package in self.Package_data])
        self.max_package_height = max([package.height for package in self.Package_data])

        # Buffer container for excluded packages
        self.__buffer_container = ULD(
            "Buffer", self.M, self.M, self.M, self.M
        )  # ["Buffer",self.M, self.M, self.M, self.M,[]]
        self.container_data.append(self.__buffer_container)

        # initialize the model
        self.model = model

        # Initialize variables
        self.set_variables()

        # Set constraints
        self.set_constraints()

        # Set objective
        self.set_objective()

    def set_variables(self):
        """
        Variables:
        x, y, z: 1D array denoting the bottom left front corner of the package
        lx, ly, lz: 1D array denoting the orientation of the package
        wx, wy, wz: 1D array denoting the orientation of the package
        hx, hy, hz: 1D array denoting the orientation of the package
        container_content: 1D array of sets denoting the content of the container
        """
        n_packages = len(self.Package_data)
        n_containers = len(self.container_data)

        # Container content assignment
        self.container_content = [
            self.model.set(n_packages) for _ in range(n_containers)
        ]
        self.Priority_packages = self.model.array(
            [i for i, j in enumerate(self.Package_data) if j.priority]
        )

        # Package positioning variables
        self.x = [self.model.int(0, self.max_package_length) for _ in range(n_packages)]
        self.y = [self.model.int(0, self.max_package_width) for _ in range(n_packages)]
        self.z = [self.model.int(0, self.max_package_height) for _ in range(n_packages)]

        # Enhanced orientation variables --> (where lx[i] denotes whether length is assumed in x direction or not for the ith package)
        self.lx = [self.model.bool() for _ in range(n_packages)]
        self.ly = [self.model.bool() for _ in range(n_packages)]
        self.lz = [self.model.bool() for _ in range(n_packages)]
        self.wx = [self.model.bool() for _ in range(n_packages)]
        self.wy = [self.model.bool() for _ in range(n_packages)]
        self.wz = [self.model.bool() for _ in range(n_packages)]
        self.hx = [self.model.bool() for _ in range(n_packages)]
        self.hy = [self.model.bool() for _ in range(n_packages)]
        self.hz = [self.model.bool() for _ in range(n_packages)]

        # Package-to-container assignment variable
        self.s = [
            [self.model.bool() for _ in range(n_containers)] for _ in range(n_packages)
        ]

        # Overlap and relative positioning flags
        self.a = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]
        self.b = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]
        self.c = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]
        self.d = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]
        self.e = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]
        self.f = [
            [self.model.bool() for _ in range(n_packages)] for _ in range(n_packages)
        ]

        self.used_l = [self.model.int(0, 100000) for _ in range(n_packages)]
        self.used_w = [self.model.int(0, 100000) for _ in range(n_packages)]
        self.used_h = [self.model.int(0, 100000) for _ in range(n_packages)]

    def set_constraints(self):
        """
        Constraints:
        - Unique package assignment constraint
        - Overlap and positioning constraints
        - Package assignment constraints
        - Container boundary constraints
        - Orientation selection constraints
        - Weight constraints
        """
        n_packages = len(self.Package_data)
        n_containers = len(self.container_data)

        # Unique package assignment constraint
        self.model.constraint(self.model.partition(self.container_content))

        # overlap and positioning constraints
        for i in range(n_packages):
            for k in range(i + 1, n_packages):

                # x-direction overlap
                x1 = (
                    self.x[i]
                    + self.Package_data[i].length * self.lx[i]
                    + self.Package_data[i].width * self.wx[i]
                    + self.Package_data[i].height * self.hx[i]
                    - self.x[k]
                    - (1 - self.a[i][k]) * self.M
                )
                x2 = (
                    self.x[k]
                    + self.Package_data[k].length * self.lx[k]
                    + self.Package_data[k].width * self.wx[k]
                    + self.Package_data[k].height * self.hx[k]
                    - self.x[i]
                    - (1 - self.b[i][k]) * self.M
                )

                self.model.constraint(x1 <= 0)
                self.model.constraint(x2 <= 0)

                # y-direction overlap
                y1 = (
                    self.y[i]
                    + self.Package_data[i].length * self.ly[i]
                    + self.Package_data[i].width * self.wy[i]
                    + self.Package_data[i].height * self.hy[i]
                    - self.y[k]
                    - (1 - self.c[i][k]) * self.M
                )
                y2 = (
                    self.y[k]
                    + self.Package_data[k].length * self.ly[k]
                    + self.Package_data[k].width * self.wy[k]
                    + self.Package_data[k].height * self.hy[k]
                    - self.y[i]
                    - (1 - self.d[i][k]) * self.M
                )

                self.model.constraint(y1 <= 0)
                self.model.constraint(y2 <= 0)

                # z-direction overlap
                z1 = (
                    self.z[i]
                    + self.Package_data[i].length * self.lz[i]
                    + self.Package_data[i].width * self.wz[i]
                    + self.Package_data[i].height * self.hz[i]
                    - self.z[k]
                    - (1 - self.e[i][k]) * self.M
                )
                z2 = (
                    self.z[k]
                    + self.Package_data[k].length * self.lz[k]
                    + self.Package_data[k].width * self.wz[k]
                    + self.Package_data[k].height * self.hz[k]
                    - self.z[i]
                    - (1 - self.f[i][k]) * self.M
                )

                self.model.constraint(z1 <= 0)
                self.model.constraint(z2 <= 0)

                # for packages in the same container
                for j in range(n_containers):
                    overlapping_condition = (
                        self.a[i][k]
                        + self.b[i][k]
                        + self.c[i][k]
                        + self.d[i][k]
                        + self.e[i][k]
                        + self.f[i][k]
                        + 1
                        - self.model.contains(self.container_content[j], i)
                        - self.model.contains(self.container_content[j], k)
                    )
                    self.model.constraint(overlapping_condition >= 0)


            # Container boundary constraints

            for j in range(n_containers):
                container = self.container_data[j]
                # along container's length
                self.l1 = (
                    self.x[i]
                    + self.Package_data[i].length * self.lx[i]
                    + self.Package_data[i].width * self.wx[i]
                    + self.Package_data[i].height * self.hx[i]
                    - container.length
                ) - (1 - self.model.contains(self.container_content[j], i)) * self.M
                self.model.constraint(self.l1 <= 0)
                # along container's width
                self.l2 = (
                    self.y[i]
                    + self.Package_data[i].length * self.ly[i]
                    + self.Package_data[i].width * self.wy[i]
                    + self.Package_data[i].height * self.hy[i]
                    - container.width
                ) - (1 - self.model.contains(self.container_content[j], i)) * self.M
                self.model.constraint(self.l2 <= 0)
                # along container's Height
                self.l3 = (
                    self.z[i]
                    + self.Package_data[i].length * self.lz[i]
                    + self.Package_data[i].width * self.wz[i]
                    + self.Package_data[i].height * self.hz[i]
                    - container.height
                ) - (1 - self.model.contains(self.container_content[j], i)) * self.M
                self.model.constraint(self.l3 <= 0)

            # Orientation selection constraints
            self.model.constraint(
                self.model.sum(self.model.array([self.lx[i], self.wx[i], self.hx[i]]))
                == 1
            )
            self.model.constraint(
                self.model.sum(self.model.array([self.ly[i], self.wy[i], self.hy[i]]))
                == 1
            )
            self.model.constraint(
                self.model.sum(self.model.array([self.lz[i], self.wz[i], self.hz[i]]))
                == 1
            )
            self.model.constraint(
                self.model.sum(self.model.array([self.lx[i], self.ly[i], self.lz[i]]))
                == 1
            )
            self.model.constraint(
                self.model.sum(self.model.array([self.wx[i], self.wy[i], self.wz[i]]))
                == 1
            )
            self.model.constraint(
                self.model.sum(self.model.array([self.hx[i], self.hy[i], self.hz[i]]))
                == 1
            )

        weights = self.model.array([package.weight for package in self.Package_data])
        weight_lambda = self.model.lambda_function(lambda i: weights[i])

        # Weight constraints for each container
        container_weights = [
            self.model.sum(container, weight_lambda)
            for container in self.container_content
        ]

        for i in range(n_containers):
            self.model.constraint(
                container_weights[i] <= self.container_data[i].weight_limit
            )

        # calculating used dimension in each direction
        for i in range(n_packages):
            self.used_l[i] = (
                self.Package_data[i].length * self.lx[i]
                + self.Package_data[i].width * self.wx[i]
                + self.Package_data[i].height * self.hx[i]
            )
            self.used_w[i] = (
                self.Package_data[i].length * self.ly[i]
                + self.Package_data[i].width * self.wy[i]
                + self.Package_data[i].height * self.hy[i]
            )
            self.used_h[i] = (
                self.Package_data[i].length * self.lz[i]
                + self.Package_data[i].width * self.wz[i]
                + self.Package_data[i].height * self.hz[i]
            )

    def set_objective(self):
        """
        Objective:
        - Minimize the cost of the packages in the last container (buffer container)
        - Minimize the spread of priority packages
        """
        lost_cost = self.model.array(
            [package.delay_cost for package in self.Package_data]
        )
        lost_cost_lambda = self.model.lambda_function(lambda i: lost_cost[i])
        lost_cost_sum = self.model.sum(self.container_content[-1], lost_cost_lambda)

        has_priority_package = [
            self.model.count(self.model.intersection(self.Priority_packages, container))
            > 0
            for container in self.container_content
        ]
        priority_spread_cost = self.model.sum(has_priority_package)

        # Objective combines minimizing packages in the last container i.e. lost cost, and priority spread
        self.objective = self.K * priority_spread_cost + lost_cost_sum
        self.model.minimize(self.objective)
