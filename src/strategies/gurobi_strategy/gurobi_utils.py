import gurobipy as gp
from gurobipy import GRB

# typing
from typing import List, Any, Dict
from models.package import Package
from models.uld import ULD


class GurobiSolver:
    def __init__(
        self,
        packages: List[Package],
        ulds: List[ULD],
        k_cost: float,
        config: Dict[str, Any],
    ):
        """
        Args:
            packages: List of packages
            ulds: List of ULDs
            k_cost: Cost of priority ULDs
            config: Configuration for Gurobi
        """
        self.packages = packages
        self.ulds = ulds
        self.k_cost = k_cost
        self.env = gp.Env(empty=True)
        self.env.setParam(
            "OutputFlag", config["output flag"]
        )  # Suppress Gurobi output if flag = 0
        self.env.setParam(
            "NumericFocus", config["numeric focus"]
        )  # Enhance numerical focus
        self.env.setParam(
            "Presolve", config["presolve"]
        )  # Aggressive presolve for config["presolve"] = 3
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("LicenseID", config["license"])
        self.model.setParam("LogFile", config["log file"])
        self.n_packages = len(packages)
        self.n_ulds = len(ulds)
        self.M = 10000

        # initialize variables
        self.set_variables()

        # set constraints
        self.set_constraints()

        # set objective
        self.set_objective()

    def set_variables(self):
        """
        Variables:
        x, y, z: 1D array denoting the bottom left front corner of the package
        lx, ly, lz: 1D array denoting the orientation of the package
        wx, wy, wz: 1D array denoting the orientation of the package
        hx, hy, hz: 1D array denoting the orientation of the package
        """
        self.x = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name="x")
        self.y = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name="y")
        self.z = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name="z")

        self.lx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="lx")
        self.ly = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="ly")
        self.lz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="lz")

        self.wx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="wx")
        self.wy = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="wy")
        self.wz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="wz")

        self.hx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="hx")
        self.hy = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="hy")
        self.hz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name="hz")

        # a, b, c, d, e, f are 2D arrays denoting the overlap direction between packages
        self.a = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="a"
        )
        self.b = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="b"
        )
        self.c = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="c"
        )
        self.d = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="d"
        )
        self.e = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="e"
        )
        self.f = self.model.addVars(
            self.n_packages, self.n_packages, vtype=GRB.BINARY, name="f"
        )

        # s[i][j] indicates package i is assigned to ULD j
        self.s = self.model.addVars(
            self.n_packages, self.n_ulds, vtype=GRB.BINARY, name="s"
        )

        # Indicator[j] indicates whether any priority package is placed in ULD j
        self.Indicator = self.model.addVars(
            self.n_ulds, vtype=GRB.BINARY, name="Indicator"
        )

    def set_constraints(self):
        """
        Constraints:
        - Overlap constraints between packages
        - Packages and ULDs bound constraints
        - Priority in ULD constraint
        - Orientation selection constraints
        - Capacity constraints
        """
        for i in range(self.n_packages):
            for k in range(i + 1, self.n_packages):
                # Overlap constraints in x direction
                self.model.addConstr(
                    self.x[i]
                    + self.packages[i].length * self.lx[i]
                    + self.packages[i].width * self.wx[i]
                    + self.packages[i].height * self.hx[i]
                    - self.x[k]
                    - (1 - self.a[i, k]) * self.M
                    <= 0,
                    name=f"x_overlap_1_{i}_{k}",
                )
                self.model.addConstr(
                    self.x[k]
                    + self.packages[k].length * self.lx[k]
                    + self.packages[k].width * self.wx[k]
                    + self.packages[k].height * self.hx[k]
                    - self.x[i]
                    - (1 - self.b[i, k]) * self.M
                    <= 0,
                    name=f"x_overlap_2_{i}_{k}",
                )
                # Overlap constraints in y direction
                self.model.addConstr(
                    self.y[i]
                    + self.packages[i].length * self.ly[i]
                    + self.packages[i].width * self.wy[i]
                    + self.packages[i].height * self.hy[i]
                    - self.y[k]
                    - (1 - self.c[i, k]) * self.M
                    <= 0,
                    name=f"y_overlap_1_{i}_{k}",
                )
                self.model.addConstr(
                    self.y[k]
                    + self.packages[k].length * self.ly[k]
                    + self.packages[k].width * self.wy[k]
                    + self.packages[k].height * self.hy[k]
                    - self.y[i]
                    - (1 - self.d[i, k]) * self.M
                    <= 0,
                    name=f"y_overlap_2_{i}_{k}",
                )
                # Overlap constraints in z direction
                self.model.addConstr(
                    self.z[i]
                    + self.packages[i].length * self.lz[i]
                    + self.packages[i].width * self.wz[i]
                    + self.packages[i].height * self.hz[i]
                    - self.z[k]
                    - (1 - self.e[i, k]) * self.M
                    <= 0,
                    name=f"z_overlap_1_{i}_{k}",
                )
                self.model.addConstr(
                    self.z[k]
                    + self.packages[k].length * self.lz[k]
                    + self.packages[k].width * self.wz[k]
                    + self.packages[k].height * self.hz[k]
                    - self.z[i]
                    - (1 - self.f[i, k]) * self.M
                    <= 0,
                    name=f"z_overlap_2_{i}_{k}",
                )

                for j in range(self.n_ulds):
                    # Overlap constraints in ULD j
                    self.model.addConstr(
                        self.a[i, k]
                        + self.b[i, k]
                        + self.c[i, k]
                        + self.d[i, k]
                        + self.e[i, k]
                        + self.f[i, k]
                        + 1
                        - self.s[i, j]
                        - self.s[k, j]
                        >= 0,
                        name=f"overlap_{i}_{j}_{k}",
                    )

        for i in range(self.n_packages):
            expr = gp.LinExpr()
            for j in range(self.n_ulds):
                expr += self.s[i, j]
            if self.packages[i].priority:
                self.model.addConstr(expr == 1, name=f"priority_{i}")
            else:
                self.model.addConstr(expr <= 1, name=f"economy_{i}")

            for j in range(self.n_ulds):
                # Orientation constraints
                self.model.addConstr(
                    self.x[i]
                    + self.packages[i].length * self.lx[i]
                    + self.packages[i].width * self.wx[i]
                    + self.packages[i].height * self.hx[i]
                    - self.ulds[j].length
                    - (1 - self.s[i, j]) * self.M
                    <= 0,
                    name=f"orientation_x_{i}_{j}",
                )
                self.model.addConstr(
                    self.y[i]
                    + self.packages[i].length * self.ly[i]
                    + self.packages[i].width * self.wy[i]
                    + self.packages[i].height * self.hy[i]
                    - self.ulds[j].width
                    - (1 - self.s[i, j]) * self.M
                    <= 0,
                    name=f"orientation_y_{i}_{j}",
                )
                self.model.addConstr(
                    self.z[i]
                    + self.packages[i].length * self.lz[i]
                    + self.packages[i].width * self.wz[i]
                    + self.packages[i].height * self.hz[i]
                    - self.ulds[j].height
                    - (1 - self.s[i, j]) * self.M
                    <= 0,
                    name=f"orientation_z_{i}_{j}",
                )

            # Orientation selection constraints
            self.model.addConstr(
                self.lx[i] + self.wx[i] + self.hx[i] == 1, name=f"lx_wx_hx_{i}"
            )
            self.model.addConstr(
                self.ly[i] + self.wy[i] + self.hy[i] == 1, name=f"ly_wy_hy_{i}"
            )
            self.model.addConstr(
                self.lz[i] + self.wz[i] + self.hz[i] == 1, name=f"lz_wz_hz_{i}"
            )
            self.model.addConstr(
                self.lx[i] + self.ly[i] + self.lz[i] == 1, name=f"lx_ly_lz_{i}"
            )
            self.model.addConstr(
                self.wx[i] + self.wy[i] + self.wz[i] == 1, name=f"wx_wy_wz_{i}"
            )
            self.model.addConstr(
                self.hx[i] + self.hy[i] + self.hz[i] == 1, name=f"hx_hy_hz_{i}"
            )

        for j in range(self.n_ulds):
            # Capacity constraints
            expr = gp.LinExpr()
            for i in range(self.n_packages):
                expr += self.s[i, j] * self.packages[i].weight
            self.model.addConstr(
                expr <= self.ulds[j].weight_limit, name=f"capacity_{j}"
            )
            # priority in uld constraint
            expr = gp.LinExpr()
            for i in range(self.n_packages):
                if self.packages[i].priority:
                    expr += self.s[i, j]
            self.model.addConstr(
                expr <= self.M * self.Indicator[j], name=f"priority_cost_{j}"
            )

    def set_objective(self):
        """
        Objective function:
        min sum(Indicator[j] * k_cost) + sum((1 - sum(s[i, j])) * packages[i].delay_cost)
        """
        expr = gp.LinExpr()
        for j in range(self.n_ulds):
            expr += self.Indicator[j] * self.k_cost
        for i in range(self.n_packages):
            sum_expr = 1
            for j in range(self.n_ulds):
                sum_expr -= self.s[i, j]
            expr += self.packages[i].delay_cost * sum_expr
        self.model.setObjective(expr, GRB.MINIMIZE)

    def get_x_span(self, i):
        """
        Get the x span of package i
        """
        return (
            self.packages[i].length
            + self.lx[i].X
            + self.packages[i].width * self.wx[i].X
            + self.packages[i].height * self.hx[i].X
        )

    def get_y_span(self, i):
        """
        Get the y span of package i
        """
        return (
            self.packages[i].length
            + self.ly[i].X
            + self.packages[i].width * self.wy[i].X
            + self.packages[i].height * self.hy[i].X
        )

    def get_z_span(self, i):
        """
        Get the z span of package i
        """
        return (
            self.packages[i].length
            + self.lz[i].X
            + self.packages[i].width * self.wz[i].X
            + self.packages[i].height * self.hz[i].X
        )
