from abc import ABC, abstractmethod
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from strategies.strategy import Strategy

class MLPStrategy(Strategy):

    def __init__(self,k_cost,ulds,packages):
        super().__init__(k_cost,ulds,packages)
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)    # Suppress Gurobi output
        self.env.setParam('NumericFocus', 3)  # Enhance numerical focus
        self.env.setParam('Presolve', 2)      # Aggressive presolve
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("LicenseID", 2584853)
        self.model.setParam("LogFile", "gurobi.log")
        self.n_packages = len(packages)
        self.n_ulds = len(ulds)
        self.M = 10000

    def set_variables(self):

        self.x = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name='x')
        self.y = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name='y')
        self.z = self.model.addVars(self.n_packages, vtype=GRB.INTEGER, name='z')

        self.lx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='lx')
        self.ly = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='ly')
        self.lz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='lz')

        self.wx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='wx')
        self.wy = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='wy')
        self.wz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='wz')

        self.hx = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='hx')
        self.hy = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='hy')
        self.hz = self.model.addVars(self.n_packages, vtype=GRB.BINARY, name='hz')

        # a, b, c, d, e, f are 2D arrays
        self.a = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='a')
        self.b = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='b')
        self.c = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='c')
        self.d = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='d')
        self.e = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='e')
        self.f = self.model.addVars(self.n_packages, self.n_packages, vtype=GRB.BINARY, name='f')

        # s[i][j] indicates bin i is assigned to container j
        self.s = self.model.addVars(self.n_packages, self.n_ulds, vtype=GRB.BINARY, name='s')

        # Indicator[j] indicates whether any priority bin is placed in container j
        self.Indicator = self.model.addVars(self.n_ulds, vtype=GRB.BINARY, name='Indicator')
        
    def set_constraints(self):
        M = self.M

        for i in range(self.n_packages):
            for k in range(i+1, self.n_packages):
                # Overlap constraints in x direction
                self.model.addConstr(self.x[i] + self.bins[i].l * self.lx[i] + self.bins[i].w * self.wx[i] + self.bins[i].h * self.hx[i] - self.x[k] - (1 - self.a[i,k]) * M <= 0, name=f'x_overlap_1_{i}_{k}')
                self.model.addConstr(self.x[k] + self.bins[k].l * self.lx[k] + self.bins[k].w * self.wx[k] + self.bins[k].h * self.hx[k] - self.x[i] - (1 - self.b[i,k]) * M <= 0, name=f'x_overlap_2_{i}_{k}')
                # Overlap constraints in y direction
                self.model.addConstr(self.y[i] + self.bins[i].l * self.ly[i] + self.bins[i].w * self.wy[i] + self.bins[i].h * self.hy[i] - self.y[k] - (1 - self.c[i,k]) * M <= 0, name=f'y_overlap_1_{i}_{k}')
                self.model.addConstr(self.y[k] + self.bins[k].l * self.ly[k] + self.bins[k].w * self.wy[k] + self.bins[k].h * self.hy[k] - self.y[i] - (1 - self.d[i,k]) * M <= 0, name=f'y_overlap_2_{i}_{k}')
                # Overlap constraints in z direction
                self.model.addConstr(self.z[i] + self.bins[i].l * self.lz[i] + self.bins[i].w * self.wz[i] + self.bins[i].h * self.hz[i] - self.z[k] - (1 - self.e[i,k]) * M <= 0, name=f'z_overlap_1_{i}_{k}')
                self.model.addConstr(self.z[k] + self.bins[k].l * self.lz[k] + self.bins[k].w * self.wz[k] + self.bins[k].h * self.hz[k] - self.z[i] - (1 - self.f[i,k]) * M <= 0, name=f'z_overlap_2_{i}_{k}')

                for j in range(self.n_ulds):
                    self.model.addConstr(self.a[i,k] + self.b[i,k] + self.c[i,k] + self.d[i,k] + self.e[i,k] + self.f[i,k] + 1 - self.s[i,j] - self.s[k,j] >= 0, name=f'overlap_{i}_{j}_{k}')

        for i in range(self.n_packages):
            expr = gp.LinExpr()
            for j in range(self.n_ulds):
                expr += self.s[i,j]
            if self.bins[i].priority:
                self.model.addConstr(expr == 1, name=f'priority_{i}')
            else:
                self.model.addConstr(expr <= 1, name=f'economy_{i}')

            for j in range(self.n_ulds):
                # Orientation constraints
                self.model.addConstr(self.x[i] + self.bins[i].l * self.lx[i] + self.bins[i].w * self.wx[i] + self.bins[i].h * self.hx[i] - self.containers[j].l - (1 - self.s[i,j]) * M <= 0, name=f'orientation_x_{i}_{j}')
                self.model.addConstr(self.y[i] + self.bins[i].l * self.ly[i] + self.bins[i].w * self.wy[i] + self.bins[i].h * self.hy[i] - self.containers[j].w - (1 - self.s[i,j]) * M <= 0, name=f'orientation_y_{i}_{j}')
                self.model.addConstr(self.z[i] + self.bins[i].l * self.lz[i] + self.bins[i].w * self.wz[i] + self.bins[i].h * self.hz[i] - self.containers[j].h - (1 - self.s[i,j]) * M <= 0, name=f'orientation_z_{i}_{j}')

            # Orientation selection constraints
            self.model.addConstr(self.lx[i] + self.wx[i] + self.hx[i] == 1, name=f'lx_wx_hx_{i}')
            self.model.addConstr(self.ly[i] + self.wy[i] + self.hy[i] == 1, name=f'ly_wy_hy_{i}')
            self.model.addConstr(self.lz[i] + self.wz[i] + self.hz[i] == 1, name=f'lz_wz_hz_{i}')
            self.model.addConstr(self.lx[i] + self.ly[i] + self.lz[i] == 1, name=f'lx_ly_lz_{i}')
            self.model.addConstr(self.wx[i] + self.wy[i] + self.wz[i] == 1, name=f'wx_wy_wz_{i}')
            self.model.addConstr(self.hx[i] + self.hy[i] + self.hz[i] == 1, name=f'hx_hy_hz_{i}')

        for j in range(self.n_ulds):
            # Capacity constraints
            expr = gp.LinExpr()
            for i in range(self.n_packages):
                expr += self.s[i,j] * self.bins[i].weight
            self.model.addConstr(expr <= self.containers[j].capacity, name=f'capacity_{j}')

            expr = gp.LinExpr()
            for i in range(self.n_packages):
                if self.bins[i].priority:
                    expr += self.s[i,j]
            self.model.addConstr(expr <= M * self.Indicator[j], name=f'priority_cost_{j}')

    def set_objective(self):
        expr = gp.LinExpr()
        for j in range(self.n_ulds):
            expr += self.Indicator[j] * self.k
        for i in range(self.n_packages):
            sum_expr = 1
            for j in range(self.n_ulds):
                sum_expr -= self.s[i,j]
            expr += self.bins[i].delay_cost * sum_expr
        self.model.setObjective(expr, GRB.MINIMIZE)

    def solve(self):

        try:
            self.model.optimize()

            # Check if the model is infeasible
            if self.model.Status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                self.model.computeIIS()
                self.model.write("model.ilp")

                # Output IIS constraints
                print("Constraints in IIS:")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(c.ConstrName)

                # Exit or handle accordingly
                raise Exception("Model is infeasible. IIS computed and written to model.ilp.")
        except gp.GurobiError as e:
            print(f"Gurobi Error code = {e.errno}")
            print(e.message)
            exit(1)
        except Exception as e:
            print(f"Exception: {str(e)}")
            exit(1)

        for i in range(self.n_packages):
            for j in range(self.n_ulds):
                if self.s[i,j].X > 0.5:
                    bottom_left_front_x = int(self.x[i].X)
                    bottom_left_front_y = int(self.y[i].X)
                    bottom_left_front_z = int(self.z[i].X)
                    self.packages[i].point1 = (bottom_left_front_x, bottom_left_front_y, bottom_left_front_z)
                    
                    top_right_back_x = int(bottom_left_front_x + self.packages[i].length + self.lx[i].X + self.packages[i].width * self.wx[i].X + self.packages[i].height * self.hx[i].X)
                    top_right_back_y = int(bottom_left_front_y + self.packages[i].length + self.ly[i].X + self.packages[i].width * self.wy[i].X + self.packages[i].height * self.hy[i].X)
                    top_right_back_z = int(bottom_left_front_z + self.packages[i].length + self.lz[i].X + self.packages[i].width * self.wz[i].X + self.packages[i].height * self.hz[i].X)
                    self.packages[i].point2 = (top_right_back_x, top_right_back_y, top_right_back_z)
                    
                    self.packages[i].uld_id = "U"+str(j+1)
                    break

    

        