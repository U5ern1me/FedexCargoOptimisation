import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gurobipy import GRB, GurobiError
from strategies.strategy import Strategy
from .gurobi_utils import GurobiSolver
import logging

from utils.io import load_config

config = load_config(os.path.join(os.path.dirname(__file__), "gurobi.config"))


class GurobiStrategy(Strategy):
    async def solve(self):
        solver = GurobiSolver(
            packages=self.packages, ulds=self.ulds, k_cost=self.k_cost, config=config
        )
        try:
            solver.model.optimize()

            # Check if the model is infeasible
            if solver.model.Status == GRB.INFEASIBLE:
                logging.error("Model is infeasible. Computing IIS...")
                solver.model.computeIIS()
                solver.model.write(config["IIS file"])

                # Output IIS constraints
                logging.info("Constraints in IIS:")
                for c in solver.model.getConstrs():
                    if c.IISConstr:
                        logging.info(c.ConstrName)

                # Exit or handle accordingly
                raise Exception(
                    f"Model is infeasible. IIS computed and written to {config['IIS file']}."
                )
        except GurobiError as e:
            logging.error(f"Gurobi Error code = {e.errno}")
            logging.error(e.message)
            raise e
        except Exception as e:
            logging.error(f"Exception: {str(e)}")
            raise e

        for i in range(self.n_packages):
            # Check if package i is assigned to any ULD
            for j in range(self.n_ulds):
                # s[i, j] is not exactly 1, but close to 1
                if solver.s[i, j].X > 0.5:
                    bottom_left_front_x = round(solver.x[i].X, 0)
                    bottom_left_front_y = round(solver.y[i].X, 0)
                    bottom_left_front_z = round(solver.z[i].X, 0)

                    self.packages[i].point1 = (
                        bottom_left_front_x,
                        bottom_left_front_y,
                        bottom_left_front_z,
                    )

                    top_right_back_x = round(
                        bottom_left_front_x + solver.get_x_span(i), 0
                    )
                    top_right_back_y = round(
                        bottom_left_front_y + solver.get_y_span(i), 0
                    )
                    top_right_back_z = round(
                        bottom_left_front_z + solver.get_z_span(i), 0
                    )
                    self.packages[i].point2 = (
                        top_right_back_x,
                        top_right_back_y,
                        top_right_back_z,
                    )

                    self.packages[i].uld_id = solver.ulds[j].id
                    break
