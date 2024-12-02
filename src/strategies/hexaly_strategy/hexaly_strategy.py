import os
import sys
from utils.io import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from .hexaly_utils import *
import pandas as pd

config = load_config(os.path.join(os.path.dirname(__file__),"hexaly.config"))

class HexalyStrategy(Strategy):
    def solve(self):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            model = optimizer.model
            try:
                optimizer.param.time_limit = config["total_timesteps"]
                solver = Solver(self.packages, self.ulds, self.k_cost, model=model)
            
                model.close()

                optimizer.solve()

                for i in range(len(self.packages)):
                    for j in range(len(self.ulds)):
                        if i in solver.container_content[j]:
                            self.packages[i].uld_id = self.ulds[i].id

                for i in range(len(self.packages)):
                    self.packages[i].point1 = (solver.x[i].value, solver.y[i].value, solver.z[i].value)
                    self.packages[i].point2 = (solver.x[i].value+solver.used_l[i].value, solver.y[i].value+solver.used_w[i].value, solver.z[i].value+solver.used_h[i].value)
 
            except Exception as e:
                print("Packing failed")
                print(e)



