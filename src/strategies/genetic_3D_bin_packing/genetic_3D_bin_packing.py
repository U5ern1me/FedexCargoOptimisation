import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from .genetic_3D_bin_packing_utils import Genetic3DBinPacking

from utils.io import load_config

config = load_config(os.path.join(os.path.dirname(__file__), "genetic_3D_bin_packing.config"))


class Genetic3DBinPackingStrategy(Strategy):
    async def solve(self):
        
        inputs = {"packages": self.packages, "ulds": self.ulds}
        
        model = Genetic3DBinPacking(
            inputs,
            uld_map_for_priority=config["uld map for priority"],
            num_generations=config["number of generations"],
            num_individuals=config["number of individuals"],
            mutation_bracket_size=config["mutation bracket size"],
            min_1=config["min 1"],
            max_1=config["max 1"],
            min_2=config["min 2"],
            max_2=config["max 2"],
            eliteCProb=config["probability of choosing elite gene"],
        )
        
        await model.fit(config["number of stable generations"], verbose=True)
        