import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from .genetic_3D_bin_packing_utils import Genetic3DBinPacking

from utils.io import load_config

config = load_config(
    os.path.join(os.path.dirname(__file__), "genetic_3D_bin_packing.config")
)


class Genetic3DBinPackingStrategy(Strategy):
    async def solve(self):

        inputs = {
            "packages": self.packages,
            "ulds": self.ulds,
            "k_cost": self.k_cost,
        }

        model = Genetic3DBinPacking(
            inputs,
            num_generations=config["number of generations"],
            num_individuals=config["number of individuals"],
            mutation_bracket_size=config["mutation bracket size"],
            solver=config["solver"],
            eliteCProb=config["probability of choosing elite gene"],
        )

        if self.debug:
            logging.info(f"Solver: {config['solver']}")

        await model.fit(config["number of stable generations"], verbose=True)
