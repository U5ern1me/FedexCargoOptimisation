import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .brkga_utils import BRKGA
from strategies.strategy import Strategy

from utils.io import load_config

config = load_config(os.path.join(os.path.dirname(__file__), "brkga.config"))


class BRKGAStrategy(Strategy):
    def solve(self):

        inputs = {"packages": self.packages, "ulds": self.ulds, "k": self.k_cost}

        model = BRKGA(
            inputs,
            num_generations=config["number of generations"],
            num_individuals=config["number of individuals"],
            num_elites=config["number of elites"],
            num_mutants=config["number of mutants"],
            fraction_biased=config["fraction of biased population"],
            eliteCProb=config["probability of choosing elite gene"],
            multiProcess=True,
        )
        model.fit(patient=config["number of stable generations"], verbose=False)
        placement_decoder = model.get_placement()

        fitness_score = placement_decoder.evaluate()

        for uld in placement_decoder.Bins:
            for _package in uld.boxes:
                package = next((p for p in self.packages if p.id == _package.id), None)
                package.uld_id = uld.id
                package.point1 = _package.point1
                package.point2 = _package.point2
