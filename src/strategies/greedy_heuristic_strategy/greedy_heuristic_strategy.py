import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from solvers import solvers
from .greedy_heuristic_utils import *
from utils.io import load_config

import logging
import asyncio

config = load_config(os.path.join(os.path.dirname(__file__), "greedy_heuristic.config"))


class GreedyHeuristicStrategy(Strategy):
    async def solve(self):
        heuristics = [sorting_heuristic_1, sorting_heuristic_2, sorting_heuristic_3]
        heuristic_combinations = [(h1, h2) for h1 in heuristics for h2 in heuristics]

        tasks = [
            main_solving_wrapper(
                packages=self.packages,
                ulds=self.ulds,
                k_cost=self.k_cost,
                sorting_heuristic_1=h1,
                sorting_heuristic_2=h2,
                verbose=self.debug,
            )
            for (h1, h2) in heuristic_combinations
        ]

        if self.debug:
            logging.info(
                f"Running {len(tasks)} tasks to find best heuristic combination"
            )

        results = await asyncio.gather(*tasks)

        best_cost, best_packages, best_ulds = min(results, key=lambda x: x[0])

        if self.debug:
            logging.info(f"Best cost: {best_cost}")

        for package_num in range(len(best_packages)):
            if best_packages[package_num].uld_id is not None:
                self.packages[package_num].uld_id = best_packages[package_num].uld_id
                self.packages[package_num].point1 = best_packages[package_num].point1
                self.packages[package_num].point2 = best_packages[package_num].point2
