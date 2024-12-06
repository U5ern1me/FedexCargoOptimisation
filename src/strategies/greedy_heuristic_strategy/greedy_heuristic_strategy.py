import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.strategy import Strategy
from solvers import solvers
from .greedy_heuristic_utils import *
from utils.io import load_config

import aiohttp
import logging
import asyncio

config = load_config(os.path.join(os.path.dirname(__file__), "greedy_heuristic.config"))


class GreedyHeuristicStrategy(Strategy):
    async def solve(self):
        average_package_density = sum(
            package.weight / (package.length * package.width * package.height)
            for package in self.packages
        ) / len(self.packages)

        average_uld_density = sum(
            uld.weight_limit / (uld.length * uld.width * uld.height)
            for uld in self.ulds
        ) / len(self.ulds)

        if self.debug:
            logging.info(f"Average package density: {average_package_density}")
            logging.info(f"Average ULD density: {average_uld_density}")

        sorting_heuristic_1, sorting_heuristic_2 = get_sorting_heuristics(
            average_package_density, average_uld_density
        )

        # initialize solver
        solver = solvers[os.environ.get("SOLVER", config["solver"])]

        if self.debug:
            logging.info(f"using solver: {solver.__name__}")
        # divide packages into priority and economic
        priority_packages, economic_packages = get_divide_into_priority_and_economic(
            self.packages,
        )

        if self.debug:
            logging.info(f"Priority packages: {len(priority_packages)}")
            logging.info(f"Economic packages: {len(economic_packages)}")
            logging.info(f"Sorting heuristic 1: {sorting_heuristic_1.__name__}")
            logging.info(f"Sorting heuristic 2: {sorting_heuristic_2.__name__}")

        # sort economic packages by delay cost per kg
        sorted_economic_packages = sort_packages(economic_packages, sorting_heuristic_1)

        # get all possible splits of ulds
        uld_splits_arr = get_all_division_of_ulds(self.ulds, 1)

        if self.debug:
            logging.info(f"{uld_splits_arr}")

        # initialize best split value and packages
        best_split_value = float("inf")
        best_split_packages = (0, 0)
        best_split_uld_splits = (None, None)

        tasks = []
        results = []
        task_threshold = 5
        best_cost = float("inf")

        for num_p_uld, uld_splits in enumerate(uld_splits_arr):
            if self.debug:
                logging.info(f"Checking ULD split {num_p_uld} of {len(uld_splits_arr)}")

            if best_cost < num_p_uld * self.k_cost:
                continue

            tasks.append(
                find_best_splits_economic_packages(
                    economic_packages=sorted_economic_packages,
                    priority_packages=priority_packages,
                    ulds=self.ulds,
                    uld_splits=uld_splits,
                    solver=solver,
                    sorting_heuristic=sorting_heuristic_1,
                    k_cost=self.k_cost,
                    num_p_uld=num_p_uld,
                    verbose=self.debug,
                )
            )

            if len(tasks) >= task_threshold:
                _results = await asyncio.gather(*tasks)
                results.extend(_results)
                _best_result = min(results, key=lambda x: x[0])
                best_cost = _best_result[0]
                tasks = []

        if len(tasks) > 0:
            _results = await asyncio.gather(*tasks)
            results.extend(_results)

        best_result = min(results, key=lambda x: x[0])

        best_split_value, best_split_packages, best_split_uld_splits = best_result

        if self.debug:
            logging.info(f"Best split value: {best_split_value}")
            logging.info(f"Best split packages: {best_split_packages}")
            logging.info(
                f"Best split ULD splits: {len(best_split_uld_splits[0])}, {len(best_split_uld_splits[1])}"
            )

        # get the final partition of packages
        partition_1 = [
            *priority_packages,
            *sorted_economic_packages[: best_split_packages[0]],
        ]

        sorted_economic_packages = sorted_economic_packages[best_split_packages[0] :]
        partition_2 = sorted_economic_packages[: best_split_packages[1]]

        sorted_economic_packages = sorted_economic_packages[best_split_packages[1] :]

        # get the remaining packages
        remaining_packages = sort_packages(
            sorted_economic_packages, sorting_heuristic_2
        )

        if self.debug:
            logging.info(f"Remaining packages: {len(remaining_packages)}")

        async with aiohttp.ClientSession() as session:
            _remaining_packages = []
            tolerance = min(config["error tuning"], len(remaining_packages))
            for package_num, package in enumerate(remaining_packages[:tolerance]):
                if self.debug:
                    logging.info(
                        f"Checking package {package_num} of {config['error tuning']} packages"
                    )

                if (
                    best_split_uld_splits[0] is not None
                    and len(best_split_uld_splits[0]) > 0
                ):
                    _partition_1 = partition_1 + [package]
                    _solver_1 = solver(
                        ulds=best_split_uld_splits[0],
                        packages=_partition_1,
                    )
                    await _solver_1.solve(only_check_fits=True, session=session)
                    if await _solver_1.get_fit(session=session):
                        partition_1.append(package)
                        continue

                if (
                    best_split_uld_splits[1] is not None
                    and len(best_split_uld_splits[1]) > 0
                ):
                    _partition_2 = partition_2 + [package]
                    _solver_2 = solver(
                        ulds=best_split_uld_splits[1],
                        packages=_partition_2,
                    )
                    await _solver_2.solve(only_check_fits=True, session=session)
                    if await _solver_2.get_fit(session=session):
                        partition_2.append(package)
                        continue

                _remaining_packages.append(package)

        remaining_packages = (
            _remaining_packages + remaining_packages[config["error tuning"] :]
        )

        if self.debug:
            logging.info(f"Remaining packages: {len(remaining_packages)}")

        # solve for both partitions and assign the solution to the strategy (done in the solver)
        # check if the partition and uld split is valid
        async with aiohttp.ClientSession() as session:
            if (
                best_split_uld_splits[0] is not None
                and len(partition_1) > 0
                and len(best_split_uld_splits[0]) > 0
            ):
                solver_1 = solver(
                    ulds=best_split_uld_splits[0],
                    packages=partition_1,
                )
                await solver_1.solve(only_check_fits=False, session=session)
                await solver_1.get_fit(session=session)

            if (
                best_split_uld_splits[1] is not None
                and len(partition_2) > 0
                and len(best_split_uld_splits[1]) > 0
            ):
                solver_2 = solver(
                    ulds=best_split_uld_splits[1],
                    packages=partition_2,
                )
                await solver_2.solve(only_check_fits=False, session=session)
                await solver_2.get_fit(session=session)
