import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from solvers import solvers
from .greedy_heuristic_utils import *

from utils.io import load_config
import logging

config = load_config(os.path.join(os.path.dirname(__file__), "greedy_heuristic.config"))


class GreedyHeuristicStrategy(Strategy):
    async def solve(self):
        # initialize solver and step size
        solver = solvers[config["solver"]]

        # divide packages into priority and economic
        priority_packages, economic_packages = get_divide_into_priority_and_economic(
            self.packages
        )

        # sort economic packages by delay cost per kg
        sorted_economic_packages = sort_packages(economic_packages, sorting_heuristic)

        # get all possible splits of ulds
        uld_splits_arr = get_all_division_of_ulds(self.ulds, 1)

        if self.debug:
            logging.info(f"{uld_splits_arr}")

        # initialize best split value and packages
        best_split_value = self.k_cost * (len(self.ulds) + 1)
        best_split_packages = (0, 0)
        best_split_uld_splits = (None, None)

        for num_p_uld, uld_splits in enumerate(uld_splits_arr):

            if self.debug:
                logging.info(f"Checking {num_p_uld} ULDs")

            # check if priority packages fit in uld splits if not then it is not a valid split
            valid_uld_splits = []
            for uld_split in uld_splits:
                uld_group_1, uld_group_2 = split_ulds_into_two(self.ulds, uld_split)
                if virtual_fit_priority(priority_packages, uld_group_1):
                    valid_uld_splits.append((uld_group_1, uld_group_2))

            # if no valid splits then continue
            if len(valid_uld_splits) == 0:
                continue

            if self.debug:
                logging.info(f"Found {len(valid_uld_splits)} valid ULD splits")

            local_best_split_packages = (0, 0)
            local_best_split_uld_splits = (None, None)

            # find the best split of economic packages for each valid uld split
            for uld_group_1, uld_group_2 in valid_uld_splits:
                splits = await find_splits_economic_packages(
                    sorted_economic_packages,
                    priority_packages,
                    uld_group_1,
                    uld_group_2,
                    solver,
                    verbose=self.debug,
                )

                # if no splits then continue (priority cannot fit in uld group 1)
                if splits is None:
                    continue

                # for a split with more economic packages, update the best split (number of ulds in both uld splits must be same for this)
                if sum(splits) > sum(local_best_split_packages):
                    local_best_split_packages = splits
                    local_best_split_uld_splits = (uld_group_1, uld_group_2)

            if self.debug:
                logging.info(f"Found local best split: {local_best_split_packages}")

            # calculate the cost of the local best split
            delay_cost = sum(
                [
                    package.delay_cost
                    for package in sorted_economic_packages[
                        local_best_split_packages[0] + local_best_split_packages[1] :
                    ]
                ]
            )
            spread_cost = self.k_cost * num_p_uld
            total_cost = delay_cost + spread_cost

            # if the cost of the local best split is less than the best split value, update the best split
            if total_cost < best_split_value:
                best_split_value = total_cost
                best_split_packages = local_best_split_packages
                best_split_uld_splits = local_best_split_uld_splits

            if self.debug:
                logging.info(f"Current best split: {best_split_value}")

        # get the final partition of packages
        partition_1 = [
            *priority_packages,
            *sorted_economic_packages[: best_split_packages[0]],
        ]
        partition_2 = sorted_economic_packages[
            best_split_packages[0] : best_split_packages[0] + best_split_packages[1]
        ]

        remaining_packages = sorted_economic_packages[
            best_split_packages[0] + best_split_packages[1] :
        ]

        # get the remaining packages
        remaining_packages = sort_packages(remaining_packages, sorting_heuristic_2)

        if self.debug:
            logging.info(f"Remaining packages: {len(remaining_packages)}")

        _remaining_packages = []
        for package in remaining_packages[: config["error tuning"]]:
            if len(best_split_uld_splits[0]) > 0:
                _partition_1 = partition_1 + [package]
                _solver_1 = solver(
                    ulds=best_split_uld_splits[0],
                    packages=_partition_1,
                )
                await _solver_1.solve(only_check_fits=True)
                if await _solver_1.get_fit():
                    partition_1.append(package)
                    continue

            if len(best_split_uld_splits[1]) > 0:
                _partition_2 = partition_2 + [package]
                _solver_2 = solver(
                    ulds=best_split_uld_splits[1],
                    packages=_partition_2,
                )
                await _solver_2.solve(only_check_fits=True)
                if await _solver_2.get_fit():
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
        if (
            best_split_uld_splits[0] is not None
            and len(partition_1) > 0
            and len(best_split_uld_splits[0]) > 0
        ):
            solver_1 = solver(
                ulds=best_split_uld_splits[0],
                packages=partition_1,
            )
            await solver_1.solve(only_check_fits=False)
            await solver_1.get_fit()

        if (
            best_split_uld_splits[1] is not None
            and len(partition_2) > 0
            and len(best_split_uld_splits[1]) > 0
        ):
            solver_2 = solver(
                ulds=best_split_uld_splits[1],
                packages=partition_2,
            )
            await solver_2.solve(only_check_fits=False)
            await solver_2.get_fit()
