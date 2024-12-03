import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy import Strategy
from solvers import solvers
from .greedy_heuristic_utils import *

from utils.io import load_config

DEBUG = True

config = load_config(os.path.join(os.path.dirname(__file__), "greedy_heuristic.config"))


class GreedyHeuristicStrategy(Strategy):
    def solve(self):

        # initialize solver and step size
        solver = solvers[config["solver"]]
        step_size = config["step size"]

        # divide packages into priority and economic
        priority_packages, economic_packages = get_divide_into_priority_and_economic(
            self.packages
        )

        # sort economic packages by delay cost per kg
        sorted_economic_packages = sort_packages(economic_packages, sorting_heuristic)

        # get all possible splits of ulds
        uld_splits_arr = get_all_division_of_ulds(self.ulds)

        # initialize best split value and packages
        best_split_value = self.k_cost * (len(self.ulds) + 1)
        best_split_packages = (-1, -1)
        best_split_uld_splits = (None, None)

        for num_p_uld, uld_splits in enumerate(uld_splits_arr):

            if DEBUG:
                print(f"Checking {num_p_uld} ULDs")

            # check if priority packages fit in uld splits if not then it is not a valid split
            valid_uld_splits = []
            for uld_split in uld_splits:
                uld_group_1, uld_group_2 = split_ulds_into_two(self.ulds, uld_split)
                if virtual_fit_priority(priority_packages, uld_group_1):
                    valid_uld_splits.append((uld_group_1, uld_group_2))

            # if no valid splits then continue
            if len(valid_uld_splits) == 0:
                continue

            if DEBUG:
                print(f"Found {len(valid_uld_splits)} valid ULD splits")

            local_best_split_packages = (-1, -1)
            local_best_split_uld_splits = (None, None)

            # find the best split of economic packages for each valid uld split
            for uld_group_1, uld_group_2 in valid_uld_splits:
                splits = find_splits_economic_packages(
                    sorted_economic_packages,
                    priority_packages,
                    uld_group_1,
                    uld_group_2,
                    solver,
                    step_size,
                )

                # if no splits then continue (priority cannot fit in uld group 1)
                if splits is None:
                    continue

                # for a split with more economic packages, update the best split (number of ulds in both uld splits must be same for this)
                if sum(splits) > sum(local_best_split_packages):
                    local_best_split_packages = splits
                    local_best_split_uld_splits = (uld_group_1, uld_group_2)

            if DEBUG:
                print(f"Found local best split: {local_best_split_packages}")

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

            if DEBUG:
                print(f"Current best split: {best_split_value}")

        # get the final partition of packages
        partition_1 = [
            *priority_packages,
            *sorted_economic_packages[: best_split_packages[0]],
        ]
        partition_2 = sorted_economic_packages[
            best_split_packages[0] : best_split_packages[1] + best_split_packages[0]
        ]

        # solve for both partitions and assign the solution to the strategy (done in the solver)
        solver_1 = solver(
            ulds=best_split_uld_splits[0], packages=partition_1, only_check_fits=False
        )
        solver_1.solve()
        solver_2 = solver(
            ulds=best_split_uld_splits[1], packages=partition_2, only_check_fits=False
        )
        solver_2.solve()
