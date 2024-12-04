from itertools import combinations
import logging
import copy
import os

from utils.io import load_config
from solvers import solvers

# typing
from typing import List, Callable, Tuple
from models.uld import ULD
from models.package import Package

config = load_config(os.path.join(os.path.dirname(__file__), "greedy_heuristic.config"))


def get_all_division_of_ulds(ulds: List[ULD], check_top: int) -> List[List[List[ULD]]]:
    """
    Get all possible divisions of ulds into two groups.

    Args:
        ulds: List of ULDs
        check_top: Number of top combinations to check

    Returns:
        All possible divisions of ulds into two groups
    """

    # create a hash map to group ulds by their dimensions and weight limit
    uld_hash_map = dict()
    for uld in ulds:
        hash_value = (uld.length, uld.width, uld.height, uld.weight_limit)
        if hash_value not in uld_hash_map:
            uld_hash_map[hash_value] = []

        uld_hash_map[hash_value].append(uld.id)

    # create a representation list of ulds each unique type of uld has a unique representation
    # represation is propertional to its capacity (volume * weight limit)
    uld_list = list((key, value) for key, value in uld_hash_map.items())
    uld_list.sort(key=lambda x: (x[0][0] * x[0][1] * x[0][2] * x[0][3]))
    uld_list = [value for _, value in uld_list]
    uld_rep_list = []

    for uld_rep_idx, uld_group in enumerate(uld_list):
        for __ in range(len(uld_group)):
            uld_rep_list.append(uld_rep_idx)

    solution = []

    # get all possible combinations of ulds
    for i in range(len(ulds) + 1):
        solution.append([])

        # get all possible combinations of ulds of split size i, n - i
        possible_combinations = combinations(uld_rep_list, i)
        # get only the top check_top unique combinations
        actual_combinations = sorted(
            list(set(possible_combinations)),
            key=lambda x: sum(x),
            reverse=True,
        )
        actual_combinations = actual_combinations[:check_top]

        # convert the representation list to the actual uld ids
        for comb in actual_combinations:
            unique_values = set(comb)
            comb_list = []
            for unique_value in unique_values:
                num_ulds = comb.count(unique_value)
                for j in range(num_ulds):
                    comb_list.append(uld_list[unique_value][j])

            # add the combination to the solution
            solution[-1].append(comb_list)

    return solution


def split_ulds_into_two(
    ulds: List[ULD], group_1_ids: List[int]
) -> Tuple[List[ULD], List[ULD]]:
    """
    Split the uld list into two lists based on the ids.

    Args:
        ulds: List of ULDs
        group_1_ids: List of ids of the first group

    Returns:
        group_1, group_2
    """
    group_1 = [uld for uld in ulds if uld.id in group_1_ids]
    group_2 = [uld for uld in ulds if uld.id not in group_1_ids]
    return group_1, group_2


def sort_packages(packages: List[Package], sort_function: Callable) -> List[Package]:
    """
    Sort the packages based on the sort function.

    Args:
        packages: List of packages
        sort_function: Sort function

    Returns:
        Sorted packages
    """
    return sorted(packages, key=sort_function, reverse=True)


def get_divide_into_priority_and_economic(
    packages: List[Package],
) -> Tuple[List[Package], List[Package]]:
    """
    Divide the packages into priority and economic packages.

    Args:
        packages: List of packages

    Returns:
        priority_packages, economic_packages
    """
    priority_packages = [package for package in packages if package.priority]
    economic_packages = [package for package in packages if not package.priority]
    return priority_packages, economic_packages


def sorting_heuristic_1(package: Package) -> float:
    """
    Heuristic to sort the packages.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    volume = package.length * package.width * package.height
    return package.delay_cost / volume


def sorting_heuristic_2(package: Package) -> float:
    """
    Heuristic to sort the packages.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    return package.delay_cost / (package.weight)


def sorting_heuristic_3(package: Package) -> float:
    """
    Heuristic to sort the packages.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    return package.delay_cost / (
        package.length * package.width * package.height * (package.weight**0.15)
    )


def virtual_fit_priority(
    priority_packages: List[Package], uld_group_1: List[ULD]
) -> bool:
    """
    Check if the priority packages can fit in the uld group 1 only by means of volume and weight.

    Args:
        priority_packages: List of priority packages
        uld_group_1: List of ULDs to fit the priority packages
    """
    PACKING_EFFICIENCY_THRESHOLD = 1

    if len(uld_group_1) == 0:
        return False

    total_weight = sum([package.weight for package in priority_packages])
    total_volume = sum(
        [
            package.length * package.width * package.height
            for package in priority_packages
        ]
    )

    total_capacity = sum([uld.weight_limit for uld in uld_group_1])
    total_volume_capacity = sum(
        [uld.length * uld.width * uld.height for uld in uld_group_1]
    )

    if total_weight / total_capacity > PACKING_EFFICIENCY_THRESHOLD:
        return False

    if total_volume / total_volume_capacity > PACKING_EFFICIENCY_THRESHOLD:
        return False

    return True


async def find_splits_economic_packages(
    economic_packages: List[Package],
    priority_packages: List[Package],
    uld_group_1: List[ULD],
    uld_group_2: List[ULD],
    solver: Callable,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Find the splits of the economic packages.

    Args:
        economic_packages: List of economic packages
        priority_packages: List of priority packages
        uld_group_1: List of ULDs in group 1
        uld_group_2: List of ULDs in group 2
        solver: Solver to check if the packages fit in the ULDs
        verbose: Whether to print the steps

    Returns:
        Split 1, split 2
    """
    if verbose:
        message = "Trying to fit priority and economic packages in ULDS: "
        for uld in uld_group_1:
            message += f"{uld.id} "
        logging.info(message)
    # binary search for the split point of economic packages

    lower_bound = 0
    upper_bound = len(economic_packages)

    while upper_bound - lower_bound > 1:
        mid = (upper_bound + lower_bound) // 2

        if mid == lower_bound or mid == upper_bound:
            break

        partition_1 = [*priority_packages, *economic_packages[:mid]]
        solver1 = solver(ulds=uld_group_1, packages=partition_1)
        await solver1.solve(only_check_fits=True)
        could_fit = await solver1.get_fit()
        if could_fit:
            lower_bound = mid
        else:
            upper_bound = mid

    split_1 = lower_bound

    if split_1 == 0:
        solver_1 = solver(ulds=uld_group_1, packages=priority_packages)
        await solver_1.solve(only_check_fits=True)
        could_fit = await solver_1.get_fit()
        if not could_fit:
            return None

    if verbose:
        logging.info(f"Found split 1: {split_1}")

    remaining_economic_packages = economic_packages[split_1:]
    split_2 = 0

    if len(uld_group_2) == 0:
        return (split_1, split_2)

    # binary search for the split point of economic packages
    lower_bound = 0
    upper_bound = len(remaining_economic_packages)
    while upper_bound - lower_bound > 1:
        mid = (upper_bound + lower_bound) // 2

        if mid == lower_bound or mid == upper_bound:
            break

        partition_2 = remaining_economic_packages[:mid]
        solver2 = solver(ulds=uld_group_2, packages=partition_2)
        await solver2.solve(only_check_fits=True)
        could_fit = await solver2.get_fit()
        if could_fit:
            lower_bound = mid
        else:
            upper_bound = mid

    # the last valid split point of economic packages
    split_2 = lower_bound

    if verbose:
        logging.info(f"Found split 2: {split_2}")

    return (split_1, split_2)


def calculate_cost(packages: List[Package], k_cost: float) -> float:
    priority_ulds = set()
    delay_cost = 0
    for package in packages:
        if package.uld_id is None:
            delay_cost += package.delay_cost
        else:
            if package.priority:
                priority_ulds.add(package.uld_id)

    return delay_cost + k_cost * len(priority_ulds)


async def main_solving_wrapper(
    packages: List[Package],
    ulds: List[ULD],
    k_cost: float,
    sorting_heuristic_1: Callable = sorting_heuristic_1,
    sorting_heuristic_2: Callable = sorting_heuristic_2,
    verbose: bool = False,
):
    packages = copy.deepcopy(packages)
    ulds = copy.deepcopy(ulds)

    # initialize solver
    solver = solvers[config["solver"]]

    if verbose:
        logging.info(f"using solver: {solver.__name__}")
    # divide packages into priority and economic
    priority_packages, economic_packages = get_divide_into_priority_and_economic(
        packages,
    )

    if verbose:
        logging.info(f"Priority packages: {len(priority_packages)}")
        logging.info(f"Economic packages: {len(economic_packages)}")
        logging.info(f"Sorting heuristic 1: {sorting_heuristic_1.__name__}")
        logging.info(f"Sorting heuristic 2: {sorting_heuristic_2.__name__}")

    # sort economic packages by delay cost per kg
    sorted_economic_packages = sort_packages(economic_packages, sorting_heuristic_1)

    # get all possible splits of ulds
    uld_splits_arr = get_all_division_of_ulds(ulds, 1)

    if verbose:
        logging.info(f"{uld_splits_arr}")

    # initialize best split value and packages
    best_split_value = k_cost * (len(ulds) + 1)
    best_split_packages = (0, 0)
    best_split_uld_splits = (None, None)

    for num_p_uld, uld_splits in enumerate(uld_splits_arr):

        if verbose:
            logging.info(f"Checking {num_p_uld} ULDs")

        # check if priority packages fit in uld splits if not then it is not a valid split
        valid_uld_splits = []
        for uld_split in uld_splits:
            uld_group_1, uld_group_2 = split_ulds_into_two(ulds, uld_split)
            if virtual_fit_priority(priority_packages, uld_group_1):
                valid_uld_splits.append((uld_group_1, uld_group_2))

        # if no valid splits then continue
        if len(valid_uld_splits) == 0:
            continue

        if verbose:
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
                verbose=verbose,
            )

            # if no splits then continue (priority cannot fit in uld group 1)
            if splits is None:
                if verbose:
                    logging.info("Could not fit the priority packages in ULD group 1")
                continue

                # for a split with more economic packages, update the best split (number of ulds in both uld splits must be same for this)
            if sum(splits) > sum(local_best_split_packages):
                local_best_split_packages = splits
                local_best_split_uld_splits = (uld_group_1, uld_group_2)

            if verbose:
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
            spread_cost = k_cost * num_p_uld
            total_cost = delay_cost + spread_cost

            # if the cost of the local best split is less than the best split value, update the best split
            if total_cost < best_split_value:
                best_split_value = total_cost
                best_split_packages = local_best_split_packages
                best_split_uld_splits = local_best_split_uld_splits

        if verbose:
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

    if verbose:
        logging.info(f"Remaining packages: {len(remaining_packages)}")

    _remaining_packages = []
    for package_num, package in enumerate(remaining_packages[: config["error tuning"]]):
        if verbose:
            logging.info(
                f"Checking package {package_num} of {config['error tuning']} packages"
            )

        if best_split_uld_splits[0] is not None and len(best_split_uld_splits[0]) > 0:
            _partition_1 = partition_1 + [package]
            _solver_1 = solver(
                ulds=best_split_uld_splits[0],
                packages=_partition_1,
            )
            await _solver_1.solve(only_check_fits=True)
            if await _solver_1.get_fit():
                partition_1.append(package)
                continue

        if best_split_uld_splits[1] is not None and len(best_split_uld_splits[1]) > 0:
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

    if verbose:
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

    return (calculate_cost(packages, k_cost), packages, ulds)
