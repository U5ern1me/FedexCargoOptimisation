from itertools import combinations
import logging
import os

# typing
from typing import List, Callable, Tuple
from models.uld import ULD
from models.package import Package


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


def base_sorting_heuristic_1(package: Package) -> float:
    """
    Heuristic to sort the packages.
    Priority is given to volume.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    volume = package.length * package.width * package.height
    return package.delay_cost / volume


def base_sorting_heuristic_2(package: Package) -> float:
    """
    Heuristic to sort the packages.
    Priority is given to weight.
    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    return package.delay_cost / (package.weight)


def base_sorting_heuristic_3(package: Package) -> float:
    """
    Heuristic to sort the packages.
    Priority majorly to volume with a small weight component.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    return package.delay_cost / (
        package.length * package.width * package.height * (package.weight**0.15)
    )


def base_sorting_heuristic_4(package: Package) -> float:
    """
    Heuristic to sort the packages.
    Priority majorly to weight with a small volume component.

    Args:
        package: Package

    Returns:
        Value to sort the packages
    """
    return package.delay_cost / (
        package.weight * (package.length * package.width * package.height) ** 0.15
    )


def get_sorting_heuristics(
    average_package_density: float, average_uld_density: float
) -> Tuple[Callable, Callable]:
    """
    Get the sorting heuristics based on the average package and uld densities.

    Args:
        average_package_density: Average package density
        average_uld_density: Average ULD density

    Returns:
        sorting_heuristic_1, sorting_heuristic_2
    """
    ratio = average_package_density / average_uld_density
    if ratio < 1.3:
        # Give more priority to volume
        sorting_heuristic_1 = base_sorting_heuristic_1
        sorting_heuristic_2 = base_sorting_heuristic_3
    elif ratio < 2.1:
        # Give mixed priority to volume and weight
        sorting_heuristic_1 = base_sorting_heuristic_3
        sorting_heuristic_2 = base_sorting_heuristic_4
    else:
        # Give more priority to weight
        sorting_heuristic_1 = base_sorting_heuristic_2
        sorting_heuristic_2 = base_sorting_heuristic_4

    return sorting_heuristic_1, sorting_heuristic_2


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
