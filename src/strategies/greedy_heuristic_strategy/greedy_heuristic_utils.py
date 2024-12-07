from itertools import combinations
import logging
import aiohttp

# typing
from typing import List, Any, Callable, Tuple
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
        sorting_heuristic_1 = base_sorting_heuristic_1  # cost / volume
        sorting_heuristic_2 = base_sorting_heuristic_3  # cost / volume * weight^0.15
    elif ratio < 2.1:
        # Give mixed priority to volume and weight
        sorting_heuristic_1 = base_sorting_heuristic_3  # cost / volume * weight^0.15
        sorting_heuristic_2 = base_sorting_heuristic_4  # cost / weight * volume^0.15
    else:
        # Give more priority to weight
        sorting_heuristic_1 = base_sorting_heuristic_2  # cost / weight
        sorting_heuristic_2 = base_sorting_heuristic_4  # cost / weight * volume^0.15

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
    sorting_heuristic: Callable,
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

    # check if priority packages can fit in uld group 1
    async with aiohttp.ClientSession() as session:
        solver_1 = solver(ulds=uld_group_1, packages=priority_packages)
        await solver_1.solve(only_check_fits=True, session=session)
        could_fit = await solver_1.get_fit(session=session)
        if not could_fit:
            if verbose:
                logging.info("Could not fit the priority packages in ULD group 1")
            return None

    # binary search for the split point of economic packages
    lower_bound = 1
    upper_bound = len(economic_packages)

    split_1 = 0

    async with aiohttp.ClientSession() as session:
        while lower_bound <= upper_bound and len(uld_group_1) > 0:
            mid = (upper_bound + lower_bound) // 2

            partition_1 = [*priority_packages, *economic_packages[:mid]]
            solver1 = solver(ulds=uld_group_1, packages=partition_1)
            await solver1.solve(only_check_fits=True, session=session)
            could_fit = await solver1.get_fit(session=session)
            if could_fit:
                split_1 = mid
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1

    if verbose:
        logging.info(f"Found split 1: {split_1}")

    remaining_economic_packages = economic_packages[split_1:]
    remaining_economic_packages = sort_packages(
        remaining_economic_packages, sorting_heuristic
    )
    split_2 = 0

    if len(uld_group_2) == 0:
        return (split_1, split_2)

    # binary search for the split point of economic packages
    lower_bound = 1
    upper_bound = len(remaining_economic_packages)

    async with aiohttp.ClientSession() as session:
        while lower_bound <= upper_bound:
            mid = (upper_bound + lower_bound) // 2

            partition_2 = remaining_economic_packages[:mid]
            solver2 = solver(ulds=uld_group_2, packages=partition_2)
            await solver2.solve(only_check_fits=True, session=session)
            could_fit = await solver2.get_fit(session=session)
            if could_fit:
                split_2 = mid
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1

    if verbose:
        logging.info(f"Found split 2: {split_2}")

    return (split_1, split_2)


async def find_best_splits_economic_packages(
    economic_packages: List[Package],
    priority_packages: List[Package],
    ulds: List[ULD],
    uld_splits: List[List[str]],
    solver: Callable,
    sorting_heuristic: Callable,
    k_cost: float,
    num_p_uld: int,
    verbose: bool = False,
) -> Tuple[int, Tuple[int, int], Tuple[List[ULD], List[ULD]]]:
    """
    Find the best splits of the economic packages among the given uld splits.

    Args:
        economic_packages: List of economic packages
        priority_packages: List of priority packages
        uld_splits: List of uld splits
        solver: Solver to check if the packages fit in the ULDs
        sorting_heuristic: Sorting heuristic to sort the packages
        verbose: Whether to print the steps

    Returns:
        cost, (split 1, split 2), (uld group 1, uld group 2)
    """

    # check if priority packages fit in uld splits if not then it is not a valid split
    valid_uld_splits = []
    for uld_split in uld_splits:
        uld_group_1, uld_group_2 = split_ulds_into_two(ulds, uld_split)
        if virtual_fit_priority(priority_packages, uld_group_1):
            valid_uld_splits.append((uld_group_1, uld_group_2))

    if len(valid_uld_splits) == 0:
        return float("inf"), (0, 0), (None, None)

    if verbose:
        logging.info(f"Found {len(valid_uld_splits)} valid ULD splits")

    best_split = (0, 0)
    best_uld_splits = (None, None)

    sorted_economic_packages = sort_packages(economic_packages, sorting_heuristic)

    # find the best split of economic packages for each valid uld split
    for uld_group_1, uld_group_2 in valid_uld_splits:
        splits = await find_splits_economic_packages(
            sorted_economic_packages,
            priority_packages,
            uld_group_1,
            uld_group_2,
            solver,
            sorting_heuristic,
            verbose=verbose,
        )

        if splits is None:
            if verbose:
                logging.info("Could not fit the priority packages in ULD group 1")
            continue

        # for a split with more economic packages, update the best split
        if sum(splits) >= sum(best_split):
            best_split = splits
            best_uld_splits = (uld_group_1, uld_group_2)

    # calculate the cost of the best split
    delay_cost = sum(
        [
            package.delay_cost
            for package in sorted_economic_packages[best_split[0] + best_split[1] :]
        ]
    )
    spread_cost = k_cost * num_p_uld
    total_cost = delay_cost + spread_cost

    return total_cost, best_split, best_uld_splits
