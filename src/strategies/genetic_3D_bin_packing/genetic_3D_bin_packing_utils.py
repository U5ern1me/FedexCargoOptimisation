import numpy as np
from models.package import Package
from models.uld import ULD
from typing import List, Tuple, Dict, Any, Optional
import random
import time
import logging
import asyncio
import aiohttp
from solvers import solvers
from itertools import combinations


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


def calculate_min_1_max_2(packages: List[Package], ulds: List[ULD]) -> Tuple[int, int]:
    """
    Calculate the minimum number of economy packages that can be packed in the economy ulds and the maximum number of priority packages that can be packed in the priority ulds.

    Args:
        packages: List of packages
        ulds: List of ULDs

    Returns:
        Tuple[int, int]: (min_1, max_2)
    """
    priority_packages = [pkg for pkg in packages if pkg.priority == 1]
    economy_packages = [pkg for pkg in packages if pkg.priority == 0]

    priority_packages_volume = sum(
        pkg.length * pkg.width * pkg.height for pkg in priority_packages
    )
    priority_packages_weight = sum(pkg.weight for pkg in priority_packages)

    uld_volume = 0
    uld_weight = 0

    ulds = sorted(ulds, key=lambda x: x.length * x.width * x.height, reverse=True)

    for i in range(len(ulds)):
        if (
            uld_volume >= priority_packages_volume
            and uld_weight >= priority_packages_weight
        ):
            break
        uld_volume += ulds[i].length * ulds[i].width * ulds[i].height
        uld_weight += ulds[i].weight_limit

    economy_packages = sorted(
        economy_packages, key=lambda x: (x.length * x.width * x.height)
    )

    for j in range(len(economy_packages)):
        uld_volume -= (
            economy_packages[j].length
            * economy_packages[j].width
            * economy_packages[j].height
        )
        uld_weight -= economy_packages[j].weight
        if (
            uld_volume < priority_packages_volume
            and uld_weight < priority_packages_weight
        ):
            break

    uld_volume = sum(uld.length * uld.width * uld.height for uld in ulds[i:])
    uld_weight = sum(uld.weight_limit for uld in ulds[i:])

    economy_packages = sorted(
        economy_packages[j:],
        key=lambda x: (x.length * x.width * x.height),
        reverse=True,
    )

    for k in range(len(economy_packages)):
        uld_volume -= (
            economy_packages[k].length
            * economy_packages[k].width
            * economy_packages[k].height
        )
        uld_weight -= economy_packages[k].weight
        if uld_volume < 0 and uld_weight < 0:
            break

    return k, j


def calculate_max_1_min_2(packages: List[Package], ulds: List[ULD]) -> Tuple[int, int]:
    """
    Calculate the maximum number of economy packages that can be packed in the economy ulds and the minimum number of priority packages that can be packed in the priority ulds.

    Args:
        packages: List of packages
        ulds: List of ULDs

    Returns:
        Tuple[int, int]: (max_1, min_2)
    """
    priority_packages = [pkg for pkg in packages if pkg.priority == 1]
    economy_packages = [pkg for pkg in packages if pkg.priority == 0]

    priority_packages_volume = sum(
        pkg.length * pkg.width * pkg.height for pkg in priority_packages
    )
    priority_packages_weight = sum(pkg.weight for pkg in priority_packages)

    uld_volume = 0
    uld_weight = 0

    ulds = sorted(ulds, key=lambda x: x.length * x.width * x.height)

    for i in range(len(ulds)):
        if (
            uld_volume >= priority_packages_volume
            and uld_weight >= priority_packages_weight
        ):
            break
        uld_volume += ulds[i].length * ulds[i].width * ulds[i].height
        uld_weight += ulds[i].weight_limit

    economy_packages = sorted(
        economy_packages, key=lambda x: (x.length * x.width * x.height), reverse=True
    )

    for j in range(len(economy_packages)):
        uld_volume -= (
            economy_packages[j].length
            * economy_packages[j].width
            * economy_packages[j].height
        )
        uld_weight -= economy_packages[j].weight
        if (
            uld_volume < priority_packages_volume
            and uld_weight < priority_packages_weight
        ):
            break

    uld_volume = sum(uld.length * uld.width * uld.height for uld in ulds[i:])
    uld_weight = sum(uld.weight_limit for uld in ulds[i:])

    economy_packages = sorted(
        economy_packages[j:], key=lambda x: (x.length * x.width * x.height)
    )

    for k in range(len(economy_packages)):
        uld_volume -= (
            economy_packages[k].length
            * economy_packages[k].width
            * economy_packages[k].height
        )
        uld_weight -= economy_packages[k].weight
        if uld_volume < 0 and uld_weight < 0:
            break

    return k, j


def calculate_best_score(packages: List[Package], ulds: List[ULD], k_cost: int) -> int:
    """
    Calculate the theoretical best score for the given packages and ulds.
    """
    sorted_packages = sorted(
        packages,
        key=lambda x: x.delay_cost / (x.length * x.width * x.height),
        reverse=True,
    )
    total_cost_by_volume = 0
    total_volume = sum(uld.length * uld.width * uld.height for uld in ulds)
    for pkg in sorted_packages:
        if total_volume >= pkg.length * pkg.width * pkg.height:
            total_volume -= pkg.length * pkg.width * pkg.height
        else:
            total_cost_by_volume += pkg.delay_cost

    sorted_packages = sorted(
        packages, key=lambda x: x.delay_cost / x.weight, reverse=True
    )
    total_cost_by_weight = 0
    total_weight = sum(uld.weight_limit for uld in ulds)
    for pkg in sorted_packages:
        if total_weight >= pkg.weight:
            total_weight -= pkg.weight
        else:
            total_cost_by_weight += pkg.delay_cost

    priority_weight = sum(pkg.weight for pkg in packages if pkg.priority == 1)
    priority_volume = sum(
        pkg.length * pkg.width * pkg.height for pkg in packages if pkg.priority == 1
    )

    sorted_ulds_volume = sorted(
        ulds, key=lambda x: x.length * x.width * x.height, reverse=True
    )
    num_ulds_volume = 0
    for uld in sorted_ulds_volume:
        if priority_volume >= 0:
            priority_volume -= uld.length * uld.width * uld.height
            num_ulds_volume += 1
        else:
            break

    sorted_ulds_weight = sorted(ulds, key=lambda x: x.weight_limit, reverse=True)
    num_ulds_weight = 0
    for uld in sorted_ulds_weight:
        if priority_weight >= 0:
            priority_weight -= uld.weight_limit
            num_ulds_weight += 1
        else:
            break

    return (
        min(total_cost_by_volume, total_cost_by_weight)
        + max(num_ulds_volume, num_ulds_weight) * k_cost
    )


class Genetic3DBinPacking:
    def __init__(
        self,
        inputs: Dict[str, Any],
        # uld_map_for_priority: Dict[str, int],
        num_generations: int = 200,
        num_individuals: int = 120,
        mutation_bracket_size: int = 5,
        solver: str = "sardine_can",
        eliteCProb: float = 0.7,
        seed: Optional[int] = None,
    ):

        self.packages = inputs["packages"]
        self.ulds = inputs["ulds"]
        self.k_cost = inputs["k_cost"]
        self.num_generations = num_generations
        self.num_individuals = num_individuals
        self.best_score = calculate_best_score(self.packages, self.ulds, self.k_cost)

        # Separate priority and economy packages
        self.priority_packages = []
        self.economy_packages = []
        self.package_map = {}
        for package in self.packages:
            if package.priority == 1:
                self.priority_packages.append(package)
            else:
                self.economy_packages.append(package)
            self.package_map[package.id] = package

        self.priority_packages = np.array(self.priority_packages)
        self.economy_packages = np.array(self.economy_packages)

        self.solver = solvers[solver]

        # Mapping from package ID to index for fast access
        self.package_id_to_index = {
            package.id: i for i, package in enumerate(self.economy_packages)
        }

        # Number of genes corresponds to the number of economy packages
        self.num_genes = len(self.economy_packages)

        # Total economic cost
        self.total_economic_cost = sum(
            package.delay_cost for package in self.economy_packages
        )

        self.mutation_bracket_size = mutation_bracket_size
        self.eliteCProb = eliteCProb

        self.min_1, self.max_2 = calculate_min_1_max_2(self.packages, self.ulds)
        self.max_1, self.min_2 = calculate_max_1_min_2(self.packages, self.ulds)

        self.mutation_probs = np.array([0.01, 0.02, 0.03, 0.04])

        # Initialize seed
        if seed is not None:
            self.seed = round(seed) % (2**31 - 1)
        else:
            self.seed = round(random.uniform(0, 1) * time.time()) % (2**31 - 1)

        random.seed(self.seed)
        np.random.seed(self.seed)

    async def get_best_ulds(self) -> Tuple[List[ULD], List[ULD]]:
        """
        Get the best ulds for packing the priority packages.

        Returns:
            Tuple of two lists of ULDs
        """
        uld_splits_arr = get_all_division_of_ulds(self.ulds, 1)

        async with aiohttp.ClientSession() as session:
            for uld_split in uld_splits_arr:
                uld_group_1, uld_group_2 = split_ulds_into_two(self.ulds, uld_split[0])

                # check if weight or volume constraints are violated
                if virtual_fit_priority(self.priority_packages, uld_group_1):

                    # if not violated, check if the priority packages are packed correctly using the solver
                    solver = self.solver(uld_group_1, self.priority_packages)
                    await solver.solve(session=session)
                    solution = await solver.get_packing_json(session=session)
                    is_valid, _ = await self.check_validity(solution)
                    if is_valid:
                        return uld_group_1, uld_group_2

    async def adjust_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        Adjust an individual's genes to satisfy the constraints on gene counts.

        Args:
            individual: Individual to adjust

        Returns:
            np.ndarray: Adjusted individual
        """
        assert len(individual) == self.num_genes

        num_1 = np.sum(individual == 1)
        num_2 = np.sum(individual == 2)

        tasks = []

        # Define helper tasks for asynchronous adjustment
        if num_2 < self.min_2:
            deficit = self.min_2 - num_2
            tasks.append(self._set_genes(individual, deficit, current=0, new=2))

        if num_2 > self.max_2:
            excess = num_2 - self.max_2
            tasks.append(self._set_genes(individual, excess, current=2, new=1))
            num_1 += excess

        if num_1 < self.min_1:
            deficit = self.min_1 - num_1
            tasks.append(self._set_genes(individual, deficit, current=0, new=1))

        if num_1 > self.max_1:
            excess = num_1 - self.max_1
            tasks.append(self._set_genes(individual, excess, current=1, new=0))

        if tasks:
            await asyncio.gather(*tasks)

        return individual

    async def _set_genes(
        self, individual: np.ndarray, count: int, current: int, new: int
    ) -> None:
        """
        Helper method to set a specified number of genes from 'current' to 'new'.

        Args:
            individual: Individual to set genes
            count: Number of genes to set
            current: Current gene value
            new: New gene value
        """
        indices = np.where(individual == current)[0]
        if len(indices) == 0:
            return
        chosen_indices = np.random.choice(
            indices, size=min(count, len(indices)), replace=False
        )
        individual[chosen_indices] = new

    async def check_validity(
        self, packing: Dict[str, Any]
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Check if all priority packages are packed correctly.

        Args:
            packing: Packing solution

        Returns:
            Tuple[bool, Optional[List[str]]]: (IsValid, List of Economy Package IDs in Priority ULDs)
        """
        economy_packages_in_priority_ulds = []
        cnt_priority_packed = 0
        for _uld in packing.get("ulds", []):
            for package_id in _uld.get("packages", []):
                if self.package_map[package_id].priority == 1:
                    cnt_priority_packed += 1
                else:
                    economy_packages_in_priority_ulds.append(package_id)

        if cnt_priority_packed != len(self.priority_packages):
            return False, None
        else:
            return True, economy_packages_in_priority_ulds

    async def rectify_individual(
        self,
        individual: np.ndarray,
        priority_ulds: np.ndarray,
        session: aiohttp.ClientSession,
    ) -> np.ndarray:
        """
        Rectify an individual's genes to ensure valid packing.

        Args:
            individual: Individual to rectify
            priority_ulds: List of priority ULDs
            session: aiohttp.ClientSession

        Returns:
            Rectified individual
        """
        economy_to_pack = self.economy_packages[individual == 2]

        # Sort economy packages based on delay_cost/volume ratio
        economy_sorted = sorted(
            economy_to_pack,
            key=lambda x: x.delay_cost / (x.length * x.width * x.height),
        )

        pos_economy_id = set((i, pkg.id) for i, pkg in enumerate(economy_sorted))
        id_to_index = {pkg.id: i for i, pkg in enumerate(economy_sorted)}

        num_economy = len(economy_sorted)

        lower_bound = self.min_2
        upper_bound = min(self.max_2, num_economy)
        economy_packed = None

        # Binary search to find the maximum number of economy packages that can be packed
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            packages_to_pack = np.array(economy_sorted[-mid:])
            packages_to_pack = np.concatenate(
                (packages_to_pack, self.priority_packages), axis=0
            )

            solver = self.solver(priority_ulds, packages_to_pack)
            await solver.solve(session=session)
            solution = await solver.get_packing_json(session=session)

            is_valid, packed = await self.check_validity(solution)

            if is_valid:
                economy_packed = packed
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1

        if economy_packed:
            for package_id in economy_packed:
                pos_economy_id.discard((id_to_index.get(package_id, -1), package_id))

        cnt_1 = np.sum(individual == 1)

        # Iterate over remaining economy packages and adjust genes
        for i, package_id in sorted(pos_economy_id, reverse=True):
            if cnt_1 < self.max_1:
                individual[self.package_id_to_index.get(package_id, -1)] = 1
                cnt_1 += 1
            else:
                individual[self.package_id_to_index.get(package_id, -1)] = 0

        return individual

    async def calculate_cost(
        self, individual: np.ndarray, packing: Dict[str, Any]
    ) -> int:
        """
        Calculate the cost of an individual based on the packing.

        Args:
            individual: Individual to calculate cost
            packing: Packing solution

        Returns:
            int: The fitness cost.
        """
        cost_of_fitted = np.sum(
            [pkg.delay_cost for pkg in self.economy_packages[individual == 2]]
        )
        for _uld in packing.get("ulds", []):
            for package_id in _uld.get("packages", []):
                cost_of_fitted += self.package_map[package_id].delay_cost
        return self.total_economic_cost - cost_of_fitted

    async def calculate_fitness(
        self, population: np.ndarray, economy_ulds: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Calculate fitness for the entire population.

        Args:
            population: Population to calculate fitness
            economy_ulds: List of economy ULDs

        Returns:
            Tuple[np.ndarray, List[dict]]: (Costs Array, Solutions List)
        """
        async with aiohttp.ClientSession() as session:
            fitness_solvers = [
                self.solver(economy_ulds, self.economy_packages[individual == 1])
                for individual in population
            ]

            tasks = [
                fitness_solver.solve(session=session, only_check_fits=True)
                for fitness_solver in fitness_solvers
            ]
            await asyncio.gather(*tasks)

            tasks = [
                fitness_solver.get_packing_json(session=session)
                for fitness_solver in fitness_solvers
            ]
            packing_responses = await asyncio.gather(*tasks)

            fitness_list = [
                (
                    await self.calculate_cost(individual, packing_response),
                    packing_response,
                )
                for individual, packing_response in zip(population, packing_responses)
            ]

        costs = np.array(
            [
                cost + (len(self.ulds) - len(economy_ulds)) * self.k_cost
                for cost, _ in fitness_list
            ]
        )
        solutions = [solution for _, solution in fitness_list]
        return costs, solutions

    async def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents to produce an offspring.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Offspring
        """
        mask = np.random.rand(self.num_genes) < self.eliteCProb
        offspring = np.where(mask, parent1, parent2)
        return offspring

    async def mating(
        self, population: np.ndarray, fitness_costs: np.ndarray
    ) -> np.ndarray:
        """
        Perform mating to produce offspring based on fitness.

        Args:
            population: Population to mate
            fitness_costs: Fitness costs of the population

        Returns:
            Offspring
        """
        # Calculate selection probabilities
        with np.errstate(divide="ignore"):
            selection_probabilities = 1 / (fitness_costs - self.best_score)
            selection_probabilities = np.where(
                fitness_costs > self.best_score, selection_probabilities, 1e6
            )

        # Normalize probabilities
        sum_prob = np.sum(selection_probabilities)
        selection_probabilities /= sum_prob

        # Select parent indices for each offspring
        parent_indices = np.array(
            [
                np.random.choice(
                    self.num_individuals,
                    size=2,
                    replace=False,
                    p=selection_probabilities,
                )
                for _ in range(self.num_individuals)
            ]
        )

        # Create a list of crossover tasks
        tasks = [
            self.crossover(population[idx1], population[idx2])
            for idx1, idx2 in parent_indices
        ]

        # Execute crossovers concurrently
        offspring_list = await asyncio.gather(*tasks)
        return np.array(offspring_list)

    async def mutate_individual(
        self, individual: np.ndarray, mutation_prob: float
    ) -> np.ndarray:
        """
        Mutate a single individual based on mutation probability.

        Args:
            individual: Individual to mutate
            mutation_prob: Mutation probability

        Returns:
            Mutated individual
        """
        mutation_mask = np.random.rand(self.num_genes) < mutation_prob
        # Define mutation choices based on current gene values
        mutated_individual = np.copy(individual)

        # Apply mutations where mask is True
        for gene_idx in np.where(mutation_mask)[0]:
            current_gene = mutated_individual[gene_idx]
            if current_gene == 0:
                mutated_individual[gene_idx] = np.random.choice([1, 2])
            elif current_gene == 1:
                mutated_individual[gene_idx] = np.random.choice([0, 2])
            elif current_gene == 2:
                mutated_individual[gene_idx] = np.random.choice([0, 1])
        return mutated_individual

    async def mutate(self, population: np.ndarray) -> np.ndarray:
        """
        Perform mutation on the entire population.

        Args:
            population: Population to mutate

        Returns:
            Mutated population
        """
        num_brackets = self.num_individuals // self.mutation_bracket_size
        tasks = []
        for i in range(2 * num_brackets):
            bracket = population[
                i * self.mutation_bracket_size : (i + 1) * self.mutation_bracket_size
            ]
            mutation_prob = self.mutation_probs[i]
            for individual in bracket:
                tasks.append(self.mutate_individual(individual, mutation_prob))

        # Execute mutations concurrently
        mutated_population = await asyncio.gather(*tasks)
        return np.array(mutated_population)

    async def fit(self, patience: int, verbose: bool = True):
        """
        Run the genetic algorithm to find the best packing solution.

        Args:
            patience: Patience for early stopping
            verbose: Whether to print logs
        """
        # Initialize population with random genes (0, 1, 2)
        population = np.random.randint(3, size=(self.num_individuals, self.num_genes))

        if verbose:
            logging.info(f"Population: {len(population)}")
            logging.info(f"Solver: {self.solver.__name__}")

        priority_ulds, economy_ulds = await self.get_best_ulds()

        if verbose:
            logging.info(f"Priority ULDs: {len(priority_ulds)}")
            logging.info(f"Economy ULDs: {len(economy_ulds)}")

        # Adjust all individuals concurrently
        tasks = [self.adjust_individual(individual) for individual in population]
        population = await asyncio.gather(*tasks)
        population = np.array(population)

        # Rectify all individuals concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.rectify_individual(individual, priority_ulds, session=session)
                for individual in population
            ]
            population = await asyncio.gather(*tasks)
            population = np.array(population)

        # Calculate initial fitness
        fitness_costs, fitness_solutions = await self.calculate_fitness(
            population, economy_ulds
        )

        # Sort population based on fitness (ascending cost)
        sorted_indices = np.argsort(fitness_costs)
        fitness_costs = fitness_costs[sorted_indices]
        fitness_solutions = [fitness_solutions[i] for i in sorted_indices]
        population = population[sorted_indices]

        # Initialize best fitness and solution
        self.best_fitness = fitness_costs[0]
        self.best_solution = fitness_solutions[0]
        best_generation = 0

        if verbose:
            logging.info(f"Generation: 0, Best Score: {self.best_fitness}")

        for generation in range(1, self.num_generations + 1):
            # Early stopping check
            if generation - best_generation > patience:
                if verbose:
                    logging.info(f"Early Stopping at Generation {generation}")
                break

            # Mating: Generate offspring
            offspring = await self.mating(population, fitness_costs)

            # Mutate offspring
            combined_population = np.concatenate((population, offspring), axis=0)
            mutants = await self.mutate(combined_population)

            # Adjust all mutants concurrently
            tasks = [self.adjust_individual(individual) for individual in mutants]
            mutants = await asyncio.gather(*tasks)
            mutants = np.array(mutants)

            # Rectify all mutants concurrently
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.rectify_individual(individual, priority_ulds, session=session)
                    for individual in mutants
                ]
                mutants = await asyncio.gather(*tasks)
                mutants = np.array(mutants)

            # Calculate fitness for mutants
            fitness_costs_mutants, fitness_solutions_mutants = (
                await self.calculate_fitness(mutants, economy_ulds)
            )

            # Sort mutants based on fitness (ascending cost)
            sorted_indices = np.argsort(fitness_costs_mutants)
            fitness_costs_mutants = fitness_costs_mutants[sorted_indices]
            fitness_solutions_mutants = [
                fitness_solutions_mutants[i] for i in sorted_indices
            ]
            mutants = mutants[sorted_indices]

            # Select the top individuals to form the new population
            population = mutants[: self.num_individuals]
            fitness_costs = fitness_costs_mutants[: self.num_individuals]
            fitness_solutions = fitness_solutions_mutants[: self.num_individuals]

            # Update the best fitness and solution if a better one is found
            if fitness_costs[0] < self.best_fitness:
                self.best_fitness = fitness_costs[0]
                self.best_solution = fitness_solutions[0]

                best_generation = generation

            if verbose:
                logging.info(
                    f"Generation: {generation}, Best Score: {self.best_fitness}"
                )
