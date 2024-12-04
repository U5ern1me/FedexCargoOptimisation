import numpy as np
from models.package import Package
from models.uld import ULD
from typing import List, Tuple, Dict, Any, Optional
import random
import time
import logging
import json
import asyncio
from solvers.threeD_bin_packing_solver.threeD_bin_packing_solver import ThreeDBinPackingSolver

DUMP = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="genetic_3D_bin_packing.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Genetic3DBinPacking:
    def __init__(
        self,
        inputs: Dict[str, Any],
        uld_map_for_priority: Dict[str, int],
        num_generations: int = 200,
        num_individuals: int = 120,
        mutation_bracket_size: int = 5,
        min_1: int = 1,
        max_1: int = 5,
        min_2: int = 1,
        max_2: int = 5,
        eliteCProb: float = 0.7,
        best_score: int = 0,
        seed: Optional[int] = None,
    ):
        self.packages: List[Package] = inputs["packages"]
        self.ulds: List[ULD] = inputs["ulds"]
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.best_score: int = best_score

        # Separate priority and economy packages
        self.priority_packages: List[Package] = []
        self.economy_packages: List[Package] = []
        self.package_map: Dict[str, Package] = {}
        for package in self.packages:
            if package.priority == 1:
                self.priority_packages.append(package)
            else:
                self.economy_packages.append(package)
            self.package_map[package.id] = package

        self.priority_packages = np.array(self.priority_packages)
        self.economy_packages = np.array(self.economy_packages)

        # Separate priority and economy ULDs
        self.economy_ulds: List[ULD] = []
        self.priority_ulds: List[ULD] = []
        for uld in self.ulds:
            if uld_map_for_priority.get(uld.id, 0) == 0:
                self.economy_ulds.append(uld)
            else:
                self.priority_ulds.append(uld)
        self.economy_ulds = np.array(self.economy_ulds)

        # Mapping from package ID to index for fast access
        self.package_id_to_index: Dict[str, int] = {
            package.id: i for i, package in enumerate(self.economy_packages)
        }

        # Number of genes corresponds to the number of economy packages
        self.num_genes: int = len(self.economy_packages)

        # Total economic cost
        self.total_economic_cost: int = sum(
            package.delay_cost for package in self.economy_packages
        )

        self.mutation_bracket_size: int = mutation_bracket_size
        self.eliteCProb: float = eliteCProb

        self.min_1: int = min_1
        self.max_1: int = max_1
        self.min_2: int = min_2
        self.max_2: int = max_2

        self.mutation_probs: np.ndarray = np.array([0.01, 0.02, 0.03, 0.04])

        # Initialize seed
        if seed is not None:
            self.seed = round(seed) % (2**31 - 1)
        else:
            self.seed = round(random.uniform(0, 1) * time.time()) % (2**31 - 1)

        random.seed(self.seed)
        np.random.seed(self.seed)

    async def adjust_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        Adjust an individual's genes to satisfy the constraints on gene counts.
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

        Returns:
            Tuple[bool, Optional[List[str]]]: (IsValid, List of Economy Package IDs in Priority ULDs)
        """
        economy_packages_in_priority_ulds: List[str] = []
        cnt_priority_packed: int = 0
        for _uld in packing.get("bins_packed", []):
            for package in _uld.get("items", []):
                package_id = package["id"]
                if self.package_map[package_id].priority == 1:
                    cnt_priority_packed += 1
                else:
                    economy_packages_in_priority_ulds.append(package_id)

        if cnt_priority_packed != len(self.priority_packages):
            return False, None
        else:
            return True, economy_packages_in_priority_ulds

    async def rectify_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        Rectify an individual's genes to ensure valid packing.
        """
        economy_to_pack = self.economy_packages[individual == 2]

        # Sort economy packages based on delay_cost/volume ratio
        economy_sorted = sorted(
            economy_to_pack,
            key=lambda x: x.delay_cost / (x.length * x.width * x.height)
        )

        pos_economy_id: set = set((i, pkg.id) for i, pkg in enumerate(economy_sorted))
        id_to_index: Dict[str, int] = {pkg.id: i for i, pkg in enumerate(economy_sorted)}

        num_economy = len(economy_sorted)

        lower_bound = self.min_2
        upper_bound = min(self.max_2, num_economy)
        economy_packed: Optional[List[str]] = None

        # Binary search to find the maximum number of economy packages that can be packed
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            packages_to_pack = np.array(economy_sorted[-mid:])
            packages_to_pack = np.concatenate((packages_to_pack, self.priority_packages),axis=0)

            async with ThreeDBinPackingSolver(self.priority_ulds, packages_to_pack) as solver:
                await solver.solve()
                solution = await solver._get_result()

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

        Returns:
            int: The fitness cost.
        """
        cost_of_fitted: int = np.sum(
            [pkg.delay_cost for pkg in self.economy_packages[individual == 2]]
        )
        for _uld in packing.get("bins_packed", []):
            for package in _uld.get("items", []):
                cost_of_fitted += self.package_map[package["id"]].delay_cost
        return self.total_economic_cost - cost_of_fitted

    async def evaluate_individual_fitness(
        self, individual_index: int, individual: np.ndarray
    ) -> Tuple[int, dict]:
        """
        Evaluate the fitness of a single individual.

        Returns:
            Tuple[int, dict]: (Cost, Packing Solution)
        """
        chosen_packages = self.economy_packages[individual == 1]
        async with ThreeDBinPackingSolver(self.economy_ulds, chosen_packages) as solver:
            await solver.solve()
            solution = await solver._get_result()
        cost = await self.calculate_cost(individual, solution)
        logging.info(f"Individual: {individual_index}, Cost: {cost}")
        return (cost, solution)

    async def calculate_fitness(
        self, population: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Calculate fitness for the entire population.

        Returns:
            Tuple[np.ndarray, List[dict]]: (Costs Array, Solutions List)
        """
        tasks = [
            self.evaluate_individual_fitness(i, individual)
            for i, individual in enumerate(population)
        ]
        fitness_list = await asyncio.gather(*tasks)
        costs = np.array([cost for cost, _ in fitness_list])
        solutions = [solution for _, solution in fitness_list]
        return costs, solutions

    async def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """
        Perform crossover between two parents to produce an offspring.
        """
        mask = np.random.rand(self.num_genes) < self.eliteCProb
        offspring = np.where(mask, parent1, parent2)
        return offspring

    async def mating(
        self, population: np.ndarray, fitness_costs: np.ndarray
    ) -> np.ndarray:
        """
        Perform mating to produce offspring based on fitness.
        """
        # Calculate selection probabilities
        with np.errstate(divide='ignore'):
            selection_probabilities = 1 / (fitness_costs - self.best_score)
            selection_probabilities = np.where(
                fitness_costs > self.best_score, selection_probabilities, 1e6
            )

        # Normalize probabilities
        sum_prob = np.sum(selection_probabilities)
        selection_probabilities /= sum_prob

        # Select parent indices for each offspring
        parent_indices = np.array([
            np.random.choice(self.num_individuals, size=2, replace=False, p=selection_probabilities)
            for _ in range(self.num_individuals)
        ])

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
        """
        mutation_mask = np.random.rand(self.num_genes) < mutation_prob
        # Define mutation choices based on current gene values
        mutation_choices = np.random.choice([0, 1, 2], size=self.num_genes)
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

    async def mutate(
        self, population: np.ndarray
    ) -> np.ndarray:
        """
        Perform mutation on the entire population.
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

    async def fit(
        self, patience: int, verbose: bool = True
    ):
        """
        Run the genetic algorithm to find the best packing solution.
        """
        # Initialize population with random genes (0, 1, 2)
        population = np.random.randint(3, size=(self.num_individuals, self.num_genes))

        # Adjust all individuals concurrently
        tasks = [self.adjust_individual(individual) for individual in population]
        population = await asyncio.gather(*tasks)
        population = np.array(population)

        # Rectify all individuals concurrently
        tasks = [self.rectify_individual(individual) for individual in population]
        population = await asyncio.gather(*tasks)
        population = np.array(population)

        # Calculate initial fitness
        fitness_costs, fitness_solutions = await self.calculate_fitness(population)

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
            # Save population for debugging (remove in production)
            np.save("population.npy", population)  # TODO: Remove this line

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
            tasks = [self.rectify_individual(individual) for individual in mutants]
            mutants = await asyncio.gather(*tasks)
            mutants = np.array(mutants)

            # Calculate fitness for mutants
            fitness_costs_mutants, fitness_solutions_mutants = await self.calculate_fitness(
                mutants
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

                if DUMP:
                    with open("best_solution.json", "w") as f:
                        json.dump(self.best_solution, f)

                best_generation = generation

            if verbose:
                logging.info(
                    f"Generation: {generation}, Best Score: {self.best_fitness}"
                )


