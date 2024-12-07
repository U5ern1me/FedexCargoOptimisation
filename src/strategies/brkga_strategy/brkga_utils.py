import copy
import math
import random
import time
import multiprocessing
import numpy as np
import logging

# type hinting
from typing import List, Tuple, Dict, Any, Optional
from models.package import Package
from models.uld import ULD


class Bin:
    def __init__(self, uld: ULD, verbose: bool = False):
        """
        Args:
            uld: ULD to be initialized
            verbose: Whether to print the progress
        """
        self.dimensions = (uld.length, uld.width, uld.height)
        self.EMSs = [[np.array((0, 0, 0)), np.array(self.dimensions)]]
        self.weight_capacity = uld.weight_limit
        self.id = uld.id
        self.boxes = []
        self.weight = 0
        self.has_priority = False

        if verbose:
            logging.info("Init EMSs:", self.EMSs)

    def __getitem__(self, index: int) -> List[np.ndarray]:
        return self.EMSs[index]

    def __len__(self) -> int:
        return len(self.EMSs)

    def update(
        self,
        box: Package,
        box_dimensions: Tuple[int, int, int],
        selected_EMS: List[np.ndarray],
        min_vol: int,
        min_dim: int,
        verbose: bool = False,
    ):
        """
        Add the box to the bin and update the EMSs of the bin.

        Args:
            box: Package to be added to the bin
            box_dimensions: Rotated dimensions of the box
            selected_EMS: EMS selected to place the box
            min_vol: Minimum volume of the remaining boxes
            min_dim: Minimum dimension of the remaining boxes
            verbose: Whether to print verbose output
        """
        new_box = np.array(box_dimensions)
        isValid = True
        if min_dim is not None and np.any(
            new_box < min_dim
        ):  # Ensure min_dim is not None
            isValid = False
            if verbose:
                logging.info("Box is invalid due to dimensions:", new_box)

        # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
        if np.prod(new_box) < min_vol:  # Change from np.product to np.prod
            isValid = False
            if verbose:
                logging.info("Box is invalid due to volume:", new_box)

        # Other update logic...

        # 1. place box in a EMS
        boxToPlace = np.array(box_dimensions)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        box.point1 = tuple(selected_min)
        box.point2 = tuple(selected_min + boxToPlace)
        self.boxes.append(box)
        box.uld_id = self.id
        self.weight += box.weight
        if box.priority:
            self.has_priority = True

        if verbose:
            logging.info("*Place Box*: EMS:", list(map(tuple, ems)))

        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):

                # eliminate overlapped EMS
                self.eliminate(EMS)

                if verbose:
                    logging.info(
                        f"*Elimination*:\nRemove overlapped EMS: {list(map(tuple, EMS))}\nEMSs left: {list(map(lambda x: list(map(tuple, x)), self.EMSs))}"
                    )

                # six new EMSs in 3 dimensionsc
                x1, y1, z1 = EMS[0]
                x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]
                x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))],
                ]

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True

                    if verbose:
                        logging.info(f"*New* EMS: {list(map(tuple, new_EMS))}")

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                logging.info(
                                    f"-> Totally inscribed by: {list(map(tuple, other_EMS))}"
                                )

                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.any(
                        new_box < min_dim
                    ):  # Change from np.min(new_box) < min_dim to np.any(new_box < min_dim)
                        isValid = False
                        if verbose:
                            logging.info(f"Box is invalid due to dimensions: {new_box}")

                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.prod(new_box) < min_vol:  # Change from np.product to np.prod
                        isValid = False
                        if verbose:
                            logging.info(f"Box is invalid due to volume: {new_box}")

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            logging.info(
                                f"-> Success\nAdd new EMS: {list(map(tuple, new_EMS))}"
                            )

        if verbose:
            logging.info(
                f"End:\nEMSs: {list(map(lambda x: list(map(tuple, x)), self.EMSs))}"
            )

    def overlapped(self, ems: np.ndarray, EMS: np.ndarray) -> bool:
        """
        Check if the EMSs are overlapped.

        Args:
            ems: EMS to be checked
            EMS: EMS to be checked against

        Returns:
            True if the EMSs are overlapped, False otherwise
        """
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False

    def inscribed(self, ems: np.ndarray, EMS: np.ndarray) -> bool:
        """
        Check if the ems is inscribed by another EMS.

        Args:
            ems: EMS to be checked
            EMS: EMS to be checked against

        Returns:
            True if the EMS is inscribed by another EMS, False otherwise
        """
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False

    def eliminate(self, ems: np.ndarray):
        """
        Eliminate the EMS from the list of EMSs.

        Args:
            ems: EMS to be eliminated
        """
        ems = list(map(tuple, ems))
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return

    def get_EMSs(self) -> List[List[Tuple[int, int, int]]]:
        """
        Get the EMSs of the bin.

        Returns:
            List of EMSs
        """
        return list(map(lambda x: list(map(tuple, x)), self.EMSs))

    def weight_packing_efficiency(self) -> float:
        """
        Get the weight packing efficiency of the bin.

        Returns:
            Weight packing efficiency
        """
        return sum(box.weight for box in self.boxes) / (self.weight_capacity)

    def volume_packing_efficiency(self) -> float:
        """
        Get the volume packing efficiency of the bin.

        Returns:
            Volume packing efficiency
        """
        return sum(
            (box.length * box.width * box.height) for box in self.boxes
        ) / np.prod(self.dimensions)


class PlacementProcedure:
    def __init__(
        self,
        inputs: Dict[str, Any],
        solution: np.ndarray,
        verbose: bool = False,
    ):
        """
        Args:
            inputs: Inputs of the problem
            solution: Solution to be decoded
            verbose: Whether to print the progress
        """
        BinOrder = np.argsort(solution[: len(inputs["ulds"])])
        self.Bins = [Bin(inputs["ulds"][i]) for i in BinOrder]
        self.boxes = copy.deepcopy(inputs["packages"])
        self.BPS = np.argsort(
            solution[len(self.Bins) : len(self.Bins) + len(self.boxes)]
        )
        self.VBO = solution[len(self.Bins) + len(self.boxes) :]
        self.num_opend_bins = 1
        self.k_cost = inputs["k"]

        self.verbose = verbose
        if self.verbose:
            logging.info(
                f"""------------------------------------------------------------------
                |   Placement Procedure
                |    -> Bins: {self.Bins}
                |    -> Bin Order: {self.BinOrder}
                |    -> Boxes: {self.boxes}
                |    -> Box Packing Sequence: {self.BPS}
                |    -> Vector of Box Orientations: {self.VBO}
                ------------------------------------------------------------------
                """
            )

        self.placement()

    def placement(self):
        """
        Place the boxes in the bins using the packing order.
        """
        items_sorted = [self.boxes[i] for i in self.BPS]

        # Box Selection
        for i, box in enumerate(items_sorted):
            if self.verbose:
                logging.info(f"Select Box: {box}")

            # Bin and EMS selection
            selected_bin = None
            selected_EMS = None
            for bin_num in range(self.num_opend_bins):
                # select EMS using DFTRC-2
                if (
                    box.weight + self.Bins[bin_num].weight
                    > self.Bins[bin_num].weight_capacity
                ):
                    continue

                EMS = self.DFTRC_2(box, bin_num)

                # update selection if "packable"
                if EMS != None:
                    selected_bin = bin_num
                    selected_EMS = EMS
                    break

            # Open new empty bin
            if selected_bin == None:
                self.num_opend_bins += 1

                bin_length = len(self.Bins)

                while (
                    self.num_opend_bins <= bin_length
                    and box.weight > self.Bins[self.num_opend_bins - 1].weight_capacity
                ):
                    self.num_opend_bins += 1

                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > bin_length:

                    if self.verbose:
                        logging.info(f"No more bin to open. [Cannot fit box {box.id}]")
                    return

                selected_EMS = self.Bins[selected_bin].EMSs[0]  # origin of the new bin
                if self.verbose:
                    logging.info(f"No available bin... open bin {selected_bin}")

            if self.verbose:
                logging.info(f"Select EMS: {list(map(tuple, selected_EMS))}")

            # Box orientation selection
            BO = self.selecte_box_orientaion(self.VBO[i], box, selected_EMS)

            # elimination rule for different process
            min_vol, min_dim = self.elimination_rule(items_sorted[i + 1 :])

            # pack the box to the bin & update state information
            self.Bins[selected_bin].update(
                box, self.orient(box, BO), selected_EMS, min_vol, min_dim
            )

            if self.verbose:
                logging.info(
                    f"""Add box to Bin {selected_bin}
                    -> EMSs: {self.Bins[selected_bin].get_EMSs()}"""
                )
        if self.verbose:
            logging.info(f"Number of used bins: {self.num_opend_bins}")

    def DFTRC_2(self, box: Package, k: int) -> np.ndarray:
        """
        Select the EMS that is the farthest from the front-top-right corner of the bin.
        DFRTC ~ Distance to the Front-Top-Right Corner

        Args:
            box: Box (length, width, height) to be placed
            k: Index of the bin

        Returns:
            Selected EMS
        """
        maxDist = -1
        selectedEMS = None

        for EMS in self.Bins[k].EMSs:
            D, W, H = self.Bins[k].dimensions
            for direction in [1, 2, 3, 4, 5, 6]:
                d, w, h = self.orient(box, direction)
                if self.fitin((d, w, h), EMS):
                    x, y, z = EMS[0]
                    distance = pow(D - x - d, 2) + pow(W - y - w, 2) + pow(H - z - h, 2)

                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS
        return selectedEMS

    def orient(self, box: Package, BO: int = 1) -> Tuple[int, int, int]:
        """
        Orient the box based on the orientation code.

        Args:
            box: Box (length, width, height) to be oriented
            BO: Orientation code

        Returns:
            Oriented box (length, width, height)
        """
        d, w, h = box.length, box.width, box.height
        if BO == 1:
            return (d, w, h)
        elif BO == 2:
            return (d, h, w)
        elif BO == 3:
            return (w, d, h)
        elif BO == 4:
            return (w, h, d)
        elif BO == 5:
            return (h, d, w)
        elif BO == 6:
            return (h, w, d)

    def selecte_box_orientaion(self, VBO: float, box: Package, EMS: np.ndarray) -> int:
        """
        Select the box orientation based on the VBO vector.

        Args:
            VBO: Random choice of the orientation code
            box: Box (length, width, height) to be oriented
            EMS: EMS to be checked against

        Returns:
            Selected orientation code
        """
        # feasible direction
        BOs = []
        for direction in [1, 2, 3, 4, 5, 6]:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)

        selectedBO = BOs[math.ceil(VBO * len(BOs)) - 1]

        if self.verbose:
            logging.info(f"Select VBO: {selectedBO}  (BOs {BOs}, vector {VBO})")
        return selectedBO

    def fitin(self, box: Tuple[int, int, int], EMS: np.ndarray) -> bool:
        """
        Check if the box fits in the EMS.

        Args:
            box: Box (length, width, height) to be checked
            EMS: EMS to be checked against
        """
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True

    def elimination_rule(self, remaining_boxes: List[Package]) -> Tuple[int, int]:
        """
        Find the box with the smallest volume and the smallest dimension.

        Args:
            remaining_boxes: List of packages to be checked

        Returns:
            Minimum volume and minimum dimension
        """
        min_vol = float("inf")
        min_dim = float("inf")  # Initialize min_dim to a large value
        for box in remaining_boxes:
            vol = box.length * box.width * box.height
            if vol < min_vol:
                min_vol = vol
                min_dim = min(box.length, box.width, box.height)
        return min_vol, min_dim

    def evaluate(self):
        """
        Evaluate the fitness of the solution.

        Returns:
            Fitness of the solution
        """
        num_priority_uld = sum(bin.has_priority for bin in self.Bins)
        delay_cost = 0
        for box in self.boxes:
            if box.uld_id is None:
                delay_cost += box.delay_cost

        packing_efficiency = 0

        for bin in self.Bins:
            packing_efficiency += 1 / bin.weight_packing_efficiency()
            packing_efficiency += 1 / bin.volume_packing_efficiency()

        return num_priority_uld * self.k_cost + delay_cost + packing_efficiency


class BRKGA:
    def __init__(
        self,
        inputs: Dict[str, Any],
        num_generations: int = 200,
        num_individuals: int = 120,
        num_elites: int = 12,
        num_mutants: int = 18,
        fraction_biased: float = 0.4,
        eliteCProb: float = 0.7,
        multiProcess: bool = False,
        seed: Optional[int] = None,
    ):
        """ "
        Args:
            inputs: Inputs of the problem
            num_generations: Number of generations
            num_individuals: Number of individuals
            num_elites: Number of elites
            num_mutants: Number of mutants
            fraction_biased: Fraction of biased population
            eliteCProb: Elite crossover probability
            multiProcess: Whether to use multiprocessing
            seed: Seed for the random number generator (used for reproducibility / ensemble)
        """
        # Setting
        self.multiProcess = multiProcess
        # Input
        self.inputs = inputs
        self.N = len(inputs["packages"])
        self.k = inputs["k"]
        self.M = len(inputs["ulds"])

        # Configuration
        self.num_generations = num_generations
        self.num_individuals = max(
            int(num_individuals), 100
        )  # Ensure minimum population size
        self.num_gene = 2 * self.N + self.M  # Number of genes

        self.num_elites = max(int(num_elites), 10)  # Ensure minimum elite size
        self.num_mutants = max(int(num_mutants), 15)  # Ensure minimum mutants
        self.eliteCProb = min(eliteCProb, 0.8)  # Cap elite crossover probability
        self.fraction_biased = min(fraction_biased, 0.8)  # Cap biased fraction

        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {"mean": [], "min": []}
        if seed is not None:
            self.seed = round(seed) % (2**31 - 1)
        else:
            self.seed = round(random.uniform(0, 1) * time.time()) % (2**31 - 1)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def decoder(self, solution: np.ndarray) -> float:
        """
        Decode the solution to the placement procedure and evaluate the fitness.

        Args:
            solution: Solution to be decoded

        Returns:
            Fitness of the solution
        """
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()

    def cal_fitness(self, population: np.ndarray) -> List[float]:
        """
        Calculate the fitness of the population.

        Args:
            population: Population to be evaluated

        Returns:
            List of fitness values
        """
        fitness_list = list()

        # use multiprocessing to calculate fitness
        if self.multiProcess:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                fitness_list = list(pool.map(self.decoder, population))
        else:
            for solution in population:
                fitness_list.append(self.decoder(solution))
        return fitness_list

    def partition(
        self, population: np.ndarray, fitness_list: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partition the population into elites and non-elites.

        Args:
            population: Population to be partitioned
            fitness_list: List of fitness values

        Returns:
            Tuple(elite_population, non_elite_population, elite_fitness_list)
        """
        sorted_indexs = np.argsort(fitness_list)
        population = np.array(population)
        fitness_list = np.array(fitness_list)

        num_elites = min(self.num_elites, len(population))

        return (
            population[sorted_indexs[:num_elites]],
            population[sorted_indexs[num_elites:]],
            fitness_list[sorted_indexs[:num_elites]],
        )

    def crossover_mixed(self, elite: np.ndarray, non_elite: np.ndarray) -> np.ndarray:
        """
        Chance to choose the gene from elite individual and non_elite individual for each gene.

        Args:
            elite: Elite individual
            non_elite: Non-elite individual

        Returns:
            Offspring individual
        """
        return [
            (
                elite[gene]
                if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb
                else non_elite[gene]
            )
            for gene in range(self.num_gene)
        ]

    def crossover_elite(self, elite_1: np.ndarray, elite_2: np.ndarray) -> np.ndarray:
        """
        Crossover between two elites individuals.

        Args:
            elite_1: First elite individual
            elite_2: Second elite individual

        Returns:
            Offspring individual
        """
        return [
            (
                elite_1[gene]
                if np.random.uniform(low=0.0, high=1.0) < 0.5
                else elite_2[gene]
            )
            for gene in range(self.num_gene)
        ]

    def biased_population(self, num_individuals: int) -> np.ndarray:
        """
        Generate a biased population, where the genes of priority ULDs are halved.

        Args:
            num_individuals: Number of individuals to be generated

        Returns:
            Biased population
        """
        biased_list = np.random.uniform(
            low=0.0, high=1.0, size=(num_individuals, self.num_gene)
        )

        vol_list = np.array(
            [uld.length * uld.width * uld.height for uld in self.inputs["ulds"]]
        )
        vol_list = vol_list / sum(vol_list)
        # for each individual, for indexes of priority ulds, divide gene by 2
        for i in range(num_individuals):
            for j in range(self.N):
                if self.inputs["packages"][j].priority:
                    biased_list[i][self.M + j] = biased_list[i][self.M + j] / 2
                else:
                    biased_list[i][self.M + j] = biased_list[i][self.M + j] / 2 + 0.5

            for j in range(self.M):
                biased_list[i][j] = biased_list[i][j] * vol_list[j]

        return biased_list

    def random_population(self, num_individuals: int) -> np.ndarray:
        """
        Generate a random population.

        Args:
            num_individuals: Number of individuals to be generated

        Returns:
            Random population
        """
        return np.random.uniform(
            low=0.0, high=1.0, size=(num_individuals, self.num_gene)
        )

    def mating(self, elites: np.ndarray, non_elites: np.ndarray) -> List[np.ndarray]:
        """
        Biased selection of mating parents: 1 elite & 1 non_elite.

        Args:
            elites: Elite individuals
            non_elites: Non-elite individuals

        Returns:
            List of offspring individuals
        """
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        num_elite_offsprings = int(num_offspring * self.fraction_biased)
        elite_offsprings = [
            self.crossover_elite(random.choice(elites), random.choice(non_elites))
            for i in range(num_elite_offsprings)
        ]
        non_elite_offsprings = [
            self.crossover_mixed(random.choice(elites), random.choice(non_elites))
            for i in range(num_offspring - num_elite_offsprings)
        ]
        return [*elite_offsprings, *non_elite_offsprings]

    def mutants(self):
        """
        Generate mutants by biased selection and random selection.
        """
        biased_count = int(self.num_mutants * self.fraction_biased)
        biased_mutants = self.biased_population(biased_count)
        random_mutants = self.random_population(self.num_mutants - biased_count)
        return np.concatenate((biased_mutants, random_mutants), axis=0)

    def fit(self, patient: int = 4, verbose: bool = False):
        """
        Main loop of the genetic algorithm.

        Args:
            patient: Number of generations without improvement before stopping.
            verbose: Whether to print the progress.
        """
        # Initial population & fitness
        biased_count = int(self.num_individuals * 2 * self.fraction_biased)
        population = np.concatenate(
            (
                self.biased_population(biased_count),
                self.random_population(2 * self.num_individuals - biased_count),
            ),
            axis=0,
        )
        fitness_list = self.cal_fitness(population)

        # keep only the population with fitness < 100000
        good_idx = np.where(np.array(fitness_list) < 100000)[0]
        population = population[good_idx]
        fitness_list = np.array(fitness_list)[good_idx]

        # Select best initial population
        sorted_idx = np.argsort(fitness_list)
        population = population[sorted_idx[: self.num_individuals]]
        fitness_list = fitness_list[sorted_idx[: self.num_individuals]]

        if len(population) == 0:
            raise ValueError("No feasible solution found")

        if verbose:
            logging.info(
                f"""\nInitial Population:
                ->  shape: {population.shape}
                ->  Best Fitness: {max(fitness_list)}"""
            )

        # best
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history["min"].append(np.min(fitness_list))
        self.history["mean"].append(np.mean(fitness_list))

        # Repeat generations
        best_iter = 0
        for g in range(self.num_generations):

            # early stopping
            if g - best_iter > patient:
                self.used_bins = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    logging.info(f"Early stop at iter {g} (timeout)")
                return

            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(
                population, fitness_list
            )

            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)

            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            new_population = np.concatenate((mutants, offsprings), axis=0)
            new_population_fitness_list = self.cal_fitness(new_population)

            population = np.concatenate((elites, new_population), axis=0)
            fitness_list = np.concatenate(
                (elite_fitness_list, new_population_fitness_list), axis=0
            )

            # Ensure population size consistency
            if len(population) > self.num_individuals:
                population = population[: self.num_individuals]
                fitness_list = fitness_list[: self.num_individuals]

            # Update Best Fitness
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]

            self.history["min"].append(np.min(fitness_list))
            self.history["mean"].append(np.mean(fitness_list))

            if verbose:
                logging.info(f"Generation : {g} \t(Best Fitness: {best_fitness})")

        self.best_fitness = best_fitness
        self.solution = best_solution

    def get_placement(self) -> PlacementProcedure:
        """
        Get the placement procedure of the best solution.

        Returns:
            Packing of the best solution
        """
        return PlacementProcedure(self.inputs, self.solution)
