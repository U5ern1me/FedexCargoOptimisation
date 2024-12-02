import copy
import math
import random
import time
import multiprocessing
import numpy as np


class Bin:
    def __init__(self, uld, verbose=False):
        self.dimensions = (uld.length, uld.width, uld.height)
        self.EMSs = [[np.array((0, 0, 0)), np.array(self.dimensions)]]
        self.weight_capacity = uld.weight_limit
        self.id = uld.id
        self.boxes = []
        self.weight = 0
        self.has_priority = False

        if verbose:
            print("Init EMSs:", self.EMSs)

    def __getitem__(self, index):
        return self.EMSs[index]

    def __len__(self):
        return len(self.EMSs)

    def update(
        self, box, box_dimensions, selected_EMS, min_vol, min_dim, verbose=False
    ):
        """
        Add the box to the bin and update the EMSs of the bin.
        """
        new_box = np.array(box_dimensions)
        isValid = True
        if min_dim is not None and np.any(
            new_box < min_dim
        ):  # Ensure min_dim is not None
            isValid = False
            if verbose:
                print("Box is invalid due to dimensions:", new_box)

        # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
        if np.prod(new_box) < min_vol:  # Change from np.product to np.prod
            isValid = False
            if verbose:
                print("Box is invalid due to volume:", new_box)

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
            print("------------\n*Place Box*:\nEMS:", list(map(tuple, ems)))

        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):

                # eliminate overlapped EMS
                self.eliminate(EMS)

                if verbose:
                    print(
                        "\n*Elimination*:\nRemove overlapped EMS:",
                        list(map(tuple, EMS)),
                        "\nEMSs left:",
                        list(map(lambda x: list(map(tuple, x)), self.EMSs)),
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
                        print("\n*New*\nEMS:", list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print(
                                    "-> Totally inscribed by:",
                                    list(map(tuple, other_EMS)),
                                )

                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.any(
                        new_box < min_dim
                    ):  # Change from np.min(new_box) < min_dim to np.any(new_box < min_dim)
                        isValid = False
                        if verbose:
                            print("Box is invalid due to dimensions:", new_box)

                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.prod(new_box) < min_vol:  # Change from np.product to np.prod
                        isValid = False
                        if verbose:
                            print("Box is invalid due to volume:", new_box)

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print("-> Success\nAdd new EMS:", list(map(tuple, new_EMS)))

        if verbose:
            print("\nEnd:")
            print("EMSs:", list(map(lambda x: list(map(tuple, x)), self.EMSs)))

    def overlapped(self, ems, EMS):
        """
        Check if the EMSs are overlapped.
        """
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False

    def inscribed(self, ems, EMS):
        """
        Check if the EMS is inscribed by another EMS.
        """
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False

    def eliminate(self, ems):
        """
        Eliminate the EMS from the list of EMSs.
        """
        ems = list(map(tuple, ems))
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return

    def get_EMSs(self):
        """
        Get the EMSs of the bin.
        """
        return list(map(lambda x: list(map(tuple, x)), self.EMSs))

    def weight_packing_efficiency(self):
        """
        Get the weight packing efficiency of the bin.
        """
        return sum(box.weight for box in self.boxes) / (self.weight_capacity)

    def volume_packing_efficiency(self):
        """
        Get the volume packing efficiency of the bin.
        """
        return sum(
            (box.length * box.width * box.height) for box in self.boxes
        ) / np.prod(self.dimensions)


class PlacementProcedure:
    def __init__(self, inputs, solution, verbose=False):
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
            print("------------------------------------------------------------------")
            print("|   Placement Procedure")
            print("|    -> Bins:", self.Bins)
            print("|    -> Bin Order:", self.BinOrder)
            print("|    -> Boxes:", self.boxes)
            print("|    -> Box Packing Sequence:", self.BPS)
            print("|    -> Vector of Box Orientations:", self.VBO)
            print("-------------------------------------------------------------------")

        self.placement()

    def placement(self):
        """
        Place the boxes in the bins using the packing order.
        """
        items_sorted = [self.boxes[i] for i in self.BPS]

        # Box Selection
        for i, box in enumerate(items_sorted):
            if self.verbose:
                print("Select Box:", box)

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
                        print(f"No more bin to open. [Cannot fit box {box.id}]")
                    return

                selected_EMS = self.Bins[selected_bin].EMSs[0]  # origin of the new bin
                if self.verbose:
                    print("No available bin... open bin", selected_bin)

            if self.verbose:
                print("Select EMS:", list(map(tuple, selected_EMS)))

            # Box orientation selection
            BO = self.selecte_box_orientaion(self.VBO[i], box, selected_EMS)

            # elimination rule for different process
            min_vol, min_dim = self.elimination_rule(items_sorted[i + 1 :])

            # pack the box to the bin & update state information
            self.Bins[selected_bin].update(
                box, self.orient(box, BO), selected_EMS, min_vol, min_dim
            )

            if self.verbose:
                print("Add box to Bin", selected_bin)
                print(" -> EMSs:", self.Bins[selected_bin].get_EMSs())
                print("------------------------------------------------------------")
        if self.verbose:
            print("|")
            print("|     Number of used bins:", self.num_opend_bins)
            print("|")
            print("------------------------------------------------------------")

    def DFTRC_2(self, box, k):
        """
        Select the EMS that is the farthest from the front-top-right corner of the bin.
        DFRTC ~ Distance to the Front-Top-Right Corner
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

    def orient(self, box, BO=1):
        """
        Orient the box based on the orientation code.
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

    def selecte_box_orientaion(self, VBO, box, EMS):
        """
        Select the box orientation based on the VBO vector.
        """
        # feasible direction
        BOs = []
        for direction in [1, 2, 3, 4, 5, 6]:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)

        selectedBO = BOs[math.ceil(VBO * len(BOs)) - 1]

        if self.verbose:
            print("Select VBO:", selectedBO, "  (BOs", BOs, ", vector", VBO, ")")
        return selectedBO

    def fitin(self, box, EMS):
        """
        Check if the box fits in the EMS.
        """
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True

    def elimination_rule(self, remaining_boxes):
        """
        Find the box with the smallest volume and the smallest dimension.
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
        inputs,
        num_generations=200,
        num_individuals=120,
        num_elites=12,
        num_mutants=18,
        fraction_biased=0.4,
        eliteCProb=0.7,
        multiProcess=False,
        seed=None,
    ):
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

    def decoder(self, solution):
        """
        Decode the solution to the placement procedure and evaluate the fitness.
        """
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()

    def cal_fitness(self, population):
        """
        Calculate the fitness of the population.
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

    def partition(self, population, fitness_list):
        """
        Partition the population into elites and non-elites.
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

    def crossover_mixed(self, elite, non_elite):
        """
        Chance to choose the gene from elite individual and non_elite individual for each gene.
        """
        return [
            (
                elite[gene]
                if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb
                else non_elite[gene]
            )
            for gene in range(self.num_gene)
        ]

    def crossover_elite(self, elite_1, elite_2):
        """
        Crossover between two elites individuals.
        """
        return [
            (
                elite_1[gene]
                if np.random.uniform(low=0.0, high=1.0) < 0.5
                else elite_2[gene]
            )
            for gene in range(self.num_gene)
        ]

    def biased_population(self, num_individuals):
        """
        Generate a biased population, where the genes of priority ULDs are halved.
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

    def random_population(self, num_individuals):
        """
        Generate a random population.
        """
        return np.random.uniform(
            low=0.0, high=1.0, size=(num_individuals, self.num_gene)
        )

    def mating(self, elites, non_elites):
        """
        Biased selection of mating parents: 1 elite & 1 non_elite.
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

    def fit(self, patient=4, verbose=False):
        """
        Main loop of the genetic algorithm.
        patient ~ number of generations without improvement before stopping.
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
            print("\nInitial Population:")
            print("  ->  shape:", population.shape)
            print("  ->  Best Fitness:", max(fitness_list))

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
                    print("Early stop at iter", g, "(timeout)")
                return "feasible"

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
                print("Generation :", g, " \t(Best Fitness:", best_fitness, ")")

        self.best_fitness = best_fitness
        self.solution = best_solution

    def get_placement(self):
        """
        Get the placement procedure of the best solution.
        """
        return PlacementProcedure(self.inputs, self.solution)
