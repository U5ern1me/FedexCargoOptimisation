## 3D Bin Packing using Biased Random-Key Genetic Algorithm (BRKGA).

Biased Random-Key Genetic Algorithm (BRKGA) is a genetic algorithm that uses biased random-key encoding to represent the solution. It is a population-based algorithm that uses a combination of mutation, crossover, and selection to evolve the population of solutions.

We utilized the BRKGA-based algorithm from the [paper](https://www.sciencedirect.com/science/article/pii/S0925527313001837) for the 3D bin packing problem.

## How to run

To run the Genetic Algorithm and solve the 3D Bin Packing Problem, follow the steps below:

### Method 1: Modify the Configuration
Ensure that the `main.config` file is correctly set up by setting the `default_strategy` to `brkga`. This will configure the optimizer to use the Genetic Algorithm approach.

### Method 2: Using the command line
```bash
python src/main.py -s brkga -d -v
```


### brkga_strategy.py

This file contains the main class for the BRKGA strategy. This class is used to run the BRKGA strategy.

### brkga_utils.py

This file contains all functions and classes definition along with the model definition for the BRKGA strategy.

### brkga.config

This file contains the configuration for the BRKGA strategy. Can be used to set model parameters like the total number of generations, the number of individuals, the number of elites, the number of mutants, the fraction of biased population, the probability of choosing elite gene, and the number of stable generations.

The following parameters defined in the `brkga.config` file are used to configure the BRKGA strategy:
1. `number of generations` (`int`): The total number of generations to run the algorithm.
2. `number of individuals` (`int`): The number of individuals in the population.
3. `number of elites` (`int`): The number of elites in the population.
4. `number of mutants` (`int`): The number of mutants in the population.
5. `fraction of biased population` (`float`): The fraction of the population that is biased.
6. `probability of choosing elite gene` (`float`): The probability of choosing an elite gene.
7. `number of stable generations` (`int`): The number of generations to run without changing the best solution.
