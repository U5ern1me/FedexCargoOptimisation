# Genetic Algorithm for 3D Bin Packing





## How to run

To run the Genetic Algorithm and solve the 3D Bin Packing Problem, follow the steps below:

### Method 1: Modify the Configuration
Ensure that the `main.config` file is correctly set up by setting the `default_strategy` to `genetic`. This will configure the optimizer to use the Genetic Algorithm approach.

### Method 2: Using the command line
bash
python src/main.py -s genetic -d

## genetic_3D_bin_packing.py

This file contains the main class for the Genetic Algorithm. This class is used to run the Genetic Algorithm.

## genetic_utils.py

This file contains the utility functions for the Genetic Algorithm. This class is used to run the Genetic Algorithm.

## genetic.config

This file contains the configuration for the Genetic Algorithm. This class is used to run the Genetic Algorithm.

Parameters that can be defined in the `genetic.config` file:

1. `number of generations` (`int`): The number of generations to run the Genetic Algorithm.<br  >
2. `number of individuals` (`int`): The number of individuals in the population.<br>
3. `mutation bracket size` (`int`): The size of the mutation bracket.<br>
4. `min 1` (`int`): The minimum value for the first dimension.<br>
5. `max 1` (`int`): The maximum value for the first dimension.<br>
6. `min 2` (`int`): The minimum value for the second dimension.<br>
7. `max 2` (`int`): The maximum value for the second dimension.<br>
8. `uld map for priority` (`dict`): The ULD map for priority.<br>
9. `number of stable generations` (`int`): The number of stable generations.<br>
10. `probability of choosing elite gene` (`float`): The probability of choosing an elite gene.<br>

Example:

