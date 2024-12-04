# Mixed Integer Linear Programming (MILP) Using Hexaly Optimizer

We have approached the 3D Bin Packing Problem by formulating it as a Mixed Integer Linear Programming (MILP) problem. The constraints are designed to ensure that each package is uniquely assigned to a container, no packages overlap in any dimension, and all packages fit within the boundaries and weight limits of their assigned containers. Orientation constraints enforce valid placement and alignment of packages within the containers, while additional constraints manage relative positioning of the packages. A Buffer ULD, large enough is added in which all the packages that don't fit into the main ULDs are allocated. The objective function minimizes the total cost by reducing delay costs for packages not assigned to main ULDs and minimizing the spread of priority packages across multiple containers.



## How to Run

To run the Hexaly Optimizer and solve the 3D Bin Packing Problem, follow the steps below:

### Method 1: Modify the Configuration

Ensure that the `main.config` file is correctly set up by changing the `default_strategy` to `hexaly`. This will configure the optimizer to use the Hexaly approach. 

### Method 2: Using the command line

```bash
python src/main.py -s hexaly -d
```

## hexaly.config

This file contains the configuration for the hexaly strategy. Can be used to set the parameter for the MILP problem.

 Parameters defined in the hexaly.config file:<br>
    1. `total_timesteps` (int): The total number of timesteps to train the model.




