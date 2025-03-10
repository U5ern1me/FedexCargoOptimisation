# Mixed Integer Linear Programming (MILP) Using Hexaly Optimizer

We have approached the Problem by formulating it as a Mixed Integer Linear Programming (MILP) problem and using Hexaly Optimizer as a solver. The constraints are designed to ensure that each package is uniquely assigned to a container, no packages overlap in any dimension, and all packages fit within the boundaries and weight limits of their assigned containers. Orientation constraints enforce valid placement and alignment of packages within the containers, while additional constraints manage relative positioning of the packages. A Buffer ULD, large enough is added in which all the packages that don't fit into the main ULDs are allocated. The objective function minimizes the total cost by reducing delay costs for packages not assigned to main ULDs and minimizing the spread of priority packages across multiple containers.

The main class used for this strategy can be found in `hexaly_strategy.py`. Utility functions used within the strategy have been created in `hexaly_utils.py`.

## How to Run

The strategy can be run in two ways:
### Method 1: Modify the Configuration

Modify `main.config` file by changing the value `default_strategy` to `hexaly`. This will configure the optimizer to use the Hexaly Strategy. 

### Method 2: Using the command line

```bash
python src/main.py -s hexaly -d -v
```

### Obtaining a License
To run the strategy, install [Hexaly Optimizer](https://www.hexaly.com/docs/last/installation/index.html) and obtain a license by following the instructions given on their website. 

## Configuration
The `hexaly.config` file can be modified to configure the strategy.

```
{
    "total_timesteps": integer //Total number of timesteps (in seconds) to train the model.
}
```