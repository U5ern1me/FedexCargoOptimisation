# Mixed Integer Linear Programming (MILP) Using Gurobi Optimizer

We have approached the as a Mixed Integer Linear Programming (MILP) problem and using Gurobi Optimizer as a solver. The constraints are designed to ensure that each package is uniquely assigned to a container, no packages overlap in any dimension, and all packages fit within the boundaries and weight limits of their assigned containers. Orientation constraints enforce valid placement and alignment of packages within the containers, while additional constraints manage relative positioning of the packages. A Buffer ULD, large enough is added in which all the packages that don't fit into the main ULDs are allocated. The objective function minimizes the total cost by reducing delay costs for packages not assigned to main ULDs and minimizing the spread of priority packages across multiple containers.


The main class used for this strategy can be found in `gurobi_strategy.py`. Utility functions used within the strategy have been created in `gurobi_utils.py`.

## How to run

The strategy can be run in two ways:
### Method 1: Modify the Configuration
Modify `main.config` file by changing value of the `default_strategy` to `gurobi`. This will configure the optimizer to use the Gurobi approach. 

### Method 2: Using the command line
```bash
python src/main.py -s gurobi -d -v
```

### Setting Up Gurobi
To run the strategy, obtain a [Gurobi License](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license) and install [Gurobi Optimizer](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). Update the License ID in the configuration file.

## Configuration
The `gurobi.config` file can be modified to configure the strategy.

```json
{
    "output flag": 0 | 1 | 2, //The output flag for the Gurobi strategy
    "numeric focus": 0 | 1 | 2, //The numeric focus for the Gurobi strategy
    "presolve": 0 | 1 | 2, //The presolve flag for the Gurobi strategy.
    "license" : string, //License ID
    "log file": string, //Path to the log file for the strategy
    "IIS file": "model.ilp" //Path to log the Irreducible Infeasible Subsystem for the problem
}
```