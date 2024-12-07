# Gurobi Solver for packing  

We utilise the Gurobi-based optimization solver for packing packages into Unit Load Devices (ULDs) efficiently, considering constraints and costs. The GurobiSolver class takes in a list of packages, ULDs, and configuration parameters to set up the optimization model.

The solver creates variables for the position, orientation, and assignment of packages, as well as overlap indicators to prevent collisions between packages. It enforces constraints to handle package overlap, weight limits, priority package placement, and valid orientations.

The objective function minimizes the total cost, which includes the penalty for using priority ULDs and delay costs for unassigned packages. This ensures an optimal and feasible packing strategy.

The solver integrates with the Gurobi optimizer, leveraging its efficient environment configuration and constraint-handling capabilities for large-scale optimization problems in logistics.

## How to run

To run the Gurobi Optimizer and solve the 3D Bin Packing Problem, follow the steps below:
### Step 1: 
Uncomment the gurobi strategy in `src/strategies/__init__.py`.
### Step 2:
### Method 1: Modify the Configuration
Ensure that the `main.config` file is correctly set up by setting the `default_strategy` to `gurobi`. This will configure the optimizer to use the Gurobi approach.

### Method 2: Using the command line
```bash
python src/main.py -s gurobi -d -v
```

## gurobi_strategy.py

This file contains the main class for the Gurobi strategy. This class is used to run the Gurobi strategy.

## gurobi_utils.py

This file contains all functions and classes definition along with the model definition for the Gurobi strategy.

## gurobi.config

This file contains the configuration for the Gurobi strategy. Can be used to set parameters like the output flag, numeric focus, presolve, license, log file, and IIS file.

Parameters defined in the `gurobi.config` file:

1. `output flag` (`int`): The output flag for the Gurobi strategy. Can be 0, 1, or 2. <br>
2. `numeric focus` (`int`): The numeric focus for the Gurobi strategy. Can be 0, 1, or 2. <br>
3. `presolve` (`int`): The presolve flag for the Gurobi strategy. Can be 0, 1, or 2. <br>
4. `license` (`str`): The license for the Gurobi strategy. <br>
5. `log file` (`str`): The log file for the Gurobi strategy. <br>
6. `IIS file` (`str`): The IIS file for the Gurobi strategy. <br>
