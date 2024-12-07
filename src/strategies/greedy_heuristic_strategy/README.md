# Greedy Heuristic for 3D Bin Packing

This repository implements a **Greedy Heuristic Strategy** for solving the **3D bin packing** problem. The problem involves packing **packages** into **Unit Load Devices (ULDs)** while considering constraints such as **weight**, **dimensions**, and **priority**. This heuristic approach aims to minimize delay and spread costs while ensuring an efficient packing of both priority and economic packages.

---

## Features

- **3D Bin Packing Algorithm**: Efficiently packs packages into ULDs, considering both spatial and weight constraints.
- **Priority Packages Handling**: Ensures priority packages are packed first to reduce delay and associated costs.
- **Economic Packages Sorting**: Sorts and places economic packages based on delay cost per kilogram to minimize the overall cost.
- **Heuristic Approach**: Utilizes custom sorting heuristics based on average densities of packages and ULDs.
- **Configurable Parameters**: Allows for the adjustment of solver type and error tuning through a configuration file.

---

## Installation

### Prerequisites

- Python 3.7+
- Required libraries: Install the dependencies via `pip` by running:

```bash
pip install -r requirements.txt
```
## How to run

To run the Genetic Algorithm and solve the 3D Bin Packing Problem, follow the steps below:

### Method 1: Modify the Configuration
Ensure that the `main.config` file is correctly set up by setting the `default_strategy` to `greedy_heuristic`. This will configure the optimizer to use the Genetic Algorithm approach.

### Method 2: Using the command line
```bash
python src/main.py -s greedy_heuristic -d -v
```
### Project Structure

The project contains the following files:

```
greedy_heuristic/
├── __init__.py               # Initializes the package and makes it importable
├── greedy_heuristic.config   # Configuration file to specify solver and error tuning settings
├── greedy_heuristic_strategy.py  # Contains the GreedyHeuristicStrategy class (main logic)
├── greedy_heuristic_utils.py    # Utility functions for sorting, ULD splitting, and package fitting
└── solver.py                # The underlying solver used for checking package fits
```

- `__init__.py`: Initializes the package and makes it importable as a module.
- `greedy_heuristic.config`: Configuration file to specify solver and error tuning settings.
- `greedy_heuristic_strategy.py`: Contains the main logic for the Greedy Heuristic Strategy.
- `greedy_heuristic_utils.py`: Utility functions for packing, sorting, and handling ULD splits.
- `solver.py`: The underlying solver used for checking if packages fit into ULDs.

---

## Key Components

### **GreedyHeuristicStrategy** (`greedy_heuristic_strategy.py`)

The core of this strategy is the `GreedyHeuristicStrategy` class, which implements the greedy heuristic for solving the bin packing problem. It is responsible for:

- **Calculating Average Densities**: The strategy calculates the average density for packages and ULDs based on their weight and volume to guide the sorting of packages.
- **Dividing Packages into Priority and Economic**: The strategy segregates the packages into priority packages (which must be packed first) and economic packages (which are packed second based on cost efficiency).
- **Solving the Bin Packing Problem**: The algorithm iteratively checks ULD splits and evaluates the fit of both priority and economic packages using an external solver.

### **Solver** (`solver.py`)

The solver is responsible for checking if packages fit into the ULDs. It is configured dynamically based on the selected solver in the configuration file (`greedy_heuristic.config`). The solver works in conjunction with the heuristic strategy to ensure optimal packing.

### **Utilities** (`greedy_heuristic_utils.py`)

Utility functions in this file provide support for:

- **Sorting Packages**: Packages are sorted based on various heuristics such as delay cost per kilogram and density.
- **ULD Splitting**: The ULDs are split into possible configurations, and valid splits are evaluated for packing feasibility.
- **Package Fitting**: Functions like `virtual_fit_priority` and `find_splits_economic_packages` help check if the packages can fit into the given ULDs.

---

## How It Works

### 1. **Density Calculations**
The strategy calculates the average densities of both packages and ULDs. The density is calculated as:

- **Package Density**: `package.weight / (package.length * package.width * package.height)`
- **ULD Density**: `uld.weight_limit / (uld.length * uld.width * uld.height)`

The strategy uses these values to determine how best to sort the packages and divide them into priority and economic groups.

### 2. **Sorting Heuristics**
Based on the average densities, two sorting heuristics are defined:

- **Sorting Heuristic 1**: Used for sorting economic packages based on delay costs and weight.
- **Sorting Heuristic 2**: Used for sorting the remaining packages after the first division.

### 3. **Package Division**
Packages are divided into **priority** and **economic** categories. Priority packages are packed first to ensure timely delivery, while economic packages are packed with a focus on minimizing delay cost.

### 4. **ULD Splitting and Validation**
All potential splits of ULDs are generated. For each split, the heuristic checks whether the priority packages can fit. If valid, the split is kept for further optimization of economic packages.

### 5. **Error Tuning**
The error tuning parameter allows for slight adjustments to the packing process to handle edge cases or fitting issues. This parameter controls the number of remaining packages to be checked against the ULDs.

### 6. **Final Packing**
Once the optimal split and packing strategy is determined, the solver is invoked to finalize the packing and check the fit for each partition.

---

## Configuration (`greedy_heuristic.config`)

The configuration file is crucial for customizing the solver and tuning the packing process. Here is an example configuration:

- `solver`(`int`): Specifies which solver to use for packing (e.g., `threeD_bin_packing`).
- `error tuning`(`int`): A parameter to control the flexibility in fitting packages into the ULDs (e.g., 65).

---


---

## File Overview

- **`greedy_heuristic.config`**: Configuration file where you specify the solver and tuning parameters.
  
- **`greedy_heuristic_strategy.py`**: Main file containing the `GreedyHeuristicStrategy` class, responsible for the core packing logic.
  
- **`greedy_heuristic_utils.py`**: Helper utilities for sorting, splitting ULDs, and evaluating packing validity.
  
- **`solver.py`**: Contains the underlying 3D bin packing solver, which checks if packages fit into ULDs.

---


- The packing heuristics and solver were inspired by industry best practices for logistics and transportation.
- Special thanks to the open-source community for the development of 3D bin packing algorithms.
