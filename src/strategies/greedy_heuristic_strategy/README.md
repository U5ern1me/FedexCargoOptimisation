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
