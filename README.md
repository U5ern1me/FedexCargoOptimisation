# Optimal Cargo Management for Flights by Team 27

## Introduction

This repository implements various strategies to solve the given problem. The goal is to efficiently pack a set of packages into Unit Load Devices (ULDs) while considering constraints such as dimensions, weight, and priority levels.

## Project Structure
```
.
├── data/
│   ├── k.txt
│   ├── package.csv
│   └── uld.csv
├── output/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── package.py
│   │   └── uld.py
│   ├── service.py
│   ├── solvers/
│   │   ├── slicing_algorithm_solver/
│   │   │   ├── slicing_algorithm_utils.py
│   │   │   └── ...
│   │   ├── threeD_bin_packing_solver/
│   │   ├── mhpa_solver/
│   │   └── solver.py
│   ├── strategies/
│   │   ├── brkga_strategy/
│   │   │   ├── brkga_utils.py
│   │   │   └── ...
│   │   ├── greedy_heuristic_strategy/
│   │   │   ├── greedy_heuristic_strategy.py
│   │   │   └── greedy_heuristic_utils.py
│   │   ├── gurobi_strategy/
│   │   ├── drl_strategy/
│   │   └── ...
│   ├── utils/
│   └── ...
└── venv/
```

## Prerequisites
Python version >3.12.3
```bash
pip install -r requirements.txt
```
For visualisation make sure you have `tkinter`
```bash
sudo apt install python3-tk
```

## Quickstart Guide
To run the main application:
```bash
python src/main.py
```

## How to Run Different Strategies
To specify a strategy, use the `-s` flag followed by the strategy name:

```bash
python src/main.py -s [strategy_name] -d -v
```

Replace [strategy_name] with one of the following:
- greedy_heuristic
- genetic_algorithm
- brkga
- gurobi
- hexaly
- drl

The `-d` flag enables debug mode, and `-v` enables verbose output

To run it for a different dataset, update the `package.csv`, `uld.csv` and `k.txt` files in the data folder.

## Configuration
Configuration files are located in the src directory and allow customization of each strategy:
- Main Configuration: `main.config`
- Greedy Heuristic: `greedy_heuristic.config`
- BRKGA: `brkga.config`
- Gurobi: `gurobi.config`
- DRL: `drl.config`
- Genetic Algorithm: `genetic_algorithm.config`
- Hexaly: `hexaly.config`

## Service
We also have a service that can be run by
```bash
fastapi run src/service.py
```
