# from .gurobi_strategy import GurobiStrategy
# from .drl_strategy import DRLStrategy
# from .hexaly_strategy import HexalyStrategy
from .brkga_strategy import BRKGAStrategy
from .genetic_algorithm_strategy import GeneticAlgorithmStrategy
from .greedy_heuristic_strategy import GreedyHeuristicStrategy

strategies = {
    # "drl": DRLStrategy,
    # "gurobi": GurobiStrategy,
    # "hexaly": HexalyStrategy,
    "brkga": BRKGAStrategy,
    "genetic_algorithm": GeneticAlgorithmStrategy,
    "greedy_heuristic": GreedyHeuristicStrategy,
}
