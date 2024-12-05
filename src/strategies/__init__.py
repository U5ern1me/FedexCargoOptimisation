from .gurobi_strategy import GurobiStrategy
from .drl_strategy import DRLStrategy
from .hexaly_strategy import HexalyStrategy
from .brkga_strategy import BRKGAStrategy
from .greedy_heuristic_strategy import GreedyHeuristicStrategy

strategies = {
    "drl": DRLStrategy,
    "gurobi": GurobiStrategy,
    "hexaly": HexalyStrategy,
    "brkga": BRKGAStrategy,
    "greedy_heuristic": GreedyHeuristicStrategy,
}
