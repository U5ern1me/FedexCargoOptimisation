from .gurobi_strategy import GurobiStrategy
from .drl_strategy import DRLStrategy
from .hexaly_strategy import HexalyStrategy
from .brkga_strategy import BRKGAStrategy
from .genetic_3D_bin_packing import Genetic3DBinPackingStrategy
from .greedy_heuristic_strategy import GreedyHeuristicStrategy

strategies = {
    "drl": DRLStrategy,
    "gurobi": GurobiStrategy,
    "hexaly": HexalyStrategy,
    "brkga": BRKGAStrategy,
    "genetic_3D_bin_packing": Genetic3DBinPackingStrategy,
    "greedy_heuristic": GreedyHeuristicStrategy,
}
