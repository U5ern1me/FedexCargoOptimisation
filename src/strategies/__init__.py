from .gurobi_strategy import GurobiStrategy
from .drl_strategy import DRLStrategy
from .hexaly_strategy import HexalyStrategy
from .brkga_strategy import BRKGAStrategy

strategies = {
    "drl": DRLStrategy,
    "gurobi": GurobiStrategy,
    "hexaly": HexalyStrategy,
    "brkga": BRKGAStrategy,
}
