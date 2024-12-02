from .gurobi_strategy import GurobiStrategy
from .drl_strategy import DRLStrategy

strategies = {
    "drl": DRLStrategy,
    "gurobi": GurobiStrategy
}
