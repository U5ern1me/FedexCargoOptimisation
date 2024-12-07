from .threeD_bin_packing_solver import ThreeDBinPackingSolver
from .mhpa_solver import MHPASolver
from .slicing_algorithm_solver import SlicingAlgorithmSolver

solvers = {
    "threeD_bin_packing": ThreeDBinPackingSolver,
    "mhpa": MHPASolver,
    "slicing_algorithm": SlicingAlgorithmSolver,
}
