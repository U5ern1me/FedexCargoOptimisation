from .threeD_bin_packing_solver import ThreeDBinPackingSolver
from .sardine_can_solver import SardineCanSolver
from .slicing_algorithm_solver import SlicingAlgorithmSolver

solvers = {
    "threeD_bin_packing": ThreeDBinPackingSolver,
    "sardine_can": SardineCanSolver,
    "slicing_algorithm": SlicingAlgorithmSolver,
}
