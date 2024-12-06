from .sardine_can_solver import SardineCanSolver
from .threeD_bin_packing_solver import ThreeDBinPackingSolver

from utils.api_error import APIError

# type hinting
from models.package import Package
from models.uld import ULD
from typing import List

SOLVERS = {
    "sardine_can": SardineCanSolver,
    "threeD_bin_packing": ThreeDBinPackingSolver,
}

FALLBACKS = {
    "threeD_bin_packing": "sardine_can",
    "sardine_can": "sardine_can",
}


class SolverWrapper:
    def __init__(self, packages: List[Package], ulds: List[ULD], base_solver: str):
        self.packages = packages
        self.ulds = ulds
        self.base_solver = base_solver

        if base_solver not in SOLVERS:
            raise ValueError(f"Invalid solver: {base_solver}")

        self.solver_class = SOLVERS[base_solver]

    async def solve(self, *args, **kwargs):
        fall_back_count = 0
        error = None
        self.solve_args = args
        self.solve_kwargs = kwargs
        while fall_back_count < 2:
            try:
                self.solver = self.solver_class(self.packages, self.ulds)
                return await self.solver.solve(*args, **kwargs)
            except APIError as e:
                error = e
                fall_back_count += 1
                self.base_solver = FALLBACKS[self.base_solver]
                self.solver_class = SOLVERS[self.base_solver]
            except Exception as e:
                raise e

        raise error

    async def get_fit(self, *args, **kwargs):
        try:
            return await self.solver.get_fit(*args, **kwargs)
        except APIError as e:
            await self.solve(*self.solve_args, **self.solve_kwargs)
            return await self.solver.get_fit(*args, **kwargs)
        except Exception as e:
            raise e

    async def get_packing_json(self, *args, **kwargs):
        try:
            return await self.solver.get_packing_json(*args, **kwargs)
        except APIError as e:
            await self.solve(*self.solve_args, **self.solve_kwargs)
            return await self.solver.get_packing_json(*args, **kwargs)
        except Exception as e:
            raise e
