from abc import ABC, abstractmethod

# typing
from typing import List, Dict, Any
from models.uld import ULD
from models.package import Package


class Solver(ABC):
    def __init__(self, ulds: List[ULD], packages: List[Package]):
        """
        Args:
            ulds: List of ULDs
            packages: List of packages
        """
        self.ulds = ulds
        self.packages = packages
        self.only_check_fits = False

    @abstractmethod
    def _solve(self):
        """
        Solve the problem
        """
        pass

    @abstractmethod
    def _get_result(self):
        """
        Get the result of the solving process
        """
        pass

    @abstractmethod
    def _parse_result(self, result: Dict[str, Any]):
        """
        Parse the result of the solving process
        """
        pass

    @abstractmethod
    def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        """
        Check if the packages fit in the ULDs
        """
        pass

    def solve(self, only_check_fits: bool = False):
        """
        Start the solving process

        Args:
            only_check_fits: If True, only check if the packages fit in the ULDs (does not update the Package data)
        """

        self.only_check_fits = only_check_fits

        self._solve()

    def check_all_fit(self) -> bool:
        """
        Check if all packages fit in the ULDs
        """
        for package in self.packages:
            if package.uld_id == None:
                return False

        return True

    def get_fit(self) -> bool:
        """
        Get the result of the solving process
        """

        result = self._get_result()

        if self.only_check_fits:
            valid = self._only_check_fits(result)
        else:
            self._parse_result(result)
            valid = self.check_all_fit()

        return valid
