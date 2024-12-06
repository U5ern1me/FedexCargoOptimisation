from abc import ABC, abstractmethod

import aiohttp

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
    async def _solve(self, session: aiohttp.ClientSession = None):
        """
        Solve the problem
        """
        pass

    @abstractmethod
    async def _get_result(self, session: aiohttp.ClientSession = None):
        """
        Get the result of the solving process
        """
        pass

    @abstractmethod
    async def _parse_result(self, result: Dict[str, Any]):
        """
        Parse the result of the solving process
        """
        pass

    @abstractmethod
    async def _only_check_fits(self, result: Dict[str, Any]) -> bool:
        """
        Check if the packages fit in the ULDs
        """
        pass

    async def solve(
        self, only_check_fits: bool = True, session: aiohttp.ClientSession = None
    ):
        """
        Start the solving process

        Args:
            only_check_fits: If True, only check if the packages fit in the ULDs (does not update the Package data)
            session: aiohttp.ClientSession
        """

        self.only_check_fits = only_check_fits

        await self._solve(session=session)

    def check_all_fit(self) -> bool:
        """
        Check if all packages fit in the ULDs
        """
        for package in self.packages:
            if package.uld_id == None:
                return False

        return True

    async def get_fit(self, session: aiohttp.ClientSession = None) -> bool:
        """
        Get the result of the solving process
        """

        result = await self._get_result(session=session)

        if self.only_check_fits:
            valid = await self._only_check_fits(result)
        else:
            await self._parse_result(result)
            valid = self.check_all_fit()

        return valid
