"""
SatLins layer class
Author: Yuntao Li
Date: 1/20/2024
"""
from StarV.fun.satlins import SatLins
from typing import List, Union
from StarV.set.probstar import ProbStar
from StarV.set.star import Star

class SatLinsLayer:
    """
    SatLinsLayer class for qualitative and quantitative reachability analysis.
    
    This class provides methods for evaluating SatLins activation and
    computing reachable sets for SatLins layers.
    """
    
    @staticmethod
    def evaluate(x):
        """
        Evaluate the SatLins function on input x.

        Args:
            x: Input data.

        Returns:
            Result of applying SatLins to x.
        """
        return SatLins.evaluate(x)
    
    @staticmethod
    def reach(In: List[Union[Star, ProbStar]], method: str = 'exact', 
              lp_solver: str = 'gurobi', pool = None, RF: float = 0.0) -> List[Union[Star, ProbStar]]:
        """
        Main reachability method for SatLins layer.

        Args:
            In: A list of input sets (Star or ProbStar).
            method: Reachability method: 'exact', 'approx', or 'relax'.
            lp_solver: LP solver: 'gurobi' (default), 'glpk', or 'linprog'.
            pool: Parallel pool (None or multiprocessing.pool.Pool).
            RF: Relax-factor from 0 to 1 (0 by default).

        Returns:
            A list of reachable sets.

        Raises:
            ValueError: If an unknown reachability method is specified.
            NotImplementedError: If 'approx' or 'relax' methods are used (currently under development).
        """
        print("\nSatLinsLayer reach function\n")
        
        if method == 'exact':
            return SatLins.reachExactMultiInputs(In, lp_solver, pool)
        elif method in ['approx', 'relax']:
            raise NotImplementedError(f"The '{method}' method is currently under development.")
        else:
            raise ValueError(f"Unknown reachability method: {method}")