import time
from enum import IntEnum
from typing import Any, Dict


class OrderAlgorithm(IntEnum):
    """
    Defines the possible order algorithm.

    * NONE: does not change anything
    * RANDOM: random order
    """

    NONE = 0
    RANDOM = 1


class OrderOptimization:
    """
    Optimizes the order of computation.

    :param builder: GraphBuilder holding the model
    :param algorithm: to apply
    :param verbose: verbosity
    """

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        algorithm: OrderAlgorithm = OrderAlgorithm.NONE,
        verbose: int = 0,
    ):
        self.builder = builder
        self.algorithm = algorithm

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(..., {self.algorithm})"

    def optimize(self) -> Dict[str, Any]:
        """
        Optimizes the model inplace. It optimizes the model in the builder
        itself by switching nodes.
        """
        begin = time.perf_counter()

        if self.algorithm == OrderAlgorithm.NONE:
            if self.verbose:
                print(f"[OrderOptimization.optimize] {self.algorithm}: does nothing")
            duration = time.perf_counter() - begin
            if self.verbose:
                print(f"[OrderOptimization.optimize] done in {duration}")
            return [
                dict(
                    pattern="order",
                    time_in=duration,
                )
            ]

        raise AssertionError(f"Unsupported algorithm {self.algorithm}.")
