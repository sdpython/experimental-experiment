import random
import time
from enum import IntEnum
from typing import Any, Dict, List


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
        self.verbose = verbose

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(..., {self.algorithm})"

    def optimize(self) -> List[Dict[str, Any]]:
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
            return [dict(pattern="order", time_in=duration)]
        if self.algorithm == OrderAlgorithm.RANDOM:
            if self.verbose:
                print(f"[OrderOptimization.optimize] {self.algorithm}: does nothing")
            duration = time.perf_counter() - begin
            stats = self.random_order()
            if self.verbose:
                print(f"[OrderOptimization.optimize] done in {duration}")
            stats.append(
                dict(pattern="order", algo=str(self.algorithm), time_in=duration)
            )
            return stats

        raise AssertionError(f"Unsupported algorithm {self.algorithm}.")

    def _position(self):
        output = {}
        first_input = {}
        for i, node in enumerate(self.builder.nodes):
            if node is None:
                continue
            for name in node.input:
                if name not in first_input:
                    first_input[name] = i
            for name in node.output:
                output[name] = i
        couples = []
        N = len(self.builder.nodes)
        for node in self.builder.nodes:
            if node is None:
                continue
            minp = max(first_input.get(i, 0) for i in node.input)
            maxp = min(first_input.get(i, N) for i in node.output)
            couples.append((minp, maxp))
        return couples

    def random_order(self):
        """
        Moves nodes by random.
        """
        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[OrderOptimization.random_order] start with {len(self.builder.nodes)} nodes"
            )
        n_changes = 0
        n_moved = 0
        expected = len([n for n in self.builder.nodes if n is not None])
        done = set()
        n_iter = 0
        while len(done) < expected:
            if self.verbose > 1:
                print(
                    f"[OrderOptimization.random_order] start iter={n_iter} with {len(self.builder.nodes)} nodes"
                )
            n_iter += 1
            couples = self._position()
            i = 0
            safe = 0
            while i < len(self.builder.nodes):
                node = self.builder.nodes[i]
                if node is None:
                    i += 1
                    continue
                if id(node) in done:
                    i += 1
                    continue
                mi, ma = couples[i]
                if self.verbose >= 10:
                    print(
                        f"[OrderOptimization.random_order] iter={n_iter} i={i} "
                        f"mi,ma={mi},{ma} safe={safe} op_type={node.op_type}"
                    )
                if mi < safe:
                    i += 1
                    continue
                assert mi <= ma, f"needed_at={mi}, first_at={ma}"
                if mi >= ma - 1:
                    done.add(id(node))
                    i += 1
                    continue

                new_position = random.randint(mi, ma)
                if self.verbose >= 10:
                    print(
                        f"[OrderOptimization.random_order]    i={i} new_position={new_position}"
                    )
                n_moved += abs(i - new_position)
                if i == new_position:
                    done.add(id(node))
                    i += 1
                    continue
                if i < new_position:
                    self.builder.nodes.insert(new_position, node)
                    del self.builder.nodes[i]
                    i = new_position + 1
                    safe = i
                    n_changes += 1
                    continue

                del self.builder.nodes[i]
                self.builder.nodes.insert(new_position, node)
                safe = i
                i += 1
                n_changes += 1

        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[OrderOptimization.random_order] done after {n_iter} iterations in "
                f"{time.perf_counter() -begin}s with changed={n_changes} scale={n_moved}"
            )
        return [
            dict(
                pattern="random_order",
                changed=n_changes,
                scale=n_moved,
                iter=n_iter,
                time=time.perf_counter() - begin,
            )
        ]
