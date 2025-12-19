import random
import time
from enum import IntEnum
from typing import Any, Dict, List
from ..helpers import make_idn


class OrderAlgorithm(IntEnum):
    """
    Defines the possible order algorithm.

    * ``NONE``: does not change anything
    * ``RANDOM``: random order
    * ``SHAPE``: moves every shape node just behind the node producing its input
    """

    NONE = 0
    RANDOM = 1
    SHAPE = 2


class OrderOptimization:
    """
    Optimizes the order of computation.
    It tries to minimize the distance between a producer and the consumer
    or a results. The idea is to reduce the memory usage.

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
        self.algorithm = (
            getattr(OrderAlgorithm, algorithm) if isinstance(algorithm, str) else algorithm
        )
        self.verbose = verbose

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(..., {self.algorithm})"

    def optimize(self) -> List[Dict[str, Any]]:
        """
        Optimizes the model inplace. It optimizes the model in the builder
        itself by switching nodes.
        """
        stats = []
        if self.verbose:
            print(f"[OrderOptimization.optimize] ALGO-{self.algorithm}")
        if self.algorithm == OrderAlgorithm.NONE:
            pass
        elif self.algorithm == OrderAlgorithm.RANDOM:
            if self.verbose:
                print(f"[OrderOptimization.optimize] ALGO-{self.algorithm}")
            stats = self.random_order()
            stats.append(dict(pattern="order", algo=str(self.algorithm)))
        elif self.algorithm == OrderAlgorithm.SHAPE:
            stats = self.shape_order()
            stats.append(dict(pattern="order", algo=str(self.algorithm)))
        else:
            raise AssertionError(f"Unsupported algorithm {self.algorithm}.")
        return stats

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
            minp = max(
                output.get(i, 0),
                max((first_input.get(i, 0) for i in node.input)) if node.input else 0,
            )
            maxp = min(first_input.get(i, N) for i in node.output)
            couples.append((minp, maxp))
        return couples

    def _check(self, stats, step):
        begin = time.perf_counter()
        assert (
            len(self.builder.nodes) > 0
        ), f"The onnx model is empty (step {step}, no node).\n{self.builder.get_debug_msg()}"
        known = set(n.name for n in self.builder.inputs)
        known |= set(self.builder.initializers_dict)
        for node in self.builder.nodes:
            assert (
                node.domain in self.builder.opsets
            ), f"Domain {node.domain!r} is not registered in {self.builder.opsets}"
            for i in node.input:
                if i == "":
                    continue
                assert i in known, f"Unknown input {i!r}, step {step!r} in node {node}"
            known |= set(node.output)
        for o in self.builder.outputs:
            assert o.name in known, f"Unknown output {o.name!r}, step {step!r} "
        stats.append(dict(pattern=f"check_{step}", time_in=time.perf_counter() - begin))

    def random_order(self):
        """Moves nodes by random."""

        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[OrderOptimization.random_order] -- starts with "
                f"{len(self.builder.nodes)} nodes, "
                f"{len(self.builder.initializers_dict)} initializers"
            )

        begin = time.perf_counter()
        stats = []
        self._check(stats, "orderA")
        n_changes = 0
        n_moved = 0
        expected = len([n for n in self.builder.nodes if n is not None])
        done = set()
        n_iter = 0
        while len(done) < expected and n_iter < len(self.builder.nodes) * 2:
            if self.verbose > 1:
                print(
                    f"[OrderOptimization.random_order] start iter={n_iter} "
                    f"with {len(self.builder.nodes)} nodes"
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
                if make_idn(node) in done:
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
                    done.add(make_idn(node))
                    i += 1
                    continue

                new_position = random.randint(mi, ma)
                if self.verbose >= 10:
                    print(
                        f"[OrderOptimization.random_order]    i={i} "
                        f"new_position={new_position}"
                    )
                n_moved += abs(i - new_position)
                if i == new_position:
                    done.add(make_idn(node))
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

            self._check(stats, "orderL")

        if self.verbose:
            print(
                f"[OrderOptimization.random_order] done after "
                f"{n_iter} iterations in {time.perf_counter() -begin}s "
                f"with changed={n_changes} scale={n_moved}"
            )
        stats.append(
            dict(
                pattern="random_order",
                changed=n_changes,
                scale=n_moved,
                iter=n_iter,
                time_in=time.perf_counter() - begin,
            )
        )
        return stats

    def shape_order(self):
        """Moves shape right after the node it consumes is created."""

        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[OrderOptimization.random_order] -- starts with "
                f"{len(self.builder.nodes)} nodes, "
                f"{len(self.builder.initializers_dict)} initializers"
            )

        begin = time.perf_counter()
        stats = []
        self._check(stats, "orderA")
        n_changes = 0
        n_moved = 0
        positions = {
            **{i.name: -1 for i in self.builder.inputs},
            **{i: -2 for i in self.builder.initializers_dict},  # noqa: C420
        }
        ordered_nodes = []
        for pos, node in enumerate(self.builder.nodes):
            if node.op_type in ("Shape", "Size") and not node.domain:
                assert node.input[0] in positions, f"Missing input {node.input[0]!r}"
                produced = positions[node.input[0]]
                if pos > produced + 1:
                    n_changes += 1
                    n_moved += pos - (produced + 1)
                    ordered_nodes.append(
                        (produced + 0.1 + pos * 1.0 / len(self.builder.nodes), node)
                    )
                else:
                    ordered_nodes.append((pos, node))
            else:
                ordered_nodes.append((pos, node))
            for o in node.output:
                positions[o] = pos
        ordered_nodes.sort()
        self.builder.nodes = [_[1] for _ in ordered_nodes]
        self._check(stats, "orderL")
        if self.verbose:
            print(
                f"[OrderOptimization.shape_order] done after "
                f"in {time.perf_counter() -begin}s "
                f"with changed={n_changes} scale={n_moved}"
            )
        stats.append(
            dict(
                pattern="shape_order",
                changed=n_changes,
                scale=n_moved,
                time_in=time.perf_counter() - begin,
            )
        )
        return stats
