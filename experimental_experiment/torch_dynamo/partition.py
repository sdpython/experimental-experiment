import torch
from typing import Callable, List, Mapping, Optional, Set
from torch.fx.passes.operator_support import OperatorSupport
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.tools_common import get_node_target, CALLABLE_NODE_OPS

try:
    from torch._functorch.compile import min_cut_rematerialization_partition
except ImportError:
    from functorch.compile import min_cut_rematerialization_partition


class CustomOperatorSupport(OperatorSupport):
    def __init__(self, unsupport_dict: Optional[Set[str]] = None, verbose: int = 0):
        super().__init__()
        self._unsupport_dict = unsupport_dict or set()
        self.verbose = verbose

    def is_node_supported(
        self, submodules: Mapping[str, "torch.nn.Module"], node: "torch.fx.Node"
    ) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            if self.verbose > 1:
                print(f"[CustomOperatorSupport.is_node_support] validate node.op [{node.op}]")
            return True

        target = get_node_target(submodules, node)

        if target in self._unsupport_dict:
            if self.verbose:
                print(f"[CustomOperatorSupport.is_node_support] rejected target [{target}]")

            return False

        if self.verbose > 1:
            print(f"[CustomOperatorSupport.is_node_support] validate target [{target}]")

        return True


def get_partition_fn():
    return min_cut_rematerialization_partition


class PartionedBackend:
    def __init__(
        self,
        fused_module,
        support,
        backend_function: Callable,
        use_aot_autograd: bool,
        decompositions,
        partition_fn,
        dynamic: bool,
        full_graph: bool,
        verbose: int,
    ):
        assert use_aot_autograd, "not implemented if use_aot_autograd=False"
        self.fused_module = fused_module
        self.backend_function = backend_function
        self.use_aot_autograd = use_aot_autograd
        self.decompositions = decompositions
        self.partition_fn = partition_fn
        self.compiled_model = None
        self.dynamic = dynamic
        self.full_graph = full_graph
        self.verbose = verbose
        self.support = support

    def __call__(self, *args):
        if self.compiled_model is None:
            aot_compiler = aot_autograd(
                fw_compiler=lambda *args, **kwargs: backend_partition_compile(
                    *args,
                    support=self.support,
                    backend_function=self.backend_function,
                    verbose=self.verbose,
                    use_aot_autograd=True,
                    decompositions=self.decompositions,
                    partition_fn=self.partition_fn,
                    **kwargs,
                ),
                decompositions=self.decompositions,
                partition_fn=self.partition_fn,
            )

            self.compiled_model = torch.compile(
                self.fused_module,
                backend=aot_compiler,
                dynamic=self.dynamic,
                fullgraph=self.fullgraph,
            )
        return self.compiled_model(*args)


class _WrapForPartition:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, graph_module, args):
        return self.wrapped(graph_module, args)


def backend_partition_compile(
    graph_module: torch.fx.GraphModule,
    args: List[torch.Tensor],
    support: Optional[OperatorSupport] = None,
    allows_single_node_partition: bool = True,
    backend_function: Optional[Callable] = None,
    use_aot_autograd: bool = True,
    decompositions=None,
    partition_fn=None,
    verbose: int = 1,
    dynamic: bool = False,
    full_graph: bool = True,
    **kwargs,
):
    """
    Partitions a graph module for any backend.
    """
    assert backend_function is not None, "backend_function should not be None."
    partitioner = CapabilityBasedPartitioner(
        graph_module,
        support or CustomOperatorSupport(),
        allows_single_node_partition=allows_single_node_partition,
    )

    partitioned_prim_graph_module = _WrapForPartition(partitioner.partition_and_fuse())
    # This shortcut is no longer possible as graph_module was modified.
    # if len(partitioned_prim_graph_module.wrapped.graph.nodes) == 1:
    #     if verbose:
    #         print("[backend_partition_compile] no partition")
    #     return backend_function(graph_module, args)

    for i, node in enumerate(partitioned_prim_graph_module.wrapped.graph.nodes):
        if verbose:
            print(
                f"[backend_partition_compile] node {i+1}/"
                f"{len(partitioned_prim_graph_module.wrapped.graph.nodes)}={node}, "
                f"node.op={node.op!r}, node.name={node.name!r}"
            )
        if node.op == "call_module" and "fused_" in node.name:
            fused_module = getattr(partitioned_prim_graph_module.wrapped, node.name)
            if verbose:
                print(
                    f"[backend_partition_compile] fused_node={node.name!r}, "
                    f"id={id(fused_module)}"
                )
            fused_module._wrapped_call = PartionedBackend(
                fused_module,
                support=support,
                backend_function=backend_function,
                use_aot_autograd=use_aot_autograd,
                decompositions=decompositions,
                partition_fn=partition_fn,
                dynamic=dynamic,
                full_graph=full_graph,
                verbose=verbose,
            )

    return partitioned_prim_graph_module(graph_module, args)
