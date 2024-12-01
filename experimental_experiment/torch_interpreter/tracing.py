import contextlib
import inspect
import math
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.fx import Node
from torch.fx.proxy import TracerBase


class CustomProxy(torch.fx.proxy.Proxy):
    """
    Defines a custom proxy to trace the execution of a model
    and converts it into a fx graph.
    Works with :class:`CustomTracer
    <experimental_experiment.torch_interpreter.tracing.CustomTracer>`.
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None):
        super().__init__(node, tracer=tracer)
        assert isinstance(
            self.tracer, CustomTracer
        ), f"Unexpected type {type(self.tracer)} for the tracer."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node.name})"

    def _custom_fx_repr_fn(self) -> str:
        "To avoid bugs."
        return f"CustomProxy(%{str(self.node)})"

    def __getattr__(self, k) -> "CustomAttribute":
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return CustomAttribute(self, k)

    def __setitem__(self, *args, **kwargs):
        assert not kwargs, f"Unexpected not empty kwargs={kwargs!r}"
        assert len(args) == 2, f"Unexpected number of args={len(args)}: {args}"
        indices, values = args
        if isinstance(indices, CustomProxy):
            indices = indices.node
        node = self.tracer.create_node(
            "call_function",
            operator.setitem,
            args=(self.node, indices, values.node if hasattr(values, "node") else values),
            kwargs={},
        )
        # node_to_replace = self.node
        return self.tracer.proxy(node)

    def __len__(self):
        raise RuntimeError(
            "len(.) expects an integer, len needs to be replaced. You should use _len."
        )

    def length(self):
        """Returns a proxy for the length."""
        node = self.tracer.create_node("call_method", "__len__", args=(self.node,), kwargs={})
        tt = self.tracer.proxy(node, cls=CustomProxyInt)
        return tt

    def instanceof(self, cls):
        """Tells if this proxy represents a specific class."""
        raise RuntimeError(f"Unable to know if cls is from type {cls}.")

    @classmethod
    def cat(
        cls,
        tensors: List["CustomProxy"],
        dim: int = 0,
        *,
        out=None,
        axis: Optional[int] = None,
    ) -> "CustomProxy":
        """Implements cat for tensors."""
        if axis is not None and dim == 0:
            dim = axis
        node = tensors.tracer.create_node(
            "call_function", torch.cat, args=(tensors.node, dim), kwargs={}
        )
        return tensors.tracer.proxy(node)


def _len(x: Any) -> Union[int, CustomProxy]:
    """
    Overloads `len` to return a proxy if the input is the proxy.
    """
    if isinstance(x, CustomProxy):
        return x.length()
    return len(x)


def _isinstance(x, cls):
    """
    Overloads `isinstance` to deal with CustomProxy.
    """
    if isinstance(x, CustomProxy):
        return x.instanceof(cls)
    return isinstance(x, list)


class CustomProxyInt(CustomProxy):
    "A proxy for an integer."

    def instanceof(self, cls):
        """isinstance"""
        return cls in {CustomProxyInt, CustomProxy, int}


class CustomProxyFloat(CustomProxy):
    "A proxy for a float."

    def instanceof(self, cls):
        """isinstance"""
        return cls in {CustomProxyInt, CustomProxy, float}


class CustomAttribute(CustomProxy):
    """
    To trace attributes.
    """

    def __init__(self, root: CustomProxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root, *args), kwargs)


class CustomParameterProxy(CustomProxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert isinstance(param, torch.nn.Parameter)
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def shape(self):
        return self.param.shape

    def size(self):
        return self.param.size()

    def dim(self):
        return self.param.dim()

    @property
    def ndim(self):
        return self.param.ndim

    def numel(self):
        return self.param.numel()

    def nelement(self):
        return self.param.nelement()


@contextlib.contextmanager
def replace_problematic_function_before_tracing():
    """
    Replaces function that cannot be traced with the default tracer
    such as :func:`torch.cat`.
    """
    saved = {
        "cat": torch.cat,
    }
    newf = {
        "cat": CustomProxy.cat,
    }
    for k, v in newf.items():
        setattr(torch, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(torch, k, v)


class CustomTracer(torch.fx.Tracer):
    """
    Defines a custom tracer to trace the execution of a model
    and converts it into a fx graph.
    Works with :class:`CustomProxy
    <experimental_experiment.torch_interpreter.tracing.CustomProxy>`.

    ::
        from experimental_experiment.torch_interpreter.tracing import CustomTracer

        graph = CustomTracer().trace(model)
    """

    def __init__(
        self,
        autowrap_modules: Tuple["ModuleType"] = (math,),  # noqa: F821
        autowrap_functions: Tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )

    def proxy(
        self, node: torch.fx.Node, cls: type[CustomProxy] = CustomProxy
    ) -> torch.fx.Proxy:
        """
        Overwrites this method to replace the default Proxy by CustomProxy.
        """
        return cls(node, self)

    def create_arg(self, a: Any) -> "Argument":  # noqa: F821
        """
        Overwrites this method to deal with more argument.
        """
        if a is bool:
            return torch.bool
        if a is int:
            return torch.int64
        if a is float:
            return torch.float32
        if a is complex:
            return torch.complex64
        return super().create_arg(a)

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """
        See :meth:`torch.fx.Tracer.getattr.
        """

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node, n=n, attr_val=attr_val: CustomParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        remove_inplace: bool = True,
    ) -> torch.fx.Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants to.

        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.
            remove_inplace (bool): Removes inplace nodes

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        with replace_problematic_function_before_tracing():
            graph = super().trace(root, concrete_args)
        self._replace_problematic_functions(graph)
        if not remove_inplace:
            return graph
        return self.remove_inplace(graph)

    @classmethod
    def _replace_problematic_functions(cls, graph: torch.fx.Graph):
        """
        The tracing introduced some problematic functions which need to be replaced.
        """
        replaces = {
            CustomProxy.cat: torch.cat,
        }
        for node in graph.nodes:
            if node.op == "call_function" and node.target in replaces:
                node.target = replaces[node.target]

    @classmethod
    def _inplace_nodes(cls, graph: torch.fx.Graph) -> List[Tuple[int, torch.fx.Node]]:
        """
        Returns the position and the node involved in inplace modifications.
        """
        return [
            (i, node)
            for i, node in enumerate(graph.nodes)
            if node.op != "output"
            and len(node.users) == 0
            and node.op.startswith("call_")
            and node.target not in {operator.getitem}
        ]

    @classmethod
    def _replace_meth_setitem(cls, graph: torch.fx.Graph) -> torch.fx.Graph:
        """
        The execution of ``op="call_method", target="__setitem__" `` returns None
        We replace it by ``op="call_function", target="operator.setitem"``.
        """
        for node in graph.nodes:
            if node.op == "call_method" and node.target == "__setitem__":
                node.op = "call_function"
                node.target = operator.setitem

    @classmethod
    def _replace_getattr(cls, graph: torch.fx.Graph) -> torch.fx.Graph:
        """
        Nodes such as
        ``%_tensor_constant0_1 : [num_users=1] = get_attr[target=_tensor_constant0]``
        are part of the replacement in function ``replace_all_uses_with``.
        Let's remove the duplicates first.
        """
        targets = {}
        to_replace = []
        for node in graph.nodes:
            if node.op == "get_attr":
                if node.target in targets:
                    # replacements
                    to_replace.append((node, targets[node.target]))
                else:
                    targets[node.target] = node
        if to_replace:
            for node, by in to_replace:
                node.replace_all_uses_with(by)
                graph.erase_node(node)

    def remove_inplace(self, graph: torch.fx.Graph) -> torch.fx.Graph:
        """
        Removes inplace operations.
        """
        inplace = self._inplace_nodes(graph)
        if len(inplace) == 0:
            # No inplace.
            return graph

        self._replace_getattr(graph)
        self._replace_meth_setitem(graph)

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        existing_nodes = list(enumerate(graph.nodes))
        for pos, node in reversed(inplace):
            assert node.target in {
                "add_",
                "div_",
                "mul_",
                "sub_",
                "mod_",
                operator.setitem,
            }, (
                f"Unsupported target {node.target!r} at position {pos}/{len(graph.nodes)}"
                f"\n--graph\n{graph}"
            )
            # We assume the first argument is the one modified inplace.
            new_name = node
            old_name = node.args[0]

            # class Node can be used as a key
            # We also assume a user is placed after this node.
            nodes_to_leave = {n[1] for n in existing_nodes[: pos + 1]}

            # let's replace
            changed = old_name.replace_all_uses_with(
                new_name,
                delete_user_cb=lambda n, leave=nodes_to_leave: delete_user_cb(n, leave),
            )
            assert changed, (
                f"No change applied, the inplace node [{node}] at position {pos} "
                f"does not replace [{old_name}] in \n{graph}\n-- node to keep --"
                f"\n{nodes_to_leave}"
            )

        inplace = self._inplace_nodes(graph)
        assert (
            len(inplace) == 0
        ), f"Inplace nodes remain at positions {sorted(_[0] for _ in inplace)} in\n{graph}"
        return graph
