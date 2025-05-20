import contextlib
import inspect
import math
import operator
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.fx import Node
from torch.fx.proxy import TracerBase
from ..helpers import string_type

_torch_cat = torch.cat


def setitem_with_transformation(a, b, transformations):
    """
    Extended version of setitem to deal with inplace modification.
    """
    function_table = {"exp": torch.exp_, "sigmoid": torch.sigmoid_}
    assert transformations, "transformations is empty, it means identity?"
    for name, args in transformations:
        assert not args, f"Not implemented for name={name!r} and args={args!r}"
        f = function_table[name]
        f(a[b])
    # operator.setitem(a, b, c)
    return a


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

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        if isinstance(orig_method, torch._ops.HigherOrderOperator):
            # not implemented by torch
            if orig_method is torch.cond:
                assert (
                    not kwargs
                ), f"Unexpected kwargs={kwargs}, args={args}, orig_method={orig_method}"
                assert (
                    len(args) == 4
                ), f"Unexpected kwargs={kwargs}, args={args}, orig_method={orig_method}"
                assert isinstance(
                    args[3], (list, tuple)
                ), f"Unexpected type {type(args[3])} for the last argument"
                root = args[0]
                cond_true = root.tracer.register_callable("cond", args[1])
                cond_false = root.tracer.register_callable("cond", args[2])
                node = root.tracer.create_node(
                    "call_function",
                    orig_method,
                    args=(
                        args[0].node,
                        cond_true,
                        cond_false,
                        type(args[3])(a.node for a in args[3]),
                    ),
                    kwargs={},
                )
                return root.tracer.proxy(node)

        return torch.fx.proxy.Proxy.__torch_function__(
            orig_method, types, args=args, kwargs=kwargs
        )

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
        assert out is None, "Tracing is not implementing is out is not None."
        if isinstance(tensors, list):
            return _torch_cat(tensors, dim)
        if axis is not None and dim == 0:
            dim = axis
        proxy = tensors
        node = proxy.tracer.create_node(
            "call_function", torch.cat, args=(proxy.node, dim), kwargs={}
        )
        return proxy.tracer.proxy(node)


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
    so that conditional tests on these attributes will not throw exception during tracing.
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


class CondCCOp(torch._ops.HigherOrderOperator):
    """
    Cannot be imported from torch.ops.higher_order.cond
    (function cond overwrite submodule cond).
    """

    def __init__(self):
        # we cannot use "cond" to avoid confusion with the existing cond
        super().__init__("condcc")

    def __call__(self, pred, true_fn, false_fn, operands):
        # torch._higher_order_ops.utils.validate_subgraph_args_types(operands)
        return super().__call__(pred, true_fn, false_fn, operands)


@contextlib.contextmanager
def replace_problematic_function_before_tracing():
    """
    Replaces function that cannot be traced with the default tracer
    such as :func:`torch.cat`.
    """
    saved = {
        "cat": torch.cat,
        "cond": torch.cond,
        # ("torch.ops.higher_order", "cond"): torch.ops.higher_order.cond,
    }
    newf = {
        "cat": CustomProxy.cat,
        "cond": CondCCOp(),
        # ("torch.ops.higher_order", "cond"): CondOp(),
    }
    for k, v in newf.items():
        if isinstance(k, tuple):
            setattr(k[0], k[1], v)
        else:
            setattr(torch, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if isinstance(k, tuple):
                setattr(k[0], k[1], v)
            else:
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
        self._callables = {}

    def register_callable(self, name: str, fn: Callable) -> torch.fx.Node:
        """
        Registers a function and return a unique name.

        :param name: prefix to prepend to the function name
        :param fn: function
        :return: new_name
        """
        cand = f"_cb_{name}_{fn.__name__}_0"
        if cand in self._callables:
            i = 1
            cand = f"_cb_{name}_{fn.__name__}_{i}"
            while cand in self._callables:
                i += 1
                cand = f"_cb_{name}_{fn.__name__}_{i}"
        self._callables[cand] = fn
        return self.create_node("get_attr", cand, args=(), kwargs={})

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
        See :meth:`torch.fx.Tracer.getattr`.
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
        update_model_with_callable: bool = True,
    ) -> torch.fx.Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants to.

        :param root: Either a ``Module`` or a function to be
            traced through. Backwards-compatibility for this parameter is
            guaranteed.
        :param concrete_args: Concrete arguments that should
            not be treated as Proxies. This parameter is experimental and
            its backwards-compatibility is *NOT* guaranteed.
        :param remove_inplace: Removes inplace nodes
        :param update_model_with_attribute: in some cases (control flow),
            the model needs to be
        :return: A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        assert concrete_args is None or isinstance(
            concrete_args, dict
        ), f"Unexpected type for concrete_args: {string_type(concrete_args)}"
        with replace_problematic_function_before_tracing():
            graph = super().trace(root)
        if concrete_args:
            for node in graph.nodes:
                if node.op == "placeholder":
                    if node.name in concrete_args:
                        node.meta["example_value"] = concrete_args[node.name]

        self._replace_problematic_functions(graph)
        if update_model_with_callable and self._callables:
            for k, v in self._callables.items():
                setattr(root, k, v)
        self.remove_unnecessary_slices(graph)
        if not remove_inplace:
            graph.lint()
            return graph
        self.remove_inplace(graph)
        graph.lint()
        return graph

    @classmethod
    def _replace_problematic_functions(cls, graph: torch.fx.Graph) -> int:
        """
        The tracing introduced some problematic functions which need to be replaced.

        :return: number of impacted nodes
        """
        replaces = {
            CustomProxy.cat: torch.cat,
            # CondCCOp: torch.ops.higher_order.cond,
        }
        n = 0
        for node in graph.nodes:
            if node.op == "call_function":
                if node.target in replaces:
                    n += 1
                    node.target = replaces[node.target]
                elif isinstance(node.target, CondCCOp):
                    n += 1
                    node.target = torch.ops.higher_order.cond
        return n

    @classmethod
    def _get_aten_name(cls, node: torch.fx.Node) -> str:
        """
        Returns the aten name for the target as a string.
        """
        assert hasattr(node, "target"), f"Unable to return aten name for {node}"
        if node.target == operator.getitem:
            return "getitem"
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            if node.target != torch.ops.aten.sym_size:
                raise RuntimeError(f"Unsupported function {node!r}.")
            raise NotImplementedError(f"Unsupported function {node!r} (not implemented).")

        if isinstance(node.target, types.BuiltinFunctionType):
            if node.target is math.ceil:
                # We need to distinguish between match.ceil and torch.ceil.
                # The output type is different.
                return "math_ceil"
            return str(node.target)

        if isinstance(node.target, torch._ops.OpOverload):
            return node.target.name()

        if callable(node.target):
            # a single function
            return f"aten_{node.target.__name__}"

        if isinstance(node.target, str):
            return node.target

        raise NotImplementedError(
            f"Unsupported function {node!r} (not implemented), "
            f"node.target={node.target}, type is {type(node.target)}."
        )

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
            and cls._get_aten_name(node)
            not in {
                "aten::_assert_scalar",
                "aten::sym_constrain_range_for_size",
                "aten::_log_api_usage_once",
                "aten::_enter_autocast",
                "aten::_exit_autocast",
                "aten::_set_grad_enabled",
                "aten__assert_scalar",
                "aten_sym_constrain_range_for_size",
                "aten__log_api_usage_once",
                "aten__enter_autocast",
                "aten__exit_autocast",
                "aten__set_grad_enabled",
            }
        ]

    @classmethod
    def _replace_meth_setitem(cls, graph: torch.fx.Graph) -> int:
        """
        The execution of ``op="call_method", target="__setitem__" `` returns None
        We replace it by ``op="call_function", target="operator.setitem"``.

        :return: number of impacted nodes
        """
        n = 0
        for node in graph.nodes:
            if node.op == "call_method" and node.target == "__setitem__":
                node.op = "call_function"
                node.target = operator.setitem
                n += 1
        return n

    @classmethod
    def _replace_getattr(cls, graph: torch.fx.Graph) -> int:
        """
        Nodes such as
        ``%_tensor_constant0_1 : [num_users=1] = get_attr[target=_tensor_constant0]``
        are part of the replacement in function ``replace_all_uses_with``.
        Let's remove the duplicates first.

        :return: number of impacted get_attr nodes
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
        return len(to_replace)

    @classmethod
    def remove_unnecessary_slices(cls, graph: torch.fx.Graph) -> int:
        """
        Removes unnecessary slices:

        :param graph: graph to modify
        :return: number of inplace nodes removed

        ::

            %slice_11 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
        """
        nodes = list(enumerate(graph.nodes))

        removed = 0
        for pos, node in nodes:
            if not hasattr(node.target, "name"):
                continue
            if node.target.name() != "aten::slice.Tensor":
                continue
            if (
                len(node.args) != 4
                or (node.args[2] != 0 and node.args[2] is not None)
                or (node.args[3] != 9223372036854775807 and node.args[3] is not None)
            ):
                continue

            # The first argument is the node to keep.
            new_name = node.args[0]
            old_name = node

            # Let's replace.
            changed = old_name.replace_all_uses_with(new_name)
            assert changed, (
                f"No change applied, the node [{node}] at position {pos} "
                f"can be removed and replaced by {old_name} in \n{graph}."
            )
            graph.erase_node(old_name)
            removed += 1
        return removed

    @classmethod
    def graph_erase_node(cls, graph: torch.fx.Graph, node: torch.fx.Node):
        """
        Removes a node and all predecessors with are only consumed by this one.
        """
        nodes = [node]
        while (
            node.op == "call_function"
            and node.args
            and isinstance(node.args[0], torch.fx.Node)
            and all(isinstance(_, (int, float)) for _ in node.args[1:])
            and len(node.args[0].users) == 1
        ):
            node = node.args[0]
            nodes.append(node)
        for node in nodes:
            graph.erase_node(node)

    @classmethod
    def _modify_graph_clone_copy_(  # != _modify_graph_clone_index_copy_
        cls,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        existing_nodes: List[torch.fx.Node],
        pos: int,
        exported_program: torch.export.ExportedProgram,
        err_graph: str,
        verbose: int = 0,
        exc: bool = True,
    ) -> int:
        """
        Removes inplace node ``clone`` + ``copy_`` (inplace copy).
        Then we tell the nodes using ``%clone`` to use ``%copy_``.

        :param graph: graph to modify
        :param node: node after clone
        :param existing_nodes: list of the nodes in the graph
        :param pos: position of the node in existing nodes
        :param exported_program: for debugging purpose
        :param err_graph: original graph as a string, for debugging purpose
        :param verbose: verbosity
        :param exc: if False, return -1 if unable to reach the goal
        :return: number of removed nodes
        """
        predecessor = node.args[0]
        predecessor_name = (
            predecessor.target.name() if hasattr(predecessor.target, "name") else None
        )

        if not exc and predecessor_name != "aten::clone":
            return -1

        assert predecessor_name == "aten::clone", (
            f"(inplace) Unexpected predecessor {predecessor.target!r} "
            f"(predecessor.target.name()={predecessor_name!r}) "
            f"for node {node.name!r} with args={node.args} at position "
            f"{pos}/{len(graph.nodes)}"
            f"\n--original graph--\n{err_graph}"
            f"\n--graph\n{exported_program or graph}"
        )

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        # class Node can be used as a key
        # We also assume a user is placed after this node.
        nodes_to_leave = {n[1] for n in existing_nodes[: pos + 1]}
        node_args = node.args
        p_users = predecessor.users

        # We can replace with expand then.
        with graph.inserting_before(node):
            # We assume the first argument is the one modified inplace.
            new_node = graph.call_method("expand_as", args=(node_args[1], predecessor))
            # let's replace
            changed = predecessor.replace_all_uses_with(
                new_node,
                delete_user_cb=(lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)),
            )
            graph.erase_node(node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(1, predecessor)

        assert changed, (
            f"No change applied, the inplace node [{node}] "
            f"at position {pos} with node.args={node_args}, was not replaced "
            f"by [{new_node}] with target {new_node.target!r} and "
            f"new_node.args={new_node.args}, predecessor="
            f"[{predecessor}] with target={predecessor.target!r}, "
            f"p_users={list(p_users)}, "
            f"predecessor.users={list(predecessor.users)}, "
            f"new_node.users={list(new_node.users)} in "
            f"\n{exported_program or graph}"
        )
        return 1

    @classmethod
    def _modify_graph_clone_index_copy_(
        cls,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        existing_nodes: List[torch.fx.Node],
        pos: int,
        exported_program: torch.export.ExportedProgram,
        err_graph: str,
        verbose: int = 0,
        exc: bool = True,
    ) -> int:
        """
        Removes inplace node ``clone`` + ``index.Tensor`` + ``copy_`` (inplace copy).
        Then we tell the nodes using ``%clone`` to use ``%copy_``.

        :param graph: graph to modify
        :param node: node after clone
        :param existing_nodes: list of the nodes in the graph
        :param pos: position of the node in existing nodes
        :param exported_program: for debugging purpose
        :param err_graph: original graph as a string, for debugging purpose
        :param verbose: verbosity
        :param exc: raises an exception if cannot reach the goal
        :return: number of removed nodes

        Example of nodes it may face:

        ::

            --  7 aten::slice.Tensor :: (clone, 0, 2, -2) -> slice_2
            --  8 aten::slice.Tensor :: (slice_2, 1, 2, -2) -> slice_3
            --  9 aten::slice.Tensor :: (slice_3, 2, 0, -1) -> slice_4
            --  12 aten::copy_ :: (slice_4, expand) -> copy_
            --  14 aten::slice.Tensor :: (clone, 0, 2, -2) -> slice_5
            --  15 aten::slice.Tensor :: (slice_5, 1, 2, -2) -> slice_6
            --  16 aten::select.int :: (slice_6, 2, -1) -> select
            --  17 aten::fill_.Tensor :: (select, lift_fresh_copy) -> fill_
            --  18 output :: ((clone,),) -> output

        It should be summarized into:

        ::

            %setitem : [num_users=1] = call_function[target=operator.setitem](
                args = (
                    %clone,
                    (slice(2, -2, None), slice(2, -2, None), slice(None, -1, None)),
                    %getitem
                ),
                kwargs = {}
            )
            %setitem_1 : [num_users=1] = call_function[target=operator.setitem](
                args = (
                    %setitem,
                    (slice(2, -2, None), slice(2, -2, None), -1),
                    0.0
                ),
                kwargs = {}
            )

        Example:

        .. runpython::
            :showcode:

            import torch
            from experimental_experiment.torch_interpreter.tracing import CustomTracer

            class Model(torch.nn.Module):
                def forward(self, x, sumx):
                    K_33 = x.clone()
                    K_33[2:-2, 2:-2, :-1] = sumx[None, 2:-2, None]
                    K_33[2:-2, 2:-2, -1] = 0.0
                    return K_33

            model = Model()

            graph = CustomTracer().trace(model)
            print(graph)
        """
        # Let's find the first clone node.
        given_node = node
        while (
            hasattr(node, "target")
            and node.target not in {"clone"}
            and (
                not hasattr(node.target, "name")
                or node.target.name() not in {"aten::clone", "aten::zeros"}
            )
            and node.args
        ):
            if isinstance(node.args[0], torch.fx.immutable_collections.immutable_list):
                arg = node.args[0]
                while isinstance(arg, torch.fx.immutable_collections.immutable_list):
                    # something like
                    # aten.zeros.default](args = ([%sym_size_int_18, %add_179, %add_179],)
                    if len(arg) == 0:
                        break
                    node = arg[0]
                    if not node.args:
                        break
                    arg = node.args[0]
            else:
                node = node.args[0]
        clone = node
        target_name = (
            (node.target.name() if hasattr(node.target, "name") else node.target)
            if hasattr(node, "target")
            else None
        )
        allowed_names = {"aten::clone", "clone", "aten::zeros", "zeros"}
        if not exc and target_name not in allowed_names:
            return -1
        assert target_name in allowed_names, (
            f"(inplace) Unexpected predecessor {node.target!r} "
            f"(target_name={target_name!r}) "
            f"for node {given_node.name!r} with args={given_node.args} at position "
            f"{pos}/{len(graph.nodes)}"
            f"\n--original graph--\n{err_graph}"
            f"\n--graph\n{exported_program or graph}"
        )

        # Let's find all the users until we find a node copy_
        # assuming this node has no users
        known = set()
        users = []
        unprocessed = [clone]
        while unprocessed:
            current = unprocessed.pop()
            if current.users:
                new_users = [u for u in current.users if u not in known]
                users.extend(new_users)
                unprocessed.extend(new_users)
                known |= set(new_users)

        def _str(pos_node):
            pos, node = pos_node
            return f"-- {pos} {node.args} -> {node.name} -- with {node.target}\n"

        # Let's reorder according to existing nodes
        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        def _macro_assert_index_(is_getitem=False):
            assert (
                not inplace_functions
            ), f"Unexpected inplace_functions={inplace_functions} before {n.target}"
            assert len(n.args) and n.args[0] in seen_nodes, (
                f"Unexpected node {n} at position {pos} (n.args={n.args}) "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}"
            )
            assert all(isinstance(_i, (tuple, int, torch.fx.Node)) for _i in n.args[1:]), (
                f"Unexpected arguments for target {n.target} "
                f"args={string_type(n.args)} - {n.args}\n--graph--\n{graph}"
            )
            assert not is_getitem or not set_item_args, (
                f"set_item_args should be empty for getitem at position {pos} "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}"
            )

        def _macro_get_axis_(n, set_item_args):
            axis = n.args[1]
            assert (
                axis not in set_item_args
            ), f"duplicated axis {n.args[1]} in \n{''.join(map(_str, pos_users))}"
            return axis

        def _macro_new_node_(n, current_remove, set_item_args, inplace_functions):
            # We need to replace all the nodes in seen_nodes
            # a set_item and make clone is now replace by this new one
            assert len(n.args) and n.args[0] in seen_nodes, (
                f"Unexpected node {n} at position {pos} "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}\n----{err_graph}"
            )
            seen_nodes.add(n)
            current_remove.append(n)
            max_axis = max(set_item_args)
            args_slices = [None for i in range(max_axis + 1)]
            for k, v in set_item_args.items():
                args_slices[k] = v
            new_args = [clone, tuple(args_slices), *n.args[1:]]

            nodes_to_leave = {_n[1] for _n in existing_nodes[: pos + 1]}

            # We use operator.setitem in both cases.
            # torch.ops.aten.fill.Tensor cannot work in this case.
            if not inplace_functions:
                function_op = operator.setitem
            else:
                function_op = setitem_with_transformation
                new_args.append(tuple(inplace_functions))

            # We can replace with expand then.
            with graph.inserting_after(n):
                # We assume the first argument is the one modified inplace.
                new_node = graph.call_function(function_op, args=tuple(new_args))
                nodes_to_leave.add(new_node)
                # let's replace
                changed = clone.replace_all_uses_with(
                    new_node,
                    delete_user_cb=(lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)),
                )
                assert changed, (
                    f"No change applied, for new_node={new_node} and "
                    f"args={new_node.args} in {''.join(map(_str, pos_users))}"
                    f"\n-- new graph\n{graph}"
                    f"\n-- old graph\n{err_graph}"
                )
                return new_node

        node_pos = {b: a for a, b in existing_nodes}
        pos_users = [(node_pos[n], n) for n in users]
        pos_users.sort()

        to_remove = []
        seen_nodes = {clone}
        set_item_args = {}
        current_remove = []
        inplace_functions = []
        for pos, n in pos_users:
            if n.target == operator.getitem:
                _macro_assert_index_(True)
                set_item_args = dict(enumerate(n.args[1]))
                seen_nodes.add(n)
                current_remove.append(n)
            elif hasattr(n.target, "name"):
                aten_name = n.target.name()
                if aten_name == "aten::slice.Tensor":
                    _macro_assert_index_()
                    axis = _macro_get_axis_(n, set_item_args)
                    set_item_args[axis] = slice(*n.args[2:])
                    seen_nodes.add(n)
                    current_remove.append(n)
                elif aten_name == "aten::select.int":
                    _macro_assert_index_()
                    axis = _macro_get_axis_(n, set_item_args)
                    set_item_args[axis] = n.args[2]
                    seen_nodes.add(n)
                    current_remove.append(n)
                elif aten_name[-1] != "_" and "_." not in aten_name:
                    # This is no inplace modification so all stored
                    # slice operator are cleaned.
                    set_item_args = {}
                    current_remove = []
                    seen_nodes = {clone}
                    inplace_functions = []
                elif aten_name in {"aten::copy_", "aten::fill_.Tensor"}:
                    new_node = _macro_new_node_(
                        n, current_remove, set_item_args, inplace_functions
                    )
                    # next root to use
                    clone = new_node
                    # reset
                    to_remove.extend(current_remove)
                    seen_nodes = {new_node}
                    set_item_args = {}
                    current_remove = []
                    inplace_functions = []
                elif aten_name == "aten::masked_fill_.Scalar":
                    # python -m experimental_experiment.torch_bench.bash_bench_huggingface
                    # --model MBartForConditionalGeneration --device cuda --dtype float16
                    # --export custom --opt_patterns default --verbose 1 --quiet 0
                    raise NotImplementedError(
                        f"Unable to handle target {aten_name!r} (could probably be ignored) "
                        f"with args={n.args} in\n{''.join(map(_str, pos_users))}\n----\n"
                        f"{err_graph}"
                    )
                else:
                    raise NotImplementedError(
                        f"Unable to handle target {aten_name!r} with args={n.args} "
                        f"in\n{''.join(map(_str, pos_users))}\n----\n{err_graph}"
                    )
            elif n.target in {torch.exp_, torch.sigmoid_} or (
                isinstance(n.target, str) and n.target.endswith("_")
            ):
                # One inplace modification.
                # We assume the inplace modification takes place instead of a copy.
                assert (
                    n.args
                    and isinstance(n.args[0], torch.fx.Node)
                    and all(not isinstance(_, torch.fx.Node) for _ in n.args[1:])
                ), (
                    f"Unexpected type in argument of node {n} at position "
                    f"{pos}(args={string_type(n.args)})"
                )
                function_name = {torch.exp_: "exp", torch.sigmoid_: "sigmoid"}[n.target]
                inplace_functions.append((function_name, n.args[1:]))
                # do the same as before
                new_node = _macro_new_node_(
                    n, current_remove, set_item_args, inplace_functions
                )
                # next root to use
                clone = new_node
                # reset
                to_remove.extend(current_remove)
                seen_nodes = {new_node}
                set_item_args = {}
                current_remove = []
                inplace_functions = []
            else:
                # Here this node should already been handle.
                # We skip it.
                assert pos == pos_users[-1][0], (
                    f"Unexpected node (1) at pos={pos}, node={n}, target={n.target!r} "
                    f"in {''.join(map(_str, pos_users))}"
                )
        else:
            # Here again this node should already been handle.
            # We skip it.
            assert pos == pos_users[-1][0], (
                f"Unexpected node (2) at pos={pos}, node={n}, target={n.target} "
                f"in {''.join(map(_str, pos_users))}"
            )

        # Let's replace the replace nodes.
        for n in reversed(to_remove):
            graph.erase_node(n)

        # The last
        assert len(to_remove) == len(pos_users) - 1, (
            "Some nodes were not properly handled "
            f"len(to_remove)={len(to_remove)} and "
            f"len(pos_users)={len(pos_users)},\n-- nodes\n"
            f"{''.join(map(_str, pos_users))}\n"
            f"-- new graph\n{graph}"
            f"\n-- old graph\n{err_graph}"
        )
        return len(to_remove)

    @classmethod
    def remove_inplace(
        cls,
        graph: torch.fx.Graph,
        exported_program: Optional[torch.export.ExportedProgram] = None,
        verbose: int = 0,
        exc: bool = True,
    ) -> int:
        """
        Removes inplace operations.

        :param graph: graph to modify
        :param exported_program: if available, it is used in the error message
            to make it easier to trace the code source
        :param verbose: verbosity
        :param exc: raise an exception if not possible, other return -1
        :return: number of inplace nodes removed

        The most difficult pattern is the following:

        ::

            %slice_11 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_12 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_11, 1, 0, 9223372036854775807), kwargs = {})
            %slice_13 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_12, 2, 0, 9223372036854775807), kwargs = {})
            %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default]
                (args = (%slice_13, %masked_fill), kwargs = {})
        """
        # Remove obvious unused nodes.
        rem = []
        for node in graph.nodes:
            if (
                not node.users
                and hasattr(node, "target")
                and node.target
                in {
                    torch._C._set_grad_enabled,
                    torch._C._log_api_usage_once,
                    torch.autograd.function.FunctionCtx,
                    torch.amp.autocast_mode._enter_autocast,
                    torch.amp.autocast_mode._exit_autocast,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten._assert_tensor_metadata.default,
                }
            ):
                rem.append(node)
        for node in rem[::-1]:
            graph.erase_node(node)
            if verbose > 2:
                print(f"[CustomTracer.remove_inplace] A.remove {node.target}")
            continue

        # True inplace nodes.
        inplace = cls._inplace_nodes(graph)
        if len(inplace) == 0:
            # No inplace.
            return 0

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        if verbose:
            print(f"[CustomTracer.remove_inplace] starts with {len(graph.nodes)} nodes")

        n_inplace = len(inplace)
        if verbose:
            print(f"[CustomTracer.remove_inplace] S1: {len(inplace)} inplace nodes")
        changed = cls._replace_getattr(graph)
        changed |= cls._replace_meth_setitem(graph)

        err_graph = str(graph)
        if changed:
            cls._inplace_nodes(graph)

        if len(inplace) == 0:
            # No inplace anymore.
            return 0

        # First step: we remove every unused node.
        for pos, node in reversed(inplace):
            if node.target in {
                operator.add,
                operator.floordiv,
                operator.mul,
                operator.mod,
                operator.sub,
                torch._C._set_grad_enabled,
                torch._C._log_api_usage_once,
                torch.autograd.function.FunctionCtx,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
                torch.ops.aten._assert_scalar.default,
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten.item.default,
            }:
                # This node cannot be one inplace modifications. The node is just not used.
                graph.erase_node(node)
                if verbose > 2:
                    print(
                        f"[CustomTracer.remove_inplace] A.remove "
                        f"{pos}: {node.target}({node.args}) -> {node}"
                    )
                continue

        inplace = cls._inplace_nodes(graph)
        if len(inplace) == 0:
            # No inplace left.
            return 0

        # Then the difficult ones.
        MAX_ITER = 10
        max_iter = MAX_ITER
        if verbose:
            print(
                f"[CustomTracer.remove_inplace] S2: {len(inplace)} inplace nodes "
                f"and {max_iter} iterations"
            )
        while inplace and max_iter > 0:
            if verbose > 1:
                print(
                    f"[CustomTracer.remove_inplace] loop {max_iter} "
                    f"iterations left with {len(graph.nodes)} nodes and "
                    f"{len(inplace)} inplace nodes"
                )
            existing_nodes = list(enumerate(graph.nodes))
            for pos, node in reversed(inplace):

                if verbose > 5:
                    print(
                        f"[CustomTracer.remove_inplace] handle inplace node "
                        f"{pos}/{len(graph.nodes)}: {node} with args={node.args} "
                        f"and target={node.target}"
                    )

                # if the target has a name
                if hasattr(node.target, "name"):
                    if node.target.name() in {
                        "aten::view",
                        "aten::detach_",  # output = input
                        "aten::add.Tensor",  # it happens when running
                        "aten::div.Tensor",  # z = f(x=x, y=x+1) but f does not use y
                        "aten::mul.Tensor",
                        "aten::sub.Tensor",
                        "aten::zeros",  # unused as it does not end up with '_'
                    } or not (  # not an inplace modification
                        node.target.name().endswith(("_", "_.Tensor"))
                    ):
                        # This node cannot be one inplace modification.
                        # The node is just not used.
                        cls.graph_erase_node(graph, node)
                        del existing_nodes[pos]
                        if verbose > 2:
                            print(
                                f"[CustomTracer.remove_inplace] B.remove "
                                f"{pos}: {node.target}({node.args}) -> {node}"
                            )
                        continue

                    if (
                        # y = add_(x, 1) but x is used once (here) and y is not used
                        node.target.name() in {"aten::add_.Tensor", "aten::mul_.Tensor"}
                        and isinstance(node.args[0], torch.fx.Node)
                        and not isinstance(node.args[1], torch.fx.Node)
                        and len(node.args[0].users) <= 1
                    ):
                        # This node cannot be one inplace modification.
                        # The node is just not used.
                        cls.graph_erase_node(graph, node)
                        del existing_nodes[pos]
                        if verbose > 2:
                            print(
                                f"[CustomTracer.remove_inplace] B2.remove "
                                f"{pos}: {node.target}({node.args}) -> {node}"
                            )
                        continue

                    if len(node.args) == 1:
                        # We should be able to remove this inplace modification
                        # unless the predecessor is a Tensor.
                        # %slice_21 : [num_users=1] = call_function[
                        #           target=torch.ops.aten.slice.Tensor
                        # ](
                        #   args = (%clone_2, 4, 4, 9223372036854775807),
                        #   kwargs = {}
                        # )
                        # %sigmoid__2 : [num_users=0] = call_function[
                        #           target=torch.ops.aten.sigmoid_.default
                        # ](
                        #   args = (%slice_21,), kwargs = {}
                        # )
                        predecessor = node.args[0]
                        if len(predecessor.users) == 0:
                            # We can safely remove as the precessessor
                            # is only used by this node
                            cls.graph_erase_node(graph, node)
                            del existing_nodes[pos]
                            if verbose > 2:
                                print(
                                    f"[CustomTracer.remove_inplace] C.remove "
                                    f"{pos}: {node.target}({node.args}) -> {node}"
                                )
                            continue

                    assert (
                        node.target.name() in {"aten::copy_", "aten::fill_.Tensor"}
                        and len(node.args) == 2
                    ) or node.target.name() in {"aten::sigmoid_"}, (
                        f"(inplace) Unsupported target {node.target!r}, target_name="
                        f"{node.target.name()!r}, name={node.name!r}, node.args={node.args} "
                        f"at position {pos}/{len(graph.nodes)}"
                        f"\n--original graph--\n{err_graph}"
                        f"\n--graph\n{exported_program or graph}"
                    )

                    # We check the predecessor if the node is a node copy_.
                    predecessor = node.args[0]
                    predecessor_name = (
                        predecessor.target.name()
                        if hasattr(predecessor.target, "name")
                        else predecessor.target
                    )
                    if predecessor_name in {"aten::slice.Tensor", "aten::select.int"}:
                        # We face a schema such as
                        # K_33[2:-2, 2:-2, :-1] = sumx[None, 2:-2, None]
                        do_break = cls._modify_graph_clone_index_copy_(
                            graph,
                            node,
                            existing_nodes,
                            pos,
                            exported_program,
                            err_graph,
                            verbose=verbose,
                            exc=exc,
                        )
                        if do_break == -1:
                            if verbose:
                                print(f"[remove_inplace] unable to remove (1) {node.target}")
                            return -1
                        if do_break:
                            break
                        continue

                    do_break = cls._modify_graph_clone_copy_(
                        graph,
                        node,
                        existing_nodes,
                        pos,
                        exported_program,
                        err_graph,
                        verbose=verbose,
                        exc=exc,
                    )
                    if do_break == -1:
                        if verbose:
                            print(f"[remove_inplace] unable to remove (2) {node.target}")
                        return -1
                    if do_break:
                        break
                else:
                    assert node.target in {
                        "add_",
                        "div_",
                        "mul_",
                        "mod_",
                        "sub_",
                        torch.exp_,
                        torch.sigmoid_,
                        operator.setitem,
                        # other function may be needed, let's be strict
                    }, (
                        f"Unsupported target {node.target!r}, name={node.name!r} "
                        f"at position {pos}/{len(graph.nodes)}\n--graph\n{graph}"
                    )

                    # We still need to check the predecessor.
                    # We check the predecessor if the node is a node copy_.
                    predecessor_name = (
                        node.args[0].target.name()
                        if hasattr(node.args[0].target, "name")
                        else node.args[0].target
                    )
                    if predecessor_name in {
                        "aten::slice.Tensor",
                        "aten::select.int",
                        operator.getitem,
                    }:
                        # We face a schema such as
                        # K_33[2:-2, 2:-2, :-1] = sumx[None, 2:-2, None]
                        do_break = cls._modify_graph_clone_index_copy_(
                            graph,
                            node,
                            existing_nodes,
                            pos,
                            exported_program,
                            err_graph,
                            verbose=verbose,
                            exc=exc,
                        )
                        if do_break == -1:
                            if verbose:
                                print(f"[remove_inplace] unable to remove (3) {node.target}")
                            return -1
                        if do_break:
                            break
                        continue

                    # We assume the first argument is the one modified inplace.
                    new_name = node
                    old_name = node.args[0]

                    if verbose > 2:
                        print(
                            f"[CustomTracer.remove_inplace] D.process {pos}: "
                            f"{node.target}({node.args}) -> {node}"
                        )

                    # class Node can be used as a key
                    # We also assume a user is placed after this node.
                    nodes_to_leave = {n[1] for n in existing_nodes[: pos + 1]}

                    # let's replace
                    changed = old_name.replace_all_uses_with(
                        new_name,
                        delete_user_cb=(
                            lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)
                        ),
                    )

                    assert changed, (
                        f"No change applied, the inplace node [{node}] at position {pos} "
                        f"does not replace [{old_name}] in \n{graph}\n-- node to keep --"
                        f"\n{nodes_to_leave}"
                    )

            # We need to continue in case one unused node left another one
            # after it was removed. It could be improved by looking at
            inplace = cls._inplace_nodes(graph)
            if len(inplace) == 0:
                # No inplace left.
                break
            max_iter -= 1

        if verbose:
            print(
                f"[CustomTracer.remove_inplace] end with {max_iter} "
                f"iterations and {len(graph.nodes)} nodes"
            )
        assert len(inplace) == 0, (
            f"Inplace nodes remain at positions {sorted(inplace)}"
            f"/{len(graph.nodes)} in\n{graph}\n--original graph--\n{err_graph}"
        )
        return n_inplace
