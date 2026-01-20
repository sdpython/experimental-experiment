import enum
import traceback
import torch
from .tracing_graph import Graph, Node, WrapperWalker

TRACING_TENSOR_INDEX = 0


class TracingTensorIndex:
    __slots__ = ["index"]

    def __init__(self, index: int | None = None):
        if index is None:
            global TRACING_TENSOR_INDEX
            TRACING_TENSOR_INDEX += 1
            self.index = TRACING_TENSOR_INDEX
        else:
            self.index = index

    def __str__(self) -> str:
        return f"i{self.index}"

    def __repr__(self) -> str:
        return f"TracingTensorIndex({self.index})"


class TracingKind(enum.IntEnum):
    FUNCTION = 1
    OPS = 2


class TracingShape:
    def __init__(self, tracing_tensor):
        self.tracing_tensor = tracing_tensor
        self.shape_traced_as = None

    @property
    def shape_as_tracing_tensor(self):
        if self.shape_traced_as is not None:
            return self.shape_traced_as
        stack = traceback.extract_stack()[:-1]
        self.tracing_tensor.__tracing_context__.enter_tracing_event(
            TracingKind.OPS, stack=stack, ops="shape", args=(self.tracing_tensor,)
        )
        true_shape = self.tracing_tensor.true_shape
        t = torch.tensor(true_shape, dtype=torch.int64, device="meta")
        traced = TracingTensor(t)
        traced.__tracing_context__ = self.tracing_tensor.__tracing_context__
        traced.source = self
        t.__traced_as__ = traced
        self.shape_traced_as = traced
        traced.__tracing_context__.leave_tracing_event(TracingKind.OPS, ops="shape", res=traced)
        return traced

    @property
    def true_shape(self):
        return self.tracing_tensor.true_shape

    def __repr__(self):
        return "TracedTensor"


class TracingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, **kwargs):
        global TRACING_TENSOR_INDEX
        assert not kwargs, f"Not implemented for kwargs={kwargs}"
        assert isinstance(data, torch.Tensor) or not data, f"type(data)={type(data)} not allowed"
        res = torch.Tensor._make_subclass(
            cls, data, data.requires_grad if isinstance(data, torch.Tensor) else False
        )
        res.__INDEX__ = TracingTensorIndex()
        return res

    @property
    def shape(self):
        return TracingShape(self)

    @property
    def true_shape(self):
        return super().shape

    @classmethod
    def wraps_with_tracing_tensor(cls, res, context=None):
        def make_tracing_tensor(t):
            tt = TracingTensor(t)
            if context is not None:
                tt.__tracing_context__ = context
            return tt

        def check_tracing_tensor(t):
            if not hasattr(t, "__INDEX__"):
                t.__INDEX__ = TracingTensorIndex()
            if not hasattr(t, "__tracing_context__") and context:
                t.__tracing_context__ = context
            return t

        return WrapperWalker(
            {TracingTensor: check_tracing_tensor, torch.Tensor: make_tracing_tensor}
        ).wraps(res)

    @classmethod
    def tracing_tensor_as_ids(cls, context, res):
        def wrap_tensor(t):
            context.store_tensor_constant(res)
            return t

        return WrapperWalker(
            {
                torch.Tensor: wrap_tensor,
                torch.nn.Parameter: wrap_tensor,
                TracingTensor: lambda t: t.__INDEX__,
                TracingShape: lambda t: t.shape_traced_as.__INDEX__,
            }
        ).wraps(res)


class TracingDispatch(torch.utils._python_dispatch.TorchDispatchMode):

    PREPROCESS_ARGS = {
        torch.Tensor: lambda t: t,  # for lifted constant
        torch.nn.Parameter: lambda t: t,
        TracingTensor: lambda t: t,
        TracingShape: lambda t: t.true_shape,
        int: lambda t: t,
        float: lambda t: t,
    }

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        stack = traceback.extract_stack()[:-1]
        if self._hook:
            assert len(self._hook) == 1
            name, h_args, h_kwargs = self._hook.pop()
            assert name == "ones", f"name={name!r}"
            assert len(args) == len(h_args)
            assert {
                k: v for k, v in kwargs.items() if k in h_kwargs
            } == h_kwargs, f"kwargs={kwargs}, h_kwargs={h_kwargs}"
            args = h_args

        self.tracing_context.enter_tracing_event(
            TracingKind.OPS, stack=stack, ops=func, types=types, args=args, kwargs=kwargs
        )
        args = WrapperWalker(TracingDispatch.PREPROCESS_ARGS).wraps(args)
        res = func(*args, **(kwargs or {}))
        res = TracingTensor.wraps_with_tracing_tensor(res, context=self.tracing_context)
        self.tracing_context.leave_tracing_event(TracingKind.OPS, ops=func, res=res)
        return res

    def add_hook(self, name, args, kwargs):
        self._hook.append((name, args, kwargs))


class TracingFunction(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        stack = traceback.extract_stack()[:-1]
        self.tracing_context.enter_tracing_event(
            TracingKind.FUNCTION, stack=stack, ops=func, types=types, args=args, kwargs=kwargs
        )
        res = func(*args, **(kwargs or {}))
        self.tracing_context.leave_tracing_event(TracingKind.FUNCTION, ops=func, res=res)
        return res


class TracingContextEvent:
    def __init__(
        self,
        context,
        kind,
        stack=None,
        types=None,
        ops=None,
        args=(),
        kwargs=None,
        res=None,
    ):
        self.context = context
        self.level = context.level
        self.kind = kind
        self.stack = stack
        self.ops = ops
        self.args = TracingTensor.tracing_tensor_as_ids(context, args)
        self.kwargs = TracingTensor.tracing_tensor_as_ids(context, kwargs)
        self.res = TracingTensor.tracing_tensor_as_ids(context, res)
        self.types = types


class TracingContext:
    def _torch_ones(self, *size, **kwargs):
        if len(size) == 1 and isinstance(size, tuple) and isinstance(size[0], TracingShape):
            # shape is requested
            t_shape = size[0].shape_as_tracing_tensor
            self.enter_tracing_event(TracingKind.OPS, ops="shape", args=(t_shape,), kwargs=kwargs)
            self.leave_tracing_event(TracingKind.OPS, ops="shape", res=t_shape)
            new_size = (size[0].true_shape,)
            t_shape.__tracing_context__.add_hook("ones", size, kwargs)
            return self._torch_functions["ones"](*new_size, **kwargs)
        return self._torch_functions["ones"](*size, **kwargs)

    def __init__(
        self,
        model: torch.nn.Module,
        verbose: int = 0,
        debug_counts: dict[str, int] | None = None,
    ):
        self.tracing_dispatch = TracingDispatch()
        self.tracing_function = TracingFunction()
        self.tracing_dispatch._hook = []
        self.tracing_dispatch.tracing_context = self
        self.tracing_function.tracing_context = self
        self.model = model
        self.verbose = verbose
        self.debug_counts = debug_counts or {}

    def add_hook(self, name, args, kwargs):
        self.tracing_dispatch.add_hook(name, args, kwargs)

    def add_node(self, node: Node):
        assert (
            node.op not in self.debug_counts
            or self.graph.counts.get("input", 0) < self.debug_counts[node.op]
        ), (
            f"more inputs than expected: node={node}, counts={self.graph.counts}, "
            f"index={node.value.__INDEX__}"
        )
        return self.graph.add_node(node)

    def init_graph(self):
        self.graph = Graph(verbose=self.verbose)
        for name, param in self.model.named_parameters():
            self.add_node(
                Node(
                    op="placeholder",
                    name=name,
                    val=TracingTensor(param),
                    example_value=param,
                    key=(id(param), param.dtype, param.shape, param.device),
                )
            )
        for name, param in self.model.named_buffers():
            self.add_node(
                Node(
                    op="placeholder",
                    name=name,
                    val=TracingTensor(param),
                    example_value=param,
                    key=(id(param), param.dtype, param.shape, param.device),
                )
            )

    def store_tensor_constant(self, tensor):
        self.constant_tensors.append(tensor)

    def _replace_torch_functions(self):
        self._torch_functions = {"ones": torch.ones}
        torch.ones = lambda *size, **kwargs: self._torch_ones(*size, **kwargs)

    def _move_torch_functions_back(self):
        torch.ones = self._torch_functions["ones"]

    def __enter__(self):
        self.level = 0
        self.stack = []
        self.constant_tensors = []
        self.tracing_enter = self.tracing_dispatch.__enter__(), self.tracing_function.__enter__()
        self.init_graph()
        self._current_node = None
        self._current_func = []
        self._replace_torch_functions()
        return self.tracing_enter

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._move_torch_functions_back()
        self.tracing_function.__exit__(exc_type, exc_val, exc_tb)
        self.tracing_dispatch.__exit__(exc_type, exc_val, exc_tb)
        del self.constant_tensors
        del self.tracing_enter
        del self._current_node
        del self._current_func
        return False

    def enter_tracing_event(self, kind, stack=None, ops=None, args=(), kwargs=None, types=None):
        if self.verbose:
            if kind == TracingKind.OPS:
                print("    " * (self.level + 2), ">", kind, ops)
            elif self.verbose > 1:
                print("    " * (self.level + 2), ">", kind, ops)
        if kind == TracingKind.OPS:
            # unsplittable event
            assert self._current_node is None
            self._current_op_kwargs = dict(
                op="call_function",
                target=ops,
                args=self._process_input_values_for_graph(args),
                kwargs=self._process_input_values_for_graph(kwargs),
                stack_trace=stack,
                types=types,
            )
        self.level += 1
        self.stack.append(
            TracingContextEvent(
                self, kind, stack=stack, ops=ops, args=args, kwargs=kwargs, types=types
            )
        )
        return self

    def leave_tracing_event(self, kind, ops=None, res=None):
        if kind == TracingKind.OPS:
            assert self._current_op_kwargs is not None
            assert res is not None, "not implemented yet when res is None"
            assert isinstance(
                res, TracingTensor
            ), f"not implemented when type(res)={type(res)}, ops={ops}"
            node = Node(
                name=self.graph.new_node_name(),
                key=res.__INDEX__,
                example_value=res,
                val=res,
                **self._current_op_kwargs,
            )
            self.add_node(node)
            self._current_op_kwargs = None
        self.stack.append(TracingContextEvent(self, kind, ops=ops, res=res))
        self.level -= 1
        if self.verbose:
            if kind == TracingKind.OPS:
                print("    " * (self.level + 2), "<", kind, ops)
            elif self.verbose > 1:
                print("    " * (self.level + 2), "<", kind, ops)
        return self

    def wrap_tensor(self, t):
        key = id(t), t.dtype, t.shape, t.device
        if self.graph.has_traced_constant(key):
            return self.graph.get_traced_constant(key)
        # Then it is a lifted constant, t + 1, ...
        tt = TracingTensor(t)
        tt.__INDEX__ = TracingTensorIndex()
        tt.__tracing_context__ = self
        return self.wrap_tracing_tensor(tt, new_op="lifted")

    def wrap_tracing_tensor(self, t, new_op: str = "input"):
        assert isinstance(t, TracingTensor), f"unexpected type {type(t)}"
        if self.graph.has_traced_value(t.__INDEX__):
            return self.graph.get_traced_value(t.__INDEX__)
        # It is a new input.
        node = self.add_node(
            Node(
                name=self.graph.new_node_name(),
                key=t.__INDEX__,
                op=new_op,
                val=t,
                example_value=t,
            )
        )
        t.__tracing_context__ = self
        assert self.graph.has_traced_node(node.name)
        assert self.graph.has_traced_value(t.__INDEX__)
        return node

    def wrap_tracing_shape(self, t):
        assert isinstance(t, TracingShape), f"unexpected type {type(t)}"
        source = t.tracing_tensor
        assert self.graph.has_traced_value(source.__INDEX__), (
            f"the shape comes from a for traced tensor missing for the "
            f"graph for index={source.__INDEX__}"
            f"\navailable={self.graph.traced_values}"
            f"\navailable={self.graph.traced_nodes}"
        )

        traced_as = t.shape_traced_as
        if not self.graph.has_traced_value(traced_as.__INDEX__):
            # The node computing the shape has not been registered yet.
            self.add_node(
                Node(
                    name=self.graph.new_node_name(),
                    key=traced_as.__INDEX__,
                    op="call_functon",
                    val=traced_as,
                    example_value=t,
                    ops="shape",
                )
            )
            t.__tracing_context__ = self
            traced_as.__tracing_context__ = self
        return self.graph.get_traced_value(traced_as.__INDEX__)

    def _process_input_values_for_graph(self, args):
        return WrapperWalker(
            {
                torch.Tensor: self.wrap_tensor,
                torch.nn.Parameter: self.wrap_tensor,
                TracingTensor: self.wrap_tracing_tensor,
                TracingShape: self.wrap_tracing_shape,
            }
        ).wraps(args)
