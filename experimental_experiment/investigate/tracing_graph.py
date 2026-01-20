from typing import Callable, Dict
import torch


class WrapperWalker:
    def __init__(self, wrap_classes: Dict[type, Callable]):
        self.wrap_classes = wrap_classes

    def wraps(self, res):
        if res is None:
            return None
        if type(res) in self.wrap_classes:
            return self.wrap_classes[type(res)](res)
        if isinstance(res, tuple):
            return tuple(self.wraps(t) for t in res)
        if type(res) is list:
            return [self.wraps(t) for t in res]
        if type(res) is dict:
            return {k: self.wraps(t) for k, t in res.items()}
        if isinstance(res, (int, float, bool, str)):
            return res
        if type(res) in (torch.dtype, torch.device):
            return res
        raise NotImplementedError(
            f"wraps_with_tracing_tensor not implemented for type({type(res)})"
        )


class Node:
    OP_VALUES = {
        "placeholder",
        "call_function",
        "call_method",
        "call_module",
        "input",
        "output",
        "lifted",
    }

    def __init__(
        self,
        key=None,
        name=None,
        op=None,
        target=None,
        args=(),
        kwargs=None,
        stack_trace=None,
        nn_module_stack=None,
        val=None,
        example_value=None,
        types=None,
    ):
        assert all(
            (a is None or isinstance(a, (Node, (int, float, bool)))) for a in args
        ), f"Unexpected type in ags {[type(a) for a in args]}\n{args}"
        assert key is not None, "key cannot be None"
        assert not isinstance(key, int), "key cannot be an integer"
        assert name, "name cannot be empty"
        assert op in self.OP_VALUES, f"op={op!r} not in {self.OP_VALUES}"
        self.key = key
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.types = types
        self.meta = {}
        if stack_trace:
            self.meta["stack_trace"] = stack_trace
        if nn_module_stack:
            self.meta["nn_module_stack"] = nn_module_stack
        if val is not None:
            self.meta["val"] = val
        if example_value is not None:
            self.meta["example_value"] = example_value

    def __str__(self) -> str:
        sname = f"%{self.name}[{self.value.__INDEX__}]"
        if self.op in {"call_method", "call_function"}:
            vs = [(f"{a.name}" if hasattr(a, "name") else str(a)) for a in self.args]
            vss = ", ".join(vs)
            if not self.kwargs:
                return f"[{self.op[:6]}] {sname} = {self.target}({vss})"
            return f"[{self.op[:6]}] {sname} = {self.target}({vss}, {self.kwargs})"
        if self.op == "placeholder":
            value = self.value
            return (
                f"[placeh] {sname} = {value.__class__.__name__}: "
                f"{value.dtype}: {tuple(value.true_shape)}"
            )
        if self.op == "lifted":
            value = self.value
            return (
                f"[lifted] {sname} = {value.__class__.__name__}: "
                f"{value.dtype}: {tuple(value.true_shape)}"
            )
        if self.op == "input":
            value = self.value
            return f"[input-] {sname} = {value.__class__.__name__}: {value.dtype}"
        raise NotImplementedError(f"Not implemented for op={self.op!r} in node.op={self.op!r}")

    def __repr__(self) -> str:
        return f"{self.op}%{self.name}"

    @property
    def value(self):
        assert self.has_value(), f"value is missing in node {self}"
        return self.meta["val"]

    def has_value(self) -> bool:
        return "val" in self.meta and self.meta["val"] is not None

    @property
    def index(self):
        if self.op == "placeholder":
            value = self.value
            if value.__class__.__name__ == "TracingTensor":
                return value.__INDEX__
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected type {type(value)} in node={self}"
            return id(value), value.dtype, value.shape, value.device
        raise NotImplementedError(f"index not implemented for op={self.op!r} in node={self}")


class Graph:
    def __init__(self, verbose: int = 0):
        self.nodes = []
        self.traced_constants = {}
        self.traced_nodes = {}
        self.traced_values = {}
        self.verbose = verbose
        self.counts = {}

    def new_node_name(self) -> str:
        return f"n{len(self.nodes)}"

    def has_traced_constant(self, key) -> bool:
        assert isinstance(key, tuple) and len(key) == 4, f"key={key!r}"
        return key in self.traced_constants

    def get_traced_constant(self, key):
        assert isinstance(key, tuple) and len(key) == 4, f"key={key!r}"
        return self.traced_constants[key]

    def has_traced_node(self, key):
        assert isinstance(key, str) and key, f"key={key!r}"
        return key in self.traced_nodes

    def get_traced_node(self, key):
        assert isinstance(key, str) and key, f"key={key!r}"
        return self.traced_constants[key]

    def has_traced_value(self, key):
        assert hasattr(key, "index"), f"key={key!r}"
        return key in self.traced_values

    def get_traced_value(self, key):
        assert hasattr(key, "index"), f"key={key!r}"
        return self.traced_values[key]

    def add_node(self, node: Node) -> Node:
        if self.verbose:
            print(f"[Graph.add_node] {node}")
        self.nodes.append(node)

        if node.op not in self.counts:
            self.counts[node.op] = 0
        self.counts[node.op] += 1

        assert node.has_value(), f"value is missing for node={node}"
        value = node.value
        assert (
            value.__class__.__name__ == "TracingTensor"
        ), f"Not implemented when type(value)={type(value)}"
        self.traced_nodes[node.name] = node
        self.traced_values[value.__INDEX__] = node

        if node.op == "placeholder":
            self.traced_constants[node.key] = node
            return node
        if node.op == "lifted":
            self.traced_constants[node.key] = node
            return node
        if node.op == "input":
            return node
        if node.op == "call_function":
            assert all(
                isinstance(v, (int, float, str, bool, torch.dtype, torch.device))
                for v in node.kwargs.values()
            ), f"Not implemented when kwargs types = {[type(v) for v in node.kwargs.values()]}"
            assert all(
                isinstance(v, (int, float, Node, str, bool)) for v in node.args
            ), f"Not implemented when args types = {[type(v) for v in node.args]}"
            return node
        raise NotImplementedError(f"Not implemented for op={self.op!r} in node={self}")

    def __str__(self) -> str:
        return "\n".join(["Graph()", *[f"    {n}" for n in self.nodes]])
