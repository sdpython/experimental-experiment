import inspect
import operator
import types
from typing import Any, Callable, Dict, List, Tuple
from .graph_builder import GraphBuilder
from .aten_functions import find_function


class DynamoWalker:
    def __init__(self, graph_builder: GraphBuilder, retriever: Callable):
        import torch

        self.torch = torch
        self.builder = graph_builder
        self.retriever = retriever

    def __call__(self, node: "torch.fx.Node"):  # noqa: F821
        if node.op == "placeholder":
            return self.placeholder(node)
        if node.op == "call_function":
            return self.call_function(node)
        if node.op == "output":
            return self.output(node)

        raise ValueError(f"Unable to process node kind {node.op!r} ({node}).")

    def placeholder(self, node: "torch.fx.Node"):  # noqa: F821
        val = node.meta.get("val", None)
        stack_trace = node.meta.get("stack_trace", None)
        if val is None:
            return self.builder.make_input(node.name)
        if isinstance(val, self.torch.Tensor):
            if stack_trace is None:
                return self.builder.make_tensor_input(
                    node.name, elem_type=val.dtype, shape=val.shape
                )
            value = self.retriever(node.target)
            return self.builder.make_initializer(node.name, value)
        raise RuntimeError(f"Unsupported type {type(val)} for a placeholder.")

    def output(self, node):
        output_name = node.name
        declared = node.args
        assert len(declared) == 1, "declared must have one element"
        output = declared[0]
        assert len(output) == 1, "declared[0] must have one element"
        output = output[0]
        if hasattr(output, "name"):
            output = output.name
        self.builder.make_node("Identity", [output], [output_name])
        self.builder.make_tensor_output(output_name)
        return output_name

    def _fill_in_default_kwargs(
        self, node: "torch.fx.Node"  # noqa: F821
    ) -> Tuple[List[Any], Dict[str, Any]]:
        if hasattr(node.target, "_schema"):
            node_schema = node.target._schema
        else:
            node_schema = self.torch.ops.aten.sym_size.int._schema

        complete_args = []
        complete_kwargs = {}

        if inspect.isbuiltin(node.target):
            complete_args = list(node.args)
        else:
            for i, expected_arg in enumerate(node_schema.arguments):
                if i < len(node.args):
                    complete_args.append(node.args[i])
                elif expected_arg.name in node.kwargs:
                    complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
                else:
                    # Get default from schema.
                    complete_kwargs[expected_arg.name] = expected_arg.default_value

        return complete_args, complete_kwargs

    def _get_aten_name(self, node: "torch.fx.Node") -> str:  # noqa: F821
        if node.target == operator.getitem:
            return "getitem"
        if isinstance(node.target, self.torch._ops.OpOverloadPacket):
            if node.target != self.torch.ops.aten.sym_size:
                raise RuntimeError(f"Unsupported function {node!r}.")
            raise NotImplementedError(
                f"Unsupported function {node!r} (not implemented)."
            )

        if isinstance(node.target, types.BuiltinFunctionType):
            return node.target

        if isinstance(node.target, self.torch._ops.OpOverload):
            return node.target

        raise NotImplementedError(f"Unsupported function {node!r} (not implemented).")

    def call_function(self, node: "torch.fx.Node"):  # noqa: F821
        fx_args, fx_kwargs = self._fill_in_default_kwargs(node)
        aten_name = self._get_aten_name(node)
        fct = find_function(aten_name)

        args = []
        for i in fx_args:
            if hasattr(i, "name"):
                args.append(i.name)
            else:
                args.append(i)

        return fct(self.builder, [node.name], *args, *fx_kwargs)
