import inspect
import operator
import types
from typing import Any, Callable, Dict, List, Tuple
from .aten_functions import find_function


class DynamoWalker:
    def __init__(
        self, graph_builder: "GraphBuilder", retriever: Callable  # noqa: F821
    ):
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
        if node.op == "call_module":
            return self.call_module(node)

        raise ValueError(f"Unable to process node kind {node.op!r} ({node}).")

    def placeholder(self, node: "torch.fx.Node"):  # noqa: F821
        val = node.meta.get("val", None)
        if val is None:
            example_value = node.meta.get("example_value", None)
            if example_value is None:
                raise RuntimeError(
                    f"Unable to guess what node is, node={node}, "
                    f"meta={node.meta} {node.__dict__}."
                )
            return self.builder.make_tensor_input(
                node.name, elem_type=example_value.dtype, shape=example_value.shape
            )
        if isinstance(val, self.torch.Tensor):
            stack_trace = node.meta.get("stack_trace", None)
            if stack_trace is None:
                # torch 2.1.0 and 2.2.0 behave differently.
                return self.builder.make_tensor_input(
                    node.name, elem_type=val.dtype, shape=val.shape
                )
            if "nn_module_stack" not in node.meta:
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

        val = node.meta.get("val", None)
        if val is None:
            output_name = node.name
            if output not in self.builder._known_shapes:
                raise RuntimeError(
                    f"val is None for node={node}, "
                    f"output={output}, output_name={output_name}"
                    f"\nmeta={node.meta}"
                    f"\nnode.__dict__={node.__dict__}"
                )

            elem_type = self.builder._known_types[output]
            shape = self.builder._known_shapes[output]
            self.builder.make_tensor_output(
                output_name, elem_type=elem_type, shape=shape
            )
            return output_name

        if isinstance(val, tuple):
            if len(val) > 1:
                raise NotImplementedError("Not yet implemented for multiple outputs.")
            val = val[0]

        if isinstance(val, self.torch.Tensor):
            output_name = node.name
            shape = val.shape
            dtype = self.builder._get_type(val.dtype)
            self.builder.make_tensor_output(output_name, dtype, shape)
            return output_name

        raise TypeError(f"Unexpected output type {type(val)}.")

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

    def getitem(self, node: "torch.fx.Node"):  # noqa: F821
        args = node.args
        assert len(args) == 2
        node_output, index = args
        result_name = node_output.name
        val = node.meta.get("val", None)
        if val is not None:
            if isinstance(val, self.torch.Tensor):
                shape = val.shape
                dtype = self.builder._get_type(val.dtype)
                self.builder.set_shape(node.name, shape)
                self.builder.set_type(node.name, dtype)
            else:
                raise TypeError(
                    f"Unexpected type in node {node!r}, type(val)={type(val)}."
                )
        return self.builder.make_node(
            "Identity", [f"{result_name}#{index}"], [node.name]
        )

    def call_function(self, node: "torch.fx.Node"):  # noqa: F821
        fx_args, fx_kwargs = self._fill_in_default_kwargs(node)
        aten_name = self._get_aten_name(node)
        if aten_name == "getitem":
            return self.getitem(node)
        fct = find_function(aten_name)

        args = []
        for i in fx_args:
            if hasattr(i, "name"):
                args.append(i.name)
            else:
                args.append(i)

        val = node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [f"{node.name}#{i}" for i in range(n_outputs)]
        else:
            output_names = [node.name]

        try:
            res = fct(self.builder, output_names, *args, **fx_kwargs)
        except (TypeError, AttributeError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"Unable to convertn node {node!r}, node.meta={node.meta}, "
                f"node.__dict__={node.__dict__}."
            ) from e

        if val is not None:
            # extracting shape and types
            if not isinstance(val, tuple):
                val = (val,)
                res = (res,)
            if len(val) != len(res):
                raise RuntimeError(f"Length mismatch between {val} and {res}.")
            for v, r in zip(val, res):
                if isinstance(v, self.torch.Tensor):
                    shape = v.shape
                    dtype = self.builder._get_type(v.dtype)
                    self.builder.set_shape(r, shape)
                    self.builder.set_type(r, dtype)
                else:
                    raise TypeError(
                        f"Unexpected type in node {node!r}, type(val)={type(v)}."
                    )

        return res

    def call_module(self, node: "torch.fx.Node"):  # noqa: F821
        def raise_msg():
            import pprint

            return (
                f"node={node}\n--\nnode.__dict__={pprint.pformat(node.__dict__)}"
                f"\n--\n{pprint.pformat(node.meta)}\n---\n{dir(node)}"
                f"\n---GRAPH\n{type(node.graph)}\n---GRAPH\n{node.graph}"
                f"\n---GRAPH\n{node.graph.__dict__}\n---GRAPH\n{dir(node.graph)}"
                f"\n---GRAPH.MODULE\n{type(node.graph.owning_module)}"
                f"\n---GRAPH.MODULE\n{node.graph.owning_module}"
                # f"\n---GRAPH.MODULE\n{node.graph.owning_module.__dict__}"
                f"\n---GRAPH.MODULE\n{dir(node.graph.owning_module)}"
                f"\nVALUES\n{pprint.pformat(self.example_values_)}"
            )

        import torch
        from .onnx_export import _make_builder_walker

        sub_module = node.graph.owning_module.get_submodule(node.target)

        if isinstance(sub_module, torch.nn.Module):
            named_args = node.args
            args = []
            for a in named_args:
                val = a.meta.get("example_value", None)
                print(val.dtype, val.shape)
                args.append(val)
            print("++++++", args)
            graph_module, builder, walker = _make_builder_walker(
                sub_module,
                tuple(args),
                as_function=True,
                target_opset=self.builder.opsets,
                optimization_options=self.builder.optimization_options,
            )
            builder.process(graph_module, walker)
        else:
            raise NotImplementedError(
                f"Not implemented for type {type(sub_module)}.\n{raise_msg()}"
            )
        print(builder.nodes)
        print(builder.initializers)

        stop
