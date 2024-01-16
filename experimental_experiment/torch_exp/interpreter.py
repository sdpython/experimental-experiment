import inspect
import operator
import types
from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np
from ._helper import make_hash
from .aten_functions import find_function
from .aten_methods import find_method


class DynamoInterpreter:
    def _hash(self) -> str:
        return make_hash(self)

    def __init__(
        self, graph_builder: "GraphBuilder", retriever: Callable  # noqa: F821
    ):
        import torch

        self.torch = torch
        self.builder = graph_builder
        self.retriever = retriever
        self.example_values_ = {}

    def run_node(self, node: "torch.fx.Node"):  # noqa: F821
        if hasattr(node, "meta") and "example_value" in node.meta:
            if isinstance(node.target, str) or callable(node.target):
                self.example_values_[node.target] = node.meta["example_value"]
            else:
                raise RuntimeError(
                    f"Unexpected type {type(node.target)} "
                    f"for node.target in {node}, op={node.op}, "
                    f"node.target={node.target}, node.meta={node.meta}."
                )
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.run_node] {node.op}:{node.name}")
        if node.op == "placeholder":
            return self.placeholder(node)
        if node.op == "call_function":
            return self.call_function(node)
        if node.op == "output":
            return self.output(node)
        if node.op == "call_module":
            return self.call_module(node)
        if node.op == "get_attr":
            return self.get_attr(node)
        if node.op == "call_method":
            return self.call_method(node)

        raise ValueError(f"Unable to process node kind {node.op!r} ({node}).")

    def get_attr(self, node: "torch.fx.Node"):  # noqa: F821
        try:
            init = getattr(node.graph.owning_module, node.target)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to find attribute {node.target!r} (node.name={node.name!r}) in "
                f"{list(sorted(dir(node.graph.owning_module)))}."
            ) from e
        self.builder.make_initializer(node.name, init)
        return node.name

    def placeholder(self, node: "torch.fx.Node"):  # noqa: F821
        val = node.meta.get("val", None)
        if val is None:
            example_value = node.meta.get("example_value", None)
            index_input = len(self.builder.inputs)
            if (
                example_value is None
                and self.builder.input_args
                and index_input < len(self.builder.input_args)
            ):
                example_value = self.builder.input_args[index_input]

            if self.builder.as_function and example_value is None:
                return self.builder.make_tensor_input(node.name, None, None)
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
        assert len(declared) == 1, f"declared must have one element: {declared}"
        output = declared[0]
        if hasattr(output, "name"):
            output = output.name
        else:
            assert len(output) == 1, f"declared[0] must have one element: {declared}"
            output = output[0]
            if hasattr(output, "name"):
                output = output.name
        self.builder.make_node("Identity", [output], [output_name], check=False)

        val = node.meta.get("val", None)
        if val is None:
            output_name = node.name
            if output in self.builder._known_shapes:
                elem_type = self.builder.get_type(output)
                shape = self.builder.get_shape(output)
            elif self.builder.as_function:
                elem_type = None
                shape = None
            else:
                raise RuntimeError(
                    f"val is None for node={node}, "
                    f"output={output}, output_name={output_name}"
                    f"\nmeta={node.meta}"
                    f"\nnode.__dict__={node.__dict__}"
                )

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
            node_schema = None

        complete_args = []
        complete_kwargs = {}

        if inspect.isbuiltin(node.target) or not node_schema:
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

        if callable(node.target):
            # a single function
            return f"aten_{node.target.__name__}"

        raise NotImplementedError(
            f"Unsupported function {node!r} (not implemented), "
            f"node.target={node.target}, type is {type(node.target)}."
        )

    def _getitem_slice(
        self,
        node: "torch.fx.Node",  # noqa: F821
        input_name: str,
        index_slice: slice,
        set_shape: bool,
        axis: int,
    ):
        assert isinstance(axis, int), f"Unexpected type {type(axis)} for axis"
        aaxis = np.array([axis], dtype=np.int64)
        axis_name = self.builder.unique_name(f"{node.name}_axis")
        self.builder.make_initializer(axis_name, aaxis)
        start = np.array([index_slice.start or 0], dtype=np.int64)
        end_name = self.builder.unique_name(f"{node.name}_end")
        if index_slice.stop is None:
            shape_name = self.builder.unique_name(f"{node.name}_shape")
            self.builder.make_node("Shape", [input_name], [shape_name])
            self.builder.make_node(
                "GatherElements", [input_name, axis_name], [end_name]
            )
        else:
            end = np.array([index_slice.stop], dtype=np.int64)
            self.builder.make_initializer(end_name, end)

        inputs = [
            input_name,
            self.builder.make_initializer(
                self.builder.unique_name(f"{node.name}_start"), start
            ),
            end_name,
            axis_name,
        ]
        if index_slice.step is not None:
            step = np.array([index_slice.step], dtype=np.int64)
            inputs.append(
                self.builder.make_initializer(
                    self.builder.unique_name(f"{node.name}_step"), step
                )
            )

        res = self.builder.make_node("Slice", inputs, [node.name])
        if not set_shape:
            dtype = self.builder.get_type(inputs[0])
            shape = self.builder.get_shape(inputs[0])
            self.builder.set_shape(
                node.name,
                self.builder._apply_slice_to_shape(shape, index_slice, axis=axis),
            )
            self.builder.set_type(node.name, dtype)
        return res

    def getitem(self, node: "torch.fx.Node"):  # noqa: F821
        args = node.args
        assert len(args) == 2
        node_output, index = args
        result_name = node_output.name
        val = node.meta.get("val", None)
        set_shape = False
        if val is not None:
            if isinstance(val, self.torch.Tensor):
                shape = val.shape
                dtype = self.builder._get_type(val.dtype)
                self.builder.set_shape(node.name, shape)
                self.builder.set_type(node.name, dtype)
                set_shape = True
            else:
                raise TypeError(
                    f"Unexpected type in node {node!r}, type(val)={type(val)}."
                )

        if hasattr(index, "name"):
            # A dynamic index (torch.fx.Node)
            res = self.builder.make_node(
                "Gather", [result_name, index.name], [node.name]
            )
            if not set_shape:
                dtype = self.builder.get_type(result_name)
                self.builder.set_type(node.name, dtype)
            return res

        if isinstance(index, int):
            return self.builder.make_node(
                "Identity", [f"{result_name}#{index}"], [node.name]
            )

        if isinstance(index, slice):
            return self._getitem_slice(
                node, node_output.name, index, set_shape=set_shape, axis=0
            )

        if isinstance(index, tuple):
            if all(map(lambda x: x is Ellipsis or isinstance(x, (slice, int)), index)):
                # case where there is only one slice
                n_ellipsis = sum(map(lambda x: 1 if x is Ellipsis else 0, index))
                if n_ellipsis == len(index) - 1:
                    axis = max(
                        map(
                            lambda ix: -1 if ix[1] is Ellipsis else ix[0],
                            enumerate(index),
                        )
                    )
                    return self._getitem_slice(
                        node,
                        node_output.name,
                        index[axis],
                        set_shape=set_shape,
                        axis=axis,
                    )

        raise RuntimeError(
            f"getitem: unexpected type {type(index)} for index={index}, "
            f"node={node}, args={args}, val={val}"
            f"{self.builder.get_debug_msg()}"
        )

    def call_function(self, node: "torch.fx.Node"):  # noqa: F821
        fx_args, fx_kwargs = self._fill_in_default_kwargs(node)
        aten_name = self._get_aten_name(node)
        if aten_name == "getitem":
            return self.getitem(node)
        fct = find_function(aten_name, args=node.args, kwargs=node.kwargs)

        args = []
        for i in fx_args:
            if i is None:
                args.append(None)
            elif isinstance(i, str):
                args.append(i)
            elif hasattr(i, "name"):
                args.append(i.name)
            elif isinstance(i, tuple):
                # For node cat=concat
                args.append(tuple(t.name for t in i))
            else:
                raise RuntimeError(f"Unexpected type {type(i)} in args={node.args}")

        output_names = self._get_output_names(node)
        can_set = self._can_set_shape_and_type(node)

        res = fct(self.builder, not can_set, output_names, *args, **fx_kwargs)

        self._set_shape_and_type(node, res)
        res = self._check_output_name(node, res, output_names)
        return res

    def call_method(self, node: "torch.fx.Node"):  # noqa: F821
        method_name = node.target
        assert isinstance(
            node.args, tuple
        ), f"Unexpected type {type(node.args)} for node.args."

        fct = find_method(
            f"aten_meth_{method_name}", args=node.args, kwargs=node.kwargs
        )
        args = node.args
        args = [getattr(args[0], "name", args[0]), *args[1:]]
        kwargs = node.kwargs
        output_names = self._get_output_names(node)
        can_set = self._can_set_shape_and_type(node)

        res = fct(self.builder, not can_set, output_names, *args, **kwargs)

        self._set_shape_and_type(node, res)
        res = self._check_output_name(node, res, output_names)
        return res

    def _get_output_names(self, node: "torch.fx.Node") -> List[str]:  # noqa: F821
        val = node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [f"{node.name}#{i}" for i in range(n_outputs)]
        else:
            output_names = [node.name]
        return output_names

    def _check_output_name(
        self,
        node: "torch.fx.Node",  # noqa: F821
        res: Union[str, List[str]],
        output_names: List[str],
    ) -> Union[str, List[str]]:
        if isinstance(node.name, str):
            if len(output_names) != 1:
                if output_names != list(res):
                    raise NotImplementedError(
                        f"Unexpected output_names {output_names}, res={res!r}, node.name={node.name!r}"
                    )
            elif res != node.name:
                self.builder.make_node("Identity", [res], [node.name])
                res = node.name
        else:
            raise NotImplementedError(
                f"Unexpected type {type(node.name)} for node.name={node.name!r}."
            )
        return res

    def _can_set_shape_and_type(self, node: "torch.fx.Node") -> bool:  # noqa: F821
        return node.meta.get("val", None) is not None

    def _set_shape_and_type(
        self, node: "torch.fx.Node", res: Union[str, List[str]]  # noqa: F821
    ):
        val = node.meta.get("val", None)
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

    def call_module(self, node: "torch.fx.Node"):  # noqa: F821
        def raise_msg():
            import pprint

            return (
                f"node={node}\n--\nnode.__dict__={pprint.pformat(node.__dict__)}"
                f"\n--\n{pprint.pformat(node.meta)}\n---\n{dir(node)}"
                f"\n---GRAPH\n{type(node.graph)}\n---GRAPH\n{node.graph}"
                f"\n---GRAPH\n{node.graph.__dict__}\n---GRAPH\n{dir(node.graph)}"
                f"\n---GRAPH.MODULE\n{type(node.graph.owning_module)}"
                f"\n---GRAPH.MODULE\n{id(node.graph.owning_module)}"
                f"\n---GRAPH.MODULE\n{node.graph.owning_module}"
                # f"\n---GRAPH.MODULE\n{node.graph.owning_module.__dict__}"
                f"\n---GRAPH.MODULE\n{dir(node.graph.owning_module)}"
                f"\nVALUES\n{pprint.pformat(self.example_values_)}"
            )

        import torch
        from .onnx_export import _make_builder_interpreter

        sub_module = node.graph.owning_module.get_submodule(node.target)

        if not isinstance(sub_module, torch.nn.Module):
            raise NotImplementedError(
                f"Not implemented for type {type(sub_module)}.\n{raise_msg()}"
            )

        named_args = node.args
        args = []
        for a in named_args:
            val = a.meta.get("example_value", None)
            args.append(val)

        if hasattr(sub_module, "graph") and isinstance(
            sub_module, torch.fx.GraphModule
        ):
            gm = sub_module
        else:
            # https://pytorch.org/docs/stable/fx.html
            tracer_class = torch.fx.Tracer
            graph = tracer_class().trace(sub_module)
            gm = torch.fx.GraphModule(sub_module, graph)

        graph_module, builder, interpreter = _make_builder_interpreter(
            gm,
            tuple(args),
            as_function=True,
            target_opset=self.builder.opsets,
            optimization_options=self.builder.optimization_options,
            verbose=self.builder.verbose,
        )
        builder.process(graph_module, interpreter)
        assert builder.outputs, f"No output detected for node={node}, graph={gm}"

        fx_args, _ = self._fill_in_default_kwargs(node)
        args = [getattr(i, "name", i) for i in fx_args]

        val = node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [f"{node.name}#{i}" for i in range(n_outputs)]
        else:
            output_names = [node.name]
            if val is None:
                val = node.meta.get("example_value", None)
        if val is not None and not isinstance(val, tuple):
            val = (val,)

        if val is not None:
            assert len(val) == len(
                builder.outputs
            ), f"Output mismatch {len(val)} != {len(builder.outputs)}"
            for i in range(len(val)):
                name = builder.outputs[i].name
                if name not in builder._known_shapes:
                    builder.set_shape(name, val[i].shape)
                if name not in builder._known_types:
                    builder.set_type(name, val[i].dtype)

        self.builder.make_nodes(
            builder, args, output_names, prefix=f"_sub_{sub_module.__class__.__name__}_"
        )
        return output_names
