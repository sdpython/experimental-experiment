import inspect
import operator
import pprint
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from onnx import TensorProto
from ..torch_test_helper import string_type
from ..xbuilder._shape_helper import all_int, DYNAMIC_SHAPE
from ..xbuilder._helper import make_hash
from ..xbuilder._dtype_helper import (
    torch_dtype_to_onnx_dtype,
    onnx_dtype_to_torch_dtype,
)
from ..xbuilder.model_container import _get_type
from . import LOCAL_DOMAIN
from .export_options import ExportOptions
from ._exceptions import FunctionNotFoundError
from .aten_functions import find_function
from .aten_methods import find_method


class DynamoInterpreter:
    """
    Interprets a torch graph into an ONNX graph.
    Dispatches every node to the appropriate converting function.

    :param graph_builder: a graph builder
    :param retriever: callable to help retrieve the weights in a module,
        see function `_retrieve
        <experimental_experiment.torch_interpreter.onnx_export._retrieve>`.
    :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
    :param export_options: see :class:`ExportOptions
        <experimental_experiment.torch_interpreter.ExportOptions>`
    """

    def _hash(self) -> str:
        return make_hash(self)

    def __init__(
        self,
        graph_builder: "GraphBuilder",  # noqa: F821
        retriever: Callable,
        dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
        example_inputs: Optional[Tuple["torch.Tensor", ...]] = None,  # noqa: F821
        export_options: Optional[ExportOptions] = None,
    ):
        import torch

        self.torch = torch
        self.builder = graph_builder
        self.retriever = retriever
        self.dispatcher = dispatcher
        self.export_options = export_options
        self.example_values_ = {}
        assert example_inputs is None or isinstance(
            example_inputs, tuple
        ), f"Unexpected type for example_inputs {type(example_inputs)}"
        assert example_inputs is None or all(
            (
                t is None
                or isinstance(
                    t, (torch.SymInt, torch.SymFloat, torch.Tensor, list, int, float)
                )
            )
            for t in example_inputs
        ), (
            f"Unexpected type for one input in example_inputs "
            f"{[type(t) for t in example_inputs]}"
        )
        self.example_inputs_ = example_inputs
        self.flat_example_inputs_ = self.flatten_inputs(example_inputs)
        self.current_input_ = 0

    def flatten_inputs(self, x: Any) -> List["torch.Tensor"]:  # noqa: F821
        """
        Flatten inputs.
        """
        if x is None:
            return x
        if isinstance(x, (list, tuple)):
            res = []
            for i in x:
                if i is None or isinstance(
                    i,
                    (self.torch.Tensor, self.torch.SymInt, self.torch.SymFloat, int, float),
                ):
                    res.append(i)
                else:
                    res.extend(self.flatten_inputs(i))
            return tuple(res) if isinstance(x, tuple) else res
        raise AssertionError(f"Unexpected type {type(x)} for x")

    def run_node(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Runs a node: call the approrpiate method based on the node type.
        """
        example_value = None
        if hasattr(node, "meta") and "example_value" in node.meta:
            if isinstance(node.target, str) or callable(node.target):
                self.example_values_[node.target] = node.meta["example_value"]
                example_value = self.example_values_[node.target]
            else:
                raise RuntimeError(
                    f"Unexpected type {type(node.target)} "
                    f"for node.target in {node}, op={node.op}, "
                    f"node.target={node.target}, node.meta={node.meta}."
                )
        if self.builder.verbose > 1:
            # verbose
            exa = (
                f"{torch_dtype_to_onnx_dtype(example_value.dtype)}'{tuple(example_value.shape)}"
                if hasattr(example_value, "dtype")
                else ""
            )
            v = node.meta.get("val", None) if hasattr(node, "meta") else None
            val = (
                f"{torch_dtype_to_onnx_dtype(v.dtype)}'{tuple(v.shape)}"
                if hasattr(v, "dtype")
                else ""
            )
            symbol = "#" if self._can_set_shape_and_type(node) else "-"
            a1 = "E" if hasattr(node, "meta") and "example_value" in node.meta else "-"
            a2 = "A" if hasattr(node, "meta") and "val" in node.meta else "-"
            print(
                f"[DynamoInterpreter-{self._hash()}.run_node][{symbol}{a1}{a2}] "
                f"{node.op}:{node.name}:{exa}:{val}"
            )

        # debug
        exa = (
            ("example_value", example_value.dtype, example_value.shape)
            if hasattr(example_value, "dtype")
            else ""
        )
        v = node.meta.get("val", None) if hasattr(node, "meta") else None
        val = ("val", v.dtype, v.shape) if hasattr(v, "dtype") else ""
        self.builder.set_shapes_types(node.name, "run_node", (exa, val))
        self.builder.register_users(node.name, node.users)

        if node.op == "placeholder":
            res = self.placeholder(node)
        elif node.op == "call_function":
            res = self.call_function(node)
        elif node.op == "output":
            res = self.output(node)
        elif node.op == "call_module":
            res = self.call_module(node)
        elif node.op == "get_attr":
            res = self.get_attr(node)
        elif node.op == "call_method":
            res = self.call_method(node)
        else:
            raise ValueError(f"Unable to process node kind {node.op!r} ({node}).")

        # Checks consistency of shapes and types
        name = node.name
        if val and len(val) == 3:
            exp_dtype, exp_shape = val[1:]
            if self.builder.has_type(name):
                itype = self.builder.get_type(name)
                ttype = onnx_dtype_to_torch_dtype(itype)
                aten_name = aten_name = (
                    self._get_aten_name(node) if node.op == "call_function" else "-"
                )
                assert ttype == exp_dtype, (
                    f"Type mismatch for {name!r}, node.op={node.op!r}, "
                    f"aten_name={aten_name!r}, "
                    f"onnx {ttype} != expected torch "
                    f"{exp_dtype}{self.builder.get_debug_msg()}"
                )
            if self.builder.has_shape(name):
                shape = self.builder.get_shape(name)
                self.builder._check_two_shapes_are_compatible(
                    tuple(exp_shape),
                    shape,
                    name=name,
                    register_int=False,
                )
        return res

    def get_attr(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Retrieves an attribute.
        """
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.get_attr][{node.name}]")
        try:
            init = getattr(node.graph.owning_module, node.target)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to find attribute {node.target!r} (node.name={node.name!r}) in "
                f"{list(sorted(dir(node.graph.owning_module)))}."
            ) from e

        if isinstance(init, self.torch.fx.GraphModule):
            # This function is meant to be used later.
            builder, args, kwargs, output_names = self._interpret_sub_module(
                init, None, None, source_node=node
            )
            self.builder.make_local_function(node.name, builder, domain=LOCAL_DOMAIN)
            return None

        self.builder.make_initializer(node.name, init)
        return node.name

    def _make_tensor_input(
        self,
        name: str,
        elem_type: Any,
        shape: DYNAMIC_SHAPE,
        is_dimension: bool,
        users: Iterable[str],
        fake_tensor: bool = False,
    ) -> str:
        if (
            not fake_tensor
            and self.example_inputs_ is not None
            and not self.builder.was_inputs_renamed
        ):
            assert len(self.builder.input_names) < len(self.flat_example_inputs_), (
                f"Too many inputs already ({len(self.builder.input_names)}), "
                f"self.current_input_={self.current_input_}, "
                f"unexpected {name!r} "
                f"after {self.builder.input_names}"
                f"{self.builder.get_debug_msg()}"
            )
            if self.flat_example_inputs_[self.current_input_] is None:
                # We skip it.
                assert len(users) == 0, (
                    f"Input {name!r} (index {self.current_input_}"
                    f"/{len(self.flat_example_inputs_)}) "
                    f"is None but it is used by {users}. "
                    f"Existing inputs {self.builder.input_names}. Example inputs: "
                    f"{['-' if t is None else t.shape for t in self.flat_example_inputs_]}"
                    f"{self.builder.get_debug_msg()}"
                )
                self.current_input_ += 1
                return ""

            # second check
            not_none = tuple(t for t in self.flat_example_inputs_ if t is not None)
            assert len(self.builder.input_names) < len(not_none), (
                f"Too many inputs already ({len(self.builder.input_names)}), "
                f"unexpected {name!r} "
                f"after {self.builder.input_names}"
                f"{self.builder.get_debug_msg()}"
            )

        self.current_input_ += 1
        return self.builder.make_tensor_input(
            name, elem_type, shape, is_dimension=is_dimension
        )

    def placeholder(self, node: "torch.fx.Node"):  # noqa: F821
        """
        placeholder for an input. The interpreter adds an Identity node
        between the input names he wants and the name it has in the
        graph module.
        """
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.placeholder][{node.name}]")
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
                return self._make_tensor_input(
                    node.name, None, None, is_dimension=False, users=node.users
                )

            if example_value is None:
                # The input is not defined.
                # We return.
                self.current_input_ += 1
                return

            if isinstance(
                example_value, (self.builder.torch.SymInt, self.builder.torch.SymFloat)
            ):
                # torch.SymInt
                self.builder.make_dynamic_object(node.name, example_value)
                return self._make_tensor_input(
                    node.name,
                    elem_type=self.builder.torch.int64,
                    shape=(1,),
                    is_dimension=True,
                    users=node.users,
                )

            if isinstance(example_value, (int, float)):
                # int or float
                return self._make_tensor_input(
                    node.name,
                    elem_type=(
                        self.builder.torch.int64
                        if isinstance(example_value, int)
                        else self.builder.torch.float32
                    ),
                    shape=(1,),
                    is_dimension=False,
                    users=node.users,
                )

            return self._make_tensor_input(
                node.name,
                elem_type=example_value.dtype,
                shape=example_value.shape,
                is_dimension=False,
                users=node.users,
            )

        if isinstance(val, (self.torch.Tensor, self.torch._subclasses.fake_tensor.FakeTensor)):
            stack_trace = node.meta.get("stack_trace", None)
            value = None
            if stack_trace is None and "from_node" not in node.meta:
                # torch 2.1.0 and 2.2.0 behave differently.
                # torch 2.4.0, stack_trace is None but from_node is in node.meta
                value = self.retriever(node.target, val, debug={"node": node}, exc=False)
                if value is None:
                    return self._make_tensor_input(
                        node.name,
                        elem_type=val.dtype,
                        shape=val.shape,
                        is_dimension=False,
                        users=node.users,
                        fake_tensor=isinstance(
                            val, self.torch._subclasses.fake_tensor.FakeTensor
                        ),
                    )
            if value is None:
                if "nn_module_stack" not in node.meta:
                    value = self.retriever(node.target, val, debug={"node": node})
                    if value is None:
                        return self._make_tensor_input(
                            node.name,
                            elem_type=val.dtype,
                            shape=val.shape,
                            is_dimension=False,
                            users=node.users,
                        )
                else:
                    value = self.retriever(node.target, val, debug={"node": node})
            if value is None or isinstance(
                value, self.torch._subclasses.fake_tensor.FakeTensor
            ):
                if ".FakeTensor" in str(type(val)):
                    dtype = val.dtype
                    shape = val.shape
                    return self._make_tensor_input(
                        node.name, dtype, shape, False, users=node.users, fake_tensor=True
                    )
                raise RuntimeError(f"value is None, unable to retrieve target {node.target!r}")
            return self.builder.make_initializer(node.name, value)

        if isinstance(val, (self.torch.SymInt, self.torch.SymFloat)):
            return self.builder.make_dynamic_object(node.name, val, shape_as_input=True)

        if isinstance(val, (int, float)):
            # scalar input
            return self._make_tensor_input(
                node.name,
                elem_type=TensorProto.INT64 if isinstance(val, int) else TensorProto.FLOAT,
                shape=(1,),
                is_dimension=False,
                users=node.users,
            )

        raise RuntimeError(
            f"Unsupported type {type(val)} for placeholder "
            f"{getattr(node, 'target', '?')}{self.builder.get_debug_msg()}."
        )

    def output(self, node):
        """
        Adds an output to the graph.
        """
        output_name = node.name
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.output][{output_name}]")
        declared = node.args
        assert len(declared) == 1, (
            f"declared must have one element: {declared}, output_name={output_name}"
            f"{self.builder.get_debug_msg()}"
        )
        output = declared[0]
        if hasattr(output, "name"):
            output = output.name
            self.builder.make_node(
                "Identity", [output], [output_name], check=False, name=".output"
            )
            outputs = [(output, output_name)]
        else:
            outputs = []
            for i, a in enumerate(output):
                if a is None:
                    a_name = None
                    o = f"{output_name}_{i}"
                    cst = None
                elif isinstance(a, int):
                    # The model seems to return an integer.
                    o = f"{output_name}_INT_{i}"
                    a_name = None
                    cst = self.builder.make_node(
                        "Constant", [], [o], value_int=a, name=".output_INT_{a}"
                    )
                    self.builder.set_type(o, TensorProto.INT64)
                    self.builder.set_shape(o, tuple())
                else:
                    cst = None
                    a_name = a if isinstance(a, str) else a.name
                    if self.builder.get_is_dimension(a_name, n_outputs=len(output)):
                        o = f"{output_name}_dim_{i}"
                    else:
                        o = f"{output_name}_{i}"

                if a_name is None:
                    # the gradient may need unused output
                    if cst is None:
                        o = f"{output_name}_NONE_{i}"
                        self.builder.make_node(
                            "Constant", [], [o], value_float=0.0, name=".output_NONE"
                        )
                        self.builder.set_type(o, TensorProto.FLOAT)
                        self.builder.set_shape(o, tuple())
                    outputs.append((None, o))
                else:
                    self.builder.make_node(
                        "Identity", [a_name], [o], check=False, name=".output"
                    )
                    outputs.append((a_name, o))

        val = node.meta.get("val", None)

        if isinstance(val, tuple):
            if len(val) > 1:
                raise NotImplementedError("Not yet implemented for multiple outputs.")
            val = val[0]

        if val is None:
            for a, o in outputs:
                if a is None:
                    elem_type = self.builder.get_type(o)
                    shape = self.builder.get_shape(o)
                else:
                    elem_type = self.builder.get_type(a)
                    if self.builder.has_shape(a):
                        shape = self.builder.get_shape(a)
                    elif self.builder.has_rank(a):
                        shape = tuple([None] * self.builder.get_rank(a))
                    elif self.builder.as_function:
                        shape = None
                    else:
                        raise RuntimeError(
                            f"val is None for node={node}, "
                            f"output={output}, a={a!r}, o={o!r}, "
                            f"has_type={self.builder.has_type(a)}, "
                            f"has_rank={self.builder.has_rank(a)}, "
                            f"has_shape={self.builder.has_shape(a)}, "
                            f"\nmeta={node.meta}"
                            f"\nnode.__dict__={node.__dict__}"
                            f"{self.builder.get_debug_msg()}"
                        )

                # let's avoid none
                ns = []
                for i, d in enumerate(shape):
                    if d is None:
                        d = f"d_{o}_{i}"
                        self.builder.make_dynamic_object(d, self.torch.SymInt(d))
                    ns.append(d)
                shape = tuple(ns)
                is_dimension = self.builder.get_is_dimension(
                    a or o, elem_type=elem_type, shape=shape, n_outputs=len(outputs)
                )

                self.builder.make_tensor_output(
                    o,
                    elem_type=elem_type,
                    shape=shape,
                    indexed=False,
                    is_dimension=is_dimension,
                )
            return [_[1] for _ in outputs]

        if isinstance(val, self.torch.Tensor):
            n_outputs = len(self.builder.outputs)
            output_name = f"{node.name}_{n_outputs}"
            shape = val.shape
            dtype = _get_type(val.dtype)
            self.builder.make_tensor_output(output_name, dtype, shape)
            return output_name

        raise TypeError(f"Unexpected output type {type(val)}.")

    def _fill_in_default_kwargs(
        self,
        node: "torch.fx.Node",  # noqa: F821
    ) -> Tuple[List[Any], Dict[str, Any]]:
        if hasattr(node.target, "_schema"):
            node_schema = node.target._schema
        else:
            node_schema = None

        complete_args = []
        complete_kwargs = {}

        if inspect.isbuiltin(node.target) or not node_schema:
            complete_args = list(node.args)
            complete_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, self.builder.torch.fx.Node):
                    complete_kwargs[k] = v.name
                elif v is None:
                    complete_kwargs[k] = None
                elif isinstance(
                    v, (int, float, str, self.builder.torch.device, self.builder.torch.dtype)
                ):
                    complete_kwargs[k] = v
                else:
                    raise AssertionError(f"Unexpected type {type(v)} for k={k!r} (v={v!r})")
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
            raise NotImplementedError(f"Unsupported function {node!r} (not implemented).")

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
        sts: Optional[Dict[str, Any]],
        axes: List[int],
        expand_axes: List[int],
        name: str = "_getitem_slice",
    ):
        assert isinstance(axes, list), f"Unexpected type {type(axes)} for axes"
        assert all_int(axes), f"Expected only integer axis but got {axes}"
        assert len(axes) == len(
            index_slice
        ), f"Length mismatch {len(axes)} != {len(index_slice)}"

        # axes
        aaxes = np.array(axes, dtype=np.int64)
        axes_name = self.builder.unique_name(f"{node.name}_axis")
        self.builder.make_initializer(axes_name, aaxes)

        shape_value = None
        if self.builder.has_shape(input_name):
            shape_value = self.builder.get_shape(input_name)

        starts = []
        ends = []
        steps = []
        shape_name = None
        end_name = None
        concat = False
        for axis_, aslice in zip(axes, index_slice):
            axis = axis_
            if isinstance(aslice, int):
                # integer
                starts.append(aslice)
                ends.append(aslice + 1)
                steps.append(1)
                continue

            assert isinstance(aslice, slice), f"Unexpected type {aslice} in {index_slice}"

            starts.append(aslice.start or 0)

            if aslice.stop is None:
                if shape_value is None or not isinstance(shape_value[axis], int):
                    if shape_name is None:
                        shape_name = self.builder.unique_name(f"{node.name}_shape")
                        self.builder.make_node(
                            "Shape", [input_name], [shape_name], name=f"{name}A"
                        )

                    aaxis = np.array([axis], dtype=np.int64)
                    axis_name = self.builder.unique_name(f"{node.name}_axis_{axis}")
                    self.builder.make_initializer(axis_name, aaxis)

                    end_name = self.builder.unique_name(f"{node.name}_end")
                    self.builder.make_node(
                        "GatherElements",
                        [shape_name, axis_name],
                        [end_name],
                        name=f"{name}B",
                        sts=None,
                    )
                    ends.append(end_name)
                    concat = True
                else:
                    ends.append(shape_value[axis])
            else:
                vstop = aslice.stop.name if hasattr(aslice.stop, "name") else aslice.stop
                concat |= isinstance(vstop, str)
                ends.append(vstop)

            steps.append(aslice.step if aslice.step else 1)

        # if concat: one end is coming from a shape
        if concat:
            iends = []
            for i in ends:
                if isinstance(i, str):
                    if self.builder.get_rank(i) == 0:
                        iends.append(
                            self.builder.op.UnsqueezeAnyOpset(
                                i, np.array([0], dtype=np.int64), name=f"{name}C"
                            )
                        )
                    else:
                        assert self.builder.get_rank(i) == 1, (
                            f"Unexpected rank={self.builder.get_rank(i)} for {i!r}"
                            f"{self.builder.get_debug_msg()}"
                        )
                        iends.append(i)
                else:
                    assert isinstance(
                        i, int
                    ), f"Unexpected value for end={i!r}{self.builder.get_debug_msg()}"
                    iends.append(np.array([i], dtype=np.int64))
            if len(iends) > 1:
                conc_ends = self.builder.op.Concat(*iends, axis=0, name=f"{name}D")
            else:
                conc_ends = self.builder.op.Identity(iends[0], name=f"{name}E")
        else:
            assert all_int(ends), (
                f"Unexpected value for ends={ends}: {[type(_) for _ in ends]}"
                f"{self.builder.get_debug_msg()}"
            )
            conc_ends = self.builder.make_initializer("", np.array(ends, dtype=np.int64))

        assert all_int(steps), (
            f"Not implemented for steps={steps} (types are "
            f"{[type(c) for c in steps]}){self.builder.get_debug_msg()}"
        )
        if all_int(starts):
            conc_starts = self.builder.make_initializer(
                self.builder.unique_name(f"{node.name}_start"),
                np.array(starts, dtype=np.int64),
            )
        else:
            istarts = []
            for i in starts:
                si = i.name if hasattr(i, "name") else i
                if isinstance(si, str):
                    if self.builder.get_rank(si) == 0:
                        istarts.append(
                            self.builder.op.UnsqueezeAnyOpset(
                                si, np.array([0], dtype=np.int64), name=f"{name}C"
                            )
                        )
                    else:
                        assert self.builder.get_rank(si) == 1, (
                            f"Unexpected rank={self.builder.get_rank(i)} for {si!r}"
                            f"{self.builder.get_debug_msg()}"
                        )
                        istarts.append(si)
                else:
                    assert isinstance(
                        si, int
                    ), f"Unexpected value for end={si!r}{self.builder.get_debug_msg()}"
                    istarts.append(np.array([si], dtype=np.int64))
            if len(istarts) > 1:
                conc_starts = self.builder.op.Concat(*istarts, axis=0, name=f"{name}SD")
            else:
                conc_starts = self.builder.op.Identity(istarts[0], name=f"{name}SE")

        inputs = [
            input_name,
            conc_starts,
            conc_ends,
            axes_name,
            self.builder.make_initializer(
                self.builder.unique_name(f"{node.name}_step"),
                np.array(steps, dtype=np.int64),
            ),
        ]

        if expand_axes:
            sliced = self.builder.make_node("Slice", inputs, name=f"{name}F")
            res = self.builder.op.UnsqueezeAnyOpset(
                sliced,
                np.array(expand_axes, dtype=np.int64),
                outputs=[node.name],
                name=f"{name}F",
            )
        else:
            res = self.builder.make_node("Slice", inputs, [node.name], name=f"{name}G")
        if not sts:
            dtype = self.builder.get_type(inputs[0])
            self.builder.set_type(node.name, dtype)
            if not concat and self.builder.has_shape(inputs[0]):
                shape = self.builder.get_shape(inputs[0])
                new_shape = self.builder._apply_slice_to_shape(
                    shape, index_slice, axes=axes, expand_axes=expand_axes
                )
                assert not self.builder.has_shape(
                    node.name
                ) or new_shape == self.builder.get_shape(node.name), (
                    f"Shape for node {node.name!r} is already set to "
                    f"{self.builder.get_shape(node.name)} with type "
                    f"{self.builder.get_type(node.name)} (expecting {dtype}) "
                    f"new_shape={new_shape}, shape={shape}, index_slice={index_slice}, "
                    f"axes={axes}, expand_axes={expand_axes}"
                    f"{self.builder.get_debug_msg()}"
                )
                self.builder.set_shape(node.name, new_shape)
            elif expand_axes:
                self.builder.set_rank(
                    node.name, self.builder.get_rank(inputs[0]) + len(expand_axes)
                )
        return res

    def _getitem_int1(
        self,
        node: "torch.fx.Node",  # noqa: F821
        input_name: str,
        indices: List[int],
        sts: Optional[Dict[str, Any]],
        axes: List[int],
        expand_axes: List[int],
        name: str = "_getitem_int1",
    ):
        from ._aten_functions import _aten_tensor_int1

        return _aten_tensor_int1(
            self.builder,
            sts,
            [node.name],
            input_name,
            indices,
            axes=axes,
            expand_axes=expand_axes,
            name=name,
        )

    def getitem(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Called when the brackets ``something[...]`` appears.
        The index may be another variable, an integer, a slice,
        a tuple, a list.
        """
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.getitem]")
        args = node.args
        assert len(args) == 2
        node_output, index = args
        result_name = node_output.name
        val = node.meta.get("val", None)
        sts = None
        if val is not None:
            if isinstance(val, self.torch.Tensor):
                shape = val.shape
                dtype = _get_type(val.dtype)
                self.builder.set_shape(node.name, tuple(shape))
                self.builder.set_type(node.name, dtype)
                sts = {"dtype": val.dtype}
            else:
                raise TypeError(f"Unexpected type in node {node!r}, type(val)={type(val)}.")

        if hasattr(index, "name"):
            # A dynamic index (torch.fx.Node)
            res = self.builder.make_node(
                "Gather", [result_name, index.name], [node.name], name="getitemA"
            )
            if not sts:
                self.builder.set_type(node.name, self.builder.get_type(result_name))
                self.builder.set_rank(
                    node.name,
                    self.builder.get_rank(result_name) + self.builder.get_rank(index.name) - 1,
                )
            return res

        if isinstance(index, int):
            name_index = f"{result_name}#{index}"
            if self.builder.has_name(name_index):
                # The user to get a tensor a tuple of tensors
                return self.builder.make_node(
                    "Identity", [name_index], [node.name], name="getitemB_tuple"
                )
            # The user mean to access the first element of a tensor or a sequence
            if self.builder.is_sequence(result_name):
                # A sequence
                tpos = self.builder.make_initializer("", np.array(index, dtype=np.int64))
                res = self.builder.make_node(
                    "SequenceAt",
                    [result_name, tpos],
                    [node.name],
                    name="getitemB_tuple",
                )
                if not sts:
                    info = self.builder.get_sequence_info(result_name)
                    self.builder.set_type(res, info["dtype"])
                    if info["shape"] is not None:
                        self.builder.set_shape(res, info["shape"][index])
                    else:
                        self.builder.set_rank(res, info["rank"][index])
                return res
            else:
                # A tensor.
                res = self.builder.op.SqueezeAnyOpset(
                    self.builder.op.Gather(
                        result_name,
                        np.array([index], dtype=np.int64),
                        name="getitemB_index",
                    ),
                    np.array([0], dtype=np.int64),
                    name="getitemB_index",
                    outputs=[node.name],
                )
                if not sts:
                    self.builder.set_type(node.name, self.builder.get_type(result_name))
                    if self.builder.has_shape(result_name):
                        self.builder.set_shape(
                            node.name, self.builder.get_shape(result_name)[1:]
                        )
                    else:
                        self.builder.set_rank(
                            node.name, self.builder.get_rank(result_name) - 1
                        )
                return res

        if isinstance(index, slice):
            return self._getitem_slice(
                node,
                node_output.name,
                [index],
                sts=sts,
                axes=[0],
                expand_axes=[],
                name="_getitem_slice1",
            )

        if isinstance(index, self.torch.fx.immutable_collections.immutable_list):
            # something like x[[0, 2]]
            if all_int(index):
                # something like x[[0, 1]]
                axes = [0]
                return self._getitem_int1(
                    node,
                    node_output.name,
                    index,
                    sts=sts,
                    axes=axes,
                    expand_axes=[],
                    name="_getitem_int1a",
                )

        if isinstance(index, tuple):
            if all(x is Ellipsis or x is None or isinstance(x, slice) for x in index):
                # something like x[3:4]
                axes = []
                slices = []
                expand_axes = []
                ellipsis = False
                true_slice = False
                for i, ind in enumerate(index):
                    if ind is Ellipsis:
                        assert not ellipsis, f"Second (...) found in index={index}"
                        ellipsis = True
                        continue
                    if ind is None:
                        assert (
                            not ellipsis
                        ), f"An axis cannot be inserted after (...) in index={index}"
                        expand_axes.append(i)
                        continue
                    axes.append(((i - len(index)) if ellipsis else i) - len(expand_axes))
                    if (
                        not isinstance(ind, slice)
                        or ind.start is not None
                        or ind.stop is not None
                        or ind.step is not None
                    ):
                        true_slice = True
                    slices.append(ind)
                if true_slice:
                    return self._getitem_slice(
                        node,
                        node_output.name,
                        slices,
                        sts=sts,
                        axes=axes,
                        expand_axes=expand_axes,
                        name="_getitem_slice2",
                    )
                # It is just a node unsqueeze.
                res = self.builder.op.UnsqueezeAnyOpset(
                    str(node.args[0]),
                    np.array(expand_axes, dtype=np.int64),
                    name="getitem_unsqueeze",
                    outputs=[node.name],
                )
                return res

        raise RuntimeError(
            f"getitem: unexpected type {type(index)} for index={index}, "
            f"node={node}, args={args}, val={val}"
            f"{self.builder.get_debug_msg()}"
        )

    def _process_arg(self, node, aten_name, i):
        if i is None:
            return None
        if isinstance(i, str):
            return i
        if hasattr(i, "name"):
            return i.name
        if isinstance(i, tuple):
            return tuple(self._process_arg(node, aten_name, t) for t in i)
        if isinstance(i, (float, int, tuple, slice, complex)):
            return i
        if isinstance(i, list):
            new_list = []
            for el in i:
                if hasattr(el, "name"):
                    # torch.fx.Node
                    new_list.append(el.name)
                    continue
                new_list.append(el)
            return new_list
        if i is Ellipsis:
            return i

        if isinstance(i, self.torch.dtype):
            return i
        raise RuntimeError(
            f"Unexpected type (argument {i}) {type(i)} "
            f"for function {aten_name!r} "
            f"in args={node.args}{self.builder.get_debug_msg()}"
        )

    def call_function(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Called for a function.
        """
        fx_args, fx_kwargs = self._fill_in_default_kwargs(node)
        aten_name = self._get_aten_name(node)
        self.builder.add_stat(kind="aten", name=aten_name)
        if aten_name == "getitem":
            return self.getitem(node)
        fct, lookup, lookup_names = None, None, None
        if self.dispatcher is not None:
            fct = self.dispatcher.find_function(aten_name)
            lookup_names = [aten_name]
        if fct is None:
            fct, lookup, lookup_names = find_function(aten_name)
        if self.dispatcher is not None:
            fct = self.dispatcher.fallback(
                aten_name, fct, node.args, node.kwargs, self.builder
            )

        if fct is None:
            raise FunctionNotFoundError(
                f"Unable to interpret function {type(aten_name)}: "
                f"{aten_name!r}, searched for "
                f"{lookup} and attributes {lookup_names}, "
                f"args={node.args}, kwargs={node.kwargs}"
                f"{self.builder.get_debug_msg()}"
            )
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.call_function][{fct.__name__}]")

        args = [self._process_arg(node, aten_name, a) for a in fx_args]
        output_names = self._get_output_names(node)
        can_set = self._can_set_shape_and_type(node)
        n_nodes = len(self.builder.nodes) + len(self.builder.initializers_dict)

        res = fct(self.builder, can_set, output_names, *args, **fx_kwargs)

        n_nodes_after = len(self.builder.nodes) + len(self.builder.initializers_dict)
        if res is None:
            if len(node.users) == 0:
                return
            raise RuntimeError(
                f"Unexpected return res=None, for node={node}, "
                f"output_names={output_names}"
                f"{self.builder.get_debug_msg()}"
            )
        if n_nodes_after == n_nodes:
            raise RuntimeError(
                f"No node or initializer was added ({n_nodes}=={n_nodes_after}) "
                f"for node={node}{self.builder.get_debug_msg()}"
            )

        self._set_shape_and_type(node, res, fct_name=aten_name)
        res = self._check_output_name(node, res, output_names)
        return res

    def call_method(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Called for a method.
        """
        method_name = node.target
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.call_method][{method_name}]")
        assert isinstance(
            node.args, tuple
        ), f"Unexpected type {type(node.args)} for node.args."

        fct = None
        if self.dispatcher is not None:
            fct = self.dispatcher.find_method(f"aten_meth_{method_name}")
        name_fct = f"aten_meth_{method_name}"
        fct = find_method(name_fct)
        if self.dispatcher is not None:
            fct = self.dispatcher.fallback(name_fct, fct, node.args, node.kwargs, self.builder)
        if fct is None:
            raise FunctionNotFoundError(
                f"Unable to interpret method {name_fct!r}, "
                f"args={node.args}, kwargs={node.kwargs}, "
                f"dispatcher={self.dispatcher}"
                f"{self.builder.get_debug_msg()}"
            )

        args = [getattr(node.args[0], "name", node.args[0])]
        for i in node.args[1:]:
            args.append(i.name if hasattr(i, "name") else i)

        kwargs = node.kwargs
        output_names = self._get_output_names(node)
        can_set = self._can_set_shape_and_type(node)

        res = fct(self.builder, can_set, output_names, *args, **kwargs)

        self._set_shape_and_type(node, res, fct_name=method_name)
        res = self._check_output_name(node, res, output_names)
        return res

    def _get_output_names(self, node: "torch.fx.Node") -> List[str]:  # noqa: F821
        val = node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [
                ("" if val[i] is None else f"{node.name}#{i}") for i in range(n_outputs)
            ]
        else:
            assert isinstance(
                node.name, str
            ), f"Unexpected type {type(node.name)} for node.name"
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
                        f"Unexpected output_names {output_names}, "
                        f"res={res!r}, node.name={node.name!r}"
                    )
            elif isinstance(res, list) and len(res) != 1:
                # SplitToSequence rewritten into a Split
                name = output_names[0]
                assert all(s.startswith(name) for s in res), (
                    f"Unexpected output_names={output_names}, "
                    f"res={res}, node.name={node.name}"
                    f"{self.builder.get_debug_msg()}"
                )
                # nothing to do
                res = tuple(res)
            elif res != node.name:
                assert isinstance(res, str), (
                    f"Unexpected res={res}, output_names={output_names}, "
                    f"node.name={node.name}"
                    f"{self.builder.get_debug_msg()}"
                )
                self.builder.make_node(
                    "Identity", [res], [node.name], name="_check_output_name"
                )
                res = node.name
        else:
            raise NotImplementedError(
                f"Unexpected type {type(node.name)} for node.name={node.name!r}."
            )
        return res

    def _can_set_shape_and_type(self, node: "torch.fx.Node") -> bool:  # noqa: F821
        if node.meta.get("val", None) is not None:
            dtype = self._get_node_output_type(node)
            assert dtype is not None, (
                f"dtype is null, but val={node.meta.get('val', None)}"
                f"{self.builder.get_debug_msg()} "
            )
            return {"dtype": dtype}
        return None

    def _get_node_output_type(
        self,
        node: "torch.fx.Node",  # noqa: F821
    ) -> Optional[Union["torch.dtype", Tuple["torch.dtype", ...]]]:  # noqa: F821
        val = node.meta.get("val", None)
        if val is not None:
            if isinstance(val, (tuple, list)):
                # Type list comes from SplitToSequence.
                return tuple((None if v is None else v.dtype) for v in val)
            if isinstance(val, self.torch.SymInt):
                return self.torch.SymInt
            if isinstance(val, self.torch.SymFloat):
                return self.torch.SymFloat
            exa = node.meta.get("example_value", None)
            assert exa is None or val.dtype == exa.dtype, (
                f"dtype inconsistency (val, example_value) "
                f"{val.dtype} != {exa.dtype}{self.builder.get_debug_msg()}"
            )
            assert hasattr(val, "dtype"), (
                f"Unexpected type {type(val)} for val={val}, "
                f"node={node!r}{self.builder.get_debug_msg()}"
            )
            return val.dtype
        return None

    def _set_shape_and_type(
        self,
        node: "torch.fx.Node",  # noqa: F821
        res: Union[str, List[str]],
        fct_name: Optional[str] = None,
    ):
        val = node.meta.get("val", None)
        exa = node.meta.get("example_value", None)
        if val is not None and exa is not None:
            assert val.dtype == exa.dtype, (
                f"dtype inconsistency (val, example_value) "
                f"{val.dtype} != {exa.dtype}{self.builder.get_debug_msg()}"
            )
            assert val.shape == exa.shape, (
                f"shape inconsistency (val, example_value) "
                f"{val.shape} != {exa.shape}{self.builder.get_debug_msg()}"
            )

        last_node = self.builder.last_added_node
        description = []
        if val is not None and fct_name not in {"aten_cond"}:
            # extracting shape and types
            if not isinstance(val, tuple):
                val = (val,)
                res = (res,)
            assert isinstance(
                res, (list, tuple)
            ), f"Unexpected type {type(res)}{self.builder.get_debug_msg()}"
            if len(val) != len(res):
                raise RuntimeError(
                    f"Length mismatch {len(val)} != {len(res)} "
                    f"between {val} and {res}"
                    f"{self.builder.get_debug_msg()}"
                )
            output_sets = set(last_node.output) if last_node is not None else {}

            for i, (v, r) in enumerate(zip(val, res)):
                if isinstance(v, self.torch.Tensor):
                    dtype = _get_type(v.dtype)
                    if i >= 1 and node.target.name() in {
                        "aten::_native_batch_norm_legit.no_stats",
                        "aten::_native_batch_norm_legit_no_training",
                    }:
                        # It seems the type is not very consistant
                        # and the output might not be used.
                        self.builder.set_type(r, dtype, exc=False)
                    else:
                        self.builder.set_type(r, dtype)
                    shape = tuple(v.shape)

                    for t in shape:
                        if isinstance(t, self.builder.torch.SymInt):
                            expr = str(t.node._expr)
                            if expr not in self.builder.dynamic_objects:
                                # A new shape may be given to a result.
                                self.builder.add_dynamic_object(expr, t)

                    if self.builder.is_dynamic_shape(shape):
                        # sets shape coming from the original model
                        # we must not set the existing shape as static,
                        # if it was dynamic before
                        self.builder.set_shape(r, shape, set_if_more_precise=False)
                    elif self.builder.has_rank(r):
                        assert len(shape) == self.builder.get_rank(r), (
                            f"Rank already set for {r!r}, "
                            f"but rank={self.builder.get_rank(r)} "
                            f"differs for shape={shape!r}{self.builder.get_debug_msg()}"
                        )
                    else:
                        self.builder.set_rank(r, len(shape))
                    if r in output_sets:
                        description.append(f"{r}:{dtype}:{shape}".replace(" ", ""))
                elif isinstance(v, self.torch.SymInt):
                    # this is a shape
                    self.builder.set_shape(r, (1,))
                    self.builder.set_type(r, TensorProto.INT64)
                    self.builder.make_dynamic_object(r, v)
                elif isinstance(v, self.torch.SymFloat):
                    # this is a shape
                    self.builder.set_shape(r, (1,))
                    self.builder.set_type(r, TensorProto.FLOAT)
                    self.builder.make_dynamic_object(r, v)
                elif v is None:
                    continue
                elif isinstance(v, list) and len(v) > 0:
                    if len(v) == len(r) and r[0].endswith("#0"):
                        # Operator Split was used instead of SplitToSequence.
                        for r_, v_ in zip(r, v):
                            self.builder.set_type(r_, torch_dtype_to_onnx_dtype(v_.dtype))
                            shape = tuple(v_.shape)
                            if self.builder.is_dynamic_shape(shape):
                                self.builder.set_shape(r_, shape, set_if_more_precise=False)
                            elif self.builder.has_rank(r_):
                                assert len(shape) == self.builder.get_rank(r_), (
                                    f"Rank already set for {r_!r}, "
                                    f"but rank={self.builder.get_rank(r_)} "
                                    f"differs for shape={shape!r}"
                                    f"{self.builder.get_debug_msg()}"
                                )
                            else:
                                self.builder.set_rank(r, len(shape))
                    else:
                        # This is coming from the sequence.
                        dtype = list(set(_.dtype for _ in v))
                        assert len(dtype) == 1, (
                            f"Only sequence of tensors of the same type are allowed "
                            f"but dtype={dtype}{self.builder.get_debug_msg()}"
                        )
                        itype = torch_dtype_to_onnx_dtype(dtype[0])
                        self.builder.set_sequence(r, itype, shapes=(_.shape for _ in v))
                else:
                    raise TypeError(
                        f"Unexpected type in node {node!r}, r={r!r}, "
                        f"type(val)={type(v)}{self.builder.get_debug_msg()}"
                        f"\n----\nval={val}"
                    )
        if exa is not None and not isinstance(exa, tuple):
            if hasattr(exa, "dtype"):
                # a tensor
                description.append(f"~{exa.dtype}:{exa.shape}".replace(" ", ""))
            else:
                # a SymInt
                description.append(f"~SumInt:{exa!r}".replace(" ", ""))
        if last_node is not None and description:
            last_node.doc_string += "\n".join(description)

    def _interpret_sub_module(self, sub_module, args, kwargs, source_node=None):
        from .onnx_export import _make_builder_interpreter

        if hasattr(sub_module, "graph") and isinstance(sub_module, self.torch.fx.GraphModule):
            gm = sub_module
        else:
            # https://pytorch.org/docs/stable/fx.html
            tracer_class = self.torch.fx.Tracer
            graph = tracer_class().trace(sub_module)
            gm = self.torch.fx.GraphModule(sub_module, graph)

        graph_module, builder, interpreter = _make_builder_interpreter(
            gm,
            args=None if args is None else tuple(args),
            kwargs=None if kwargs is None else kwargs,
            as_function=True,
            target_opset=self.builder.opsets,
            optimization_options=self.builder.optimization_options,
            verbose=self.builder.verbose,
            dispatcher=self.dispatcher,
            raise_list=self.builder.raise_list,
            dynamic_shapes=self.builder.dynamic_shapes,
            export_options=self.export_options,
        )
        builder.process(graph_module, interpreter)
        assert builder.outputs, f"No output detected for node={source_node}, graph={gm}"

        fx_args, fx_kwargs = self._fill_in_default_kwargs(source_node)
        args = [getattr(i, "name", i) for i in fx_args]
        kwargs = [getattr(i, "name", i) for i in fx_kwargs]

        val = source_node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [f"{source_node.name}#{i}" for i in range(n_outputs)]
        else:
            output_names = [source_node.name]
            if val is None:
                val = source_node.meta.get("example_value", None)
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
                self.builder.set_shapes_types(
                    source_node.name, "call_module", (val[i].dtype, val[i].shape)
                )

        return builder, args, kwargs, output_names

    def call_module(self, node: "torch.fx.Node"):  # noqa: F821
        """
        Called for a module.
        """

        def raise_msg():
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

        sub_module = node.graph.owning_module.get_submodule(node.target)

        assert isinstance(
            sub_module, self.torch.nn.Module
        ), f"Not implemented for type {type(sub_module)}.\n{raise_msg()}"

        named_args = node.args
        args = []
        for a in named_args:
            val = a.meta.get("example_value", None)
            args.append(val)

        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.call_module] class [{type(sub_module)}]")

        builder, args, kwargs, output_names = self._interpret_sub_module(
            sub_module, args, source_node=node
        )
        assert kwargs is None or len(kwargs) == 0, (
            f"args={string_type(args)}, kwargs={string_type(kwargs)} "
            f"is not implemented yet{self.builder.get_debug_msg()}"
        )

        self.builder.make_nodes(
            builder, args, output_names, prefix=f"_sub_{sub_module.__class__.__name__}_"
        )
        return output_names
