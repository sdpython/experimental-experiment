import inspect
import math
import operator
import os
import pprint
import re
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from onnx import TensorProto
from onnx_diagnostic.helpers.helper import flatten_object
from ..helpers import (
    string_type,
    make_hash,
    torch_dtype_to_onnx_dtype,
    onnx_dtype_to_torch_dtype,
    onnx_dtype_name,
)
from ..xbuilder import GraphBuilder, FunctionOptions, VirtualTensor
from ..xshape._shape_helper import all_int, DYNAMIC_SHAPE
from ..xbuilder.model_container import _get_type
from ..xshape.rename_expressions import parse_expression_tokens
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
    :param optimize_submodules: optimizes submodules after they are built
    :param submodule_naming: a function which returns a submodule name in the onnx graph
    :param parameter_naming: a function which returns a parameter name in the onnx graph
    :param module_name: module name (makes it easier to retrieve the parameter names)
    :param exe_path: gives information on how the :class:`torch.fx.Graph` was obtained
    """

    def _hash(self) -> str:
        return make_hash(self)

    def __init__(
        self,
        graph_builder: GraphBuilder,
        retriever: Callable,
        dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
        example_inputs: Optional[Tuple["torch.Tensor", ...]] = None,  # noqa: F821
        export_options: Optional[ExportOptions] = None,
        optimize_submodules: bool = False,
        function_options: Optional[FunctionOptions] = None,
        submodule_naming: Optional[Callable] = None,
        parameter_naming: Optional[Callable] = None,
        module_name: Optional[str] = None,
        default_values: Optional[Dict[str, Any]] = None,
        exe_path: str = "",
    ):
        import torch
        from ..xbuilder import FunctionOptions

        self.torch = torch
        self.builder = graph_builder
        self.retriever = retriever
        self.dispatcher = dispatcher
        self.export_options = export_options
        self.optimize_submodules = optimize_submodules
        self.function_options = function_options or FunctionOptions(
            name="*",
            domain="*",
            export_as_function=True,
            external_threshold=256,
            move_initializer_to_constant=True,
            return_initializer=True,
            merge_allowed=True,
            rename_allowed=True,
        )
        self.example_values_ = {}
        self.exe_path = exe_path
        assert example_inputs is None or isinstance(
            example_inputs, tuple
        ), f"Unexpected type for example_inputs {type(example_inputs)}"
        assert example_inputs is None or all(
            (
                t is None
                or isinstance(
                    t,
                    (
                        torch.SymInt,
                        torch.SymFloat,
                        torch.Tensor,
                        list,
                        int,
                        float,
                        VirtualTensor,
                    ),
                )
                or t.__class__.__name__
                in {"DynamicCache", "MambaCache", "EncoderDecoderCache", "BaseModelOutput"}
            )
            for t in example_inputs
        ), (
            f"Unexpected type for one input in example_inputs "
            f"{string_type(example_inputs, with_shape=True, limit=100)}"
        )
        self.example_inputs_ = example_inputs
        self.flat_example_inputs_ = self.flatten_inputs(example_inputs)
        self.current_input_ = 0
        self.preserved_modules = set()
        self.parent_interpreter = None
        self.parameter_naming = parameter_naming
        self.submodule_naming = submodule_naming
        self.module_name = module_name
        self.default_values = default_values or {}
        self._debug_aten_as_function = int(os.environ.get("ATENDEBUG", "0"))

    def register_named_modules(
        self,
        parent_interpreter: Optional["DynamoInterpreter"],
        preserved_modules: Optional[Set[type["torch.nn.Module"]]],  # noqa: F821
        named_modules: Dict[str, "torch.nn.Module"],  # noqa: F821
    ):
        """
        Registers a list of modules to preserve as local function
        in the onnx model. If empty, the graph is almost inlined.
        The module to convert to onnx should the output of method
        :func:`torch.export.unflatten.unflatten`.
        """
        assert parent_interpreter is None or isinstance(
            parent_interpreter, DynamoInterpreter
        ), f"Unexpected type {type(parent_interpreter)} for the interpreter"
        if self.builder.verbose > 4 and preserved_modules:
            print(
                f"[DynamoInterpreter-{self._hash()}.register] "
                f"{sorted(c.__name__ for c in preserved_modules)}"
            )
        self.named_modules = named_modules
        self.preserved_modules = preserved_modules or parent_interpreter.preserved_modules
        if parent_interpreter is not None:
            self.submodule_naming = parent_interpreter.submodule_naming
            self.parameter_naming = parent_interpreter.parameter_naming

    def flatten_inputs(self, x: Any) -> List["torch.Tensor"]:  # noqa: F821
        """Flatten inputs."""
        if x is None:
            return x
        if isinstance(x, (list, tuple)):
            res = []
            for i in x:
                if i is None or isinstance(
                    i,
                    (
                        self.torch.Tensor,
                        self.torch.SymInt,
                        self.torch.SymFloat,
                        int,
                        float,
                        VirtualTensor,
                    ),
                ):
                    res.append(i)
                else:
                    res.extend(self.flatten_inputs(i))
            return tuple(res) if isinstance(x, tuple) else res
        return flatten_object(x, drop_keys=True)

    def start_graph(self, graph: "torch.fx.Graph"):  # noqa: F821
        assert (
            not hasattr(self, "graph_begin_processed") or not self.graph_begin_processed
        ), "A graph is already being processed."
        self.graph_begin_processed = graph

    def end_graph(self, graph: "torch.fx.Graph"):  # noqa: F821
        assert hasattr(self, "graph_begin_processed") and id(self.graph_begin_processed) == id(
            graph
        ), "A graph was not processed or it is not the same greaph."
        self.graph_begin_processed = None

    def run_node(
        self,
        node: "torch.fx.Node",  # noqa: F821
        source_lines: Optional[Dict[str, Tuple[str, Tuple[int, int]]]] = None,
    ):
        """Runs a node: call the approrpiate method based on the node type."""
        assert (
            hasattr(self, "graph_begin_processed") and self.graph_begin_processed
        ), "start_graph was not called before this method."

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
                (
                    f"{torch_dtype_to_onnx_dtype(v.dtype)}'"
                    f"{tuple(self.builder._torch_sym_int_to_str(_) for _ in v.shape)}"
                )
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
        last_added = len(self.builder.nodes)

        if node.op == "placeholder":
            res = self.placeholder(node)
        elif node.op == "call_function":
            res = self.call_function(node)
        elif node.op == "output":
            res = self.output(node)
        elif node.op == "call_module":
            self.builder._check_constants(f"before-{node.op}")
            res = self.call_module(node)
            self.builder._check_constants(f"after-{node.op}")
        elif node.op == "get_attr":
            res = self.get_attr(node)
        elif node.op == "call_method":
            res = self.call_method(node)
        else:
            raise ValueError(f"Unable to process node kind {node.op!r} ({node}).")

        if (
            node.op in {"call_module", "call_function", "call_method", "get_attr"}
            and node.meta
            and ("stack_trace" in node.meta or "nn_module_stack" in node.meta)
        ):
            self._set_submodule_name_in_model_as_metadata(
                source_lines,
                node.meta.get("stack_trace", None),
                last_added,
                nn_module_stack=node.meta.get("nn_module_stack", None),
            )

        # Checks consistency of shapes and types
        name = node.name
        if val and len(val) == 3:
            exp_dtype, exp_shape = val[1:]
            if isinstance(exp_dtype, int):
                exp_dtype = onnx_dtype_to_torch_dtype(exp_dtype)
            if self.builder.has_type(name):
                itype = self.builder.get_type(name)
                ttype = onnx_dtype_to_torch_dtype(itype)
                aten_name = self._get_aten_name(node) if node.op == "call_function" else "-"
                assert ttype == exp_dtype, (
                    f"Type mismatch for {name!r}, node.op={node.op!r}, "
                    f"aten_name={aten_name!r}, "
                    f"onnx {ttype} != expected torch "
                    f"{exp_dtype}{self.builder.get_debug_msg()}"
                )
            if len(node.users) > 0 and self.builder.has_shape(name):
                # We should not raise an exception if the result is not used.
                shape = self.builder.get_shape(name)
                self.builder._check_two_shapes_are_compatible(
                    shape,  # new_shape (or shape infered with onnx node)
                    tuple(exp_shape),  # old_shape
                    name=name,
                    register_int=False,
                )
                old_shape = self.builder.verify_dynamic_shape(exp_shape, add=False)
                self.builder._check_two_shapes_are_compatible(
                    shape,  # new_shape (or shape infered with onnx node)
                    old_shape,  # old_shape
                    name=name,
                    register_int=False,
                )
        return res

    def _set_submodule_name_in_model_as_metadata(
        self,
        source_lines: Optional[Dict[str, Tuple[str, Tuple[int, int]]]],
        stack_trace: Optional[str],
        start_node: int,
        nn_module_stack: Optional[Any],
    ):
        """Adds information about where in the model the created node come from."""
        attr_names = []
        module_names = []
        if nn_module_stack:
            for name, nn_module in nn_module_stack.values():
                nn_module_name = None
                if isinstance(nn_module, str):
                    nn_module_name = nn_module
                else:
                    nn_module_name = nn_module.__module__
                    if nn_module_name is None:
                        nn_module_name = nn_module.__name__
                    else:
                        nn_module_name += f".{nn_module.__name__}"
                if nn_module_name not in module_names:
                    module_names.append(nn_module_name)
                if name:
                    attr_names.append(name)

        if module_names:
            for node in self.builder.nodes[start_node:]:
                if attr_names:
                    p = node.metadata_props.add()
                    p.key = "scope"
                    p.value = ".".join(attr_names)
                for i, name in enumerate(module_names):
                    p = node.metadata_props.add()
                    p.key = f"module[{i}]"
                    p.value = name

        if not stack_trace or not source_lines:
            return

        reg = re.compile('File "([^"]+?)", line (\\d+)')
        files = reg.findall(stack_trace)
        if not files:
            return

        names = []
        for filename, line_number in files:
            filename = filename.replace("\\", "/")
            if filename not in source_lines:
                continue
            line = int(line_number)
            for name, interval in source_lines[filename]:
                if line >= interval[0] and line <= interval[1]:
                    names.append(name)
                    break

        if names:
            for node in self.builder.nodes[start_node:]:
                for i, name in enumerate(names):
                    p = node.metadata_props.add()
                    p.key = f"source[{i}]"
                    p.value = name

    def get_attr(self, node: "torch.fx.Node"):  # noqa: F821
        """Retrieves an attribute."""
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.get_attr][{node.name}]")
        try:
            init = getattr(node.graph.owning_module, node.target)
        except AttributeError as e:
            # Maybe it is a parameter:
            init = None
            for name, p in node.graph.owning_module.named_parameters():
                if name == node.target:
                    init = p
            if init is None:
                for name, p in node.graph.owning_module.named_buffers():
                    if name == node.target:
                        init = p
            if init is None:
                raise AttributeError(
                    f"Unable to find attribute {node.target!r} (node.name={node.name!r}) in "
                    f"type(owning_module)={type(node.graph.owning_module)}, "
                    f"\nmodules="
                    f"{sorted([_[0] for _ in node.graph.owning_module.named_modules()])}"
                    f"\nparameters="
                    f"{sorted([_[0] for _ in node.graph.owning_module.named_parameters()])}"
                    f"\nbuffers="
                    f"{sorted([_[0] for _ in node.graph.owning_module.named_buffers()])}"
                    f"\nnode.__dict__={node.__dict__}{self.builder.get_debug_msg()}"
                ) from e

        if isinstance(init, self.torch.fx.GraphModule) or callable(init):
            # This function is meant to be used later.
            if "." in self.builder.local_domain:
                root, n = self.builder.local_domain.split(".")
                n = int(n) + 1
            else:
                root, n = self.builder.local_domain, 0

            if not isinstance(init, self.torch.fx.GraphModule):
                trace_init_cls = self.make_nn_module_with_callable(init)
                trace_init = trace_init_cls()
            else:
                trace_init = init

            builder, _args, _kwargs, _output_names = self._interpret_sub_module(
                trace_init, None, None, source_node=node, local_domain=f"{root}.{n}"
            )
            self.builder.make_local_function(
                builder,
                function_options=FunctionOptions(
                    name=node.name,
                    domain=self.builder.local_domain,
                    export_as_function=True,
                    return_initializer=True,
                    move_initializer_to_constant=self.function_options.move_initializer_to_constant,
                    external_threshold=self.function_options.external_threshold,
                    merge_allowed=self.function_options.merge_allowed,
                    rename_allowed=self.function_options.rename_allowed,
                ),
                optimize=self.optimize_submodules,
            )
            return None

        if isinstance(init, self.builder.torch.utils._pytree.TreeSpec):

            def print_treespec(rows, spec, indent=0):
                prefix = "  " * indent
                rows.append(f"{prefix}Node(type={spec.type}, leaves={spec.num_leaves})")
                for child in getattr(spec, "children_specs", []):
                    print_treespec(rows, child, indent + 1)

            rows = []
            print_treespec(rows, init)
            msg = "\n".join(rows)
            raise NotImplementedError(
                f"Unable to handle type {type(init)} for node.name={node.name!r}"
                f"\n{msg}\n--{self.builder.get_debug_msg()}"
            )

        parameter_name = (
            self.parameter_naming(node.name, init, node=node, prefix=self.module_name)
            if isinstance(init, self.builder.torch.nn.Parameter)
            else None
        )

        assert hasattr(init, "shape"), (
            f"Unexpected type {type(init)} for init={init}, node.name="
            f"{node.name!r}{self.builder.get_debug_msg()}"
        )

        self.builder.make_initializer(
            node.name,
            init,
            parameter_name=parameter_name,
            source=(
                f"DynamoInterpret.get_attr.1/P({parameter_name})"
                if parameter_name
                else "DynamoInterpret.get_attr.0"
            ),
            allow_empty=0 in init.shape,
        )
        return node.name

    def make_nn_module_with_callable(self, f: Callable) -> "torch.nn.Module":  # noqa: F821
        """Wraps a function into a nn Module to export it."""
        sig = inspect.signature(f)
        if len(sig.parameters) == 3:

            class LocalFunction3(self.torch.nn.Module):
                def forward(self, x, y, z):
                    return f(x, y, z)

            return LocalFunction3
        raise NotImplementedError(
            f"make_nn_module_with_callable not implemented for "
            f"{len(sig.parameters)} parameters{self.builder.get_debug_msg()}"
        )

    def _make_tensor_check(self, name: str, fake_tensor: bool, users: Any):
        if (
            not fake_tensor
            and self.example_inputs_ is not None
            and not self.builder.was_inputs_renamed
        ):
            assert len(self.builder.input_names) < len(self.flat_example_inputs_) or not users, (
                f"Too many inputs already ({len(self.builder.input_names)}), "
                f"self.current_input_={self.current_input_}, "
                f"unexpected {name!r}, users={users}, "
                f"after {self.builder.input_names}"
                f"{self.builder.get_debug_msg()}"
            )
            if not self.builder.as_function and (
                self.current_input_ < len(self.flat_example_inputs_)
                and self.flat_example_inputs_[self.current_input_] is None
            ):
                # We skip it.
                assert not users, (
                    f"Input {name!r} (index {self.current_input_}"
                    f"/{len(self.flat_example_inputs_)}) "
                    f"is None but it is used by {users}, "
                    f"as_function={self.builder.as_function}. "
                    f"Existing inputs {self.builder.input_names}. Example inputs: "
                    f"{['-' if t is None else t.shape for t in self.flat_example_inputs_]}"
                    f"{self.builder.get_debug_msg()}"
                )
                self.current_input_ += 1
                return ""

            # second check
            assert (
                self.builder.as_function
                or len(self.builder.input_names)
                < len(tuple(t for t in self.flat_example_inputs_ if t is not None))
                or not users
            ), (
                f"Too many inputs already ({len(self.builder.input_names)}), "
                f"unexpected {name!r}, users={users}, "
                f"after {self.builder.input_names}"
                f"{self.builder.get_debug_msg()}"
            )
        return None

    def _make_tensor_input(
        self,
        name: str,
        elem_type: Any,
        shape: DYNAMIC_SHAPE,
        is_dimension: bool,
        users: Iterable[str],
        fake_tensor: bool = False,
        default_initializer: Optional[Any] = None,
        device: Optional[int] = None,
    ) -> str:
        ret = self._make_tensor_check(name, fake_tensor, users)
        if ret is not None:
            return ret

        shape = self.builder.get_input_dynamic_shape(name, self.current_input_, shape)
        self.current_input_ += 1
        return self.builder.make_tensor_input(
            name,
            elem_type,
            shape,
            is_dimension=is_dimension,
            default_initializer=default_initializer,
            device=device,
            marker="DynamoInterpreter._make_tensor_input",
        )

    def _make_list_input(
        self,
        name: str,
        example_value: List["torch.Tensor"],  # noqa: F821
        users: Iterable[str],
        fake_tensor: bool = False,
    ) -> str:
        ret = self._make_tensor_check(name, fake_tensor, users)
        if ret is not None:
            return ret

        assert all(isinstance(t, self.torch.Tensor) for t in example_value), (
            f"Input {name!r}, unexpected type in example_value: "
            f"{string_type(example_value)}{self.get_debug_msg()}"
        )
        assert len(set(t.dtype for t in example_value)) == 1, (
            f"Input {name!r}, multiple element type in example_value "
            f"{[t.dtype for t in example_value]}{self.get_debug_msg()}"
        )

        shape = self.builder.get_input_dynamic_shape(
            name, self.current_input_, example_shape=None, example_value=example_value
        )
        assert isinstance(shape, list) and len(shape) == 1, (
            f"For a sequence, shapes should be specified as a list of 1 element, "
            f"shape={string_type(shape)}{self.builder.get_debug_msg()}"
        )
        elem_type = _get_type(example_value[0].dtype)
        self.current_input_ += 1
        return self.builder.make_tensor_sequence_input(
            name,
            elem_type,
            shape[0],
            marker="DynamoInterpreter._make_list_input",
        )

    def placeholder(self, node: "torch.fx.Node"):  # noqa: F821
        """
        placeholder for an input. The interpreter adds an Identity node
        between the input names he wants and the name it has in the graph module.
        """
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.placeholder][{node.name}]")
        val = node.meta.get("val", None)
        _msg = lambda _=self: _.builder.get_debug_msg()  # noqa: E731

        if self.builder.verbose > 2:
            print(
                f"[DynamoInterpreter-{self._hash()}.placeholder]"
                f"[{node.name}] val={string_type(val)}"
            )
        if val is None:
            example_value = node.meta.get("example_value", None)
            if self.builder.verbose > 2:
                print(
                    f"[DynamoInterpreter-{self._hash()}.placeholder]"
                    f"[{node.name}] example_value={string_type(val)}"
                )
            # index_input may be wrong because torch.export.export may flatten the inputs.
            # gathering the default value may not be optimal here.
            if example_value is None and node.name in self.default_values:
                example_value = self.default_values[node.name]

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
                    device=-1,  # cpu
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
                    device=-1,  # cpu
                )
            if isinstance(example_value, (self.torch.Tensor, VirtualTensor)):
                return self._make_tensor_input(
                    node.name,
                    elem_type=example_value.dtype,
                    shape=example_value.shape,
                    is_dimension=False,
                    users=node.users,
                    device=example_value.get_device(),
                )
            if isinstance(example_value, list) and all(
                isinstance(t, self.torch.Tensor) for t in example_value
            ):
                return self._make_list_input(node.name, example_value, users=node.users)

            if example_value.__class__.__name__ == "DynamicCache":
                import transformers

                assert isinstance(example_value, transformers.cache_utils.DynamicCache), (
                    f"Unexpected type {type(example_value)} for an input"
                    f"{self.builder.get_debug_msg()}"
                )
                if not example_value.key_cache:
                    # The cache is empty. We create a dummy input with a default value
                    return self._make_tensor_input(
                        node.name,
                        elem_type=np.float32,
                        shape=(1,),
                        is_dimension=False,
                        users=None,
                        default_initializer=np.array([0], dtype=np.float32),
                    )

            raise NotImplementedError(
                f"Unable to create an input {node.name!r} "
                f"with type {string_type(example_value)}"
                f"{self.builder.get_debug_msg()}"
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
                        device=val.get_device(),
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
                    value = self.retriever(node.target, val, debug={"node": node}, exc=False)
                    if value is None:
                        # This is probably one input then.
                        return self._make_tensor_input(
                            node.target,
                            elem_type=val.dtype,
                            shape=val.shape,
                            is_dimension=False,
                            users=node.users,
                        )

            if value is None or isinstance(value, self.torch._subclasses.fake_tensor.FakeTensor):
                if ".FakeTensor" in str(type(val)):
                    dtype = val.dtype
                    shape = val.shape
                    return self._make_tensor_input(
                        node.name,
                        dtype,
                        shape,
                        False,
                        users=node.users,
                        fake_tensor=True,
                        device=val.get_device(),
                    )
                raise RuntimeError(f"value is None, unable to retrieve target {node.target!r}")
            parameter_name = (
                self.parameter_naming(node.name, value, node=node, msg=_msg)
                if isinstance(value, self.builder.torch.nn.Parameter)
                else None
            )
            return self.builder.make_initializer(
                node.name,
                value,
                parameter_name=parameter_name,
                source=(
                    f"DynamoInterpret.placeholder.1/P({parameter_name})"
                    if parameter_name
                    else "DynamoInterpret.placeholder.0"
                ),
                allow_empty=True,
            )

        if isinstance(val, (self.torch.SymInt, self.torch.SymFloat)):
            return self.builder.make_dynamic_object(node.name, val, shape_as_input=True)

        if isinstance(val, (int, float)):
            # scalar input
            return self._make_tensor_input(
                node.name,
                elem_type=(
                    TensorProto.BOOL
                    if isinstance(val, bool)
                    else (TensorProto.INT64 if isinstance(val, int) else TensorProto.FLOAT)
                ),
                shape=tuple(),  # scalar should have no dimension
                is_dimension=False,
                users=node.users,
                device=-1,  # cpu
            )

        if isinstance(val, VirtualTensor):
            return self._make_tensor_input(
                node.name,
                elem_type=val.dtype,
                shape=val.shape,
                is_dimension=False,
                users=node.users,
                device=val.get_device(),
            )

        raise RuntimeError(
            f"Unsupported type {type(val)} for placeholder "
            f"{getattr(node, 'target', '?')}{self.builder.get_debug_msg()}."
        )

    def _make_name(
        self,
        node,
        prefix: str,
        index: int = -1,
        is_int: bool = False,
        is_dim: bool = False,
        is_none: bool = False,
    ) -> str:
        "Tries to improve the output node."
        if node is None or not hasattr(node, "args"):
            return prefix
        anode = node if index < 0 else node.args[0][index]
        if not hasattr(anode, "args"):
            return prefix
        if anode.args and isinstance(
            anode.args[0], self.torch.fx.immutable_collections.immutable_list
        ):
            if (
                isinstance(anode.args[0][0], self.torch.fx.node.Node)
                and hasattr(anode.args[0][0], "name")
                and anode.args[0][0].name.startswith("past_key_values")
            ):
                return f"present{anode.args[0][0].target[4:]}"
        return prefix

    def output(self, node):
        """Adds an output to the graph."""
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
            output = self._make_name(node, output.name, index=-1)
            self.builder.make_node(
                "Identity", [output], [output_name], check=False, name=".output"
            )
            outputs = [(output, output_name)]
        else:
            outputs = []
            for i, a in enumerate(output):
                if a is None:
                    a_name = None
                    o = self._make_name(node, f"{output_name}_{i}", index=i)
                    cst = None
                elif isinstance(a, int):
                    # The model seems to return an integer.
                    o = self._make_name(node, f"{output_name}_INT_{i}", is_int=True, index=i)
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
                        o = self._make_name(node, f"{output_name}_dim_{i}", is_dim=True, index=i)
                    else:
                        o = self._make_name(node, f"{output_name}_{i}", index=i)

                if a_name is None:
                    # the gradient may need unused output
                    if cst is None:
                        o = self._make_name(
                            node, f"{output_name}_NONE_{i}", is_none=True, index=i
                        )
                        self.builder.make_node(
                            "Constant", [], [o], value_float=0.0, name=".output_NONE"
                        )
                        self.builder.set_type(o, TensorProto.FLOAT)
                        self.builder.set_shape(o, tuple())
                    outputs.append((None, o))
                else:
                    self.builder.make_node("Identity", [a_name], [o], check=False, name=".output")
                    outputs.append((a_name, o))

        val = node.meta.get("val", None)

        if val is None:
            example_value = node.meta.get("example_value", None)
            if example_value is not None:
                val = example_value

        if isinstance(val, tuple):
            assert len(val) == 1, (
                f"output not yet implemented for multiple outputs, node={node}"
                f"{self.builder.get_debug_msg()}"
            )
            val = val[0]

        if val is None:
            for a, o in outputs:
                if a is None:
                    assert not self.builder.is_sequence(o), (
                        f"Output sequences are not implemented but {o!r} is one"
                        f"{self.builder.get_debug_msg()}"
                    )
                    elem_type = self.builder.get_type(o)
                    shape = self.builder.get_shape(o)
                else:
                    assert not self.builder.is_sequence(a), (
                        f"Output sequences are not implemented but {a!r} is one"
                        f"{self.builder.get_debug_msg()}"
                    )
                    elem_type = self.builder.get_type(a) if self.builder.has_type(a) else 0
                    if self.builder.has_shape(a):
                        shape = self.builder.get_shape(a)
                    elif self.builder.has_rank(a):
                        shape = tuple([None] * self.builder.get_rank(a))
                    elif self.builder.as_function:
                        shape = None
                    else:
                        shape = None
                        if a in self.builder._known_torch_value:
                            stored = self.builder._known_torch_value[a]
                            example = stored[1][0]
                            if example and len(example) > 2:
                                shape = example[2]
                        if shape is None and not self.export_options.allow_untyped_output:
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
                if shape is not None:
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
                else:
                    is_dimension = False

                self.builder.make_tensor_output(
                    o,
                    elem_type=elem_type,
                    shape=shape,
                    indexed=False,
                    is_dimension=is_dimension,
                    allow_untyped_output=self.export_options.allow_untyped_output,
                    doc_string=f"#A:{a}-{o}",
                )
            return [_[1] for _ in outputs]

        if isinstance(val, self.torch.Tensor):
            n_outputs = len(self.builder.outputs)
            output_name = self._make_name(node, f"{node.name}_{n_outputs}", index=-1)
            shape = val.shape
            dtype = _get_type(val.dtype)
            self.builder.make_tensor_output(
                output_name, dtype, shape, doc_string=f"#B:{node.name}#{n_outputs}"
            )
            return output_name

        raise TypeError(f"Unexpected output type {type(val)}.")

    def _fill_in_default_kwargs(
        self,
        node: "torch.fx.Node",  # noqa: F821
    ) -> Tuple[List[Any], Dict[str, Any]]:
        node_schema = node.target._schema if hasattr(node.target, "_schema") else None
        complete_args = []
        complete_kwargs = {}

        if inspect.isbuiltin(node.target) or not node_schema:
            complete_args = list(node.args)
            complete_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, self.torch.fx.Node):
                    complete_kwargs[k] = v.name
                elif v is None:
                    complete_kwargs[k] = None
                elif isinstance(v, (int, float, str, self.torch.device, self.torch.dtype)):
                    complete_kwargs[k] = v
                elif isinstance(v, self.torch.fx.immutable_collections.immutable_list) and all(
                    isinstance(el, self.torch.fx.Node) for el in v
                ):
                    complete_kwargs[k] = [t.name for t in v]
                else:
                    raise AssertionError(
                        f"Unexpected type {type(v)} for k={k!r} (v={v!r})"
                        f"{self.builder.get_debug_msg()}"
                    )
        else:
            for i, expected_arg in enumerate(node_schema.arguments):
                if i < len(node.args):
                    complete_args.append(node.args[i])
                elif expected_arg.name in node.kwargs:
                    v = node.kwargs[expected_arg.name]
                    complete_kwargs[expected_arg.name] = (
                        v.name if isinstance(v, self.torch.fx.Node) else v
                    )
                else:
                    # Get default from schema.
                    complete_kwargs[expected_arg.name] = expected_arg.default_value

        return complete_args, complete_kwargs

    def _get_aten_name(self, node: "torch.fx.Node") -> str:  # noqa: F821
        if node.target in {operator.getitem, operator.or_, operator.and_}:
            if node.target == operator.getitem:
                return "getitem"
            if node.target == operator.or_:
                return "aten_bitwise_or"
            if node.target == operator.and_:
                return "aten_bitwise_and"
        if isinstance(node.target, self.torch._ops.OpOverloadPacket):
            if node.target != self.torch.ops.aten.sym_size:
                raise RuntimeError(f"Unsupported function {node!r}.")
            raise NotImplementedError(f"Unsupported function {node!r} (not implemented).")

        if isinstance(node.target, types.BuiltinFunctionType):
            if node.target is math.ceil:
                # We need to distinguish between match.ceil and torch.ceil.
                # The output type is different.
                return "math_ceil"
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
        assert len(axes) == len(index_slice), f"Length mismatch {len(axes)} != {len(index_slice)}"

        # axes
        aaxes = np.array(axes, dtype=np.int64)
        axes_name = self.builder.unique_name(f"{node.name}_axis")
        self.builder.make_initializer(
            axes_name, aaxes, source="DynamoInterpreter._getitem_slice.axis.1"
        )

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

            assert isinstance(
                aslice, (slice, int, self.torch.fx.Node)
            ), f"Unexpected type {type(aslice)} ({aslice}) in {index_slice}"
            assert isinstance(aslice, slice), (
                f"One index is given as an integer {aslice!r} but this requires "
                f"to append a node 'Squeeze' after this one and this is not yet "
                f"implemented. You can replace the integer by `i:i+1`"
                f"{self.builder.get_debug_msg()}"
            )

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
                    self.builder.make_initializer(
                        axis_name, aaxis, source="DynamoInterpreter._getitem_slice.axis.2"
                    )

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
            conc_ends = self.builder.make_initializer(
                "", np.array(ends, dtype=np.int64), source="DynamoInterpreter._getitem_slice.1"
            )

        assert all_int(steps), (
            f"Not implemented for steps={steps} (types are "
            f"{[type(c) for c in steps]}){self.builder.get_debug_msg()}"
        )
        if all_int(starts):
            conc_starts = self.builder.make_initializer(
                self.builder.unique_name(f"{node.name}_start"),
                np.array(starts, dtype=np.int64),
                source="DynamoInterpreter._getitem_slice.2",
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
                source="DynamoInterpreter._getitem_slice.3",
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
                # the graph could be new if a function produces a results
                # depending on the result values
                t_shape = tuple(shape)
                self._verify_new_shape(shape, node)
                # Let's set the shape if not null shape
                if len(t_shape) > 1 and not any(i == 0 for i in t_shape if isinstance(i, int)):
                    self.builder.set_shape(
                        node.name, t_shape, allow_zero=all_int(t_shape) and t_shape == (0,)
                    )
                else:
                    self.builder.set_rank(node.name, len(t_shape))
                self.builder.set_type(node.name, dtype)
                self.builder.set_device(node.name, val.get_device())
                sts = {"dtype": val.dtype}
            elif isinstance(val, self.torch.SymInt):
                self.builder.set_shape(node.name, tuple())
                self.builder.set_type(node.name, TensorProto.INT64)
                self.builder.set_device(node.name, -1)
                sts = {"dtype": self.torch.int64}
            else:
                raise TypeError(
                    f"Unexpected type {type(val)} in node {node!r}"
                    f"\n{self.builder.pretty_text(add_fx_graph=True)}"
                )

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
                tpos = self.builder.make_initializer(
                    "", np.array(index, dtype=np.int64), source="DynamoInterpreter.getitem.1"
                )
                res = self.builder.make_node(
                    "SequenceAt",
                    [result_name, tpos],
                    [node.name],
                    name="getitemB_tuple",
                )
                if not sts:
                    info = self.builder.get_sequence(result_name)
                    dtype = info["dtype"]
                    if isinstance(dtype, tuple):
                        dtype = dtype[index]
                    self.builder.set_type(res, dtype)
                    if info["shapes"] is not None:
                        self.builder.set_shape(
                            res, info["shapes"][min(index, len(info["shapes"]) - 1)]
                        )
                    elif info["ranks"] is not None:
                        if isinstance(info["ranks"], int):
                            self.builder.set_rank(res, info["ranks"])
                        else:
                            self.builder.set_rank(
                                res, info["ranks"][min(index, len(info["ranks"]) - 1)]
                            )
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
                    if self.builder.has_type(result_name):
                        self.builder.set_type(node.name, self.builder.get_type(result_name))
                    if self.builder.has_device(result_name):
                        self.builder.set_device(node.name, self.builder.get_device(result_name))
                    if self.builder.has_shape(result_name):
                        self.builder.set_shape(node.name, self.builder.get_shape(result_name)[1:])
                    elif self.builder.has_rank(result_name):
                        self.builder.set_rank(node.name, self.builder.get_rank(result_name) - 1)
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
            if all(isinstance(x, (slice, self.torch.fx.Node)) for x in index):
                return self._getitem_slice(
                    node,
                    node_output.name,
                    list(index),
                    sts=sts,
                    axes=list(range(len(index))),
                    expand_axes=[],
                    name="_getitem_slicen",
                )

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
                f"getitem: unexpected tuple {tuple(type(x) for x in index)} "
                f"for index={index}, node={node}, args={args}, val={val}, "
                f"types={string_type(args)}{self.builder.get_debug_msg()}"
            )

        raise RuntimeError(
            f"getitem: unexpected type {type(index)} for index={index}, "
            f"node={node}, args={args}, val={val}, "
            f"types={string_type(args)}{self.builder.get_debug_msg()}"
        )

    def _verify_new_shape(self, shape, node):
        for axis, dim in enumerate(shape):
            if isinstance(dim, self.torch.SymInt):
                sdim = self.builder._torch_sym_int_to_str(dim)
                tokens = parse_expression_tokens(sdim)
                if len(tokens) == 1:
                    # Only one token, possibly knew
                    t = tokens.pop()
                    if t not in self.builder.dynamic_objects:
                        self.builder.add_dynamic_object(t, t)
                        source = dict(axis=axis, input_name=dim)
                        if t in self.builder.dynamic_dimensions_source:
                            self.builder.dynamic_dimensions_source[t].append(source)
                        else:
                            self.builder.dynamic_dimensions_source[t] = [source]

    def _process_arg(self, node, aten_name, i):
        if i is None:
            return None
        if isinstance(i, str):
            return i
        if hasattr(i, "name"):
            return i.name
        if isinstance(i, tuple):
            return tuple(self._process_arg(node, aten_name, t) for t in i)
        if isinstance(i, (float, int, tuple, complex)):
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
        if isinstance(i, slice):
            return slice(
                self._process_arg(node, aten_name, i.start),
                self._process_arg(node, aten_name, i.stop),
                self._process_arg(node, aten_name, i.step),
            )
        if i is Ellipsis:
            return i
        if isinstance(i, (self.torch.dtype, self.torch.device)):
            return i
        raise RuntimeError(
            f"Unexpected type (argument {i}) {type(i)} "
            f"for function {aten_name!r} "
            f"in args={node.args}{self.builder.get_debug_msg()}"
        )

    def call_function(self, node: "torch.fx.Node") -> Union[str, Tuple[str]]:  # noqa: F821
        """Called for a function."""
        aten_name = self._get_aten_name(node)
        fx_args, fx_kwargs = self._fill_in_default_kwargs(node)

        if aten_name in ("aten_auto_functionalized", "aten_auto_functionalized_v2"):
            # Should we make a direct call?
            aten_name = node.args[0]
            fx_args = fx_args[1:]

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
            fct = self.dispatcher.fallback(aten_name, fct, node.args, node.kwargs, self.builder)

        if fct is None:
            raise FunctionNotFoundError(
                f"Unable to interpret function {type(aten_name)}: "
                f"{aten_name!r}, searched for "
                f"{lookup} and attributes {lookup_names}, "
                f"args={node.args}, kwargs={node.kwargs}, dispatcher="
                f"{self.dispatcher.supported if self.dispatcher else None}"
                f"{self.builder.get_debug_msg()}, "
            )
        if self.builder.verbose > 1:
            name = fct.__class__.__name__ if isinstance(fct, GraphBuilder) else fct.__name__
            print(f"[DynamoInterpreter-{self._hash()}.call_function][{name}]")

        args = [self._process_arg(node, aten_name, a) for a in fx_args]
        output_names = self._get_output_names(node)
        can_set = self._can_set_shape_and_type(node)
        n_nodes = len(self.builder.nodes) + len(self.builder.initializers_dict)

        assert (
            len(node.users) > 0
            or aten_name
            in {
                self.torch._C._set_grad_enabled,
                self.torch._C._log_api_usage_once,
                self.torch.amp.autocast_mode._enter_autocast,
                self.torch.amp.autocast_mode._exit_autocast,
                self.torch.ops.aten._assert_scalar.default,
                self.torch.ops.aten._assert_tensor_metadata.default,
                self.torch.torch.sym_constrain_range_for_size,
                "aten__exit_autocast",
                "aten__enter_autocast",
                "aten_FunctionCtx",
                # ending with '_' but not inplace
                operator.or_,
                operator.and_,
                "aten_bitwise_or",
                "aten_bitwise_and",
            }
            or (
                hasattr(aten_name, "_opname")
                and aten_name._opname in {"sym_constrain_range_for_size"}
            )
            # the node is only used by this one
            or (
                aten_name == self.torch.ops.aten.native_dropout.default
                and len(node.args[0].users) == 1
            )
            # if an int, it cannot be modified inplace
            or ("val" in node.meta and isinstance(node.meta["val"], (int, self.torch.SymInt)))
        ), (
            f"This is probably one inplace function node={node!r}, "
            f"aten_name={aten_name!r}, node.meta={node.meta!r}, "
            f"aten_name._opname={getattr(aten_name, '_opname', '?')}, "
            f"len(node.args[0].users)="
            f"{len(node.args[0].users) if node.args and hasattr(node.args[0], 'users') else 0}"
            f", output_names={output_names!r}, exe_path={self.exe_path!r}"
            f"{self.builder.get_debug_msg()}"
        )

        if self.export_options.export_as_aten_function(aten_name):
            res = self.add_aten_as_function(
                f"aten_{aten_name.__name__}" if callable(aten_name) else str(aten_name),
                fct,
                can_set,
                output_names,
                args=args,
                kwargs=fx_kwargs,
                metadata_props={"inline": "0"},
            )
            allow_new_dynamic_dimension = False
        elif isinstance(fct, GraphBuilder):
            # The function is already implemented in a graph builder, we
            # add it as a local function.
            fct_builder = fct
            name_fct = str(aten_name).replace(".", "_")
            domain = "local_domain"

            inits, (fdomain, fname) = self.builder.make_local_function(
                fct_builder,
                FunctionOptions(
                    export_as_function=True,
                    name=name_fct.replace(".", "_"),
                    domain=domain,
                    inline=False,
                    merge_allowed=True,
                    rename_allowed=True,
                    move_initializer_to_constant=True,
                    return_initializer=True,
                    external_threshold=2**8,
                ),
                optimize=False,
            )
            new_inits = []
            for init in inits:
                new_init = self.builder.make_initializer(
                    init.name, init, source="add_aten_as_function"
                )
                new_inits.append(new_init)
            n_outputs = len(fct_builder.output_names)
            if len(output_names) == 1 and n_outputs > 1:
                output_names = [f"{output_names[0]}#{i}" for i in range(n_outputs)]
            else:
                assert len(output_names) == n_outputs, (
                    f"Unexpected output_names={output_names}, new functions:\n"
                    f"{fct_builder.pretty_text()}\n{self.builder.get_debug_msg()}"
                )
            self.builder.make_node(
                fname, [*args, *new_inits], output_names, domain=fdomain, name=name_fct
            )
            for bout, out in zip(fct_builder.output_names, output_names):
                if fct_builder.has_type(bout):
                    self.builder.set_type(out, fct_builder.get_type(bout))
                if fct_builder.has_device(bout):
                    self.builder.set_device(out, fct_builder.get_device(bout))
                if fct_builder.has_shape(bout):
                    self.builder.set_shape(out, fct_builder.get_shape(bout))
                elif fct_builder.has_rank(bout):
                    self.builder.set_rank(out, fct_builder.get_rank(bout))
            res = output_names[0] if len(output_names) == 1 else tuple(output_names)
            allow_new_dynamic_dimension = True
        else:
            res = fct(self.builder, can_set, output_names, *args, **fx_kwargs)
            allow_new_dynamic_dimension = False

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

        str_target = str(node.target)
        self._set_shape_and_type(
            node,
            res,
            fct_name=aten_name,
            allow_new_dynamic_dimension=allow_new_dynamic_dimension
            or (
                node.target == self.torch.ops.aten.nonzero_numpy.default
                or (node.target == self.torch.ops.aten.where.default and len(node.args) == 1)
                or ("aten." not in str_target and "aten_" not in str_target)
            ),
        )
        res = self._check_output_name(node, res, output_names)
        return res

    def call_method(self, node: "torch.fx.Node") -> Union[str, Tuple[str]]:  # noqa: F821
        """Called for a method."""
        method_name = node.target
        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.call_method][{method_name}]")
        assert isinstance(node.args, tuple), f"Unexpected type {type(node.args)} for node.args."

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

        if self.export_options.export_as_aten_function(method_name):
            res = self.add_aten_as_function(
                name_fct,
                fct,
                can_set,
                output_names,
                args,
                kwargs,
                metadata_props={"inline": "0"},
            )
        else:
            res = fct(self.builder, can_set, output_names, *args, **kwargs)

        self._set_shape_and_type(node, res, fct_name=method_name)
        res = self._check_output_name(node, res, output_names)
        return res

    def add_aten_as_function(
        self,
        name_fct: str,
        fct: Callable,
        can_set: Optional[Dict[str, Any]],
        output_names: List[str],
        args: List[Any],
        kwargs: Dict[str, Any],
        domain: str = "aten",
        metadata_props: Optional[Dict[str, str]] = None,
    ) -> Union[str, Tuple[str]]:
        """
        Converts a function into a local function and adds this local function to the graph.
        """
        assert isinstance(name_fct, str), (
            f"Unexpected type {type(name_fct)} for name_fct={name_fct}"
            f"{self.builder.get_debug_msg()}"
        )
        assert "<" not in name_fct, f"Unexpected name {name_fct!r}{self.builder.get_debug_msg()}"
        # Collects inputs
        input_names = self.builder.extract_input_names_from_args(args)

        for k, v in kwargs.items():
            if isinstance(v, str):
                raise NotImplementedError(
                    f"This option is not implemented yet for k={k!r} "
                    f"with type={type(v)}{self.builder.get_debug_msg()}"
                )

        if self.builder.verbose > 1 or self._debug_aten_as_function:
            print(
                f"[DynamoInterpreter.add_aten_as_function] {name_fct}"
                f"({', '.join(input_names)}) -> {', '.join(output_names)}"
            )

        self.builder._check_function_order()
        new_builder = self.builder.make_subset_builder(
            input_names, name=name_fct.replace(".", "_"), domain=domain
        )
        try:
            res = fct(new_builder, can_set, output_names, *args, **kwargs)
        except AssertionError as e:
            raise AssertionError(
                f"The conversion of operator {name_fct!r} into a local function\n--ERROR--\n"
                f"{e}{self.builder.get_debug_msg()}"
            ) from e

        assert (len(output_names) == 1 and res == output_names[0]) or res == output_names, (
            f"Mismatch issue res={res!r}, output_names={output_names!r} "
            f"for function {name_fct!r}{self.builder.get_debug_msg()}"
        )
        for o in output_names:
            new_builder.make_tensor_output(
                o,
                indexed=False,
                is_dimension=self.builder.get_is_dimension(o, exc=False),
                doc_string=f"#C:{o}",
            )
        new_builder._check_function_order()
        inits, (fdomain, fname) = self.builder.make_local_function(
            new_builder,
            FunctionOptions(
                export_as_function=True,
                name=name_fct.replace(".", "_"),
                domain=domain,
                inline=False,
                merge_allowed=True,
                rename_allowed=True,
                move_initializer_to_constant=True,
                return_initializer=True,
                external_threshold=2**8,
            ),
            optimize=False,
            metadata_props=metadata_props,
        )
        new_inits = []
        for init in inits:
            if isinstance(init, str):
                assert (
                    init in self.builder.initializers_dict
                ), f"Missing initializer {init!r}{self.builder.get_debug_msg()}."
                new_inits.append(init)
            else:
                new_init = self.builder.make_initializer(
                    init.name, init, source="add_aten_as_function"
                )
                new_inits.append(new_init)

        self.builder.make_node(
            fname,
            [*input_names, *new_inits],
            output_names,
            domain=fdomain,
            name=name_fct,
            metadata_props=dict(aten_name=name_fct, args=str(args), kwargs=str(kwargs)),
        )
        assert len(output_names) == len(new_builder.outputs), (
            f"Function output mismatch {output_names} != {new_builder.outputs}"
            f"{self.get_debug_msg()}"
        )
        for out, lout in zip(output_names, new_builder.outputs):
            if new_builder.has_type(lout.name):
                self.builder.set_type(out, new_builder.get_type(lout.name))
            if new_builder.has_device(lout.name):
                self.builder.set_device(out, new_builder.get_device(lout.name))
            if new_builder.has_shape(lout.name):
                self.builder.set_shape(out, new_builder.get_shape(lout.name))
            elif new_builder.has_rank(lout.name):
                self.builder.set_rank(out, new_builder.get_rank(lout.name))
        return output_names[0] if len(output_names) == 1 else output_names

    def _get_output_names(self, node: "torch.fx.Node") -> List[str]:  # noqa: F821
        val = node.meta.get("val", None)
        if val is not None and isinstance(val, tuple):
            n_outputs = len(val)
            output_names = [
                ("" if val[i] is None else f"{node.name}#{i}") for i in range(n_outputs)
            ]
        else:
            assert isinstance(node.name, str), f"Unexpected type {type(node.name)} for node.name"
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
                    for i, (a, b) in enumerate(zip(output_names, res)):
                        if a != b and a:
                            raise NotImplementedError(
                                f"Unexpected output_names at position {i}, a={a!r}, "
                                f"b={b!r},\noutput_names={output_names}"
                                f"\n         res={res!r}\n"
                                f"node.name={node.name!r}"
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
            elif (
                isinstance(res, list)
                and len(res) == 1
                and str(getattr(node, "target", None))
                in {"scan", "aten.split_with_sizes.default"}
            ):
                # Scan allows that
                name = output_names[0]
                assert all(s.startswith(name) for s in res), (
                    f"Unexpected output_names={output_names}, "
                    f"res={res}, node.name={node.name}, "
                    f"target(str)={str(getattr(node, 'target', None))!r}"
                    f"{self.builder.get_debug_msg()}"
                )
                # nothing to do
                res = tuple(res)
            elif res != node.name:
                assert isinstance(res, str), (
                    f"Unexpected res={res}, output_names={output_names}, "
                    f"node.name={node.name}, target={getattr(node, 'target', '?')!r}, "
                    f"target(str)={str(getattr(node, 'target', None))!r}"
                    f"{self.builder.get_debug_msg()}"
                )
                self.builder.make_node("Identity", [res], [node.name], name="_check_output_name")
                res = node.name
        else:
            raise NotImplementedError(
                f"Unexpected type {type(node.name)} for node.name={node.name!r}."
            )
        return res

    def _can_set_shape_and_type(
        self, node: "torch.fx.Node"  # noqa: F821
    ) -> Optional[Dict[str, Any]]:
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
                import torch

                return tuple(
                    (None if v is None else (torch.int64 if isinstance(v, int) else v.dtype))
                    for v in val
                )
            if isinstance(val, (int, self.torch.SymInt)):
                return self.torch.SymInt
            if isinstance(val, self.torch.SymBool):
                return self.torch.SymBool
            if isinstance(val, (float, self.torch.SymFloat)):
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
        allow_new_dynamic_dimension: bool = False,
    ):
        """
        Sets shape and type for a result.r
        This information is coming from the torch exporter.
        ``allow_new_dynamic_dimension`` can be use to bypass errors
        related to new shape.
        """
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
        if val is not None and fct_name not in {
            "aten_cond",
            "aten_simple_loop_for",
            "aten_wrap_with_autocast",
            "aten_wrap_with_set_grad_enabled",
        }:
            # extracting shape and types
            if not isinstance(val, tuple):
                val = (val,)
                res = (res,)
            assert isinstance(res, (list, tuple)), (
                f"fct_name={fct_name!r}, unexpected type {type(res)}, "
                f"res={string_type(res, with_shape=True)}, "
                f"val={string_type(val, with_shape=True)}, "
                f"node={node}, node.target={node.target}, "
                f"res={res!r}{self.builder.get_debug_msg()}"
            )
            if len(val) != len(res):
                raise RuntimeError(
                    f"Length mismatch {len(val)} != {len(res)} "
                    f"between {val} and {res}"
                    f"{self.builder.get_debug_msg()}"
                )
            output_sets = set(last_node.output) if last_node is not None else {}

            # specific case, a loop was not exported as a loop because if it has one element only.
            if (
                len(val) == 1
                and isinstance(val[0], list)
                and len(val[0]) == 1
                and isinstance(res, tuple)
                and len(res) == 1
                and isinstance(res[0], str)
                and not self.builder.is_sequence(res[0])
                and hasattr(node.target, "name")
                and node.target.name()
                in {"aten::unbind.int", "aten::nonzero_numpy", "aten::where"}
            ):
                val = val[0]

            for i, (v, r) in enumerate(zip(val, res)):
                if isinstance(v, self.torch.Tensor):
                    self.builder.set_device(r, v.get_device(), keep_this_device=True)
                    dtype = _get_type(v.dtype)
                    if i >= 1 and node.target.name() in {
                        "aten::_native_batch_norm_legit.no_stats",
                        "aten::_native_batch_norm_legit_no_training",
                        "aten::_scaled_dot_product_efficient_attention",
                        "aten::_scaled_dot_product_flash_attention",
                    }:
                        # It seems the type is not very consistant
                        # and the output might not be used.
                        self.builder.set_type(r, dtype, exc=False)
                    else:
                        self.builder.set_type(r, dtype)
                    shape = tuple(v.shape)

                    for t in shape:
                        if isinstance(t, self.builder.torch.SymInt):
                            expr = str(t.node._expr).replace(" ", "")
                            if expr not in self.builder.dynamic_objects:
                                # A new shape may be given to a result.
                                self.builder.add_dynamic_object(expr, t, parse=True)

                    if self.builder.is_dynamic_shape(shape):
                        # sets shape coming from the original model
                        # we must not set the existing shape as static,
                        # if it was dynamic before
                        self.builder.set_shape(
                            r,
                            shape,
                            set_if_more_precise=False,
                            allow_zero=any(isinstance(s, int) and s == 0 for s in shape),
                        )
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
                    if self.builder.has_shape(r):
                        assert self.builder.get_shape(r) == tuple(), (
                            f"Shape mismatch for {r!r}, got {self.builder.get_shape(r)}, "
                            f"it should be empty{self.builder.get_debug_msg()}"
                        )
                    else:
                        self.builder.set_shape(r, tuple())
                    if self.builder.has_type(r):
                        t = self.builder.get_type_known(r)
                        assert t is None or t == self.builder.get_type(r), (
                            f"Unexpected type {onnx_dtype_name(self.builder.get_type(r))}, "
                            f"know by torch is {onnx_dtype_name(t)}, "
                            f"for r={r!r}, expected INT32 or INT64{self.builder.get_debug_msg()}"
                        )
                    else:
                        t = self.builder.get_type_known(r)
                        if t is not None:
                            assert t in (TensorProto.INT64, TensorProto.INT32), (
                                f"Unexpected type "
                                f"{onnx_dtype_name(self.builder.get_type_known(r))} "
                                f"for r={r!r}, expected INT32 or INT64"
                                f"{self.builder.get_debug_msg()}"
                            )
                            self.builder.set_type(r, t)
                        else:
                            self.builder.set_type(r, TensorProto.INT64)
                    self.builder.make_dynamic_object(r, v)
                elif isinstance(v, self.torch.SymBool):
                    # this is a shape
                    self.builder.set_shape(r, tuple())
                    self.builder.set_type(r, TensorProto.BOOL)
                    self.builder.make_dynamic_object(r, v)
                elif isinstance(v, self.torch.SymFloat):
                    # this is a shape
                    self.builder.set_shape(r, tuple())
                    self.builder.set_type(r, TensorProto.FLOAT)
                    self.builder.make_dynamic_object(r, v)
                elif isinstance(v, bool):
                    # check for boolean first because a boolean is an int.
                    self.builder.set_shape(r, tuple())
                    self.builder.set_type(r, TensorProto.BOOL)
                elif isinstance(v, int):
                    # this is unknown
                    self.builder.set_shape(r, tuple())
                    self.builder.set_type(r, TensorProto.INT64)
                elif v is None:
                    continue
                elif isinstance(v, list) and len(v) > 0:
                    assert isinstance(r, (list, tuple)), (
                        f"Expected a list but type is {type(r)}, v={v!r}, "
                        f"r={r!r}, val={val}, res={res}, node={node!r}, "
                        f"node.target={node.target!r}{self.builder.get_debug_msg()}"
                    )
                    if len(v) == len(r) and (r[0].endswith("#0") or len(v) == 1):
                        # Operator Split was used instead of SplitToSequence.
                        # Or any other node producing multiple results.
                        for r_, v_ in zip(r, v):
                            assert not self.builder.is_sequence(r_), (
                                f"{r_!r} is defined as a sequence already"
                                f"{self.builder.get_debug_msg()}"
                            )
                            self.builder.set_type(r_, torch_dtype_to_onnx_dtype(v_.dtype))
                            self.builder.set_device(r_, v_.get_device())
                            shape = tuple(v_.shape)
                            if not any(
                                i == 0 for i in shape if isinstance(i, int)
                            ) and self.builder.is_dynamic_shape(
                                shape, allow_new_dynamic_dimension=allow_new_dynamic_dimension
                            ):
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
                            f"but dtype={dtype}, node={node!r}, target={node.target!r}"
                            f"{self.builder.get_debug_msg()}"
                        )
                        itype = torch_dtype_to_onnx_dtype(dtype[0])
                        self.builder.set_sequence(
                            r,
                            itype,
                            shapes=tuple(
                                tuple(map(self.builder._torch_sym_int_to_str, _.shape)) for _ in v
                            ),
                        )
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

    def _interpret_sub_module(
        self,
        sub_module,
        args,
        kwargs,
        source_node=None,
        local_domain=None,
        preserve_as_submodule=False,
    ):
        from .onnx_export import _make_builder_interpreter

        assert not kwargs, (
            f"This functionality is not implemented kwargs={string_type(kwargs)}"
            f"{self.get_debug_msg()}"
        )
        if args is None:
            new_args = None
        else:
            new_args = []
            for a in args:
                if isinstance(a, self.torch.fx.Node):
                    name = a.name
                    dtype = self.builder.get_type(name) if self.builder.has_type(name) else 0
                    shape = (
                        self.builder.get_shape(name)
                        if self.builder.has_shape(name)
                        else (
                            self.builder.make_new_dynamic_shape(
                                self.builder.get_rank(name), prefix=name
                            )
                            if self.builder.has_rank(name)
                            else None
                        )
                    )
                    # shape is None if the module is traced.
                    device = (
                        self.builder.get_device(name) if self.builder.has_device(name) else None
                    )
                    new_args.append(
                        VirtualTensor(name=name, dtype=dtype, shape=shape, device=device)
                    )
                elif isinstance(a, self.torch.Tensor):
                    new_args.append(a)
                else:
                    raise NotImplementedError(
                        f"Unable to process argument {type(a)}{self.get_debug_msg()}"
                    )

        if hasattr(sub_module, "graph") and isinstance(sub_module, self.torch.fx.GraphModule):
            gm = sub_module
        elif (
            hasattr(sub_module, "graph")
            and isinstance(sub_module, self.torch.nn.Module)
            and sub_module.__class__.__name__ == "InterpreterModule"
        ):
            gm = sub_module
        else:
            # https://docs.pytorch.org/docs/stable/fx.html
            tracer_class = self.torch.fx.Tracer
            tracer = tracer_class()
            graph = tracer.trace(sub_module)
            # Let's propulate with type
            if new_args:
                ii = 0
                for node in graph.nodes:
                    if node.op == "placeholder":
                        if ii >= len(new_args) or "val" in node.meta:
                            ii += 1
                            continue
                        ag = new_args[ii]
                        if isinstance(ag, VirtualTensor):
                            node.meta["val"] = ag
                        else:
                            node.meta["example_value"] = ag
                        ii += 1
            gm = self.torch.fx.GraphModule(
                getattr(tracer, "traced_model", None) or sub_module, graph
            )

        graph_module, builder, interpreter, mask_outputs = _make_builder_interpreter(
            gm,
            args=None if new_args is None else tuple(new_args),
            kwargs=None if kwargs is None else kwargs,
            as_function=True,
            target_opset=self.builder.opsets,
            optimization_options=self.builder.optimization_options,
            verbose=max(0, self.builder.verbose - 1),
            dispatcher=self.dispatcher,
            raise_list=self.builder.raise_list,
            # dynamic shapes applies on the inner graph, not on the subgraph
            # dynamic_shapes=self.builder.dynamic_shapes,
            export_options=self.export_options,
            optimize_submodules=self.optimize_submodules,
            function_options=self.function_options,
            local_domain=local_domain,
            submodule_naming=self.submodule_naming,
            parameter_naming=self.parameter_naming,
            module_name=(
                None
                if (self.module_name is None or source_node is None)
                else (
                    source_node.target
                    if self.module_name == ""
                    else f"{self.module_name}.{source_node.target}"
                )
            ),
        )
        builder._parent = self.builder
        assert mask_outputs is None or all(
            mask_outputs
        ), f"Unexpected value for mask_outputs={mask_outputs}{self.get_debug_msg()}"

        if not preserve_as_submodule:
            # We register the dynamic elements in case the submodule is using them.
            # But only if it not preserved as a submodule. In that case,
            # these information are likely to become inputs.
            for k, v in self.builder.dynamic_objects.items():
                # We assume the list of dynamic objects is valid.
                if not self.builder.has_name(k):
                    builder.add_dynamic_object(k, v, check_tokens=False)
                    if self.builder.has_type(k):
                        builder.set_type(k, self.builder.get_type(k))
                    if self.builder.has_device(k):
                        builder.set_device(k, self.builder.get_device(k))
                    if self.builder.has_shape(k):
                        builder.set_shape(k, self.builder.get_shape(k))

        if self.preserved_modules and hasattr(self, "named_modules"):
            assert (
                source_node is not None
            ), f"For this option, source_node cannot be None{self.builder.get_debug_msg()}"
            module_name = source_node.target
            if module_name in self.named_modules:
                module_child = self.named_modules[module_name]
                interpreter.register_named_modules(self, None, dict(module_child.named_modules()))

        # processes the submodules
        builder.process(graph_module, interpreter)
        assert builder.outputs, f"No output detected for node={source_node}, graph={gm}"

        # processing args, kwargs
        fx_args, fx_kwargs = self._fill_in_default_kwargs(source_node)
        args = [getattr(i, "name", i) for i in fx_args]
        kwargs = [getattr(i, "name", i) for i in fx_kwargs]

        # looking at the sample example
        val = source_node.meta.get("val", None)
        if val is not None and isinstance(val, (tuple, list)):
            n_outputs = len(val)
            output_names = [f"{source_node.name}#{i}" for i in range(n_outputs)]
        elif (
            preserve_as_submodule
            and self.preserved_modules
            and val is not None
            and isinstance(val, list)
        ):
            n_outputs = len(val)
            output_names = [f"{source_node.name}#{i}" for i in range(n_outputs)]
            val = tuple(val)
        else:
            output_names = [source_node.name]
            if val is None:
                val = source_node.meta.get("example_value", None)
        if val is not None and not isinstance(val, tuple):
            val = (val,)

        # if not none
        if val is not None and not preserve_as_submodule:
            # regular node
            if isinstance(val, tuple) and len(val) == 1 and isinstance(val[0], list):
                val = val[0]
            assert len(val) == len(builder.outputs), (
                f"Output mismatch {len(val)} != {len(builder.outputs)}, "
                f"source_node.name={source_node.name!r}, target={source_node.target!r}"
                f"type(val)={string_type(val)}, "
                f"builder.outputs={string_type(builder.outputs)}"
                f"{self.builder.get_debug_msg()}"
            )
            for i in range(len(val)):
                name = builder.outputs[i].name
                if not builder.has_shape(name):
                    builder.set_shape(name, val[i].shape)
                if not builder.has_type(name):
                    builder.set_type(name, val[i].dtype)
                if isinstance(val[i], self.builder.torch.Tensor):
                    self.builder.set_shapes_types(
                        source_node.name, "call_module", (val[i].dtype, val[i].shape)
                    )
                    if not builder.has_device(name):
                        builder.set_device(name, val[i].get_device())
                elif isinstance(val[i], (self.builder.torch.SymInt)):
                    self.builder.set_shapes_types(
                        source_node.name,
                        "call_module",
                        (self.builder.torch.SymInt, tuple()),
                    )
                elif isinstance(val[i], (self.builder.torch.SymFloat)):
                    self.builder.set_shapes_types(
                        source_node.name,
                        "call_module",
                        (self.builder.torch.SymFloat, tuple()),
                    )
        return builder, args, kwargs, output_names

    def get_submodule_name(
        self, module_name: str, module: "torch.nn.Module"  # noqa: F821
    ) -> str:
        """Gets a submodule name, simple but unique."""
        assert self.submodule_naming, "submodule_naming is null"
        assert self.parameter_naming, "parameter_naming is null"
        return self.submodule_naming(module_name, module).replace(".", "_")

    def call_module(self, node: "torch.fx.Node"):  # noqa: F821
        """Called for a module."""

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

        owning_module = node.graph.owning_module
        assert owning_module is not None, f"owning_module is None\n{raise_msg()}"
        sub_module = owning_module.get_submodule(node.target)

        assert isinstance(
            sub_module, self.torch.nn.Module
        ), f"Not implemented for type {type(sub_module)}.\n{raise_msg()}"

        if self.builder.verbose > 1:
            print(f"[DynamoInterpreter-{self._hash()}.call_module] class [{type(sub_module)}]")
            print(
                f"[DynamoInterpreter-{self._hash()}.call_module] with "
                f"node.args={string_type(node.args)}]"
            )
            print(
                f"[DynamoInterpreter-{self._hash()}.call_module] with "
                f"kwargs={string_type(node.kwargs)}]"
            )

        # This function is meant to be used later.
        if "." in self.builder.local_domain:
            root, n = self.builder.local_domain.split(".")
            n = int(n) + 1
        else:
            root, n = self.builder.local_domain, 0

        self.builder._check_constants("before-_interpret_sub_module")

        node_module = (
            self.named_modules[node.target]
            if node.target
            and hasattr(self, "named_modules")
            and node.target in self.named_modules
            else None
        )
        preserve_as_submodule = node_module and type(node_module) in self.preserved_modules

        builder, args, kwargs, output_names = self._interpret_sub_module(
            sub_module,
            node.args,
            node.kwargs,
            source_node=node,
            local_domain=f"{root}.{n}",
            preserve_as_submodule=preserve_as_submodule,
        )

        self.builder._check_constants("after-_interpret_sub_module")

        assert kwargs is None or len(kwargs) == 0, (
            f"args={string_type(args)}, kwargs={string_type(kwargs)} "
            f"is not implemented yet{self.builder.get_debug_msg()}"
        )

        name = sub_module.__class__.__name__
        prefix = f"_sub_{name}_{node.name}__"
        local_function_name = None
        if sub_module.__class__.__name__ == "InterpreterModule":
            # a local function is added.
            assert node_module, (
                f"Unable to find module name {node.target!r} in "
                f"{sorted(self.named_modules)}{self.builder.get_debug_msg()}"
            )
            if preserve_as_submodule:
                # Which name to give the submodule?
                # The class, the module name, ...?
                local_function_name = name = self.get_submodule_name(node.target, node_module)
                assert local_function_name, (
                    f"empty value for local_function_name={local_function_name!r}, "
                    f"type(m)={type(node_module)}, "
                    f"self.preserved_modules={self.preserved_modules}, "
                    f"node={node!r}, node.target={node.target}{self.builder.get_debug_msg()}"
                )

                self.builder._check_constants("before-make_nodes")

                # let's create a function under the appropriate name
                prefix = f"_sub_IM_{node.name}__"
                self.builder.make_nodes(
                    builder,
                    args,
                    output_names,
                    prefix=prefix,
                    function_options=FunctionOptions(
                        name=local_function_name,
                        domain=LOCAL_DOMAIN,
                        export_as_function=True,
                        return_initializer=True,
                        move_initializer_to_constant=self.function_options.move_initializer_to_constant,
                        external_threshold=self.function_options.external_threshold,
                        merge_allowed=self.function_options.merge_allowed,
                        rename_allowed=self.function_options.rename_allowed,
                    ),
                    optimize=self.optimize_submodules,
                )

                self.builder._check_constants("after-make_nodes")

                if len(output_names) == len(builder.outputs):
                    # One output, both tensor
                    for name, out_name in zip(builder.output_names, output_names):
                        if builder.has_type(name):
                            self.builder.set_type(out_name, builder.get_type(name))
                        if builder.has_device(name):
                            self.builder.set_device(out_name, builder.get_device(name))
                        if builder.has_shape(name):
                            existing_shape = builder.get_shape(name)
                            # We need to move any dynamic objects necessary from the submodules
                            # to the parent module.
                            self.builder.register_dynamic_objects_from_shape(existing_shape)
                            self.builder.set_shape(out_name, existing_shape)
                        elif builder.has_rank(name):
                            self.builder.set_rank(out_name, builder.get_rank(name))
                elif len(output_names) == 1 and len(builder.outputs) > 1:
                    # The module outputs more than one output
                    itypes, shapes, ranks = [], [], []
                    for name in builder.output_names:
                        itypes.append(builder.get_type(name) if builder.has_type(name) else None)
                        shapes.append(
                            builder.get_shape(name) if builder.has_shape(name) else None
                        )
                        ranks.append(builder.get_rank(name) if builder.has_rank(name) else None)
                    self.builder.set_sequence(
                        output_names[0], tuple(itypes), shapes=tuple(shapes), ranks=ranks
                    )
                else:
                    raise AssertionError(
                        f"Unexpected number of outputs, output_names={output_names}, "
                        f"len(builder.outputs)={len(builder.outputs)}, "
                        f"builder.output_names={builder.output_names}"
                        f"{builder.get_debug_msg()}\n--\n--\n--"
                        f"{self.builder.get_debug_msg()}\n------\n"
                    )
                self._set_shape_and_type(
                    node,
                    output_names[0] if len(output_names) == 1 else tuple(output_names),
                    allow_new_dynamic_dimension=True,
                )
                return output_names
            prefix = f"_sub_ime__{node.name}_"
        else:
            assert not preserve_as_submodule, (
                f"Unable to preserve module class {type(node_module)} for node {node!r} "
                f"type(sub_module)={type(sub_module)}{self.builder.get_debug_msg()}"
            )

        self.builder._check_constants("before-make_nodes(2)")
        self.builder.make_nodes(
            builder, args, output_names, prefix=prefix, force_rename_with_prefix=node.name
        )
        self.builder._check_constants("after-make_nodes(2)")
        return output_names
