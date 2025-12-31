import contextlib
import os
import pprint
import textwrap
import time
import sys
from collections import Counter
from enum import IntEnum
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
)
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.external_data_helper import uses_external_data
from onnx.model_container import make_large_tensor_proto
from onnx.shape_inference import infer_shapes as onnx_infer_shapes
from ..helpers import (
    dtype_to_tensor_dtype,
    from_array_extended,
    make_hash,
    string_sig,
    onnx_dtype_to_torch_dtype,
    pretty_onnx,
    size_type,
    string_signature,
    string_type,
    torch_dtype_to_onnx_dtype,
    make_idn,
    make_idg,
    onnx_dtype_name,
)
from ..xshape.rename_expressions import rename_dynamic_dimensions, rename_dynamic_expression
from ..xshape._shape_helper import (
    DYNAMIC_SHAPE,
    STATIC_SHAPE,
    all_int,
    all_int_or_str,
    is_static_dimension,
    is_static_shape,
)
from ..xshape.shape_type_compute import set_shape_type_op_any, set_shape_type_custom
from ..xshape._builder_runtime import _BuilderRuntime
from ..xshape._shape_runtime import _ShapeRuntime
from ..xshape._inference_runtime import _InferenceRuntime
from ..xshape._onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
    str_tensor_proto_type,
)
from ..xshape.simplify_expressions import simplify_expression
from ._onnx_helper import (
    _default_OPSET_TO_IR_VERSION,
    _nice_shape,
    choose_consistent_domain_opset,
    compatible_opsets,
    same_function_proto,
)
from .model_container import TorchModelContainer, proto_from_array, _get_type
from .optimization_options import OptimizationOptions
from ..xshape.rename_expressions import rename_expression, parse_expression_tokens
from ..xshape.simplify_expressions import simplify_two_expressions
from ..xshape.expressions_torch import Expression, parse_expression
from .graph_builder_opset import Opset
from .virtual_tensor import VirtualTensor


# To help finding bugs.
def assert_sorted(inputs):
    try:
        return sorted(inputs)
    except TypeError:
        return list(inputs)


@contextlib.contextmanager
def _unset_fake_temporarily() -> Generator:
    import torch

    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


class FunctionOptions:
    """
    Defines how local functions must behave.

    :param name: function name
    :param domain: function domain
    :param export_as_function: export the onnx as functions or keep local function
    :param external_threshold: whether or not keep initializer as input for the function
        or move them as constant of the function
    :param move_initializer_to_constant: move initializers as constant first before
        creating the function proto, that depends on the size defined by
        external_threshold
    :param return_initializer: return the remaining initializer and add them as input
        to the function
    :param inline: inline functions
    :param rename_allowed: allow to rename the function if a duplicate is detected
    :param merge_allowed: allow to merge a function in case the same code is detected
    """

    empty_names = (None, "", "*")

    def __init__(
        self,
        export_as_function: bool = False,
        name: str = "",
        domain: str = "",
        external_threshold: int = 2**25,
        move_initializer_to_constant: bool = False,
        return_initializer: bool = False,
        inline: bool = False,
        merge_allowed: bool = False,
        rename_allowed: bool = False,
    ):
        if name:
            export_as_function = True
        assert not export_as_function or name, (
            f"to be removed help track bugs, name={name!r}, domain={domain!r}, "
            f"export_as_function={export_as_function!r}"
        )
        assert export_as_function or (not name and not domain), (
            f"to be removed help track bugs, name={name!r}, domain={domain!r}, "
            f"export_as_function={export_as_function!r}"
        )
        assert isinstance(
            external_threshold, int
        ), f"Unexpected type {type(external_threshold)} for external_threshold"
        self.export_as_function = export_as_function
        self.external_threshold = external_threshold
        self.move_initializer_to_constant = move_initializer_to_constant
        self.name = name
        self.domain = domain
        self.return_initializer = return_initializer
        self.inline = inline
        self.rename_allowed = rename_allowed
        self.merge_allowed = merge_allowed

    def __repr__(self) -> str:
        "usual"
        return string_sig(self)


class InferShapesOptions(IntEnum):
    """
    Defines options when running shape inference on an existing model.
    Options ``NEW`` means shapes informations is removed by running it again.
    """

    NONE = 0
    NEW = 1
    ONNX = 2
    DATA_PROP = 4
    BUILDER = 8


class GraphBuilder(_BuilderRuntime, _ShapeRuntime, _InferenceRuntime):
    """
    Simplifies the creation of a model.

    :param target_opset_or_existing_proto: a ModelProto, an integer,
        a dictionary of domain, version
    :param input_names: input names
    :param as_function: export as a function or a model
       there are less assert when as_function is True
    :param optimization_options: optimizations options,
        see :class:`OptimizationOptions`
    :param args: example of inputs
    :param kwargs: example of inputs
    :param ir_version: ir version when exporting
    :param verbose: verbosity
    :param infer_shapes_options: options when running
        shape inference for an existing model
    :param raise_list: raise an exception if a new operator belongs to that list
    :param dynamic_shapes: dynamic shapes
    :param local_domain: domain name to use for local functions if not specified
    :param signature: the signature is unused but helps for debugging purposes
    :param check_empty_source: checks source are not empty
    :param graph_module: only used for debugging purpose
    :param exe_path: gives information on how the :class:`torch.fx.Graph` was obtained
    :param output_names: output names
    :param output_dynamic_shapes: same as dynamic_shapes but for the output
    :param _opsets: to overwrite opsets when the builder receives a `GraphProto`
    :param _context: known names from the parent graph is any
    :param _parent: parent owning the builder is any

    Important attributes:

    - `input_names: List[str]`: list of input names
    - `output_names: List[str]`: list of output names
    - `as_function: bool`: the model must be exported as a function or as a model,
      there are less assert when as_function is True
    - `optimization_options: OptimizationOptions`:
    - `nodes: List[NodeProto]`: list of nodes
    - `initializers_dict: Dict[str, Any]`: initializers
    - `initializers_dict_sources: Dict[str, InitializerInfo]`:
      information about where the initiliazers was created
    - `inputs: List[ValueInfoTensorProto]`: inputs
    - `outputs: List[ValueInfoTensorProto]`: outputs
    - `ir_version: int`: ir version
    - `opsets: Dict[str, int]`: declared opsets
    - `input_args: List[T]`: input tensors when
      the class is used to convert an existing model
    - `input_kwargs: Dict[str, T]`: input tensors when
      the class is used to convert an existing model
    - `functions: Dict[Tuple[str,str], FunctionProto]`:
      dictionary of functions to add to the model
    - `value_info: List[ValueInfoProto]`: value info of the original model
    - `dynamic_shapes: Union[Dict[str, Any], Tuple[Any]]]`: dynamic_shapes informations
    - `_parameter_renaming: Dict[str, str]`: to rename parameter and give them
      a name which can be found in ``module.named_parameter``

    Computed attributes:

    - `_unique_names`: used to create unused result names
    - `_unique_node_names`: used to create unused node names
    - `_known_names`: set of existing results names
    - `_known_shapes: Dict[str, DYNAMIC_SHAPE]`: declared shapes
    - `_known_types: Dict[str, int]`: declared element types
    - `_known_devices: Dict[str, int]`: declared devices
    - `_known_value_shape: Dict[str, Any]`: if a result is a shape or not
      (for example the output of operator Shape)
    - `_known_ranks: Dict[str, int]`: declared ranks
    - `_known_sequences: Dict[str, Dict[str, Any]]`: known sequences
    - `_dynamic_examples: Dict[str, Set[Union[int,float]]]`: example of dynamic dimensions
    - `constants_node_: Dict[bytes, NodeProto]`: constant node
    - `constants_alias_: Dict[str, str]`: alias for constant
    - `constants_: Dict[str, Any]`: constant values
    - `constants_computed_: Dict[str, Any]`: computed constant values
    - `dynamic_objects: Dict[str, torch.SymInt]`: list of dynamic dimension
    - `dynamic_objects_rev: Dict[str, str]`: reverse dictionary to fasten lookups
    - `_cache_shape: Dict[key,str]`: cache concatenation of shapes
    - `_values: Dict[key,str]`: cache initializer value to merge those which are equal
    - `_dynamic_alias: Dict[str,str]`: used when the user gives a different
        name to the dynamic shapes
    - `constraints_: Dict[str, Set[Any]]`:
        if a broadcast implies a constraints on a dynamic shape,
        it is stored here
    - `_events`: is used to retrieve any information useful to debug

    Debugging attributes:

    - `_raise_list: Set[str]`: the builder stop if a result falls in that list
      (debugging tool)

    You can setup environment variable ``ONNXSTOP``, ``ONNXSTOPSHAPE``, ``ONNXSTOPTYPE``,
    ``ONNXSTOPVALUESHAPE``, ``ONNXSTOPOUTPUT`` to raise an exception when the type or shape
    of a variable is set. Example: ``ONNXSTOP=attn_output python ...``.
    ``ONNXCST=1`` shows which constant is computed,
    ``NULLSHAPE=1`` raises an exception as soon as a null shape occurs. The code includes:

    ::

        self._debug_null_shape = int(os.environ.get("NULLSHAPE", "0"))
        self._debug_stop = os.environ.get("ONNXSTOP", "#?#")
        self._debug_stop_shape = os.environ.get("ONNXSTOPSHAPE", "#?#")
        self._debug_stop_type = os.environ.get("ONNXSTOPTYPE", "#?#")
        self._debug_stop_sequence = os.environ.get("ONNXSTOPSEQUENCE", "#?#")
        self._debug_get_constant = int(os.environ.get("ONNXCST", "0"))
        self._debug_foldnot = int(os.environ.get("ONNXFOLDNOT", "0"))
        self._debug_local_function = int(os.environ.get("ONNXFUNC", "0"))
        self._debug_value_shape = os.environ.get("ONNXSTOPVALUESHAPE", "")
        self._debug_node_output = os.environ.get("ONNXSTOPOUTPUT", "")
        self._debug_node_type = os.environ.get("ONNXNODETYPE", "")
        self._debug_quiet = int(os.environ.get("ONNXQUIET", "0"))
        self._debug_shape_missing = int(os.environ.get("ONNXSHAPECOMPUTE", "0"))
        self._debug_constant_folding = int(os.environ.get("ONNXCONSTANTFOLD", "0"))
        self._debug_dyn_dim = os.environ.get("ONNXDYNDIM","").split(",")

    .. warning:: ModelProto may be modified

        The builder tries to minimize the copies of protos. The original
        model may be modified if the builder is initialized with a model.
    """

    MINUS_ONE = np.array([-1], dtype=np.int64)
    ONE = np.array([1], dtype=np.int64)
    ONE_NO_DIM = np.array(1, dtype=np.int64)
    ZERO = np.array([0], dtype=np.int64)
    ZERO_NO_DIM = np.array(0, dtype=np.int64)
    # MAXINT for int64, means a slice goes up to the last element
    END = np.array([9223372036854775807], dtype=np.int64)

    class ShapeConstant:
        """Wraps a constant shape even if the input producing the shape is not."""

        def __init__(self, name: str, shape: Tuple[int, ...], node: NodeProto):
            self.name = name
            self.shape = shape
            self.node = node

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}({self.name!r}, shape={self.shape!r}, "
                f"node=<{self.node.op_type})"
            )

    class WrapSym:
        """Wraps a symbolic int (a dimension for example)."""

        def __init__(self, sym: Union["torch.SymInt", "torch.SymFloat"]):  # noqa: F821
            if isinstance(sym, GraphBuilder.WrapDim):
                self.sym = sym.name
            else:
                assert isinstance(sym, str) or hasattr(
                    sym, "node"
                ), f"Missing attribute node for type {type(sym)}"
                self.sym = sym

        def __repr__(self) -> str:
            return f"WrapSym({self._dynamic_to_str(self.sym)})"

        @property
        def name(self):
            return self._dynamic_to_str(self.sym)

        def _dynamic_to_str(self, obj: Any) -> Optional[str]:
            if isinstance(obj, str):
                return obj
            import torch

            if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
                return obj.__name__
            if isinstance(obj, torch.export.dynamic_shapes._Dim):
                return obj.__name__
            if isinstance(obj, (torch.SymInt, torch.SymFloat)):
                if isinstance(obj.node, str):
                    return obj.node
                i = obj.node._expr
                if "sympy" in str(type(i)):
                    return str(i).replace(" ", "")
                return None
            raise AssertionError(f"Unexpected type {type(obj)} to convert into string")

    class WrapDim:
        """Wraps a string considered as a ``torch.export.Dim``."""

        def __init__(self, name: str):
            self.name = name

        def __repr__(self) -> str:
            return f"WrapDim({self.name!r})"

        @property
        def name_as_string(self):
            if isinstance(self.name, str):
                return self.name
            if self.name.__class__.__name__ == "Dim":
                # It should be torch.export.dynamic_shapes.Dim
                return self.name.__name__
            raise AssertionError(
                f"Unable to return the dimension as a string type is "
                f"{type(self.name)}, name={self.name!r}"
            )

    class InitializerInfo:
        """
        Tracks the location where the initializer was created.

        :param name: initializer name
        :param source: information
        :param same_as: same as an existing initializers
        """

        def __init__(self, name: str, source: str, same_as: Optional[str] = None):
            self.name = name
            self.source = source
            self.same_as = same_as

        def __repr__(self) -> str:
            if self.same_as:
                return (
                    f"InitializerInfo({self.name!r}, source={self.source!r}, "
                    f"same_as={self.same_as!r})"
                )
            return f"InitializerInfo({self.name!r}, source={self.source!r})"

        def add_source(self, source: str):
            """
            Adds other sources.
            """
            self.source += f"##{source}"

    # Size of a tensor kept in the onnx file and not stored as exrternal weight.
    SMALL_TENSOR = 1024

    _op_type_element_wise_types = element_wise_binary_op_types()
    _op_type_element_wise_cmp_types = element_wise_op_cmp_types()
    _op_type_unary_like = unary_like_op_types()

    def __init__(
        self,
        target_opset_or_existing_proto: Union[
            int, Dict[str, int], GraphProto, ModelProto, FunctionProto
        ],
        input_names: Optional[Sequence[str]] = None,
        as_function: bool = False,
        optimization_options: Optional[OptimizationOptions] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        ir_version: Optional[int] = None,
        verbose: int = 0,
        infer_shapes_options: InferShapesOptions = InferShapesOptions.NONE,
        raise_list: Optional[Set[str]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        local_domain: str = "local_function",
        signature: Optional[Any] = None,
        check_empty_source: bool = False,
        graph_module: Optional["torch.fx.GraphModule"] = None,  # noqa: F821
        exe_path: str = "",
        output_names: Optional[List[str]] = None,
        output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        _opsets: Optional[Dict[str, int]] = None,
        _context: Optional[Set[str]] = None,
        _parent: Optional[Union["GraphBuilder", Tuple["GraphBuilder", NodeProto]]] = None,
    ):
        import torch
        from . import TEMPLATE_TYPE

        self.TEMPLATE_TYPE = TEMPLATE_TYPE

        self.torch = torch
        self.maybe_disable_fake_tensor_mode = _unset_fake_temporarily
        self.optimization_options = optimization_options or OptimizationOptions(verbose=verbose)
        self.local_domain = local_domain
        self.as_function = as_function
        self.input_args = args
        self.input_kwargs = kwargs
        self.verbose = verbose
        self.ir_version = ir_version
        self._debug_msg = {"EXEPATH": exe_path}
        self.dynamic_dimensions_source = {}
        self.dynamic_dimensions_source_flat = None
        self.output_dynamic_dimensions_source_flat = None
        self.dynamic_shapes = self._pre_process_dynamic_shape(dynamic_shapes)
        self.output_dynamic_shapes = self._pre_process_dynamic_shape(output_dynamic_shapes)
        self.dynamic_objects = {}
        self.dynamic_objects_rev = {}
        self.functions = {}
        self.functions_builder = {}
        self.value_info = []
        self.raise_list = raise_list
        self._raise_list = raise_list or set()
        self.constants_computed_ = {}
        self._cache_shape = {}
        self._values = {}
        self._dynamic_alias = {}
        self.constants_node_ = {}
        self.constants_alias_ = {}
        self.graph_module = graph_module
        self._events = {}
        self.signature = signature
        self.check_empty_source = check_empty_source
        self.user_defined_output_names = output_names or []
        self._context = _context or set()
        self._do_not_turn_constant_initializers = False

        if isinstance(_parent, tuple):
            self._parent = _parent[0]
            self._parent_node = _parent[1]
        else:
            self._parent = _parent

        self.nodes = []
        self.initializers_dict = {}
        self.initializers_dict_sources = {}
        self.inputs = []
        self.outputs = []

        self._known_shapes = {}
        self._known_types = {}
        self._known_ranks = {}
        self._known_devices = {}
        self._known_torch_value = {}
        self._known_names = set()
        self._known_sequences = {}
        self._unique_names = set()
        self._unique_node_names = set()
        self._known_value_shape = {}
        self._dynamic_examples = {}
        self._parameter_renaming = {}
        self._parameter_norename = set()
        self.constants_ = {}
        self.op = Opset(self)
        self.anyop = Opset(self, allow_unknown=True)

        self._debug_null_shape = int(os.environ.get("NULLSHAPE", "0"))
        self._debug_stop = os.environ.get("ONNXSTOP", "#?#")
        self._debug_stop_shape = os.environ.get("ONNXSTOPSHAPE", "#?#")
        self._debug_stop_type = os.environ.get("ONNXSTOPTYPE", "#?#")
        self._debug_stop_sequence = os.environ.get("ONNXSTOPSEQUENCE", "#?#")
        self._debug_get_constant = int(os.environ.get("ONNXCST", "0"))
        self._debug_local_function = int(os.environ.get("ONNXFUNC", "0"))
        self._debug_value_shape = os.environ.get("ONNXSTOPVALUESHAPE", "")
        self._debug_node_output = os.environ.get("ONNXSTOPOUTPUT", "")
        self._debug_node_type = os.environ.get("ONNXNODETYPE", "")
        self._debug_quiet = int(os.environ.get("ONNXQUIET", "0"))
        self._debug_shape_missing = int(os.environ.get("ONNXSHAPECOMPUTE", "0"))
        self._debug_constant_folding = int(os.environ.get("ONNXCONSTANTFOLD", "0"))
        self._debug_foldnot = int(os.environ.get("ONNXFOLDNOT", "0"))
        self._debug_dyn_dim = set(os.environ.get("ONNXDYNDIM", "").split(","))
        self.CONTINUE = int(os.environ.get("CONTINUE", "0"))
        if self._debug_dyn_dim == {""}:
            self._debug_dyn_dim = set()

        self.time_evaluation_constants_ = 0
        self.statistics_ = {}
        self.constraints_ = {}
        self._registered_users = {}
        self.was_inputs_renamed = input_names is not None and input_names
        self.update_dynamic_shape_when_input_name_is_defined = False
        self._implicit_decisions = []

        assert dynamic_shapes is None or isinstance(dynamic_shapes, (dict, tuple)), (
            f"dynamic_shapes is expected to be empty or a dictionary or a tuple "
            f"not {type(dynamic_shapes)}, dynamic_shapes={dynamic_shapes}"
        )

        if self.dynamic_shapes:
            self._register_dynamic_object_from_dynamic_shapes()
        if self.output_dynamic_shapes:
            self._register_dynamic_object_from_dynamic_shapes(output=True)

        if isinstance(infer_shapes_options, bool):
            infer_shapes_options = (
                InferShapesOptions.ONNX if infer_shapes_options else InferShapesOptions.NONE
            )
        if isinstance(target_opset_or_existing_proto, (int, dict)):
            # starts a model from nothing
            assert (
                not infer_shapes_options
            ), "infer_shapes_options is used if an existing model is loaded"
            self.opsets = (
                {"": target_opset_or_existing_proto}
                if isinstance(target_opset_or_existing_proto, int)
                else target_opset_or_existing_proto
            )
            self.input_names = input_names or []
            self.current_input = 0
            self._unique_names = set(self.input_names)
        elif isinstance(target_opset_or_existing_proto, (GraphProto, ModelProto, FunctionProto)):
            # loads a model from nothing
            if input_names:
                raise ValueError("input_names must be empty if the input is an existing model.")
            self.current_input = len(self.inputs)
            if isinstance(target_opset_or_existing_proto, GraphProto):
                assert (
                    _opsets is not None
                ), "_opsets must be specified if the input is a GraphProto"
                self.opsets = _opsets
            else:
                assert _opsets is None, "_opsets must be None if the input is not a GraphProto"
            self._update_structures_with_proto(
                target_opset_or_existing_proto, infer_shapes_options
            )
            self.constant_folding(
                (optimization_options or OptimizationOptions()).constant_folding,
                convert_into_initializer=False,
            )
            if not isinstance(target_opset_or_existing_proto, FunctionProto):
                self._update_shape_types_with_proto(
                    target_opset_or_existing_proto, infer_shapes_options
                )
        else:
            raise NotImplementedError(f"{type(target_opset_or_existing_proto)} is not supported.")

    def has_exact_same_constant_in_context(self, name: str) -> Optional[bool]:
        """
        Tells if this constant already exiting in the context of the main graph.
        """
        cst = self.is_constant(name)
        cstp = self._parent.is_constant_part_of_previous_context(name)
        if cst and cstp:
            # Maybe it the same constant and then it is ok.
            # Should be compare (only if the cost if not two much)
            if (
                self.has_shape(name)
                and self.has_type(name)
                and self._parent.has_shape(name)
                and self._parent.has_type(name)
            ):
                shape1 = self.get_shape(name)
                shape2 = self._parent.get_shape(name)
                if shape1 != shape2:
                    return False
                if self.get_type(name) != self._parent.get_type(name):
                    return False
                n_elem = np.prod(shape1)
                if n_elem < 128:
                    # Then the code is not much.
                    cst1 = self.get_constant(name, exc=False, computed_value=True)
                    cst2 = self.get_constant_from_parent(name, exc=False, computed_value=True)
                    if cst1 is not None and cst2 is not None:
                        d = cst1 - cst2
                        if d.min() == d.max() == 0:
                            return True
                else:
                    return None
        return False

    def do_not_turn_constant_initializers_maybe_because_of_showing(self, name: str) -> bool:
        """
        Constants are turned into initializers in the main graph.
        It is not possible in function or not always safe inside subgraphs
        (due to shadowing).
        """
        if self._do_not_turn_constant_initializers:
            return True
        if self._parent:
            # The current behaviour could be improved because
            # the name could been create after this subgraph is really used
            # in the parent graph.
            if not self._parent.has_name(name):
                return False
            r = self.has_exact_same_constant_in_context(name)
            if r is not None:
                return not r
            return self._parent.do_not_turn_constant_initializers_maybe_because_of_showing(name)
        return False

    def make_subset_builder(
        self,
        input_names: List[str],
        name: str,
        domain: str,
    ) -> "GraphBuilder":
        """
        Creates a copy of the existing builder but with information reduced to the input_names
        considered as inputs.

        :param input_names: new inputs
        :param name: function name
        :param domain: domain name for the function
        :return: shortened builder
        """
        # new dynamic_shapes
        ds2 = []
        for n in input_names:
            assert not self.is_sequence(n), (
                f"Input {n!r} is sequence but that's not yet implemented"
                f"{self.get_debug_msg()}"
            )
            if not self.has_shape(n):
                ds2 = (None,)
                break
            shape = self.get_shape(n)
            ds = {i: s for i, s in enumerate(shape) if isinstance(s, str)}
            ds2.append(ds)
        ds2 = tuple(ds2)

        new_builder = GraphBuilder(
            target_opset_or_existing_proto=self.opsets,
            input_names=input_names,
            as_function=True,
            optimization_options=self.optimization_options,
            ir_version=self.ir_version,
            verbose=max(self.verbose - 1, 0),
            infer_shapes_options=False,
            raise_list=self.raise_list,
            local_domain=self.local_domain,
            dynamic_shapes=ds2,
            _parent=self,
        )
        new_builder.dynamic_objects = self.dynamic_objects.copy()

        for n in input_names:
            assert not self.is_sequence(n), (
                f"Input {n!r} is sequence but that's not yet implemented"
                f"{self.get_debug_msg()}"
            )
            new_builder.make_tensor_input(
                n,
                self.get_type(n),
                self.get_shape(n) if self.has_shape(n) else None,
                is_dimension=self.get_is_dimension(n),
                marker=f"make_subset_builder-{name}",
            )
        for k, v in self.functions.items():
            if v.domain != domain:
                new_builder.functions[k] = v
        return new_builder

    @classmethod
    def _pre_process_dynamic_shape(
        cls,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        unique_names: Optional[Set[str]] = None,
    ) -> Optional[Union[Dict[str, Any], Tuple[Any]]]:
        """Replaces Hints by true DynamicShapes."""
        import torch

        if isinstance(dynamic_shapes, torch.export.dynamic_shapes._Dim):
            return dynamic_shapes
        if isinstance(dynamic_shapes, str):
            # Not allowed by pytorch but allowed here.
            return cls.WrapDim(dynamic_shapes)
        if isinstance(dynamic_shapes, cls.WrapDim):
            return dynamic_shapes
        if not dynamic_shapes:
            return dynamic_shapes
        if unique_names is None:
            unique_names = set()
        if isinstance(dynamic_shapes, (tuple, list)):
            return type(dynamic_shapes)(
                cls._pre_process_dynamic_shape(i, unique_names) for i in dynamic_shapes
            )
        if isinstance(dynamic_shapes, dict):
            return {
                k: cls._pre_process_dynamic_shape(v, unique_names)
                for k, v in dynamic_shapes.items()
            }
        if dynamic_shapes in (torch.export.Dim.DYNAMIC, torch.export.Dim.AUTO):
            i = 0
            name = "DYN0"
            while name in unique_names:
                i += 1
                name = f"DYN{i}"
            unique_names.add(name)
            return torch.export.Dim(name)
        raise AssertionError(f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes")

    def _register_dynamic_object_from_dynamic_shapes_dict(self, input_name, shape_dict):
        # example:
        # args_0 {0: <class '._bash_bench_model_runner.batch'>}

        def handle_dim(_k, _v, input_name=input_name):
            if isinstance(_v, self.torch.SymInt):
                self.make_dynamic_object(
                    _v.__name__,
                    _v,
                    axis=_k,
                    input_name=input_name,
                )
            elif isinstance(_v, self.torch.export.dynamic_shapes._DerivedDim):
                self.make_dynamic_object(
                    _v.__name__,
                    self.torch.SymInt(_v.__name__),
                    axis=_k,
                    input_name=input_name,
                )
                # It should be recursive.
                # maybe add a constraint here
                self.make_dynamic_object(
                    _v.root.__name__,
                    self.torch.SymInt(_v.root.__name__),
                    axis=None,
                    input_name=None,
                )
            elif isinstance(_v, self.torch.export.dynamic_shapes._Dim):
                self.make_dynamic_object(
                    _v.__name__,
                    self.torch.SymInt(_v.__name__),
                    axis=_k,
                    input_name=input_name,
                )
            elif isinstance(_v, self.WrapDim):
                self.make_dynamic_object(
                    _v.name,
                    self.torch.SymInt(_v.name),
                    axis=_k,
                    input_name=input_name,
                )
            elif _v is not None:
                raise AssertionError(
                    f"Unexpected type {type(_v)} for dynamic "
                    f"dimension for axis {_k}, input_name={input_name!r}, "
                    f"pos_vv={shape_dict!r}, "
                    f"self.dynamic_shapes={self.dynamic_shapes}, "
                    f"self.output_dynamic_shapes={self.output_dynamic_shapes}"
                )

        if not isinstance(shape_dict, dict):
            # A scalar, no axis.
            handle_dim(None, shape_dict)
            return

        for _k, _v in shape_dict.items():
            handle_dim(_k, _v)

    def _register_dynamic_object_from_dynamic_shapes(self, output: bool = False):
        dynamic_shapes = self.output_dynamic_shapes if output else self.dynamic_shapes
        assert (
            dynamic_shapes is not None
        ), f"Call this method if dynamic_shapes is not None, output={output!r}"
        if not output:
            self.update_dynamic_shape_when_input_name_is_defined = not isinstance(
                dynamic_shapes, dict
            )
        _prefixes = []

        def _unfold(obj, prefix=None):
            assert prefix is None or isinstance(
                prefix, (str, tuple, int)
            ), f"Unexpected type prefix={prefix!r}, obj={obj!r}"
            if isinstance(obj, dict) and all_int(obj):
                assert prefix is not None, f"prefix is None for obj={obj}"
                _prefixes.append(prefix)
                return [(prefix, obj)]
            if isinstance(obj, dict):
                res = []
                for k, v in obj.items():
                    if v is None:
                        continue
                    p = (
                        k
                        if prefix is None
                        else ((prefix, k) if isinstance(prefix, (str, int)) else (*prefix, k))
                    )
                    u = _unfold(v, p)
                    res.extend(u)
                return res
            if isinstance(obj, (list, tuple)):
                res = []
                for k, v in enumerate(obj):
                    if v is None:
                        continue
                    p = (
                        k
                        if prefix is None
                        else ((prefix, k) if isinstance(prefix, (str, int)) else (*prefix, k))
                    )
                    u = _unfold(v, p)
                    res.extend(u)
                return res
            if isinstance(obj, (str, self.torch.export.dynamic_shapes._Dim)):
                _prefixes.append(prefix)
                return [(prefix, obj)]
            raise TypeError(f"Unexpected type {type(obj)} and prefix={prefix!r}")

        seq_dynamic_shapes = _unfold(dynamic_shapes)

        for input_name_or_position, pos_vv in seq_dynamic_shapes:
            assert isinstance(pos_vv, (dict, str, self.torch.export.dynamic_shapes._Dim)), (
                f"Unexpected pos_vv={pos_vv} (type is {type(pos_vv)}), "
                f"input_name_or_position={input_name_or_position}"
            )
            self._register_dynamic_object_from_dynamic_shapes_dict(input_name_or_position, pos_vv)
        if not output:
            self.dynamic_dimensions_source_flat = _prefixes
        else:
            self.output_dynamic_dimensions_source_flat = _prefixes

    def add_stat(self, kind: str, name: str):
        """Increments a counter."""
        if kind not in self.statistics_:
            self.statistics_[kind] = {}
        stat = self.statistics_[kind]
        if name not in stat:
            stat[name] = 1
        else:
            stat[name] += 1

    @property
    def output_names(self) -> List[str]:
        return [o.name for o in self.outputs]

    def empty_copy(
        self, as_function: bool = False, constant_size: int = 2**24, _shapable: bool = True
    ) -> "GraphBuilder":
        """
        Creates an empty copy but with the same opsets.
        This is used in pattern matching.

        :param as_function: to create a function
        :param constant_size: maximum size of a constant
        :param _shapable: disable check on shapes if enable
        :return: new graph
        """
        assert isinstance(as_function, bool), f"wrong type {type(as_function)} for as_function"
        opt = OptimizationOptions(
            constant_size=constant_size,
            constant_fusing=False,
            remove_identity=False,
            remove_duplicated_shape=False,
            patterns=None,
        )
        g = GraphBuilder(
            self.opsets.copy(),
            verbose=self.verbose,
            ir_version=self.ir_version,
            as_function=as_function,
            optimization_options=opt,
        )
        if not _shapable:
            g._debug_shape_missing = False
        return g

    def make_key(self, value: Any) -> Optional[Tuple[Union[str, int], ...]]:
        """
        Builds a key identifying a value.
        Returns None if it is not possible.
        """
        if isinstance(value, TensorProto):
            if uses_external_data(value):
                key = None
                for info in value.external_data:
                    if info.key == "location":
                        key = info.value
                        break
                assert key is not None, (
                    f"External tensor {value.name!r} has no location, "
                    f"external_data is {value.external_data}"
                )
                return None
            return self.make_key(onh.to_array(value))
        if isinstance(value, self.torch.Tensor):
            if value.dtype == self.torch.int64 and value.numel() < 8:
                return self.make_key(value.detach().cpu().numpy())
            return None
        if isinstance(value, int):
            return int, value
        if isinstance(value, np.ndarray):
            if value.size < self.SMALL_TENSOR:
                return (str(value.dtype), tuple(value.shape), tuple(value.ravel().tolist()))
        return None

    def pretty_tensor(self, tensor: Any) -> str:
        if hasattr(tensor, "shape"):
            return f"{tensor.dtype}:{tuple(tensor.shape)}"
        return f"no pretty: {type(tensor)}"

    def pretty_node(
        self,
        node: Optional[NodeProto],
        limit: int = 80,
        short: bool = True,
        shape: bool = False,
    ) -> str:
        """
        Pretty rendering for a node.

        :param node: node to render
        :param limit: to show type and shapes after the limit
        :param short: do not display shape information on the left
        :param shape: show shape information below
        :return: string
        """
        if node is None:
            return "None"
        if shape:
            st = []
            for i in node.input:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            st.append("->")
            for i in node.output:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            shape_info = " ".join(st)
        else:
            shape_info = ""

        if node.op_type in {"Shape", "Reshape", "Unsqueeze", "Squeeze", "Cast", "Transpose"}:
            atts = []
            for att in node.attribute:
                if att.type == AttributeProto.INT:
                    atts.append(f"{att.name}={att.i}")
                elif att.type == AttributeProto.FLOAT:
                    atts.append(f"{att.name}={att.f}")
                elif att.type == AttributeProto.INTS:
                    atts.append(f"{att.name}={att.ints}")
            suffix = ""
            if len(node.input) == 2 and self.is_constant(node.input[1]):
                cst = self.get_constant(node.input[1], exc=False)
                if cst is not None:
                    suffix = f", {cst.tolist()}"
            satts = ", ".join(atts)
            if satts:
                satts = f", {satts}"
            text = (
                f"{node.op_type}(.{suffix}{satts}): {', '.join(node.input)} -> "
                f"{', '.join(node.output)}"
            )
        else:
            text = (
                (
                    f"{node.op_type}[{node.domain}]: "
                    f"{', '.join(node.input)} -> {', '.join(node.output)}"
                )
                if node.domain
                else f"{node.op_type}: {', '.join(node.input)} -> {', '.join(node.output)}"
            )

        if shape_info:
            text = f"{text} ## {shape_info}"
        if short:
            return text
        add = " " * abs(80 - len(text))
        text += add
        info = []
        for o in node.output:
            t = f"T{self.get_type(o)}" if self.has_type(o) else ""
            s = " x ".join(map(str, self.get_shape(o))) if self.has_shape(o) else ""
            info.append(": ".join([t, s]))
        if node.name:
            s = f"{text}|{' '.join(info)}"
            return f"{s}{' ' * (110 - len(s))}- {node.name}"
        return f"{text}|{' '.join(info)}"

    def pretty_text(self, add_fx_graph: bool = False, recursive: bool = True) -> str:
        """
        Pretty rendering of the graph.

        :param add_fx_graph: add the fx Graph to the rendering
        :param recursive: dig into subgraphs (not implemented yet)
        :return: string
        """

        def _d(d1):
            if isinstance(d1, self.torch.SymInt):
                return f"SymInt({self._torch_sym_int_to_str(d1)})"
            if isinstance(d1, self.torch.SymBool):
                return f"SymBool({self._torch_sym_int_to_str(d1)})"
            if isinstance(d1, (self.WrapDim, self.WrapSym)):
                return repr(d1)
            if isinstance(d1, self.torch.export.dynamic_shapes._DerivedDim):
                return f"_DerivedDim({d1.__name__})"
            if isinstance(d1, self.torch.export.dynamic_shapes._Dim):
                return f"_Dim({d1.__name__})"
            if isinstance(d1, list):
                s = ", ".join(map(_d, d1))
                return f"[{s}]"
            if isinstance(d1, tuple):
                s = ", ".join(map(_d, d1))
                return f"({s})"
            if isinstance(d1, dict):
                s = ", ".join(f"{k}:{_d(d)}" for d in d1.items())
                return f"{{{s}}}"
            if isinstance(d1, (str, int)):
                return f"{d1!r}"
            if d1 is None:
                return "None"
            raise AssertionError(f"Unexpected type for {type(d1)}")

        def _v(v):
            if hasattr(v, "shape"):
                shape = " x ".join(map(str, v.shape))
                dtype = str(v.dtype)
                prefix = "A" if isinstance(v, np.ndarray) else "T"
                d = "G" if hasattr(v, "is_cuda") and v.is_cuda else "C"
                prefix = f"{d}{prefix}"
            elif isinstance(v, TensorProto):
                shape = tuple(v.dims)
                dtype = v.data_type
                prefix = "CP"
            else:
                shape = "?"
                dtype = "?"
                prefix = ""
            return f"{prefix}{dtype}: {shape}"

        def _io(o, prefix):
            text = f"{prefix}: {o}"
            add = " " * abs(80 - len(text))
            text += add
            t = f"T{self.get_type(o)}" if self.has_type(o) else ""
            s = " x ".join(map(str, self.get_shape(o))) if self.has_shape(o) else ""
            return f"{text}|{': '.join([t,s])}"

        rows = [""]
        # signature
        if self.signature:
            rows.append(string_signature(self.signature))
        # dynamic shapes
        for k, v in assert_sorted(self.dynamic_objects.items()):
            rows.append(f"dyn---: {k} -> {_d(v)}")
        for k, v in assert_sorted(self.dynamic_objects_rev.items()):
            rows.append(f"dynrev: {k} -> {_d(v)}")
        for k, v in assert_sorted(self.dynamic_dimensions_source.items()):
            rows.append(f"dynsrc: {k} -> {_d(v)}")
        for k, v in assert_sorted(self._dynamic_alias.items()):
            rows.append(f"dynals: {k} -> {_d(v)}")
        if self.dynamic_shapes:
            if isinstance(self.dynamic_shapes, dict):
                for k, v in assert_sorted(self.dynamic_shapes.items()):
                    rows.append(f"d-dynshp: {k} -> {_d(v)}")
            else:
                for k, v in enumerate(self.dynamic_shapes):
                    rows.append(f"t-dynshp: {k} -> {_d(v)}")
        # the rest
        for k, v in self.opsets.items():
            rows.append(f"opset: {k}: {v}")
        for k, v in self.initializers_dict.items():
            if k in self.initializers_dict_sources and self.initializers_dict_sources[k].source:
                t = f"init: {k}: {_v(v)}"
                rows.append(
                    f"{t}{' '*max(2,70-len(t))} -- {self.initializers_dict_sources[k].source}"
                )
            else:
                rows.append(f"init: {k}: {_v(v)}")
        for k, v in self.initializers_dict_sources.items():
            if (
                k not in self.initializers_dict_sources
                or not self.initializers_dict_sources[k].source
            ):
                rows.append(f"init-source: {k}: {v}")
        for i in self.input_names:
            rows.append(_io(i, "input:"))
        for node in self.nodes:
            rows.append(self.pretty_node(node, short=False))
        for i in self.output_names:
            rows.append(_io(i, "output:"))
        for k, f in self.functions.items():
            rows.append("")
            rows.append(f"FUNCKEY: {k}")
            rows.append(f"FUNC {f.name}[{f.domain}]: {f.input} -> {f.output}")
            for op in f.opset_import:
                n = op.domain if op.domain else "''"
                rows.append(f"  opset: {n}: {op.version}")
            for node in f.node:
                rows.append(f"  {self.pretty_node(node, short=False)}")
            if k in self.functions_builder:
                rows.append(
                    self.functions_builder[k].pretty_text(
                        add_fx_graph=add_fx_graph, recursive=False
                    )
                )
        if add_fx_graph:
            fx = self._debug_msg.get("process.graph_module.graph")
            if fx:
                rows.append("-- FX.GRAPH-- ")
                rows.append(str(fx))
                rows.append("--")
                for node in fx.nodes:
                    val = node.meta.get("val", None)
                    if val is None:
                        rows.append(f"{node.op}:[{node.name}:{node.target}]")
                    elif isinstance(val, list):
                        rows.append(f"{node.op}:[{node.name}:{node.target}] - list")
                    elif isinstance(val, self.torch.Tensor):
                        rows.append(
                            f"{node.op}:[{node.name}:{node.target}] - {val.dtype}:{val.shape}"
                        )
                    else:
                        rows.append(f"{node.op}:[{node.name}:{node.target}] - {type(val)}")
        return "\n".join(rows)

    @property
    def main_opset(self):
        "Returns the opset for the main domain (assuming it is used)."
        return self.opsets[""]

    def get_opset(self, domain: str, exc: bool = True) -> int:
        """
        Returns the opset version for a specific domain.

        :param domain: domain name
        :param exc: raise an exception if missing
        :return: version
        """
        assert (
            exc or domain in self.opsets
        ), f"Domain {domain!r} is not registered{self.get_debug_msg()}."
        return self.opsets.get(domain, None)

    def add_domain(self, domain: str, version: int = 1):
        """
        Adds a domain to the list of supported ones.
        Checks the version is the same if it exists.
        """
        if domain in self.opsets:
            assert version == self.opsets[domain], (
                f"Version mismatch for domain={domain!r}, current is "
                f"{self.opsets[domain]}, new is {version}{self.get_debug_msg()}"
            )
            return
        self.opsets[domain] = version

    def _hash(self) -> str:
        return make_hash(self)

    def _get_tensor_shape(self, proto: Union[NodeProto, TensorProto]) -> STATIC_SHAPE:
        if isinstance(proto, TensorProto):
            return tuple(proto.dims)
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                assert att.type != AttributeProto.GRAPH, (
                    f"Unexpected attribute type for attribute {att.name!r} "
                    f"attribute list is {[a.name for a in proto.attribute]} "
                    f"in node type {proto.op_type!r}, name={proto.name!r}, "
                    f"doc_string={proto.doc_string!r}"
                )
                if att.name in ("value_float", "value_int"):
                    return tuple()
                if att.name == "value_floats":
                    return (len(att.floats),)
                if att.name == "value_ints":
                    return (len(att.ints),)
                if att.name == "value":
                    return tuple(att.t.dims)
        raise TypeError(
            f"Unexpected or unsupported scenario type {type(proto)}: "
            f"{proto}, attribute={proto.attribute}."
        )

    def _get_tensor_type(self, proto: Union[NodeProto, TensorProto]) -> int:
        if isinstance(proto, TensorProto):
            return proto.data_type
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return TensorProto.FLOAT
                if att.name == "value_int":
                    return TensorProto.INT64
                if att.name == "value_floats":
                    return TensorProto.FLOAT
                if att.name == "value_ints":
                    return TensorProto.INT64
                if att.name == "value":
                    return att.t.data_type
        raise ValueError(f"Unexpected type or value {type(proto)}: {proto}.")

    def is_constant(self, name: str) -> bool:
        """Tells if a result is a constant."""
        return name in self.constants_

    def compute_constant(
        self, name: str, exc: bool = True, only_array: bool = False, allow_empty: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Computes a constant.

        :param name: constant name
        :param exc: raises an exception if any failure
        :param only_array: do not return TensorProto
        :param allow_empty: allow empty result
        :return: constant

        If returns None if the constant is a FakeTensor.
        """
        assert self.is_constant(name), f"Name {name!r} is not a constant"
        if name in self.initializers_dict:
            value = self.initializers_dict[name]
            assert not isinstance(
                value, tuple
            ), f"Unexpected type {type(value)} for name={name!r}"
            if only_array and isinstance(value, TensorProto):
                # Should reuse memory buffer here.
                v = onh.to_array(value)
                src = (
                    ""
                    if name not in self.initializers_dict_sources
                    or not self.initializers_dict_sources[name].source
                    else f"##{self.initializers_dict_sources[name].source}"
                )
                assert (
                    src.count("GraphBuilder.compute_constant/from") < 10
                ), f"Unexpected source={src!r} for initializer {name!r}{self.get_debug_msg()}"

                self.add_initializer(
                    name,
                    v,
                    existing=True,
                    allow_empty=allow_empty,
                    source=f"GraphBuilder.compute_constant/from({name}){src}",
                )
                return v, None
            if isinstance(value, self.torch._subclasses.fake_tensor.FakeTensor):
                assert not self._debug_constant_folding, (
                    f"Unable to compute constant because value is a FakeTensor"
                    f"{string_type(value, with_shape=True)}"
                    f"in node {self.pretty_node(v)}{self.get_debug_msg()}"
                )
                return None, None
            return value, None
        return _InferenceRuntime.compute_constant(
            self, name=name, exc=exc, only_array=only_array, allow_empty=allow_empty
        )

    def get_constant(
        self,
        name: str,
        exc: bool = True,
        computed_value: bool = False,
        as_shape: bool = False,
        multiple_outputs: bool = False,
    ) -> Union[np.ndarray, NodeProto]:
        """
        The method returns the constant *name*. It is a tensor (numpy array)
        or a NodeProto which must be evaluated.
        If *computed_value* is True, the NodeProto is evaluated with the
        ReferenceEvaluator.

        :param name: constant name
        :param exc: raise an exception if anything is impossible to do
        :param computed_value: compute the value if not a constant
        :param as_shape: returns a tuple for a shape
        :param multiple_outputs: allow multiple outputs
        :return: value
        """
        if self._debug_get_constant:
            print(
                f"[GraphBuilder-{self._hash()}.get_constant] name={name}, "
                f"computed_value={computed_value}, as_shape={as_shape}, "
                f"exc={exc}"
            )
        if as_shape:
            assert not multiple_outputs, "multiple outputs not allowed with as_shape=True"
            res = self.get_constant(name, exc, computed_value=computed_value, as_shape=False)
            if res is None:
                assert not exc, (
                    f"No constant for name={name!r}, exc={exc}, "
                    f"computed_value={computed_value}, as_shape={as_shape}, "
                    f"multiple_outputs={multiple_outputs}{self.get_debug_msg()}"
                )
                if self._debug_get_constant:
                    print(f"[GraphBuilder-{self._hash()}.get_constant]   A: None, name={name!r}")
                return None
            assert multiple_outputs or not isinstance(
                res, tuple
            ), f"Multiple output is not allowed but type is {type(res)} for name={name!r}"
            new_res = []
            with self.maybe_disable_fake_tensor_mode():
                for i in res:
                    new_res.append(i if isinstance(i, str) else int(i))
            if self._debug_get_constant:
                print(f"[GraphBuilder-{self._hash()}.get_constant]   SHAPE: {tuple(new_res)}")
            return tuple(new_res)

        if not self.is_constant(name):
            if exc and not self._parent:  # subgraphs not fully working
                raise ValueError(f"Result {name!r} is not a constant{self.get_debug_msg()}")
            if self._debug_get_constant:
                print(f"[GraphBuilder-{self._hash()}.get_constant]   C: None, name={name!r}")
            return None
        possible_value = self.constants_[name]
        if name in self.constants_computed_:
            value = self.constants_computed_[name]
            assert value is not None, f"Constant is empty for name={name!r}"
            assert multiple_outputs or not isinstance(
                value, tuple
            ), f"Multiple output is not allowed but type is {type(value)} for name={name!r}"
            if self._debug_get_constant:
                print(
                    f"[GraphBuilder-{self._hash()}.get_constant]   D: value: "
                    f"{type(value)}, name={name!r}"
                )
            assert (
                not exc or value is not None
            ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
            return value

        if possible_value is not None:
            assert isinstance(
                possible_value, (np.ndarray, self.torch.Tensor, NodeProto)
            ), f"Unexpected type {type(possible_value)} for a constant{self.get_debug_msg()}"
            if computed_value and isinstance(possible_value, NodeProto):
                res, _ = self.compute_constant(name, exc=exc)
                if res is None:
                    # The constant is too big to be computed.
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.get_constant]   E: None, name={name!r}"
                        )
                    return None

                assert multiple_outputs or not isinstance(res, tuple), (
                    f"Multiple output is not allowed but type is {type(res)} "
                    f"for name={name!r}"
                )
                assert (
                    not multiple_outputs
                ), f"get_constants not implemented when multiple_outputs=True, name={name!r}"
                if not isinstance(res, tuple):
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.get_constant]   "
                            f"F: tuple, name={name!r}"
                        )
                    return res

                assert isinstance(res, tuple), (
                    f"Expecting multiple outputs for name={name!r}, "
                    f"op_type={possible_value.op_type!r}"
                )
                if len(res) == 1:
                    assert multiple_outputs or not isinstance(value, tuple), (
                        f"Multiple output is not allowed but type is {type(value)} "
                        f"for name={name!r}"
                    )
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.get_constant]   "
                            f"G: {type(res[0])}, name={name!r}"
                        )
                    assert (
                        not exc or res[0] is not None
                    ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
                    return res[0]

                index = list(possible_value.output).index(name)
                value = res[index]
                assert value is not None, f"Constant is empty for name={name!r}"
                assert multiple_outputs or not isinstance(value, tuple), (
                    f"Multiple output is not allowed but type is {type(value)} "
                    f"for name={name!r}"
                )
                if self._debug_get_constant:
                    print(
                        f"[GraphBuilder-{self._hash()}.get_constant]   "
                        f"H: {type(value)}, name={name!r}"
                    )
                assert (
                    not exc or value is not None
                ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
                return value

            assert possible_value is not None, f"Constant is empty for name={name!r}"
            assert multiple_outputs or not isinstance(possible_value, tuple), (
                f"Multiple output is not allowed but type is {type(possible_value)} "
                f"for name={name!r}"
            )
            if self._debug_get_constant:
                print(
                    f"[GraphBuilder-{self._hash()}.get_constant]   "
                    f"I: {type(possible_value)}, name={name!r}"
                )
            assert (
                not exc or possible_value is not None
            ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
            return possible_value

        if name in self.initializers_dict:
            value = self.initializers_dict[name]
        else:
            # So it should be stored in a parent.
            value = self.get_constant_from_parent(
                name,
                exc=False,
                computed_value=computed_value,
                as_shape=as_shape,
                multiple_outputs=multiple_outputs,
            )
            if value is None and exc:
                raise ValueError(
                    f"Result {name!r} was never evaluated within method 'constant_folding', "
                    f"known initializers={sorted(self.initializers_dict)}{self.get_debug_msg()}"
                )
            if self._debug_get_constant:
                print(f"[GraphBuilder-{self._hash()}.get_constant]   J: None, name={name!r}")
            if value is None:
                return value

        if isinstance(value, np.ndarray):
            if self._debug_get_constant:
                if value.size < 10:
                    print(
                        f"[GraphBuilder-{self._hash()}.get_constant] "
                        f"  K: np.ndarray {value.shape}, "
                        f"{value.dtype}, {value}"
                    )
                else:
                    print(
                        f"[GraphBuilder-{self._hash()}.get_constant] "
                        f"  K: np.ndarray {value.shape}, "
                        f"{value.dtype}"
                    )
            return value

        if isinstance(value, self.torch.Tensor):
            with self.maybe_disable_fake_tensor_mode():
                v = value.detach().cpu()
                self.constants_computed_[name] = v
            assert not multiple_outputs, f"Multiple output is not allowed for name={name!r}"
            if self._debug_get_constant:
                print(
                    f"[GraphBuilder-{self._hash()}.get_constant] "
                    f"  L: nptorch.Tensor {v.shape}, {v.dtype}"
                )
            return v

        if isinstance(value, TensorProto):
            if uses_external_data(value):
                if exc:
                    raise TypeError(
                        f"Tensor is using external data, data_type={value.data_type}, "
                        f"dims={value.dims}"
                    )
                if self._debug_get_constant:
                    print("[GraphBuilder.get_constant]   M: None")
                return None
            v = onh.to_array(value)
            assert not multiple_outputs, f"Multiple output is not allowed for name={name!r}"
            self.constants_computed_[name] = v
            if self._debug_get_constant:
                print("[GraphBuilder.get_constant]   O: TensorProto")
            return v

        if isinstance(value, np.float32):
            # This should not be needed.
            if self._debug_get_constant:
                print(f"[GraphBuilder-{self._hash()}.get_constant]   P: np.float32 {value}")
            return np.array(value, dtype=np.float32)

        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def get_constant_from_parent(
        self,
        name: str,
        exc: bool = True,
        computed_value: bool = False,
        as_shape: bool = False,
        multiple_outputs: bool = False,
    ) -> Union[np.ndarray, NodeProto]:
        """Tells if this name is part of the previous context."""
        if not self._parent:
            assert (
                not exc
            ), f"No parent found, unable to retrieve constant name {name!r}{self.get_debug_msg()}"
            return None
        value = self._parent.get_constant(
            name,
            exc=False,
            computed_value=computed_value,
            as_shape=as_shape,
            multiple_outputs=multiple_outputs,
        )
        if value is not None:
            return value
        return self._parent.get_constant_from_parent(
            name,
            exc=False,
            computed_value=computed_value,
            as_shape=as_shape,
            multiple_outputs=multiple_outputs,
        )

    def is_constant_part_of_previous_context(self, name: str) -> bool:
        """Tells if this name is part of the previous context and a constant as well."""
        if self.is_constant(name):
            return True
        if self._parent:
            return self._parent.is_constant_part_of_previous_context(name)
        return False

    def is_sequence(self, name: str) -> bool:
        """Tells if a result is a sequence."""
        if name in self._known_sequences:
            return True
        assert self.has_name(name), f"Unknown name={name!r}{self.get_debug_msg()}"
        return False

    def get_sequence(self, name: str) -> Dict[str, Any]:
        """Returns sequence information"""
        assert (
            name in self._known_sequences
        ), f"Sequence {name!r} is not known{self.get_debug_msg()}"
        return self._known_sequences[name]

    def set_sequence(
        self,
        name: str,
        dtype: Union[int, Tuple[int, ...]],
        shapes: Optional[Tuple[DYNAMIC_SHAPE, ...]] = None,
        ranks: Optional[Tuple[int, ...]] = None,
        unknown: bool = False,
    ):
        """Defines a result as a sequence."""
        if name in (self._debug_stop_sequence, self._debug_stop_type):
            raise AssertionError(
                f"Requested stop, sequence name={name!r}, dtype={dtype}{self.get_debug_msg()}"
            )
        assert self.has_name(name), f"No result name={name!r}{self.get_debug_msg()}"
        assert isinstance(dtype, (int, tuple)), (
            f"Only one type is allowed in onnx sequences but dtype={dtype!r}, "
            f"the interpret allows multiple types for simplicity"
            f"{self.get_debug_msg()}"
        )
        if isinstance(dtype, dict):
            self._known_sequences[name] = dtype
        else:
            d = dict(dtype=dtype, shapes=shapes, ranks=ranks)
            if shapes is not None and ranks is None:
                d["ranks"] = tuple(len(s) for s in shapes)
            if name not in self._known_sequences:
                self._known_sequences[name] = d
            else:
                old = self._known_sequences[name]
                new_value = {}
                for k in ["dtype", "shapes", "ranks"]:
                    e = old.get(k, None)
                    n = new_value.get(k, None)
                    if e is None:
                        new_value[k] = n
                    elif n is None:
                        new_value[k] = e
                    else:
                        assert n == e, (
                            f"Sequence {name!r} was already declared with a different value "
                            f"for k={k!r}, existing={e!r}, new={n!r}, declared={old}, "
                            f"new={d}{self.get_debug_msg()}"
                        )
                self._known_sequences[name] = new_value

    def set_name(self, name: str, marker: str):
        """Adds a name to the list of known names."""
        assert name != "", (
            f"Empty name {name!r} cannot be registered, "
            f"marker={marker!r}{self.get_debug_msg()}"
        )
        assert len(name) == len(name.strip()), (
            f"No space should be added at the extremities of the name {name!r}"
            f"{self.get_debug_msg()}"
        )
        assert name not in self._raise_list, (
            f"Name {name!r} is one of the name declared in "
            f"the stop list, marker={marker!r}{self.get_debug_msg()}"
        )
        assert isinstance(name, str), (
            f"Unexpected type {type(name)} for name, "
            f"marker={marker!r}, existing marker={self._events[name, 'set_name']}"
            f"{self.get_debug_msg()}"
        )
        assert name not in self._known_names, (
            f"Name {name!r} already exists, marker={marker!r}, "
            f"existing marker is {self._events[name, 'set_name']}"
            f"{self.get_debug_msg()}"
        )
        self._known_names.add(name)
        self._unique_names.add(name)
        self._events[name, "set_name"] = marker

    def set_rank(self, name: str, value: int):
        """
        Sets the rank for a result.

        :param name: result name
        :param value: rank
        """
        if name == self._debug_stop or name == self._debug_stop_shape:
            # Set ONNXSTOP or ONNXSTOPSHAPE to stop here.
            raise AssertionError(f"Requested stop, name={name!r}, rank={value}")
        assert isinstance(value, int), f"Unexpected rank type {type(value)} for {name!r}"
        assert not isinstance(value, bool), f"Unexpected rank type {type(value)} for {name!r}"
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name in self._known_ranks:
            assert value == self._known_ranks[name], (
                f"Inconsistent ranks for {name!r}, previous value is "
                f"{self._known_ranks[name]}, new value is {value}{self.get_debug_msg()}"
            )
            if self.verbose > 5:
                print(f"[GraphBuilder-{self._hash()}.set_rank] (again) {name}:{value}")
            return
        self._known_ranks[name] = value
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_rank] {name}:{value}")

    def is_more_precise(self, shape: STATIC_SHAPE, base: STATIC_SHAPE) -> bool:
        assert len(shape) == len(
            base
        ), f"Cannot compare shapes with different ranks {shape} and {base}"
        for a, b in zip(shape, base):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    return False
            if isinstance(a, str) and isinstance(b, str):
                if a != b:
                    return False
        return True

    def get_is_dimension(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[STATIC_SHAPE] = None,
        n_outputs: Optional[int] = None,
        exc: bool = True,
    ) -> bool:
        """Tells if a result is a dynamic dimension or not."""
        if name in self.dynamic_objects:
            res = True
        elif name in self._known_torch_value:
            value = self._known_torch_value[name]
            if value[0] == "run_node":
                val1 = value[1]
                _exa, val = val1
                if val is not None and len(val) == 3:
                    el_type, size = val[1:]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                        self.torch.bool,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
                    if el_type in (self.torch.int64, self.torch.int32) and len(size) == 0:
                        # A single integer with no shape, it looks like a dimension.
                        # Let's assume it is. It is more efficient to consider it as
                        # a dimension.
                        return True
                    # In another case, let's assume it is not.
                    return False
                else:
                    if elem_type is not None and elem_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                        self.torch.bool,
                    ):
                        return False
                    if not self.has_type(name) or (shape is not None and len(shape) >= 2):
                        return False
                    dtype = self.get_type(name)
                    if dtype in {
                        TensorProto.FLOAT16,
                        TensorProto.FLOAT,
                        TensorProto.DOUBLE,
                        TensorProto.BFLOAT16,
                        TensorProto.BOOL,
                    }:
                        return False
                    if self.has_shape(name):
                        shape = self.get_shape(name)
                        if dtype == TensorProto.INT64 and shape == (1,):
                            return True
                    elif self.has_rank(name):
                        if self.get_rank(name) > 1:
                            return False
                if isinstance(val1[0], tuple) and len(val1[0]) >= 1:
                    v = val1[0]
                    if (
                        isinstance(v, tuple)
                        and len(v) == 3
                        and v[0] == "example_value"
                        and len(self.dynamic_objects) == 0
                    ):
                        # No dynamic shape as input, so there
                        # shoud not be any dynamic shape as output.
                        return False
                if val1 == ("", ""):
                    # Another case where it seems False.
                    return False
                raise RuntimeError(
                    f"Not implemented for name={name!r}, value={value!r} ({type(value)}), "
                    f"val1={val1}, elem_type={elem_type}, shape={shape}, n_outputs={n_outputs}"
                    f"{self.get_debug_msg()}"
                )
            elif value[0] == "call_module":
                if isinstance(value[1], tuple) and len(value[1]) == 2:
                    el_type, size = value[1]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
            if n_outputs == 1:
                # We may assume a model would not output just one dimension.
                return False
            raise RuntimeError(
                f"Not implemented for name={name!r}, value={value!r} ({type(value)}), "
                f"elem_type={elem_type}, shape={shape}, n_outputs={n_outputs}"
                f"{self.get_debug_msg()}"
            )
        elif "_INT_" in name:
            # This is most likely a dimension but not marked as such for the time being.
            return False
        else:
            if elem_type in {
                TensorProto.FLOAT16,
                TensorProto.FLOAT,
                TensorProto.DOUBLE,
                TensorProto.BFLOAT16,
                TensorProto.BOOL,
            }:
                return False
            if not exc:
                # We return false by default.
                return False
            raise RuntimeError(
                f"Unable to guess if {name!r}, elem_type={elem_type}, "
                f"shape={shape} is a dimension{self.get_debug_msg()}"
            )
        assert not res or (
            (
                elem_type is None
                or elem_type
                in {
                    TensorProto.INT64,
                    TensorProto.INT32,
                    TensorProto.UINT64,
                    TensorProto.UINT32,
                    # not a dimension but a result of a computation involving a dimension
                    TensorProto.FLOAT,
                }
            )
            and (shape is None or (isinstance(shape, tuple) and len(shape) == 0))
        ), (
            f"Inconsistent result type for name={name!r}, is_dimension={res}, "
            f"elem_type={elem_type}, shape={shape}{self.get_debug_msg()}"
        )
        return res

    def reset_types_and_shapes(self):
        """Removes shapes, and types if not an initializer."""
        for k in self._known_names:
            if self.is_constant(k):
                continue
            if self.has_type(k):
                del self._known_types[k]
            if self.has_device(k):
                del self._known_devices[k]
            if self.has_shape(k):
                del self._known_shapes[k]
            if self.has_rank(k):
                del self._known_ranks[k]
            if k in self._known_value_shape:
                del self._known_value_shape[k]

    def set_shapes_types(
        self,
        name: Union[str, "torch.fx.Node"],  # noqa: F821
        where: str,
        value: Any,
    ):
        if hasattr(name, "name"):
            name = name.name
        self._known_torch_value[name] = (where, value)

    def _torch_sym_int_to_str(self, value: "torch.SymInt") -> Union[int, str]:  #  noqa: F821
        if isinstance(value, str):
            return value
        if hasattr(value, "node") and isinstance(value.node, str):
            return f"{value.node}"

        from torch.fx.experimental.sym_node import SymNode

        if hasattr(value, "node") and isinstance(value.node, SymNode):
            # '_expr' is safer than expr
            return str(value.node._expr).replace(" ", "")

        try:
            val_int = int(value)
            return val_int
        except (
            TypeError,
            ValueError,
            AttributeError,
            self.torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
        ):
            pass

        raise AssertionError(f"Unable to convert {value!r} into string")

    def _check_two_shapes_are_compatible(
        self,
        old_shape: Tuple[Any, ...],
        shape: Tuple[Any, ...],
        register_int: bool = True,
        name: Optional[str] = None,
    ):
        """Raises an exception if two shapes are not compatible."""
        assert len(old_shape) == len(shape), (
            f"Rank mismatch between previous shape {old_shape} and "
            f"new shape {shape} (name={name!r}){self.get_debug_msg()}"
        )
        if self._debug_dyn_dim:
            try:
                sshape = set(str(_) for _ in shape)
            except AttributeError:
                sshape = set()
            if (set(old_shape) | sshape) & self._debug_dyn_dim:
                print(
                    f"[GraphBuilder-{self._hash()}._check_two_shapes_are_compatible] "
                    f"old_shape={old_shape}, shape={shape}, register_int={register_int}, "
                    f"name={name!r}"
                )
        for d1, d2 in zip(old_shape, shape):
            if isinstance(d1, int) and isinstance(d2, int):
                assert d1 == d2, (
                    f"Shape {name!r} already exists and one dimension "
                    f"is not compatible existing {old_shape} != {shape} "
                    f"(new) {self.get_debug_msg()}"
                )
                continue

            d1_, d2_ = d1, d2
            if isinstance(d1, self.torch.SymInt):
                d1 = self._torch_sym_int_to_str(d1)
            elif isinstance(d1, self.torch.export.dynamic_shapes._Dim):
                d1 = self._torch_sym_int_to_str(d1)
            if isinstance(d2, self.torch.SymInt):
                d2 = self._torch_sym_int_to_str(d2)
            elif isinstance(d2, self.torch.export.dynamic_shapes._Dim):
                d2 = self._torch_sym_int_to_str(d2)

            if isinstance(d1, (int, str)) and isinstance(d2, (int, str)):
                if d1 == d2:
                    continue
                if isinstance(d1, str) and isinstance(d2, str):
                    self.register_constraint_dimension(d1, d2)
                    self.register_constraint_dimension(d2, d1)
                continue

            raise RuntimeError(
                f"Shape {name!r} already exists "
                f"and it is not compatible "
                f"existing {old_shape} != {shape} (new) "
                f"d1={d1_!r}, d2={d2_!r}, dim types="
                f"({type(d1_)}, {type(d2_)}), "
                f"d1={d1!r}, d2={d2!r}"
                f"{self.get_debug_msg()}"
            )

    def register_dynamic_objects_from_shape(self, shape: DYNAMIC_SHAPE):
        """Registers all the dynamic objects required in this shape."""
        for dim in shape:
            if isinstance(dim, str):
                self.register_dynamic_objects_from_dim(dim)

    def register_dynamic_objects_from_dim(self, dim: str):
        """Registers all the dynamic objects required in a dimension."""
        if isinstance(dim, self.WrapDim):
            token = dim.name
            if token not in self.dynamic_objects:
                self.add_dynamic_object(token, token)
            if dim not in self.dynamic_objects:
                self.add_dynamic_object(dim, dim)
            return

        assert isinstance(dim, str) and " " not in dim and dim.count("(") == dim.count(")"), (
            f"type(dim)={type(dim)} must be a str and should not contain "
            f"a comma or a space dim={dim!r} and the same number of opened and closed "
            f"brackets{self.get_debug_msg()}"
        )
        for token in parse_expression_tokens(dim):
            if token not in self.dynamic_objects:
                self.add_dynamic_object(token, token)
        if dim not in self.dynamic_objects:
            self.add_dynamic_object(dim, dim)

    def set_shape(
        self,
        name: str,
        shape: DYNAMIC_SHAPE,
        set_rank: bool = True,
        set_if_more_precise: bool = False,
        exc: bool = False,
        allow_zero: bool = False,
    ):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: result name
        :param shape: shape
        :param set_rank: set the rank as well
        :param set_if_more_precise: change the shape if it is more precise
        :param exc: raise an exception if inconsistency
        :param allow_zero: the shape should not have a value equal to zero
        """
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        if (self._debug_stop or self._debug_stop_shape) and name in (
            self._debug_stop,
            self._debug_stop_shape,
        ):
            # Set ONNXSTOP or ONNXSTOPSHAPE to stop here.
            raise AssertionError(
                f"Requested stop, name={name!r}, shape={shape}{self.get_debug_msg()}"
            )
        assert not shape or not isinstance(shape[0], tuple), f"Unexpected shape {shape}"
        assert "torch.Size" not in str(shape), f"Unexpected shape {shape}"
        for sdim in shape:
            if not isinstance(sdim, str):
                continue
            self.register_dynamic_objects_from_dim(sdim)
        shape0 = shape
        shape = self.verify_shape(shape, 0, name=name)
        assert allow_zero or 0 not in shape or shape == (0,), (
            f"Unexpected null shape {shape!r} (or {shape0!r}) for name={name!r}, "
            f"this case usually happens before a concatenation, "
            f"allow_zero={allow_zero}{self.get_debug_msg()}"
        )

        # costly
        # assert all(not isinstance(t, self.torch.SymInt) for t in shape), (
        #     f"Unexpected type for a shape, shape={shape}, types={[type(_) for _ in shape]}"
        #     f"{self.get_debug_msg()}"
        # )
        # shape_int = [d for d in shape if isinstance(d, int)]
        # assert (
        #     len(shape) == 0 or not shape_int or min(shape_int) >= 0
        # ), f"Negative value in shape {shape} for {name!r}{self.get_debug_msg()}"
        # assert (
        #     not self._debug_null_shape
        #     or len(shape) == 0
        #     or not shape_int
        #    or min(shape_int) > 0
        # ), f"Zero value in shape {shape} for {name!r}{self.get_debug_msg()}"
        if name in self._known_shapes:
            old_shape = self._known_shapes[name]
            if self._debug_dyn_dim and self._debug_dyn_dim & (set(shape) | set(old_shape)):
                print(
                    f"[GraphBuilder-{self._hash()}.set_shape] set_shape({name!r}, {shape}), "
                    f"old_shape={old_shape}"
                )
            if len(shape) == len(old_shape) and set_if_more_precise:
                is_more_precise = self.is_more_precise(shape=shape, base=old_shape)
                if not is_more_precise:
                    if old_shape == (0,) or shape == (0,):
                        if "warnings" not in self._debug_msg:
                            self._debug_msg["warnings"] = []
                        self._debug_msg["warnings"].append(
                            f"Name {name!r} already exists and it is not compatible "
                            f"existing {old_shape} != {shape} (new)"
                        )
                    else:
                        self._check_two_shapes_are_compatible(
                            old_shape, shape, name=name, register_int=False
                        )

            elif shape != old_shape:
                if exc:
                    raise RuntimeError(
                        f"Name {name!r} already exists and its shape different "
                        f"{old_shape} (old) != {shape}{self.get_debug_msg()}"
                    )
                return
            else:
                return

        if self._debug_dyn_dim and set(shape) & self._debug_dyn_dim:
            print(f"[GraphBuilder-{self._hash()}.set_shape] set_shape({name!r}, {shape})")
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_shape] {name}:{shape}")
        for s in shape:
            if isinstance(s, str) and s not in self.dynamic_objects:
                self.add_dynamic_object(s, s)
        self._known_shapes[name] = shape
        if hasattr(self, "replacements_dimensions_"):
            self.replacements_dimensions_[name] = tuple(
                (
                    rename_dynamic_expression(_, self.replacements_for_replacements_dimensions_)
                    if isinstance(_, str)
                    else _
                )
                for _ in shape
            )
        if set_rank and not self.has_rank(name):
            self.set_rank(name, len(shape))

    def set_type_shape_or_rank(self, name: str, like: str):
        """Sets the type and the shape of *name* like *like*."""
        if self.has_device(like):
            self.set_device(name, self.get_device(like))
        if self.has_type(like):
            self.set_type(name, self.get_type(like))
        if self.has_shape(like):
            self.set_shape(name, self.get_shape(like))
        elif self.has_rank(like):
            self.set_rank(name, self.get_rank(like))

    def set_device(
        self,
        name: str,
        device: Union[int, "torch.dtype"],  # noqa: F821
        exc: bool = True,
        keep_this_device: bool = False,
    ):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: name
        :param device: an integer or a torch device then converted into an integer
        :param exc: raises an exception
        :param keep_this_device: keeps this device anyway, this may happen if a constant
            is created, then the device may not be the one from the input (a shape)
            but the most frequent for any result tensor
        """
        assert exc, f"not implemented when exc={exc}"
        if not isinstance(device, int):
            device = -1 if device.type == "cpu" else device.index
        if name in self._known_devices and not keep_this_device:
            assert self._known_devices[name] == device, (
                f"device mismatch for name={name!r}, got {self._known_devices[name]}, "
                f"new device is {device}{self.get_debug_msg()}"
            )
            return
        self._known_devices[name] = device

    def has_device(self, name) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_devices

    def get_device(self, name) -> int:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert (
            name in self._known_devices
        ), f"Unknown device for name={name!r}{self.get_debug_msg()}"
        return self._known_devices[name]

    def set_type(self, name: str, dtype: int, exc: bool = True):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: name
        :param dtype: element type (an integer, ONNX), 0 (unknonw is a possible value)
        :param exc: raises an exception
        """
        if name in (self._debug_stop, self._debug_stop_type):
            raise AssertionError(
                f"Requested stop, name={name!r}, dtype={dtype}{self.get_debug_msg()}"
            )
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if isinstance(dtype, int):
            int_type = dtype
        else:
            int_type = _get_type(dtype)
        if name in self._known_types:
            # 0 is undefined
            if self._known_types[name] != 0 and int_type != self._known_types[name]:
                assert not exc, (
                    f"Type for name {name!r} already exists and it is different, "
                    f"known is {self._known_types[name]} != {int_type} (new) - "
                    f"(mapping={str_tensor_proto_type()}){self.get_debug_msg()}"
                )
                if "warnings" not in self._debug_msg:
                    self._debug_msg["warnings"] = []
                self._debug_msg["warnings"].append(
                    f"Type for name {name!r} already exists and it is different, "
                    f"known is {self._known_types[name]} != {int_type} (new) - "
                )
                if self.verbose > 5:
                    print(
                        f"Type for name {name!r} already exists and it is different, "
                        f"known is {self._known_types[name]} != {int_type} (new) - "
                    )
                return

        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_type] {name}:{int_type}")
        self._known_types[name] = int_type

    def rank(self, name: str) -> int:
        """Shortcut to :meth:`get_rank`."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return self.get_rank(name)

    def has_name(self, name: str) -> bool:
        """Tells if a result exists."""
        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name (name={name!r}){self.get_debug_msg()}"
        return name in self._known_names

    def has_rank(self, name: str) -> bool:
        """Tells if a result has a rank."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_ranks

    def has_shape(self, name: str, full=False) -> bool:
        """Tells if a result has a shape."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_shapes:
            return False
        if full:
            shape = self._known_shapes[name]
            return is_static_shape(shape) and min(shape) >= 0
        return True

    def has_type(self, name: str) -> Union[bool, int]:
        """Tells if a result has a type. This should be always true."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_types:
            return False
        # If the type is undefined, then it has no type.
        return self._known_types[name]

    def get_rank(self, name: str) -> int:
        """Returns the rank of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_ranks, (
            f"rank is unknown for result {name!r}, has_shape={self.has_shape(name)}, "
            f"has_rank={self.has_rank(name)}, "
            f"known_ranks={self._known_ranks}{self.get_debug_msg()}"
        )
        return self._known_ranks[name]

    def _get_shape_(self, name) -> str:
        "Use to debug."
        if self.has_shape(name):
            return f"{name}:{self.get_shape(name)}"
        if self.has_rank(name):
            s = tuple(["?" for _ in range(self.get_rank(name))])
            return f"{name}:{s}"
        return f"{name}:?"

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        """Returns the shape of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_shapes, (
            f"Shape is unknown for result {name!r}, "
            f"known_shapes={self._known_shapes}{self.get_debug_msg()}"
        )
        return self._known_shapes[name]

    def get_type_known(self, name: str, exc: bool = False) -> Optional[int]:
        """Returns the type known by torch to help solve mismatches."""
        if name in self._known_torch_value:
            value = self._known_torch_value[name]
            # something like (
            #                   'run_node',
            #                   (
            #                       '',
            #                       ('val', torch.float16, torch.Size([2, 12, 2048, 2048]))
            #                   )
            #                )
            assert not exc or (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], tuple)
                and len(value[1][1]) == 3
            ), (
                f"Unexpected output value {value} for {name!r}, "
                f"No information provided by torch"
                f"{self.get_debug_msg()}"
            )

            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], tuple)
                and len(value[1][1]) == 3
            ):
                dtype = value[1][1][1]
                itype = torch_dtype_to_onnx_dtype(dtype)
                return itype
        return None

    def get_type(self, name: str) -> int:
        """Returns the type of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_types, (
            f"Type is unknown for result {name!r}, "
            f"known_types={self._known_types}{self.get_debug_msg()}."
        )
        return self._known_types[name]

    def _dims_equal_set(self, d):
        done = set()
        ds = {d}
        while True:
            nds = set()
            for d in ds:
                if d in done:
                    continue
                done.add(d)
                if d in self.constraints_:
                    nds |= self.constraints_[d]
            if len(nds) == 0:
                break
            ds |= nds
        return ds

    def same_dimension(self, a, b) -> bool:
        """Checks if the dimensions are the same. Looks into constraints as well."""
        if a == b:
            return True
        return bool(self._dims_equal_set(a) & self._dims_equal_set(b))

    def same_shape(self, x: str, y: str) -> bool:
        """Checks if the shapes are the same. Looks into constraints as well."""
        assert self.has_shape(x), f"missing shape for {x!r}{self.get_debug_msg()}"
        assert self.has_shape(y), f"missing shape for {y!r}{self.get_debug_msg()}"
        shape1 = self.get_shape(x)
        shape2 = self.get_shape(y)
        if shape1 == shape2:
            return True
        if len(shape1) != len(shape2):
            return False
        return all(self.same_dimension(a, b) for a, b in zip(shape1, shape2))

    def value_as_shape(self, name: str) -> bool:
        """Returns the value of a result if it is a shape."""
        if name in self._known_value_shape:
            return self._known_value_shape[name]
        if (
            self.has_type(name)
            and self.get_type(name) == TensorProto.INT64
            and self.is_constant(name)
        ):
            # It is probably a shape because the user requested it as a shape.
            cst = self.get_constant(name, exc=False, computed_value=True)
            if cst is not None and len(cst.shape) == 1 and cst.dtype == np.int64:
                value = tuple(map(int, cst))
                self._known_value_shape[name] = value
                return value
        return None

    def set_value_shape(self, name: str, value: Any, equal_to: Optional[Tuple[str, str]] = None):
        """
        Sets the value for a shape result.

        :param name: name
        :param value: it cannot be empty
        :param equal_to: if specified, the value is also equal to this value

        A value can be a string (for an unknown shape, a tuple for a shape,
        an integer for a single scalar.
        """
        if self._debug_value_shape and name == self._debug_value_shape:
            raise AssertionError(
                f"Requested stop, name={name!r}, value={value!r}, equal_to={equal_to!r}"
            )

        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name={name!r}{self.get_debug_msg()}"
        assert not isinstance(value, tuple) or all(isinstance(d, (str, int)) for d in value), (
            f"Unexpected value for shape {name!r}, value={value!r}, "
            f"types={string_type(value)}{self.get_debug_msg()}"
        )
        if not self.has_rank(name):
            self.set_shape(name, (len(value),) if isinstance(value, tuple) else tuple())
        assert self.has_rank(name), (
            f"name={name!r}, has no rank, but it should, value={value!r}"
            f"{self.get_debug_msg()}"
        )
        assert self.get_rank(name) in (0, 1) or (
            isinstance(value, tuple) and len(value) <= 2048
        ), (
            f"name={name!r} is not a shape, its rank is {self.get_rank(name)}, "
            f"shape={self.get_shape(name) if self.has_shape(name) else '?'}, "
            f"value={string_type(value, with_shape=True)}{self.get_debug_msg()}"
        )
        assert not isinstance(value, (int, float)) or self.get_rank(name) == 0, (
            f"Mismatch between value={value!r} and rank="
            f"{self.get_rank(name)} for name={name!r}"
            f"{self.get_debug_msg()}"
        )
        if equal_to is None:
            if name in self._known_value_shape:
                existing = self._known_value_shape[name]
                if (
                    isinstance(existing, tuple)
                    and isinstance(value, tuple)
                    and len(existing) == len(value) == 1
                    and isinstance(existing[0], str)
                ):
                    self.register_constraint_dimension("existing", value)
                    return
            assert (
                name not in self._known_value_shape or self._known_value_shape[name] == value
            ), (
                f"Shape value for {name!r} (value={value!r}) is already "
                f"registered and is different from the existing "
                f"value which is {self._known_value_shape.get(name, None)!r} "
                f"(equal_to={equal_to!r}){self.get_debug_msg()}"
            )
            if self.verbose > 2:
                print(f"[GraphBuilder-{self._hash()}.set_value_shape] {name}:{value}")
            self._known_value_shape[name] = value
            return

        assert (
            name in equal_to
        ), f"Unexpected name={name!r}, it should be in equal_to={equal_to!r}."
        values = (
            self._known_value_shape.get(equal_to[0], None),
            self._known_value_shape.get(equal_to[1], None),
        )
        assert value in values, (
            f"Unexpected value={value} for name={name!r}, equal_to={equal_to}, "
            f"values={values}{self.get_debug_msg()}"
        )
        assert equal_to[0] in self._known_value_shape, (
            f"{equal_to[0]!r} should already registered, name={name!r}, "
            f"value={value!r}, equal_to={equal_to!r}{self.get_debug_msg()}"
        )
        # The logic is to get rid of one value instead of keeping
        # a mapping between equivalent values.
        new_value = self._known_value_shape[equal_to[0]]
        for n in equal_to:
            if n not in self._known_value_shape:
                self._known_value_shape[n] = new_value

    def unique_dimension_name(self, prefix: str) -> str:
        """
        Returns a unique dimension name.
        If by any change, a dimension has the same name as a dimension,
        the builder may be confused as a dynamic dimension can take part
        of a known value for a shape.
        """
        existing = set(self.dynamic_objects) | set(self._unique_names)
        for shape in self._known_shapes:
            existing |= set(shape)
        if prefix not in existing:
            self._unique_names.add(prefix)
            return prefix
        sug = prefix
        i = 0
        while sug in existing:
            i += 1
            sug = f"{prefix}{i}"
        self._unique_names.add(sug)
        return sug

    def unique_name(self, prefix: str) -> str:
        "Returns a unique result name."
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            self._unique_names.add(sug)
            return sug
        self._unique_names.add(prefix)
        return prefix

    def unique_node_name(self, name: str) -> str:
        "Returns a unique node name."
        if name in self._unique_node_names:
            i = 2
            sug = f"{name}2"
            while sug in self._unique_node_names:
                i += 1
                sug = f"{name}{i}"
            self._unique_node_names.add(sug)
            return sug
        self._unique_node_names.add(name)
        return name

    def elem_size(self, elem_type: int) -> int:
        "Returns the size in byte of the an element of this size."
        if elem_type in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if elem_type in {
            TensorProto.DOUBLE,
            TensorProto.INT64,
            TensorProto.UINT64,
            TensorProto.COMPLEX64,
        }:
            return 8
        if elem_type in {TensorProto.COMPLEX128}:
            return 16
        if elem_type in {
            TensorProto.INT16,
            TensorProto.UINT16,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }:
            return 2
        if elem_type in {TensorProto.BOOL, TensorProto.UINT8, TensorProto.INT8}:
            return 1
        raise AssertionError(
            f"elem_size not implemented for elem_type={elem_type}, "
            f"among {str_tensor_proto_type()}"
        )

    def make_dynamic_object(
        self,
        name: str,
        value: Any,
        shape_as_input: bool = False,
        input_name: Optional[str] = None,
        axis: Optional[int] = None,
    ) -> str:
        """
        Creates a dynamic shapes.

        :param name: name
        :param value: value
        :param shape_as_input: adds the name to the list of the inputs
            of the onnx model
        :param input_name: the dimension comes from this input
        :param axis: the dimension comes this axis
        :return: the name
        """

        def _append_to_source(name, input_name, axis, value):
            if input_name is not None and isinstance(value, self.torch.SymInt):
                # axis can be None for a scaler.
                assert name != input_name, (
                    f"Name {name!r} (input_name={input_name!r}) "
                    f"cannot be defined from itself (axis={axis}), "
                    f"value type is {type(value)}{self.get_debug_msg()}"
                )
                source = dict(input_name=input_name, axis=axis)
                if name in self.dynamic_dimensions_source:
                    self.dynamic_dimensions_source[name].append(source)
                else:
                    self.dynamic_dimensions_source[name] = [source]

        if name in self.dynamic_objects:
            # The dimension is already registered but it is used for another input.
            _append_to_source(name, input_name, axis, value)
            return None

        assert (
            name not in self.dynamic_objects
        ), f"Dynamic object {name!r}, value={value!r} is already there{self.get_debug_msg()}"
        if isinstance(value, self.WrapSym):
            value = value.sym
        assert isinstance(
            value, (str, self.torch.SymInt, self.torch.SymFloat, self.torch.SymBool)
        ), f"Unexpected type {type(value)} for value{self.get_debug_msg()}"
        _append_to_source(name, input_name, axis, value)

        self.add_dynamic_object(name, value, parse=True)
        if (
            shape_as_input
            and isinstance(value, self.torch.SymInt)
            and value.node.maybe_as_int() is None
        ):
            # Then an input is a shape.
            self.add_dynamic_object(str(value), value)

        # Do we need this?
        # if name not in self._known_value_shape:
        #    self._known_value_shape[name] = name

        key = None
        if isinstance(value, (self.torch.SymInt, self.torch.SymFloat)):
            if hasattr(value, "node"):
                from torch.fx.experimental.sym_node import SymNode

                if isinstance(value.node, str):
                    key = f"{value.node}"
                elif isinstance(value.node, SymNode):
                    key = f"{value.node}"
                    key2 = value.node.maybe_as_int()
                    if key2 is not None:
                        self._add_dynamic_example(key, key2)
                else:
                    raise AssertionError(
                        f"Unexpected type {type(value.node)} for value.node={value.node}"
                    )
                    # key = str(value)
        if key is None:
            key = str(value)

        self.add_dynamic_objects_rev(key, (name, value))

        if shape_as_input:
            assert isinstance(value, (self.torch.SymInt, self.torch.SymFloat)), (
                f"shape_as_input={shape_as_input}, unexpected type "
                f"{type(value)} for value{self.get_debug_msg()}"
            )
            # torch.compile adds input for dynamic shapes
            if not self.value_as_shape(name):
                # Let's mark  this input as a shape
                self.set_value_shape(name, name)
            return self.make_tensor_input(
                self._known_value_shape[name],
                (
                    TensorProto.INT64
                    if isinstance(value, self.torch.SymInt)
                    else TensorProto.FLOAT
                ),
                tuple(),
                is_dimension=True,
                marker="make_dynamic_object",
            )
        return self._known_value_shape.get(name, name)

    def get_dimension_as_result(self, name: str) -> str:
        if self.has_name(name):
            return name
        assert (
            name in self.dynamic_dimensions_source
            and len(self.dynamic_dimensions_source[name]) > 0
        ), (
            f"Dimension {name!r} has no registered source "
            f"it cannot be created as a result{self.get_debug_msg()}"
        )
        source = self.dynamic_dimensions_source[name][0]
        axis = source["axis"]
        input_name = source["input_name"]
        shape_name = self.unique_name(f"_onx_shape_{name}")
        self.make_node("Shape", [input_name], [shape_name], name="_get_dimension_as_result")
        axis_name = self.make_initializer(
            "",
            np.array([axis], dtype=np.int64),
            source="GraphBuilder.get_dimension_as_result.axis_name",
        )
        self.make_node("Gather", [shape_name, axis_name], [name], name="_get_dimension_as_result")
        return name

    def make_shape_from_results(self, shape: DYNAMIC_SHAPE, name="") -> str:
        """Creates a shape coming from intermediate results."""
        assert isinstance(
            shape, (list, tuple)
        ), f"Unexpected type {type(shape)} for shape{self.get_debug_msg()}"
        if all_int(shape):
            return self.make_initializer(
                "",
                np.array(shape, dtype=np.int64),
                source="GraphBuilder.make_shape_from_results.shape",
            )

        key = []
        for d in shape:
            if isinstance(d, int):
                key.append(d)
            elif isinstance(d, self.torch.SymInt):
                value = self._torch_sym_int(d)
                key.append(value)
            elif isinstance(d, (str, self.torch.SymInt)):
                assert self._debug_quiet or (
                    self.has_shape(d) or (self.has_rank(d) and self.get_rank(d) == 0)
                ), (
                    f"Missing shape for {d!r} in {shape!r}, has_rank={self.has_rank(d)}, "
                    f"has_type={self.has_type(d)}{self.get_debug_msg()}"
                )
                key.append(d)
            else:
                raise RuntimeError(
                    f"Unexpected type {type(d)} for a dimension in "
                    f"{shape}{self.get_debug_msg()}"
                )

        assert all_int_or_str(key), (
            f"Unexpected key {key} type are {[type(_) for _ in key]}, "
            f"shape={shape}{self.get_debug_msg()}"
        )
        key = ("Concat", *key)
        if key in self._cache_shape:
            # The same shape was already requested.
            return self._cache_shape[key]

        shape_shape = 0
        conc = []
        for d in shape:
            if isinstance(d, int):
                conc.append(
                    self.make_initializer(
                        "",
                        np.array([d], dtype=np.int64),
                        source="GraphBuilder.make_shape_from_results.conc",
                    ),
                )
                if shape_shape is not None:
                    shape_shape += 1
            elif isinstance(d, str):
                value = d
                if value in self.dynamic_objects_rev:
                    assert len(self.dynamic_objects_rev[value]) >= 1
                    name = self.dynamic_objects_rev[value][0][0]
                    assert not isinstance(name, tuple), (
                        f"Unexpected type {type(name)}, name={name!r}, value={value!r}"
                        f"{self.get_debug_msg()}"
                    )
                    name = self.get_dimension_as_result(name)
                else:
                    name = value

                assert name in self.dynamic_objects or self.has_name(
                    name
                ), f"Unknown dynamic object {d!r} (or {name!r}){self.get_debug_msg()}"

                if self.has_rank(name):
                    assert self.get_rank(name) <= 1, (
                        f"Unexpected rank={self.get_rank(name)} "
                        "for a shape{self.get_debug_msg()}"
                    )
                    if self.get_rank(name) == 0:
                        r = self.op.UnsqueezeAnyOpset(name, self.ZERO, name=f"_mkshape1_{name}")
                        if self.has_device(name):
                            self.set_device(r, self.get_device(name))
                        self.set_type(r, self.get_type(name))
                        self.set_shape(r, (1,))
                        conc.append(r)
                        shape_shape += 1
                    else:
                        # We assume rank is one.
                        if self.has_shape(name):
                            if shape_shape is not None:
                                shape_shape += self.get_shape(name)[0]
                        else:
                            shape_shape = None
                        conc.append(name)
                else:
                    shape_shape = None
                    conc.append(name)

            elif isinstance(d, self.torch.SymInt):
                value = self._torch_sym_int(d)
                if value in self.dynamic_objects_rev:
                    assert len(self.dynamic_objects_rev[value]) >= 1
                    name = self.dynamic_objects_rev[value][0][0]
                    assert not isinstance(name, tuple), (
                        f"Unexpected type {type(name)}, name={name!r}, value={value!r}"
                        f"{self.get_debug_msg()}"
                    )
                    name = self.get_dimension_as_result(name)
                else:
                    name = value
                if isinstance(name, self.torch.SymInt):
                    name = self._torch_sym_int(name)
                    assert not isinstance(name, self.torch.SymInt)
                assert not isinstance(name, self.torch.SymInt)
                assert name in self.dynamic_objects or self.has_name(
                    name
                ), f"Unknown dynamic object {d!r}-{name!r}{self.get_debug_msg()}"
                if self.has_rank(name):
                    assert self.get_rank(name) <= 1, (
                        f"Unexpected rank={self.get_rank(name)} "
                        "for a shape{self.get_debug_msg()}"
                    )
                    if self.get_rank(name) == 0:
                        r = self.op.UnsqueezeAnyOpset(name, self.ZERO, name=f"_mkshape2_{name}")
                        if self.has_device(name):
                            self.set_device(r, self.get_device(name))
                        self.set_type(r, self.get_type(name))
                        self.set_shape(r, (1,))
                        conc.append(r)
                    else:
                        conc.append(name)
                else:
                    conc.append(name)
                if shape_shape is not None:
                    shape_shape += 1
            else:
                raise RuntimeError(
                    f"Unexpected type {type(d)} for a dimension in "
                    f"{shape}{self.get_debug_msg()}"
                )

        if len(conc) > 1:
            res = self.make_node("Concat", conc, axis=0, name=f"_mkshape_{name}")
            if shape_shape is None:
                self.set_rank(res, 1)
            else:
                self.set_shape(res, (shape_shape,))
        else:
            assert len(conc) > 0, f"No shape to concatenate{self.get_debug_msg()}"
            res = self.make_node("Identity", conc[0], name=f"_mkshape1_{name}")
        self._cache_shape[key] = res
        return res

    def make_initializer(
        self,
        name: str,
        value: Any,
        external: bool = False,
        msg: str = "",
        parameter_name: Optional[str] = None,
        source: str = "",
        allow_empty: bool = False,
        give_unique: bool = False,
    ) -> str:
        """
        Adds an initializer to the graph.
        The function detects duplicated small containers, only if they are
        integers. Other type might be used as weights. Even similar, they could
        change after training.

        :param name: name, if empty (`""`), a unique names is given, if not empty,
            it is more like a prefix, the method might change it to make it unique
        :param value: value (TensorProto)
        :param external: external initializer or not (not stored in the graph model)
        :param msg: added to the error message if something goes wrong
        :param parameter_name: the parameter name is different than its name in the fx graph,
            they are restored when the model is finally exported into onnx,
            until then, the mapping is kept in attribute ``_parameter_renaming``
        :param source: any additional information, this field is usually used to
            let the number know where the initializer was created.
        :param allow_empty: allow_empty value
        :param give_unique: give a unique name
        :return: name of the initializer
        """
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if isinstance(value, int):
            value = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            value = np.array(value, dtype=np.float32)
        elif isinstance(value, TensorProto):
            pass
        elif isinstance(value, np.ndarray):
            assert not self.has_shape(name) or self.get_shape(name) == value.shape, (
                f"Shape {value.shape} does not match the registered one {self.get_shape(name)} "
                f"for name {name!r}{self.get_debug_msg()}"
            )
        elif isinstance(value, (np.float16, np.float32, np.float64, np.int64)):
            value = np.array(value)
            assert not self.has_shape(name) or self.get_shape(name) == value.shape, (
                f"Shape {value.shape} does not match the registered one {self.get_shape(name)} "
                f"for name {name!r}{self.get_debug_msg()}"
            )
        elif isinstance(value, self.torch.Tensor):
            # torch.nn.parameter.Parameter -> np.ndarray
            assert "FakeTensor" not in str(type(value)), (
                f"FakeTensor {name!r} cannot be an initializer {type(value)}"
                f"{self.get_debug_msg()}"
            )
            assert not self.has_shape(name) or self.get_shape(name) == value.shape, (
                f"Shape {value.shape} does not match the registered one {self.get_shape(name)} "
                f"for name {name!r}{self.get_debug_msg()}"
            )
        else:
            raise RuntimeError(
                f"Initializer name={name!r}, "
                f"unexpected type {type(value)} for value={value!r} ({msg})"
                f"{self.get_debug_msg()}"
            )

        key = self.make_key(value)
        if key and key in self._values:
            # exception with obivous names
            if name in {"", "ONES", "ZEROS", "ZERO", "ONE"} or name.startswith("ONES"):
                assert not parameter_name, (
                    f"Empty name cannot be used with parameter_name={parameter_name!r}, "
                    f"key={key!r}"
                )
                new_name = self._values[key]
                assert new_name in self.initializers_dict, f"Unable to find {new_name!r}"
                self._append_initializer_source(new_name, source, existing=True)
                return new_name
            self._append_initializer_source(name, source, same_as=key)
            res = self.make_node(
                "Identity",
                [self._values[key]],
                [name],
                name="make_initializer",
                insert_position="HEAD",
            )
            return res

        if isinstance(value, TensorProto):
            itype = value.data_type
            shape = tuple(value.dims)
        else:
            itype = _get_type(value.dtype)
            shape = tuple(value.shape)

        if name == "" or give_unique:
            if name:
                name = self.unique_name(name)
            else:
                sh = "x".join(map(str, shape))
                size = np.prod(value.size()) if hasattr(value, "detach") else value.size
                sh2 = (
                    "_".join(map(str, value.ravel().tolist()))
                    if size <= 5 and value.dtype == np.int64
                    else ""
                )
                name = self.unique_name(f"init{itype}_s{sh}_{sh2}")

        self.add_initializer(
            name,
            value,
            itype=itype,
            shape=shape,
            key=key,
            parameter_name=parameter_name,
            source=source,
            allow_empty=allow_empty,
        )
        return name

    def _append_initializer_source(
        self, name: str, source: str, same_as: Optional[str] = None, existing: bool = False
    ):
        """
        Gathers information related to an initializer.

        :param name: name of the initializer
        :param source: any kind of string, it should not be empty
        :param same_as: the initializer was detected as a duplicate of an existing one,
            this field reflects that.
        :param existing: does it already exists?
        """
        if existing:
            assert name in self.initializers_dict_sources, (
                f"Initializer {name!r} does not exist, source={source!r}, "
                f"{self.get_debug_msg()}"
            )
            self.initializers_dict_sources[name].add_source(source)
            return
        assert existing is None or name not in self.initializers_dict_sources, (
            f"Initializer {name!r} was already added to the model, source={source!r}, "
            f"existing is {self.initializers_dict_sources[name]!r}{self.get_debug_msg()}"
        )
        assert not self.check_empty_source or source, f"source is null for initializer {name!r}"
        self.initializers_dict_sources[name] = GraphBuilder.InitializerInfo(
            name, source, same_as=same_as
        )

    def add_initializer(
        self,
        name: str,
        value: Any,
        itype: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
        cst: Optional[Any] = None,
        key: Optional[Any] = None,
        existing: Optional[bool] = False,
        allow_empty: bool = False,
        parameter_name: Optional[str] = None,
        source: str = "",
    ):
        """
        Adds an initializer.

        :param name: constant name
        :param value: initializer
        :param itype: to overwrite the type
        :param shape: to overwrite the shape
        :param cst: value to send to :meth:`update_node_constant
            <experimental_experiment.xshape._inference_runtime._InferenceRuntime.update_node_constant>`
        :param key: used to register the initializer
        :param existing: if True, shape and type should exist,
            if False, it should not exist, if None, both case are allowed
        :param allow_empty: allow empty tensor anyway
        :param parameter_name: the parameter name is different than its name in the fx graph,
            they are restored when the model is finally exported into onnx,
            until then, the mapping is kept in attribute ``_parameter_renaming``
        :param source: any additional information, this field is usually used to
            let the number know where the initializer was created.
        """
        assert (
            not self.has_shape(name)
            or not hasattr(value, "shape")
            or self.get_shape(name) == value.shape
            or any(isinstance(s, str) and s.startswith("NEWDIM") for s in self.get_shape(name))
        ), (
            f"Shape {value.shape} does not match the registered one {self.get_shape(name)} "
            f"for name {name!r}{self.get_debug_msg()}"
        )
        is_proto = isinstance(value, (TensorProto, NodeProto))
        if shape is None:
            shape = (
                self._get_tensor_shape(value) if is_proto else tuple(int(i) for i in value.shape)
            )
        if itype is None:
            itype = (
                self._get_tensor_type(value) if is_proto else dtype_to_tensor_dtype(value.dtype)
            )
        if existing:
            assert allow_empty or len(shape) == 0 or min(shape) > 0, (
                f"Initializer {name!r} has an empty shape={shape}, itype={itype}, "
                f"existing shape={self.get_shape(name) if self.has_shape(name) else '?'}, "
                f"type={type(value)}{self.get_debug_msg()}"
            )
            assert self.has_name(name), (
                f"value {name!r} is replaced by an initializer "
                f"already exists{self.get_debug_msg()}"
            )
            assert not self.has_type(name) or itype == self.get_type(name), (
                f"Type mismatch for {name!r}, existing type {self.get_type(name)}, "
                f"new type {itype}{self.get_debug_msg()}"
            )
            assert not self.has_shape(name) or shape == self.get_shape(name), (
                f"Type mismatch for {name!r}, existing shape "
                f"{self.get_shape(name)}, new shape {shape}{self.get_debug_msg()}"
            )
            self.set_shape(name, shape, allow_zero=allow_empty)
            self.set_type(name, itype)
        else:
            assert allow_empty or len(shape) == 0 or min(shape) > 0, (
                f"Initializer {name!r} has an empty shape={shape}, itype={itype}, "
                f"type={type(value)}, "
                f"value={string_type(value, with_shape=True, with_min_max=True)}"
                f"{self.get_debug_msg()}"
            )
            assert existing is None or name not in self.initializers_dict, (
                f"initializer {name!r} was already added (itype={itype}, shape={shape})"
                f"{self.get_debug_msg()}"
            )
            assert existing is None or not self.has_name(
                name
            ), f"initializer {name!r} already exists{self.get_debug_msg()}"
            self.set_shape(name, shape, allow_zero=allow_empty)
            self.set_type(name, itype)
            if not self.has_name(name):
                self.set_name(name, "make_initializer")
            self._unique_names.add(name)

        if key is None:
            key = self.make_key(value)
        if name in self.initializers_dict:
            assert self.constant_is_equal_to(name, value)
            if existing:
                self.initializers_dict[name] = value
        else:
            self.initializers_dict[name] = value
        self._append_initializer_source(name, source, existing=existing)

        if parameter_name and parameter_name != name:
            # We want a specific name for this one, let's keep that information in
            # main so that we can rename them later.
            assert not self.has_name(parameter_name), (
                f"Parameter {name!r} cannot be renamed int {parameter_name!r} as it "
                f"is already taken{self.get_debug_msg()}"
            )
            self._parameter_renaming[name] = parameter_name
            self._parameter_norename.add(parameter_name)
            self.initializers_dict[parameter_name] = value
            self._append_initializer_source(
                parameter_name, source, existing=existing, same_as=name
            )
            self.set_name(parameter_name, marker="parameter_name")
            self.set_shape(parameter_name, self.get_shape(name))
            self.set_type(parameter_name, self.get_type(name))
            if self.has_device(name):
                self.set_device(parameter_name, self.get_device(name))
            if self.verbose and (self.verbose > 3 or (self.verbose > 2 and np.prod(shape) > 100)):
                print(
                    f"[GraphBuilder-{self._hash()}.make_initializer] "
                    f"{name}[{itype}:{shape}] -> {parameter_name}"
                )
        else:
            if self.verbose and (self.verbose > 3 or (self.verbose > 2 and np.prod(shape) > 100)):
                print(f"[GraphBuilder-{self._hash()}.make_initializer] {name}[{itype}:{shape}]")

        if cst is None and isinstance(value, NodeProto):
            cst = value
        is_constant = self.update_node_constant(name, cst)
        if is_constant and parameter_name:
            self.update_node_constant(parameter_name, cst)

        if key is not None and key not in self._values:
            self._values[key] = name
        return name

    def _shape_type(self, value: Any) -> Tuple[Any, int, STATIC_SHAPE]:
        if isinstance(value, np.ndarray):
            return oh.np_dtype_to_tensor_dtype(value.dtype), value.size, value.shape
        if isinstance(value, self.torch.Tensor):
            return torch_dtype_to_onnx_dtype(value.dtype), value.nelement(), value.shape
        if isinstance(value, TensorProto):
            shape = tuple(value.dims)
            return value.data_type, np.prod(shape), shape
        raise TypeError(f"Unsupported type for a constant {type(value)}")

    def constant_is_equal_to(self, name, value):
        assert name in self.initializers_dict, f"intializer {name!r} not found."
        cst = self.initializers_dict[name]
        itype1, size1, shape1 = self._shape_type(cst)
        itype2, size2, shape2 = self._shape_type(value)
        if itype1 != itype2 or shape1 != shape2:
            return False

        if max(size1, size2) >= 30:
            # No check. Too long.
            return True

        if isinstance(cst, TensorProto):
            cst = onh.to_array(cst)
        if isinstance(value, TensorProto):
            value = onh.to_array(value)

        if isinstance(cst, np.ndarray) and isinstance(value, np.ndarray):
            if cst.dtype != value.dtype:
                return False
            if cst.shape != value.shape:
                return False
            if cst.shape == (0,):
                return True
            if len(cst.shape) == 0:
                return cst == value
            return np.abs(cst - value).max() == 0

        t1 = self.torch.from_numpy(cst) if isinstance(cst, np.ndarray) else cst
        t2 = self.torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        assert hasattr(t1, "dtype"), f"Unexpected type {type(t1)} for name={name!r}"
        assert hasattr(t2, "dtype"), f"Unexpected type {type(t2)}, {type(value)}"
        if t1.dtype != t2.dtype:
            return False
        if t1.shape != t2.shape:
            return False
        if t1.shape == (0,):
            return True
        if len(t1.shape) == 0:
            return t1 == t2
        return np.abs(t1 - t2).max() == 0

    def is_dynamic_shape(
        self,
        shape: DYNAMIC_SHAPE,
        verify: bool = True,
        allow_none: bool = False,
        allow_new_dynamic_dimension: bool = False,
    ) -> bool:
        """Tells if a shape is dynamic or static (only integers)."""
        return all(
            self.is_dynamic_dimension(
                x,
                verify=verify,
                allow_none=allow_none,
                allow_new_dynamic_dimension=allow_new_dynamic_dimension,
            )
            for x in shape
        )

    def is_constant_or_attribute(self, node: NodeProto, input_index: int, att_name: str) -> bool:
        """
        Tells if an input is a constant or returns true if in an older
        opset, it was named as an attribute.
        """
        if input_index < len(node.input):
            return self.is_constant(node.input[input_index])
        return True

    def get_constant_or_attribute(self, node: NodeProto, input_index: int, att_name: str) -> Any:
        """
        Tells if an input is a constant or returns true if in an older
        opset, it was named as an attribute.
        """
        if input_index < len(node.input):
            return self.get_constant(node.input[input_index])
        for att in node.attribute:
            if att.name == att_name:
                assert (
                    att.type == AttributeProto.INTS
                ), f"Not Implemented when att.type={att.type}{self.get_debug_msg()}"
                return np.array(list(att.ints), dtype=np.int64)
        return None

    def is_dynamic_dimension(
        self,
        dim: Any,
        verify: bool = True,
        allow_none: bool = False,
        allow_new_dynamic_dimension: bool = False,
    ) -> bool:
        if allow_none and dim is None:
            return True
        if not isinstance(dim, (int, self.torch.SymInt, str)):
            return False
        if str(dim) in self._dynamic_alias:
            dim = self._dynamic_alias[str(dim)]
        assert (
            not verify
            or is_static_dimension(dim)
            or str(dim) in self.dynamic_objects
            or str(dim) in self.dynamic_objects_rev
            or self.has_name(str(dim))
            or self.parse_dimension_expression(
                dim, allow_new_dynamic_dimension=allow_new_dynamic_dimension
            )
        ), (
            f"dim={dim!r} (type={type(dim)}) not in found in "
            f"{self.dynamic_objects}, self.dynamic_shapes={self.dynamic_shapes}, "
            f"self._dynamic_alias={self._dynamic_alias}{self.get_debug_msg()}"
        )
        return True

    def get_dynamic_dimension(self, dim: Any, keep_const: bool = True) -> Any:
        """Returns a dynamic dimension as a 1D tensor of one element."""
        if isinstance(dim, int):
            if keep_const:
                return np.array([dim], dtype=np.int64)
            return self.make_initializer(
                "",
                np.array([dim], dtype=np.int64),
                source="GraphBuilder.get_dynamic_dimension.dim",
            )
        assert isinstance(
            dim, (str, self.torch.SymInt)
        ), f"Unexpected type {type(dim)} for dim={dim}{self.get_debug_msg()}"
        if isinstance(dim, str):
            if self.has_name(dim):
                assert self.has_rank(dim), f"dim={dim!r} has no rank{self.get_debug_msg()}"
                if self.get_rank(dim) == 0:
                    return self.op.UnsqueezeAnyOpset(
                        dim,
                        self.ZERO,
                        name=f"get_dynamic_dimension_a_{dim}",
                    )
                return dim
            assert dim in self.dynamic_objects, (
                f"Unable to find a dynamic object for {dim:r}, "
                f"list={list(self.dynamic_objects)}"
                f"{self.get_debug_msg()}"
            )
            assert dim in self.dynamic_dimensions_source, (
                f"Unable to find a result to express dim {dim:r}, "
                f"sources={list(self.dynamic_dimensions_source)}"
                f"{self.get_debug_msg()}"
            )
            raise NotImplementedError(
                f"Source is available for {dim!r}, "
                f"source={self.dynamic_dimensions_source[dim]}"
            )
        name = self._torch_sym_int_to_str(dim)
        assert name, f"Unable to expression a dynamic dimension{self.get_debug_msg()}"
        if self.has_name(name):
            assert self.has_rank(name), f"name={name!r} has no rank{self.get_debug_msg()}"
            if self.get_rank(name) == 0:
                return self.op.UnsqueezeAnyOpset(
                    dim, self.ZERO, name=f"get_dynamic_dimension_b_{name}"
                )
            return dim
        assert name in self.dynamic_objects, (
            f"Unable to find a dynamic object for {name:r}, "
            f"list={list(self.dynamic_objects)}"
            f"{self.get_debug_msg()}"
        )
        assert name in self.dynamic_dimensions_source, (
            f"Unable to find a result to express dim {name:r}, "
            f"sources={list(self.dynamic_dimensions_source)}"
            f"{self.get_debug_msg()}"
        )
        raise NotImplementedError(
            f"Source is available for {dim!r}, source={self.dynamic_dimensions_source[dim]}"
        )

    def _get_dynamic_dimension(self, name: str, dim: int) -> Optional[Union[str, "WrapDim"]]:
        if self.dynamic_shapes is None:
            return None
        if not self.dynamic_shapes or name not in self.dynamic_shapes:
            return None
        dyn = self.dynamic_shapes[name]
        if dim not in dyn:
            return None
        v = dyn[dim]
        st = str(type(v))
        name = (
            v.__name__
            if "_Dim" in st or "_DerivedDim" in st or "torch.export.dynamic_shapes.Dim" in st
            else v
        )
        assert isinstance(name, (str, self.WrapDim)), (
            f"_get_dynamic_dimension must return a string but name is {name!r}, "
            f"type(name)={type(name)}"
        )
        return name

    def add_dynamic_object(
        self,
        key: str,
        value: Any,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        parse: bool = False,
        check_tokens: bool = True,
    ):
        """
        Registers a dynamic object such as a dynamic dimension.

        :param key: string
        :param value: SymInt, Dim, _DerivedDim
        :param name: input name it comes from
        :param dim: dimension for this dimension in input
        :param parse: parse the expression add pieces of it as well
        :param check_token: check that the subtoken are
            registered prior to this addition
        """
        assert not isinstance(value, self.torch.export.dynamic_shapes._Dim), (
            f"Unexpected dimension type {type(value)}:{value!r}, "
            f"class is {value.__class__.__name__!r} for key={key!r}{self.get_debug_msg()}"
        )
        if isinstance(key, self.WrapDim):
            keykey = key.name
        else:
            assert isinstance(
                key, str
            ), f"Unexpected type {type(key)} for key{self.get_debug_msg()}"
            keykey = key
        if isinstance(value, (self.torch.SymInt, self.torch.SymFloat, self.WrapDim)):
            wrapped_value = self.WrapSym(value)
            self.dynamic_objects[keykey] = wrapped_value
            wrapped_name = wrapped_value.name
            if wrapped_name and wrapped_name not in self.dynamic_objects:
                # This means this informations is a dynamic dimension
                # used for the first time.
                self.dynamic_objects[wrapped_name] = wrapped_value
        else:
            self.dynamic_objects[keykey] = value

        if name is not None and name in self.dynamic_shapes and dim is not None:
            dyn_shape = self.dynamic_shapes[name]
            if dim in dyn_shape:
                dyndim = dyn_shape[dim]
                self._add_dynamic_alias(key, dyndim.__name__)

        if parse:
            self.register_dynamic_objects_from_dim(keykey)
        elif check_tokens:
            tokens = parse_expression_tokens(key)
            for t in tokens:
                if isinstance(t, str):
                    assert t in self.dynamic_objects, (
                        f"Token {t!r} from {key!r} is not registered "
                        f"among {list(self.dynamic_objects)}{self.get_debug_msg()}"
                    )

    def has_dynamic_object(self, name: str) -> bool:
        """Tells if a result is a dynamic object, `torch.SymInt` for torch."""
        return name in self.dynamic_objects

    def add_dynamic_objects_rev(self, key, name_value):
        if key not in self.dynamic_objects_rev:
            self.dynamic_objects_rev[key] = []
        self.dynamic_objects_rev[key].append(name_value)

    def _add_dynamic_example(self, name: str, value: Union[int, float]):
        if name not in self._dynamic_examples:
            self._dynamic_examples[name] = set()
        self._dynamic_examples[name].add(value)

    def _torch_sym_int(self, d, add: bool = False) -> Optional[Union[int, str, float]]:
        assert isinstance(
            d, (self.torch.SymInt, str, self.torch.SymFloat)
        ), f"unexpected type for d={d}, type={type(d)}"
        value = None
        try:
            # don't use 'expr'
            dyn_val = str(d.node._expr).replace(" ", "")
            value = dyn_val
        except AttributeError:
            pass

        if isinstance(d, (str, int, float)):
            return d
        if isinstance(d.node, str):
            return d.node
        if value is None:
            # Is it an integer?
            value = d.node.maybe_as_int()
        else:
            # maybe an expression which is a single integer
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        if isinstance(value, int):
            assert not isinstance(value, self.torch.SymInt)
            return value

        if value is None:
            value = str(d)

        if value in self._dynamic_alias:
            value = self._dynamic_alias[value]

        if value not in self.dynamic_objects_rev:
            # The dynamic dimension does not seem to be registered.
            # Maybe it is constant.
            try:
                val_int = d.node.maybe_as_int()
                if val_int is not None:
                    self._add_dynamic_example(value, val_int)
                return value
            except (TypeError, ValueError):
                pass

        if value in self.dynamic_objects:
            assert not isinstance(value, self.torch.SymInt)
            return value

        if value not in self.dynamic_objects_rev and value not in self._known_value_shape and add:
            self.add_dynamic_object(value, value)
            self.add_dynamic_object_rev(value, value)

        assert value in self.dynamic_objects_rev or value in self._known_value_shape, (
            f"value={value!r}, unable to find dimension {d!r} ({type(d)}) "
            f"(str(d)={str(d)!r}) in {self.dynamic_objects_rev} "
            f"or {self._dynamic_alias} or {self._known_value_shape}"
            f"{dir(d)}{self.get_debug_msg()}"
        )
        assert not isinstance(
            value, self.torch.SymInt
        ), f"Unexpected type {type(value)} for d={d!r}"
        if value in self.dynamic_objects_rev:
            new_value = self.dynamic_objects_rev[value]
            assert isinstance(new_value, list), (
                f"Unexpected type {type(new_value)} for value={value!r}, d={d}"
                f"{self.get_debug_msg()}"
            )
            assert len(new_value) >= 1, (
                f"Unexpected number of items in {new_value}, value={value!r}, d={d}"
                f"{self.get_debug_msg()}"
            )
            # We assume if len(new_value) > 1 that all names are equivalent.
            # The graph is doing the same computation multiple times.
            final = new_value[0]
            assert isinstance(final, str) or (isinstance(final, tuple) and len(final) == 2), (
                f"Unexpected type {type(final)}, final={final}, value={value}, d={d}"
                f"new_value={new_value}, {self.get_debug_msg()}"
            )
            if isinstance(final, str):
                # A formula
                return final

            # An alias
            name = final[0]
            assert isinstance(name, str), (
                f"Unexpected type {type(name)}, name={final}, value={value}, d={d}, "
                f"new_value={new_value}, {self.get_debug_msg()}"
            )
            return name

        # Its value is in self._known_value_shape. We still return its name.
        return value

    def verify_dynamic_shape(
        self,
        shape: Any,
        name: Optional[str] = None,
        add: bool = True,
    ) -> DYNAMIC_SHAPE:
        """The implementation of this method should be revisited."""
        if shape is None:
            return None
        try:
            is_static = is_static_shape(shape)
        except AssertionError as e:
            raise AssertionError(
                f"Unable to check static shape {string_type(shape)}{self.get_debug_msg()}"
            ) from e
        if is_static:
            return tuple(int(i) for i in shape)
        new_shape = []
        for dim, d in enumerate(shape):
            if isinstance(d, (self.torch.SymInt, str, self.WrapDim)):
                dyn_name = None if name is None else self._get_dynamic_dimension(name, dim)
                if dyn_name is not None:
                    if add:
                        self.add_dynamic_object(dyn_name, dyn_name, parse=True)
                    new_shape.append(
                        dyn_name.name_as_string
                        if isinstance(dyn_name, self.WrapDim)
                        else dyn_name
                    )
                    continue

                value = self._torch_sym_int(d, add=add)
                assert (
                    value is not None
                ), f"Unexpected type {type(d)} in shape={shape}{self.get_debug_msg()}"
                new_shape.append(value)
                continue
            if (
                isinstance(d, self.torch.export.dynamic_shapes._Dim)
                or "_DerivedDim" in str(d)
                or "_Dim" in str(d)
            ):
                raise NotImplementedError(
                    f"verify_dynamic_shape not yet implemented for type(d)={type(d)}, d={d}"
                )
            if isinstance(d, int):
                new_shape.append(d)
                continue
            assert (
                d is not None
            ), f"Unexpected type {type(d)} in shape={shape}{self.get_debug_msg()}"
        assert all(isinstance(d, (int, str)) for d in new_shape), (
            f"Issue with shape={new_shape}, types={[type(d) for d in new_shape]}"
            f"{self.get_debug_msg()}"
        )
        new_new_shape = []
        for dim in new_shape:
            if isinstance(dim, int):
                new_new_shape.append(dim)
                continue
            dim = self._simplifies_dynamic_dimension_string(dim)
            new_new_shape.append(dim)
        return tuple(new_new_shape)

    def _simplifies_dynamic_dimension_string(self, dim: str) -> str:
        dim = dim.replace(" ", "")
        while dim[0] == "(" and dim[-1] == ")" and "(" not in dim[1:-1]:
            dim = dim[1:-1]
        if dim.startswith("0+"):
            dim = dim[2:]
        return dim

    def register_users(self, name: str, users: Iterable[str]):
        """
        Registers users. This is not used except to check the conversion
        is valid.
        """
        assert (
            name not in self._registered_users
        ), f"{name!r} is already registered{self.get_debug_msg()}"
        self._registered_users[name] = set(users)

    def make_tensor_input(
        self,
        name: Union[str, Tuple[str]],
        elem_type: Optional[Any] = None,
        shape: Optional[DYNAMIC_SHAPE] = None,
        is_dimension: bool = False,
        marker: str = "",
        default_initializer: Optional[Any] = None,
        device: Optional[int] = None,
    ) -> str:
        """
        Adds a tensor input to the onnx graph.

        :param name: name or tuple of names, in case, all inputs are create
            with the same element type and shape
        :param elem_type: element type
        :param shape: shape
        :param is_dimension: torch is using ``torch.SymInt`` to add a dynamic input
            to the graph
        :param marker: to known from this input was created
        :param default_initializer: add an initializer with the same name of the input
        :param device: device if known
        :return: input name
        """
        if isinstance(name, (tuple, list)):
            res = []
            for n in name:
                res.append(
                    self.make_tensor_input(
                        n,
                        elem_type,
                        shape,
                        is_dimension=is_dimension,
                        marker=marker,
                        default_initializer=default_initializer,
                    )
                )
            return res

        assert (
            self.as_function or elem_type
        ), f"elem_type is unknown for name={name!r}{self.get_debug_msg()}"
        add_node = lambda: None  # noqa: E731
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            if input_name != name:
                add_node = lambda: self.make_node(  # noqa: E731
                    "Identity",
                    [input_name],
                    [name],
                    name="make_tensor_input_id",
                )
        else:
            if is_dimension:
                # The convention is to have _dim_ in the name to tell
                # it is a dimension.
                input_name = f"{name}_dim_"
                if input_name != name:
                    add_node = lambda: self.make_node(  # noqa: E731
                        "Identity",
                        [input_name],
                        [name],
                        name="make_tensor_input_id",
                    )
            else:
                self.input_names.append(name)
                input_name = name

        self.current_input += 1
        elem_type = _get_type(elem_type)

        if self.update_dynamic_shape_when_input_name_is_defined:
            #
            # dynamic shapes were defined as tuple,
            # we need to propagate the information to the names
            # dynamic_dimensions_source={'dim': [{'axis': 1, 'input_name': 0}]}
            # Let's replace None by strings.
            #
            for dim_name, v in self.dynamic_dimensions_source.items():
                for d in v:
                    if isinstance(d["input_name"], int) and d["input_name"] == len(self.inputs):
                        d["input_name"] = input_name
                        if shape:
                            axis = d["axis"]
                            assert axis < len(shape), (
                                f"Unexpected shape={shape!r} and axis={axis}, "
                                f"dim_name={dim_name!r}, self.dynamic_dimensions_source="
                                f"{self.dynamic_dimensions_source} "
                                f"name={name!r}, input_name={input_name!r} "
                                f"self.dynamic_shapes={self.dynamic_shapes} "
                                f"self.input_names={self.input_names}"
                                f"{self.get_debug_msg()}"
                            )
                            dim_name_axis = self._torch_sym_int_to_str(shape[axis])
                            if dim_name != dim_name_axis and not isinstance(dim_name_axis, int):
                                assert (
                                    dim_name_axis not in self._dynamic_alias
                                    or self._dynamic_alias[dim_name_axis] == dim_name
                                ), (
                                    f"Alias mismatch for {dim_name_axis!r}, axis={axis}, "
                                    f"existing is {self._dynamic_alias[dim_name_axis]!r}, "
                                    f"new is {dim_name!r} "
                                    f"for input {input_name!r} and shape {shape!r}"
                                    f"{self.get_debug_msg()}"
                                )
                                self._add_dynamic_alias(dim_name_axis, dim_name)
                            shape = tuple(
                                dim_name if i == axis else shape[i] for i in range(len(shape))
                            )

        if shape is not None:
            shape = tuple(
                (sh if sh is not None else self.unique_dimension_name(input_name)) for sh in shape
            )
        dyn_shape = self.verify_dynamic_shape(shape, name=input_name, add=True)
        self._fill_dynamic_alias(shape, name)
        new_dyn_shape = self._fill_dynamic_alias(dyn_shape, name)
        if new_dyn_shape is not None:
            dyn_shape = tuple(
                (a if a is not None else b) for a, b in zip(new_dyn_shape, dyn_shape)
            )

        node = oh.make_tensor_value_info(input_name, elem_type, dyn_shape)
        self.inputs.append(node)
        self.set_name(input_name, marker=f"make_tensor_input_{marker}")
        if shape is not None:
            self._make_tensor_input_finalize(name, shape, dyn_shape)

        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_input] "
                f"{input_name}[{elem_type}:{dyn_shape}] -- marker={marker}"
            )
        assert (
            self.as_function or elem_type
        ), f"elem_type={elem_type!r} must be specified for input {name!r}"
        if shape is not None:
            self.set_shape(name, dyn_shape)
            if input_name != name:
                self.set_shape(input_name, dyn_shape)
        if elem_type is not None:
            self.set_type(name, elem_type)
            if input_name != name:
                self.set_type(input_name, elem_type)
        if device is not None:
            self.set_device(name, device)
            if input_name != name:
                self.set_device(input_name, device)

        node.doc_string += ".\n" + self._info_shape_type([name]) + "\n"
        add_node()

        if default_initializer is not None:
            init_name = self.add_initializer(
                name, value=default_initializer, source="default_initilizer"
            )
            assert init_name == name, (
                f"The initializer {name!r} should not be renamed into {init_name!r}"
                f"{self.get_debug_msg()}"
            )
        return name

    def make_tensor_sequence_input(
        self,
        name: str,
        elem_type: Any,
        shape: DYNAMIC_SHAPE,
        marker: str = "",
    ) -> str:
        """
        Adds a tensor input to the onnx graph.

        :param name: name
        :param elem_type: element type
        :param shape: shape
        :param marker: to known from this input was created
        :return: input name
        """
        add_node = lambda: None  # noqa: E731
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            if input_name != name:
                add_node = lambda: self.make_node(  # noqa: E731
                    "Identity",
                    [input_name],
                    [name],
                    name="make_tensor_input_id",
                )
        else:
            self.input_names.append(name)
            input_name = name

        self.current_input += 1
        elem_type = _get_type(elem_type)

        if self.update_dynamic_shape_when_input_name_is_defined:
            #
            # dynamic shapes were defined as tuple,
            # we need to propagate the information to the names
            # dynamic_dimensions_source={'dim': [{'axis': 1, 'input_name': 0}]}
            for dim_name, v in self.dynamic_dimensions_source.items():
                for d in v:
                    if isinstance(d["input_name"], int) and d["input_name"] == len(self.inputs):
                        d["input_name"] = input_name
                        if shape:
                            axis = d["axis"]
                            assert axis < len(shape), (
                                f"Unexpected shape={shape!r} and axis={axis}, "
                                f"dim_name={dim_name!r}, self.dynamic_dimensions_source="
                                f"{self.dynamic_dimensions_source} "
                                f"name={name!r}, input_name={input_name!r} "
                                f"self.dynamic_shapes={self.dynamic_shapes} "
                                f"self.input_names={self.input_names}"
                                f"{self.get_debug_msg()}"
                            )
                            dim_name_axis = self._torch_sym_int_to_str(shape[axis])
                            if dim_name != dim_name_axis:
                                assert (
                                    dim_name_axis not in self._dynamic_alias
                                    or self._dynamic_alias[dim_name_axis] == dim_name
                                ), (
                                    "Alias mismatch for {dim_name_axis!r}, existing is "
                                    f"{self._dynamic_alias[dim_name_axis]!r}, "
                                    f"new is {dim_name!r} "
                                    f"for input {input_name!r} and shape {shape!r}"
                                    f"{self.get_debug_msg()}"
                                )
                                self._add_dynamic_alias(dim_name_axis, dim_name)
                            shape = tuple(
                                dim_name if i == axis else shape[i] for i in range(len(shape))
                            )

        dyn_shape = self.verify_dynamic_shape(shape, name=input_name, add=True)
        self._fill_dynamic_alias(shape, name)
        new_dyn_shape = self._fill_dynamic_alias(dyn_shape, name)
        if new_dyn_shape is not None:
            dyn_shape = tuple(
                (a if a is not None else b) for a, b in zip(new_dyn_shape, dyn_shape)
            )

        node = oh.make_tensor_sequence_value_info(input_name, elem_type, dyn_shape)
        self.inputs.append(node)
        self.set_name(input_name, marker=f"make_tensor_sequence_input_{marker}")
        if shape is not None:
            self._make_tensor_input_finalize(name, shape, dyn_shape)

        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_sequence_input] "
                f"{input_name}[{elem_type}:{dyn_shape}] -- marker={marker}"
            )
        assert (
            self.as_function or elem_type
        ), f"elem_type={elem_type!r} must be specified for input {name!r}"
        self.set_sequence(name, elem_type, [shape] if shape else None)
        node.doc_string += ".\n" + self._info_shape_type([name]) + "\n"
        add_node()
        return name

    def _dynamic_to_str(
        self, obj: Any, exc: bool = True, register_if_not_exist: bool = False
    ) -> Optional[str]:
        if register_if_not_exist:
            res = self._dynamic_to_str(obj, exc, register_if_not_exist=False)
            if res is None:
                return res
            if res not in self.dynamic_objects:
                self.add_dynamic_object(res, res, parse=register_if_not_exist)
            return res

        if isinstance(obj, str):
            return obj
        if isinstance(obj, self.torch.export.dynamic_shapes._DerivedDim):
            return obj.__name__
        if isinstance(obj, self.torch.export.dynamic_shapes._Dim):
            return obj.__name__
        if isinstance(obj, self.torch.SymInt):
            if isinstance(obj.node, str):
                return obj.node
            i = obj.node._expr
            if "sympy" in str(type(i)) or (
                hasattr(i, "__module__") and i.__module__ == "torch.utils._sympy.functions"
            ):
                return str(i).replace(" ", "")
            if isinstance(i, self.torch.SymInt):
                return self._dynamic_to_str(
                    i, exc=exc, register_if_not_exist=register_if_not_exist
                )
            if exc:
                raise AssertionError(
                    f"Object has {type(obj)} but could not find a dynamic interpretation"
                )
            return None
        if isinstance(obj, self.WrapDim):
            return obj.name
        raise AssertionError(f"Unexpected type {type(obj)} to convert into string")

    def _is_dynamic_dimension(self, dyn: Any) -> bool:
        return isinstance(
            dyn,
            (
                str,
                self.torch.export.dynamic_shapes._DerivedDim,
                self.torch.export.dynamic_shapes._Dim,
                self.torch.SymInt,
            ),
        )

    def _add_dynamic_alias(self, key, value):
        assert not isinstance(key, int), (
            f"An integer should not be an alias, key={key}, value={value}"
            f"{self.get_debug_msg()}"
        )
        self._dynamic_alias[key] = value

    def _fill_dynamic_alias(
        self, dyn_shape: Optional[Tuple[Any, ...]], name: str
    ) -> Optional[Tuple[Any, ...]]:
        if dyn_shape is None:
            return None
        res = []
        for pos, k in enumerate(dyn_shape):
            if not self._is_dynamic_dimension(k):
                res.append(None)
                continue
            alias = self._dynamic_to_str(k, exc=False)
            if alias is None:
                res.append(None)
                continue
            if not self.dynamic_shapes or name not in self.dynamic_shapes:
                res.append(None)
                continue
            ds = self.dynamic_shapes[name]
            if pos not in ds:
                res.append(None)
                continue
            dim = ds[pos]
            sdim = self._dynamic_to_str(dim, register_if_not_exist=True)
            if sdim != alias:
                self._add_dynamic_alias(alias, sdim)
            res.append(sdim)
        return tuple(res)

    def _make_tensor_input_finalize(self, name: str, shape: Any, dyn_shape: Any):
        tuple_shape = tuple(shape)
        assert len(tuple_shape) == len(
            dyn_shape
        ), f"mismatch between shape={shape}, dynamic_shape={dyn_shape}"
        for _idim, (a, b) in enumerate(zip(tuple_shape, dyn_shape)):
            if isinstance(a, int) and isinstance(b, int):
                assert a == b, (
                    f"Unexpected shape mismatch shape={shape}, "
                    f"dyn_shape={dyn_shape}{self.get_debug_msg()}"
                )
                continue

            if isinstance(a, str) and isinstance(b, str):
                assert a == b, (
                    f"Unexpected shape mismatch shape={shape}, "
                    f"dyn_shape={dyn_shape}{self.get_debug_msg()}"
                )
                sb = b
                if sb not in self.dynamic_objects:
                    self.add_dynamic_object(sb, sb)
                continue

            self._dynamic_to_str(b, register_if_not_exist=True)
            self._dynamic_to_str(a, register_if_not_exist=True)

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[STATIC_SHAPE] = None,
        indexed: bool = True,
        is_dimension: Optional[bool] = None,
        allow_untyped_output: bool = False,
        doc_string: str = "",
        device: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Adds a tensor output to the onnx graph.

        :param name: name
        :param elem_type: element type
        :param shape: shape
        :param indexed: the name must be indexed?
        :param is_dimension: torch is using torch.SymInt to add a dynamic input
            to the graph
        :param allow_untyped_output: allow untyped output even if it is not a function
        :param doc_string: doc string
        :param device: device if known
        :return: output name
        """
        assert self.as_function or is_dimension is not None, (
            f"is_dimension must be specified for output name={name!r}, "
            f"elem_type={elem_type}, shape={shape!r}."
        )
        if isinstance(name, list):
            assert not is_dimension, f"name={name!r} not compatible with is_dimension=True"
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
                assert self.has_name(
                    n
                ), f"Output {n!r} among {name} not found{self.get_debug_msg()}"
            return res

        assert (
            self.as_function or not indexed or "_" in name
        ), f"Name {name!r} is not indexed like 'output_0'{self.get_debug_msg()}"
        assert (is_dimension and "_dim_" in name) or (not is_dimension and "_dim_" not in name), (
            f"Inconsistence for input {name!r}, "
            f"elem_type={elem_type}, shape={shape!r}, "
            f"is_dimension={is_dimension}"
        )

        elem_type = _get_type(elem_type, False)
        if not self.as_function and not allow_untyped_output and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
        if shape is not None:
            shape = tuple(
                (sh if sh is not None else self.unique_dimension_name(name)) for sh in shape
            )
        dyn_shape = self.verify_shape(shape, name=name, elem_type=elem_type)

        index_output = len(self.outputs)
        if index_output < len(self.user_defined_output_names):
            new_name = self.user_defined_output_names[index_output]
            assert not self.has_name(new_name), (
                f"Cannot rename one output into {new_name!r} as it is already used"
                f"{self.get_debug_msg()}"
            )
            self.make_node(
                "Identity",
                [name],
                [new_name],
                name=self.unique_node_name("make_tensor_output_rename"),
            )
            if self.verbose > 1:
                print(
                    f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                    f"rename {name} into {new_name}"
                )
            old_name = name
            name = new_name
        else:
            old_name = None

        if elem_type != 0:
            node = oh.make_tensor_value_info(name, elem_type, dyn_shape, doc_string=doc_string)
        else:
            # We skip the shape as well.
            node = ValueInfoProto()
            node.name = name
            if doc_string:
                node.doc_string = doc_string

        node.doc_string += ".\n" + self._info_shape_type([name]) + "\n"
        self.outputs.append(node)
        assert self.has_name(name), f"Output {name!r} not found{self.get_debug_msg()}"
        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                f"{name}[{elem_type}: {dyn_shape}]"
            )
        if dyn_shape:
            self.set_shape(name, dyn_shape, allow_zero=True)
            if old_name:
                self.set_shape(old_name, dyn_shape)
        if elem_type:
            self.set_type(name, elem_type)
            if old_name:
                self.set_type(old_name, elem_type)
        if device is not None:
            self.set_device(name, device)
            if old_name:
                self.set_device(old_name, device)
        return name

    def make_tensor_value_info_from_name(self, name: str, verbose: int = 0) -> ValueInfoProto:
        """
        Creates a ValueInfoProto based on the information available on
        the graph.
        """
        if self.has_type(name) and self.has_rank(name):
            if self.has_shape(name):
                out = oh.make_tensor_value_info(name, self.get_type(name), self.get_shape(name))
                if verbose > 1:
                    print(
                        f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                        f"{name}[{self.get_type(name)}:R{self.get_shape(name)}]"
                    )
            else:
                out = oh.make_tensor_value_info(
                    name, self.get_type(name), [None] * self.get_rank(name)
                )
                if verbose > 1:
                    print(
                        f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                        f"{name}[{self.get_type(name)}:R{self.get_rank(name)}]"
                    )
        else:
            out = oh.make_value_info(name, TypeProto())
            if verbose > 1:
                print(f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}")
        return out

    def select_outputs(self, output_names: List[str]):
        """
        Selects new outputs. The type is assumed to be unknown.
        The method only wipes out the outputs to replace them by
        others. It assumes the unused nodes are removed afterwards.

        :param output_names: new outputs
        """
        new_outputs = []
        for name in output_names:
            out = self.make_tensor_value_info_from_name(name, verbose=self.verbose)
            new_outputs.append(out)

        self.outputs = new_outputs

    def verify_shape(
        self,
        shape: Optional[DYNAMIC_SHAPE],
        elem_type: int,
        name: Optional[str] = None,
    ) -> Optional[DYNAMIC_SHAPE]:
        assert isinstance(elem_type, int), f"elem_type must be an integer not {type(elem_type)}"
        assert shape is None or isinstance(
            shape, tuple
        ), f"Shape must be a tuple not {type(shape)}"
        if shape is None:
            return None
        # costly
        # assert is_static_shape(shape) or self.is_dynamic_shape(shape, allow_none=True), (
        #     f"Shape={shape} is not a shape (type={[type(i) for i in shape]}), "
        #     f"name={name!r}, elem_type={elem_type}{self.get_debug_msg()}"
        # )
        new_shape = self.verify_dynamic_shape(shape, name=name)
        return tuple(
            (
                s
                if isinstance(s, int)
                else (
                    simplify_expression(s)
                    if len(s) < 100
                    else self.unique_dimension_name("NEWDIMLONG")
                )
            )
            for s in new_shape
        )

    def _get_symbol(self, i: str) -> str:
        k = 0
        if self.has_type(i):
            k += 1
        if self.has_rank(i):
            k += 2
        if self.has_shape(i):
            k += 4
        return k

    def _debug_string_inputs(
        self, inputs: List[str], outputs: List[str], align: Optional[int] = None
    ) -> str:
        """
        Meaning:

        - ``"-"``: (0) none
        - ``"T"``: (1) type
        - ``"R"``: (2) rank
        - ``"U"``: (3) rank + type
        - ``"S"``: (4) shape
        - ``"V"``: (5) shape + type
        - ``"W"``: (6) shape + rank
        - ``"#"``: (7) shape + type + rank
        """
        st = ""
        c = "-TRUSVW#"
        for i in inputs:
            st += c[self._get_symbol(i)]
        st += ":"
        for o in outputs:
            st += c[self._get_symbol(o)]
        if align and len(st) < align:
            st += " " * (align - len(st))
        return st

    def _check_op_type(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        domain: str,
        name: str,
        attributes: Optional[List[AttributeProto]] = None,
        **kwargs: Dict[str, Any],
    ):
        assert (
            not op_type.startswith("Reduce")
            or domain != ""
            or (len(inputs) == 2 and "axes" not in kwargs)
            or len(inputs) == 1
        ), (
            f"Operator {op_type!r} defines twice the axes, kwargs={kwargs}, "
            f"len(inputs)={len(inputs)}{self.get_debug_msg()}"
        )
        assert (
            op_type != "Cast"
            or domain != ""
            or ("to" in kwargs and kwargs["to"] is not None)
            or (attributes is not None and "to" in set(att.name for att in attributes))
        ), (
            f"Operator Cast needs arguments to but kwargs={kwargs}, name={name!r}"
            f"{self.get_debug_msg()}"
        )
        assert op_type != "Cast" or domain != "" or (len(inputs) == 1 and inputs[0]), (
            f"Operator Cast has one empty input, name={name!r}, inputs={inputs}"
            f"{self.get_debug_msg()}"
        )
        assert op_type != "Concat" or domain != "" or len(inputs) > 1, (
            f"Concatenation of zero or one input is not necessary, "
            f"len(inputs)={len(inputs)}{self.get_debug_msg()} "
        )
        if self.main_opset <= 11:
            assert op_type != "Squeeze" or domain != "" or len(inputs) == 1, (
                f"Operator Squeeze is not correclty specified for opset "
                f"{self.main_opset}, inputs={inputs}, kwargs={kwargs}, "
                f"atts={attributes}{self.get_debug_msg()}"
            )
        else:
            n_entries = len(inputs) + len(attributes or []) + len(kwargs)
            assert op_type != "Squeeze" or domain != "" or n_entries in (1, 2), (
                f"Operator Squeeze is not correclty specified for opset "
                f"{self.main_opset}, n_entries={n_entries}, "
                f"inputs={inputs}, kwargs={kwargs}, "
                f"atts={attributes}{self.get_debug_msg()}"
            )
        assert (
            op_type not in {"NegXplus1", "ReplaceZero"} or domain != ""
        ), f"Type={op_type!r} and domain {domain!r} mismatch{self.get_debug_msg()}"
        assert (
            op_type != "If"
            or domain != ""
            or (
                len(kwargs["then_branch"].output)
                == len(outputs)
                == len(kwargs["else_branch"].output)
                and len(kwargs["then_branch"].input) == 0 == len(kwargs["else_branch"].input)
            )
        ), (
            f"Node 'If' has an unexpected number of inputs or outputs."
            f"\nIf({', '.join(inputs)}) -> {outputs}"
            f"\nkwargs={pprint.pformat(kwargs)}{self.get_debug_msg()}"
        )
        assert outputs != [""], (
            f"Unexpected value for outputs={outputs}, op_type={op_type!r}, domain={domain!r}, "
            f"inputs={inputs}, attributes={attributes}{self.get_debug_msg()}"
        )

    def do_not_remove(self, node: NodeProto) -> bool:
        """Tells if a node should be removed or not."""
        return node.name.startswith("_DONOTREMOVE_")

    def unique_names(
        self,
        outputs: int,
        op_type: str,
        inputs: List[str],
        attributes: Optional[List[AttributeProto]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if (
            inputs
            and isinstance(inputs[0], str)
            and op_type
            in {
                "Reshape",
                "Shape",
                "Size",
                "Transpose",
                "Cast",
                "Squeeze",
                "Unsqueeze",
            }
        ):

            def _get(name, attributes, kwargs):
                if kwargs and name in kwargs:
                    return kwargs[name]
                for att in attributes or []:
                    if att.name == name:
                        if att.type == AttributeProto.INTS:
                            return tuple(att.ints)
                        if att.type == AttributeProto.INT:
                            return att.i
                return None

            if op_type == "Cast":
                to = _get("to", attributes, kwargs) or "?"
                name = f"{inputs[0]}::C{to}"
            elif op_type == "Transpose":
                perm = _get("perm", attributes, kwargs) or "?"
                name = f"{inputs[0]}::T{''.join(map(str,perm))}"
            elif op_type == "Shape":
                start = _get("start", attributes, kwargs)
                end = _get("end", attributes, kwargs)
                name = f"{inputs[0]}::Shape{start or ''}:{end or ''}"
            elif op_type == "Size":
                name = f"{inputs[0]}::Size"
            else:
                short = {"Reshape": "RSh", "Squeeze": "Sq", "Unsqueeze": "UnSq"}[op_type]
                cst = (
                    self.compute_constant(
                        inputs[1], exc=False, only_array=True, allow_empty=True
                    )[0]
                    if len(inputs) > 1 and self.is_constant(inputs[1])
                    else None
                )
                name = (
                    f"{inputs[0]}::{short}"
                    if cst is None
                    else f"{inputs[0]}::{short}{'x'.join(map(str,cst))}"
                )
            output_names = [name]
        else:
            lower = f"_onx_{op_type.lower()}_"
            if inputs and isinstance(inputs[0], str) and inputs[0]:
                # The new name tries to keep track of the first input,
                # usually the most meaningful.
                lower = (
                    f"{lower}{inputs[0][5:]}"
                    if inputs[0].startswith("_onx_")
                    else f"{lower}{inputs[0]}"
                )
            output_names = [lower] if outputs == 1 else [f"{lower}_{i}" for i in range(outputs)]
        return [self.unique_name(o) for o in output_names]

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        check: Optional[bool] = None,
        name: Optional[str] = None,
        sts: Optional[Dict[str, Any]] = None,
        do_not_remove: bool = False,
        insert_position: Optional[int] = None,
        metadata_props: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Adds a node in the graph.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: output names, may be None, in that case,
            the builder chooses them for the user
        :param domain: domain
        :param attributes: list of attributes to add as AttributeProto
        :param check: do some verification
        :param name: node name
        :param sts: if not specified, tries to set the shape and the type of
            the new results aftr the node is added, it is not possible
            for every node, there is no tool which determines the output shape
            of just one node
        :param do_not_remove: prevent this node from being removed
        :param insert_position: insert the node at the end (None) or
            at the top (HEAD).
        :param kwargs: additional attributes to add the node
        :param metadata_props: additional information to store into the node
        :return: output names
        """
        if self._debug_local_function and domain:
            print(
                f"[GraphBuilder-{self._hash()}.make_node-f?] {op_type}[{domain}] "
                f"({', '.join(inputs)}) -> {', '.join(outputs)}"
            )
        assert name is not None and not name.startswith("None"), (
            f"It is good practice to give every node a name so that is "
            f"easier to see where this node is created but name={name!r} "
            f"and op_type={op_type!r}."
        )
        assert (
            not kwargs or not attributes
        ), f"Only attributes or kwargs can be filled for node {op_type!r}."
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if isinstance(outputs, int):
            assert outputs > 0, f"outputs={outputs} must be > 0."
            assert isinstance(op_type, str), f"Unexpected type {type(op_type)}: {op_type}"
            output_names = self.unique_names(outputs, op_type, inputs, attributes, kwargs)
        elif isinstance(outputs, str):
            output_names = [outputs]
        else:
            output_names = outputs
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs, kwargs = self._partial_rewrite_opset_version(
            op_type, inputs, kwargs, domain=domain
        )

        if self.verbose == 2:
            print(
                f"[GraphBuilder-{self._hash()}.1.make_node]"
                f"[{self._debug_string_inputs(inputs, output_names)}] "
                f"{op_type}: {inputs}->{outputs}"
            )

        if op_type == "Identity" and inputs == output_names:
            # No need.
            return output_names[0]

        if check is not False:
            for i in inputs:
                assert isinstance(i, str), (
                    f"Unexpected type {type(i)} in {inputs}, op_type={op_type!r}, "
                    f"name={name!r}, {self.get_debug_msg()}"
                )
                if i == "":
                    # Optional input.
                    continue
                assert self.has_name(i), (
                    f"Input {i!r} does not exist for operator {op_type!r}, "
                    f"inputs={inputs}, outputs={outputs}, name={name!r} "
                    f"({self._hash()}){self.get_debug_msg()}"
                )
            for i in output_names:
                assert isinstance(
                    i, str
                ), f"Unexpected type {type(i)} in {output_names}{self.get_debug_msg()}"
                if i == "":
                    # Optional output.
                    continue
                assert not self.has_name(i), (
                    f"Output {i!r} already exists for operator {op_type!r} "
                    f"({self._hash()}), output_names={output_names}, inputs={inputs}"
                    f"{self.get_debug_msg()}"
                )
        if check is True:
            for i in inputs:
                assert self.has_shape(i), f"Input {i!r} has no known shape."
                assert self.has_type(i), f"Input {i!r} has no known type."

        if do_not_remove:
            name = self.unique_node_name(f"_DONOTREMOVE_{name or ''}")
        elif name:
            name = self.unique_node_name(name)

        self._check_op_type(
            op_type,
            inputs,
            outputs,
            domain=domain,
            name=name,
            attributes=attributes,
            **kwargs,
        )

        # break?
        # if op_type == "Sub":
        #     print("op_type", op_type, inputs, output_names)
        #     raise AssertionError(f"MANUAL BREAK{self.get_debug_msg()}")
        if self._debug_node_output and self._debug_node_output in output_names:
            raise AssertionError(
                f"Stop requested as {self._debug_node_output!r} appears in "
                f"{op_type}({', '.join(inputs)}) -> {', '.join(output_names)}"
                f"{self.get_debug_msg()}"
            )

        if self._debug_node_type and op_type == self._debug_node_type:
            raise AssertionError(
                f"Stop requested as {self._debug_node_type!r} appears in "
                f"{op_type}({', '.join(inputs)}) -> {', '.join(output_names)}"
                f"\nkwargs={pprint.pformat(kwargs)}{self.get_debug_msg()}"
            )

        # next
        try:
            node = oh.make_node(op_type, inputs, output_names, domain=domain, name=name, **kwargs)
        except TypeError as e:
            iti = [type(i) for i in inputs]
            ito = [type(o) for o in outputs] if isinstance(outputs, (tuple, list)) else outputs
            raise TypeError(
                f"A node {op_type!r} cannot be created with "
                f"inputs={inputs} (types={iti}), outputs={outputs} (types={ito}), "
                f"domain={domain!r}, kwargs={kwargs}."
            ) from e

        assert len(node.output) == len(set(node.output)) or "" in node.output, (
            f"Repeated outputs for node {node.op_type}({', '.join(node.input)}) -> "
            f"{', '.join(node.output)}"
        )
        if metadata_props:
            oh.set_metadata_props(node, metadata_props)

        if attributes:
            node.attribute.extend(attributes)

        if node.domain == "" and node.op_type in {"Constant", "ConstantOfShape"}:
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                t = node.attribute[0].t
                size = np.prod(t.dims)
                assert size < self.optimization_options.constant_size, (
                    f"A node Constant is created with a size {size} greater than "
                    f"the limit {self.optimization_options.constant_size}"
                    f"{self.get_debug_msg()}"
                )

            # A exact constant may be already existing,
            # In that case, we just return an identity node.
            origin = self.is_exact_same_constant(node)
            if origin is not None:
                if self.verbose > 2:
                    print(
                        f"[GraphBuilder-{self._hash()}.2.make_node] "
                        f"duplicated constant detected for "
                        f"{node.op_type}: {node.input}->{node.output}"
                    )
                node = oh.make_node(
                    "Identity",
                    [origin.output[0]],
                    [node.output[0]],
                    name=self.unique_node_name(".make_node"),
                )
                self.constants_alias_[node.output[0]] = origin.output[0]
            else:
                self.add_constant_node(node)

        # constant handling, shape, type
        self._make_node_set_type_shape_constant(node, sts=sts)

        if self.verbose > 3:
            print(
                f"[GraphBuilder-{self._hash()}.3.make_node] "
                f"[{self._debug_string_inputs(node.input, output_names)}] "
                f"{node.op_type}: {node.input}->{node.output}"
            )

        # shape inference
        shape_set = self.simple_update_value_shape_with_node(node)

        # add the node
        for o in node.output:
            if o == "":
                continue
            self.set_name(o, f"make_node_{op_type}_{o}")
        if insert_position == "HEAD":
            self.nodes.insert(0, node)
        else:
            self.nodes.append(node)

        if not shape_set or (node.output and not self.has_shape(node.output[0])):
            # second try
            self._make_node_set_type_shape(node)

        node.doc_string += ".\n" + self._info_shape_type(node.output) + "\n"

        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def _another_pass_at_shape_inference(self):
        for node in self.nodes:
            if any(not self.has_shape(o) for o in node.output):
                self._make_node_set_type_shape(node)

    def _info_shape_type(self, outputs: List[str]) -> str:
        rows = []
        for o in outputs:
            st = f"-T{self.get_type(o)}:" if self.has_type(o) else "T?:"
            if self.has_shape(o):
                st += "x".join(map(str, self.get_shape(o)))
            elif self.has_rank(o):
                st += f"R{self.get_rank(o)}"
            rows.append(st)
        return "/".join(rows)

    @property
    def last_added_node(self):
        return self.nodes[-1] if self.nodes else None

    def _partial_rewrite_opset_version(
        self, op_type: str, inputs: List[str], kwargs: Dict[str, Any], domain: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        if domain == "":
            opset = self.opsets[""]
            if op_type == "Unsqueeze":
                if opset < 13 and len(inputs) == 2:
                    assert isinstance(
                        inputs[1], (list, np.ndarray)
                    ), f"Unexpected type for axis={inputs[1]} and operator Unsqueeze"
                    kwargs["axes"] = (
                        [inputs[1]] if isinstance(inputs[1], int) else inputs[1].tolist()
                    )
                    inputs = inputs[:1]
                elif opset >= 13 and len(inputs) == 1:
                    name = self.make_initializer(
                        "",
                        np.array(kwargs["axes"], dtype=np.int64),
                        source="GraphBuilder._partial_rewrite_opset_version.axes",
                    )
                    inputs.append(name)
                    del kwargs["axes"]
        return inputs, kwargs

    def get_attribute(
        self, node: NodeProto, att_name: str, exc: bool = True
    ) -> Optional[AttributeProto]:
        """Returns an attribute for a node."""
        for att in node.attribute:
            if att.name == att_name:
                return att
        assert not exc, (
            f"Unable to find attribute {att_name!r} for node "
            f"type {node.op_type!r} in node {node}"
        )
        return None

    def get_attribute_with_default(self, node: NodeProto, name: str, default_value: Any) -> Any:
        """
        Returns an attribute or its default value if missing.

        :param node: node
        :param name: attribute name
        :param default_value: default value
        :return: value
        """
        for att in node.attribute:
            if att.name == name:
                if att.type == AttributeProto.INT:
                    return att.i
                if att.type == AttributeProto.INTS:
                    return list(att.ints)
                if att.type == AttributeProto.FLOAT:
                    return att.f
                if att.type == AttributeProto.FLOATS:
                    return list(att.floats)
                if att.type == AttributeProto.STRING:
                    return att.s
                raise TypeError(
                    f"Not implemented for attribute name {att.name!r}, attribute={att}"
                )
        return default_value

    def get_attributes_with_default(self, node: NodeProto, **default_values) -> Dict[str, Any]:
        """
        Returns int or float attributes. If missing, the default value is returned
        if it is not None.

        :param node: node
        :param default_values: default values
        """
        res = {}
        for att in node.attribute:
            if att.name in default_values:
                if att.type == AttributeProto.INT:
                    res[att.name] = att.i
                elif att.type == AttributeProto.INTS:
                    res[att.name] = list(att.ints)
                elif att.type == AttributeProto.FLOAT:
                    res[att.name] = att.f
                elif att.type == AttributeProto.FLOATS:
                    res[att.name] = list(att.floats)
                elif att.type == AttributeProto.STRING:
                    res[att.name] = att.s
                else:
                    raise TypeError(
                        f"Not implemented for attribute name {att.name!r}, attribute={att}"
                    )
        for k, v in default_values.items():
            if k not in res and v is not None:
                res[k] = v
        res = {k: v for k, v in res.items() if v is not None}
        return res

    def make_nodes(
        self,
        builder: "GraphBuilder",
        input_names: List[str],
        output_names: List[str],
        prefix: str = "",
        function_options: Optional[FunctionOptions] = None,
        optimize: bool = False,
        force_rename_with_prefix: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """
        Appends all nodes and initializers from another builder.
        Handles the renaming of results.
        The content stored in 'builder' is modified inplace to avoid copying.

        :param builder: other builder
        :param input_names: input names
        :param output_names: output names
        :param prefix: prefix all name from this builder if `function_options` is None
        :param function_options: defines how to create a local function if needed
        :param optimize: optimize the function
        :param force_rename_with_prefix: even if a parameter name should be renamed,
            the prefix *name* is used
        :return: output names
        """
        if function_options is not None and function_options.export_as_function:
            if self._debug_local_function:
                print(
                    f"[GraphBuilder-{self._hash()}.make_nodes-f] "
                    f"function_options={function_options}, "
                    f"optimize={optimize}"
                )
                print(f"[GraphBuilder-{self._hash()}.make_nodes-f] input_names={input_names}")
            new_inits, (fdomain, fname) = self.make_local_function(
                builder, function_options=function_options, optimize=optimize
            )
            if self._debug_local_function:
                print(f"[GraphBuilder-{self._hash()}.make_nodes-f] new_inits={new_inits}")
            self.make_node(
                fname,
                [*input_names, *new_inits],
                output_names,
                domain=fdomain,
                check=False,
                name=fname,
            )

            # Shape information, needs to handle multiple outputs
            # hopefully, the interpreter fills this information with what it knows
            # fproto = self.functions[fdomain, fname]
            # for o, no in zip(fproto.output, output_names):
            #    if builder.has_shape(o):
            #        shape = builder.get_shape(o)
            #        if None in shape:
            #            self.set_rank(no, len(shape))
            #        else:
            #            self.set_shape(no, shape)
            #    if builder.has_type(o):
            #        self.set_type(no, builder.get_type(o))

            if fdomain not in self.opsets:
                self.opsets[fdomain] = 1
        else:
            renaming = {}
            for init, value in builder.initializers_dict.items():
                if init in builder._parameter_renaming:
                    # Its copy already exists.
                    continue
                if init in builder._parameter_norename:
                    name = init
                    if self.has_name(init):
                        if force_rename_with_prefix:
                            name = f"{force_rename_with_prefix}.{name}"
                            assert not self.has_name(name), (
                                f"Parameter {init!r} was renamed into {name!r} "
                                f"but it already exists{self.get_debug_msg()}"
                            )
                        else:
                            assert not self.has_name(init), (
                                f"Parameter {init!r} must be renamed as another one "
                                f"already exists{self.get_debug_msg()}"
                            )
                else:
                    name = self.unique_name(f"{prefix}{init}")
                renaming[init] = name
                if isinstance(value, TensorProto):
                    value.name = name
                else:
                    assert "FakeTensor" not in str(type(value)), (
                        f"FakeTensor {name!r} cannot be an initializer {type(value)}"
                        f"{self.get_debug_msg()}"
                    )
                src = (
                    ""
                    if init not in builder.initializers_dict_sources
                    or not builder.initializers_dict_sources[init].source
                    else f"##{builder.initializers_dict_sources[init].source}"
                )
                self.add_initializer(
                    name,
                    value,
                    itype=builder._known_types[init],
                    shape=builder._known_shapes[init],
                    source=f"GraphBuilder.make_nodes/from{init}{src}",
                )

            for k, v in builder.dynamic_objects.items():
                self.make_dynamic_object(k, v)

            assert len(input_names) == len(
                builder.inputs
            ), f"Inconsistency between input_names={input_names} and inputs={builder.inputs}"
            for name, inp in zip(input_names, builder.inputs):
                new_name = self.unique_name(f"{prefix}{inp.name}")
                renaming[inp.name] = new_name
                self.make_node("Identity", [name], [new_name], name=".make_nodes")

            for node in builder.nodes:
                assert name is not None and not name.startswith("None"), (
                    f"It is good practice to give every node a name so that is "
                    f"easier to see where this node is created but name={name!r} "
                    f"and op_type={node.op_type!r}."
                )
                new_inputs = [renaming[builder._parameter_renaming.get(i, i)] for i in node.input]
                new_outputs = [self.unique_name(f"{prefix}{o}") for o in node.output]
                for o, no in zip(node.output, new_outputs):
                    renaming[o] = no
                self.make_node(
                    node.op_type,
                    new_inputs,
                    new_outputs,
                    domain=node.domain,
                    attributes=node.attribute,
                    check=False,
                    name=node.name,
                )
                for o, no in zip(node.output, new_outputs):
                    if builder.has_shape(o):
                        shape = builder.get_shape(o)
                        if None in shape:
                            self.set_rank(no, len(shape))
                        else:
                            self.set_shape(no, shape)
                    if builder.has_type(o):
                        self.set_type(no, builder.get_type(o))
                    if builder.has_device(o):
                        self.set_device(no, builder.get_device(o))

            assert len(output_names) == len(builder.outputs), (
                f"Inconsistency between output_names={output_names} and "
                f"outputs={builder.outputs}, renaming={renaming}{self.get_debug_msg()}"
            )
            for name, out in zip(output_names, builder.outputs):
                self.make_node("Identity", [renaming[out.name]], [name], name=".make_nodes")

        # opsets and domains
        for o, v in builder.opsets.items():
            if o in self.opsets:
                assert self.opsets[o] == builder.opsets[o], (
                    f"Opset mismatch for domain {o!r}, "
                    f"{self.opsets[o]} != {builder.opsets[o]}."
                )
                continue
            self.opsets[o] = v

        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def _build_large_initializers(
        self,
        external_threshold: int,
        full_parameter_name: bool = True,
        subset: Optional[Set[str]] = None,
    ):
        assert isinstance(
            external_threshold, int
        ), f"Unexpected type {type(external_threshold)} for external_threshold"
        new_inits = {}
        large_inits = {}
        for k, v in self.initializers_dict.items():
            if subset and k not in subset:
                continue
            if self._parameter_renaming and (
                (full_parameter_name and k in self._parameter_renaming)
            ):
                # Those parameters are present under another name already.
                continue
            doc_string = (
                self.initializers_dict_sources[k].source
                if k in self.initializers_dict_sources
                and self.initializers_dict_sources[k].source
                else ""
            )
            itype = self.get_type(k)
            shape = self.get_shape(k)
            vshape = v.shape if hasattr(v, "shape") else tuple(v.dims)  # TensorProto
            assert not is_static_shape(shape) or shape == vshape, (
                f"Unexpected shape for initializer {k!r}, shape={shape!r}, itype={itype}, "
                f"initializer shape={v.shape}, dtype={v.dtype}, "
                f"{self.get_debug_msg()}"
            )
            shape = vshape
            size = np.prod(shape) * self.elem_size(itype)
            if size < external_threshold:
                new_inits[k] = v
            else:
                location = f"#{k}"
                nt = make_large_tensor_proto(location, k, itype, shape)
                nt.doc_string += doc_string
                new_inits[k] = nt
                large_inits[location] = v
        return new_inits, large_inits

    def _build_initializers(
        self,
        large_model: bool,
        switch_low_high: bool,
        external_threshold: Union[bool, int],
        full_parameter_name: bool = True,
        subset: Optional[Set[str]] = None,
    ) -> Tuple[List[TensorProto], Dict[str, TensorProto]]:
        """
        Builds initializers.

        :param large_model: build with a large container
        :param switch_low_high: invert low, high precision
        :param external_threshold: size to use when moving a tensor to the list of tensors
            stored outside the model, if can be False for none of them, true for all of them
            or a number, if the threshold is specified and large_model is False,
            then all tensors above this threshold are ignored
        :param full_parameter_name: keeps the full name for the parameters or not
        :param subset: only consider a subset of the initializers
        :return: a list of tensors to stored in the model,
            another list to tensors stored outside the model
        """
        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._build_initializers] "
                f"start with {len(self.initializers_dict)} initializers, "
                f"large_model={large_model}, external_threshold={external_threshold}"
            )

        init_dict, large_inits = (
            self._build_large_initializers(
                external_threshold, full_parameter_name=full_parameter_name, subset=subset
            )
            if large_model
            else (self.initializers_dict, {})
        )

        if switch_low_high:
            # Let's try to minimize the time.
            if self.verbose:
                print(f"[GraphBuilder-{self._hash()}._build_initializers] switch low/high order")
            initializer = []
            for k, v in init_dict.items():
                if subset and k not in subset:
                    continue
                if self._parameter_renaming and (
                    (full_parameter_name and k in self._parameter_renaming)
                ):
                    # Those parameters are present under another name already.
                    continue
                doc_string = (
                    self.initializers_dict_sources[k].source
                    if k in self.initializers_dict_sources
                    and self.initializers_dict_sources[k].source
                    else ""
                )
                if isinstance(v, TensorProto):
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder-{self._hash()}._build_initializers] "
                            f"TensorProto-{k}:{v.data_type}[{tuple(v.dims)}]"
                        )
                    v.doc_string += doc_string
                    initializer.append(v)
                    continue

                if self.verbose > 1:
                    print(
                        f"[GraphBuilder-{self._hash()}._build_initializers] "
                        f"<{v.__class__.__name__}>-{k}:{v.dtype}[{v.shape}]"
                    )
                if isinstance(v, np.ndarray):
                    itype = dtype_to_tensor_dtype(v.dtype)
                    if itype in {
                        TensorProto.BOOL,
                        TensorProto.STRING,
                        TensorProto.UNDEFINED,
                        TensorProto.COMPLEX64,
                        TensorProto.COMPLEX128,
                        getattr(TensorProto, "UINT4", 0),
                        getattr(TensorProto, "INT4", 0),
                    }:
                        t = from_array_extended(v, name=k)
                        t.doc_string += doc_string
                        initializer.append(t)
                        continue

                    from_np = True
                elif isinstance(v, np.float32):
                    # This should not happen.
                    t = from_array_extended(np.array([v], dtype=np.float32), name=k)
                    t.doc_string += doc_string
                    initializer.append(t)
                    continue
                elif isinstance(v, np.float16):
                    # This should not happen.
                    t = from_array_extended(np.array([v], dtype=np.float16), name=k)
                    t.doc_string += doc_string
                    initializer.append(t)
                    continue
                else:
                    assert isinstance(
                        v, self.torch.Tensor
                    ), f"tensor {k!r} has un unexpected type {type(v)}"
                    assert "FakeTensor" not in str(
                        type(v)
                    ), f"tensor {k!r} cannot be a FakeTensor: {type(v)}"
                    from_np = False
                    itype = dtype_to_tensor_dtype(v.dtype)

                # How to avoid a copy?
                if from_np:
                    tensor = TensorProto()
                    tensor.name = k
                    tensor.dims.extend(v.shape)
                    tensor.data_type = itype
                    tensor.raw_data = v.tobytes()
                else:
                    with self.maybe_disable_fake_tensor_mode():
                        tensor = proto_from_array(v, name=k, verbose=self.verbose)

                if hasattr(tensor, "doc_string"):
                    tensor.doc_string += doc_string
                initializer.append(tensor)

            if self.verbose:
                begin = time.perf_counter()
                print(
                    f"[GraphBuilder-{self._hash()}._build_initializers] "
                    f"done in {time.perf_counter() - begin}s "
                    f"with {len(initializer)} initializers, "
                    f"{len(large_inits)} large initializers"
                )
            return initializer, large_inits

        assert (
            not self.large_model
        ), "_build_initializers not implemented when large_model is True"
        large_inits = {}
        res = []
        for k, v in init_dict.items():
            if subset and k not in subset:
                continue
            if self._parameter_renaming and (
                (full_parameter_name and k in self._parameter_renaming)
            ):
                # Those parameters are present under another name already.
                continue
            doc_string = (
                self.initializers_dict_sources[k].source
                if k in self.initializers_dict_sources
                and self.initializers_dict_sources[k].source
                else ""
            )
            if isinstance(v, TensorProto):
                v.doc_string += doc_string
                res.append(v)
                continue
            if isinstance(v, self.torch.Tensor):
                # no string tensor
                t = self.from_array(v, name=k)
                t.doc_string += doc_string
                res.append(t)
                continue
            if isinstance(v, np.ndarray):
                if self.verbose > 2 and np.prod(v.shape) > 100:
                    print(
                        f"[GraphBuilder-{self._hash()}._build_initializers]"
                        f"from_array_extended:{k}:{v.dtype}[{v.shape}]"
                    )
                t = from_array_extended(v, name=k)
                t.doc_string += doc_string
                res.append(t)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type {type(v)} into a TensorProto."
            )
        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._build_initializers] "
                f"done in {time.perf_counter() - begin}s "
                f"with {len(res)} initializers, "
                f"{len(large_inits)} large initializers"
            )
        return res, large_inits

    def get_initializer_size(self, name: str) -> int:
        """
        Returns the size of an initializer.

        :param name: name
        :return: size
        """
        assert name in self.initializers_dict, f"Initializer {name!r} was not found."
        init = self.initializers_dict[name]
        if hasattr(init, "numel"):
            # torch.Tensor
            return init.numel()
        if hasattr(init, "size"):
            # numpy array
            return init.size
        if hasattr(init, "size"):
            # TensorProto
            return np.prod(init.dims)
        raise AssertionError(f"Unexpected type {type(init)} for initializer {name!r}")

    def get_debug_msg(self, limit: int = 1000) -> str:
        """
        Returns a string providing as much information as possible
        to help the developper understand why a conversion failed.

        :param limit: limit the string if the model is big
        :return: many pieces of informations about the on going conversion
        """

        def _align(s, length):
            if len(s) < length:
                s += " " * (length - len(s))
            return s

        def _dtype(t):
            if hasattr(t, "dtype"):
                return t.dtype
            if hasattr(t, "data_type"):
                return t.data_type
            raise RuntimeError(f"dtype unknown for type {type(t)}-{t}.")

        def _shape(t):
            if hasattr(t, "shape"):
                return t.dtype
            if hasattr(t, "dims"):
                return tuple(t.dims)
            raise RuntimeError(f"dtype unknown for type {type(t)}-{t}.")

        def _size(t):
            if hasattr(t, "numel"):
                return t.numel()
            if hasattr(t, "size"):
                return t.size
            if hasattr(t, "dims"):
                return np.prod(tuple(t.dims))
            raise RuntimeError(f"Size unknown for type {type(t)}-{t}.")

        def _values(t):
            if hasattr(t, "detach"):

                def is_allow_non_fake_inputs_enabled():
                    from torch._guards import detect_fake_mode

                    return detect_fake_mode(t)

                if is_allow_non_fake_inputs_enabled():
                    return "FakeTensorMode enabled"
                return t.detach().cpu().flatten().tolist()
            if hasattr(t, "size"):
                return t.ravel().tolist()
            if hasattr(t, "dims"):
                a = onh.to_array(t)
                return a.ravel().tolist()
            raise RuntimeError(f"Values unknown for type {type(t)}-{t}.")

        rows = ["", "--DEBUG--", "-- to print the exported program: PRINT_EXPORTED_PROGRAM=1", ""]
        hs = self._hash()
        rows.append(
            f"[GraphBuilder-{hs}] Message starts, there are "
            f"{len(self.initializers_dict)} initializers, "
            f"{len(self.nodes)} nodes, {len(self.inputs)} inputs, "
            f"{len(self.inputs)} outputs."
        )

        if self._implicit_decisions:
            rows.append("--IMPLICIT DECISIONS--")
            rows.extend(map(str, self._implicit_decisions))
        if self.functions:
            rows.append("--LOCAL FUNCTIONS--")
            for k, v in self.functions.items():
                rows.append(f"    {k[0]},{k[1]}({v.input}) -> {v.output}")
        if self.constraints_:
            rows.append("--CONSTRAINTS--")
            for a, b in assert_sorted(self.constraints_.items()):
                rows.append(f"    {a} = {b}")
        rows.append("--SHAPE--")
        rows.append("_dynamic_examples=")
        for i, (k, v) in enumerate(assert_sorted(self._dynamic_examples.items())):
            try:
                rows.append(f"   {k} = {v!r}")
            except AttributeError:
                rows.append(f"   {k} = ERR: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}")
            if i >= 10000:
                break
        rows.append("dynamic_objects=")
        for i, (k, v) in enumerate(assert_sorted(self.dynamic_objects.items())):
            try:
                rows.append(f"   {k} = {v!r}")
            except AttributeError:
                rows.append(f"   {k} = ERR: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}")
            if i >= 10000:
                break

        rows.append("dynamic_objects_rev=")
        for i, (k, v) in enumerate(assert_sorted(self.dynamic_objects_rev.items())):
            if isinstance(v, (list, tuple)):
                rows.append(f"   {k!r} = {type(v)}")
                for vv in v:
                    if isinstance(vv, tuple):
                        rows.append("     tuple")
                        for vvv in vv:
                            try:
                                rows.append(f"       {vvv!r}")
                            except AttributeError:
                                rows.append(
                                    f"       ERR**: {type(vvv)!r}:"
                                    f"{getattr(vvv, 'node', 'node=?')!r}"
                                )
                    else:
                        try:
                            rows.append(f"       {vv!r}")
                        except AttributeError:
                            rows.append(
                                f"       ERR*: {type(vv)!r}:{getattr(vv, 'node', 'node=?')!r}"
                            )
            else:
                try:
                    rows.append(f"   {k} = {v!r}")
                except AttributeError:
                    rows.append(f"   {k} = ERR-: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}")
            if i >= 10000:
                break

        rows.append(f"dynamic_dimensions_source={pprint.pformat(self.dynamic_dimensions_source)}")
        rows.append(
            f"dynamic_dimensions_source_flat="
            f"{pprint.pformat(self.dynamic_dimensions_source_flat)}"
        )
        rows.append(
            f"output_dynamic_dimensions_source_flat="
            f"{pprint.pformat(self.output_dynamic_dimensions_source_flat)}"
        )
        rows.append(f"dynamic_alias={pprint.pformat(self._dynamic_alias)[:10000]}")
        rows.append(f"dynamic_shapes={pprint.pformat(self.dynamic_shapes)[:10000]}")
        rows.append(f"_known_shapes={pprint.pformat(self._known_shapes)[:10000]}")
        rows.append(f"_known_types={pprint.pformat(self._known_types)[:10000]}")
        rows.append(f"_known_devices={pprint.pformat(self._known_devices)[:10000]}")
        rows.append(f"_context={pprint.pformat(sorted(self._context))[:10000]}")
        short_sh = {
            k: (v if (isinstance(v, tuple) and len(v) < 10) else string_type(v, with_shape=True))
            for k, v in self._known_value_shape.items()
        }
        rows.append(f"_known_value_shape={pprint.pformat(short_sh)[:10000]}")
        rows.append(
            f"_known_constants={pprint.pformat(list(assert_sorted(self.constants_))[:10000])}"
        )
        reminaing_ranks = {
            k: v for k, v in self._known_ranks.items() if k not in self._known_shapes
        }
        rows.append(f"_known_ranks (with no shape)={pprint.pformat(reminaing_ranks )[:10000]}")

        rows.append("--PARAMETERS--")
        rows.append("_parameter_renaming=")
        for i, (k, v) in enumerate(assert_sorted(self._parameter_renaming.items())):
            rows.append(f"   {k} = {v!r}")
            if i >= 10000:
                break
        rows.append("--TORCH-USERS--")
        for k, v in assert_sorted(self._registered_users.items()):
            rows.append(f"    {k} -> {v}")

        rows.append("--TORCH-SHAPES--")
        for kk, vv in self._known_torch_value.items():
            rows.append(
                f"    {kk}: {vv} --- "
                f"{self.get_type(kk) if self.has_type(kk) else ''}:"
                f"{self.get_rank(kk) if self.has_rank(kk) else ''}:"
                f"{self.get_shape(kk) if self.has_shape(kk) else ''}:"
            )
            if len(rows) > limit:
                rows.append("...")
                break

        rows.append("--ONNX--")
        for k, v in self._debug_msg.items():
            rows.append(f"-- {k} --")
            rows.append(pprint.pformat(v) if isinstance(v, dict) else str(v))
            if len(rows) > limit:
                rows.append("...")
                break

        rows.append(f"-- {len(self.inputs)} INPUTS")
        for io in self.inputs:
            shh = _nice_shape(io.type.tensor_type.shape)
            rows.append(
                f"[GraphBuilder-{hs}.1.make_tensor_input] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
            )
        rows.append(f"-- {len(self.initializers_dict)} INITIALIZERS")
        for name, init in self.initializers_dict.items():
            sval = "" if _size(init) > 5 else f":{_values(init)}"
            source = (
                self.initializers_dict_sources[name].source
                if name in self.initializers_dict_sources
                else "?"
            )
            rows.append(
                f"[GraphBuilder-{hs}.1.make_initializer] "
                f"{name}[{_dtype(init)}:{_shape(init)}{sval}] "
                f"- SOURCE: {source}"
            )
            if len(rows) > limit:
                rows.append("...")
                break

        for node in self.nodes:
            if node is None:
                continue
            if node.op_type == "Cast":
                ext = f">{node.attribute[0].i}"
            else:
                ext = ""
            rows.append(
                f"[GraphBuilder-{hs}.4.make_node] "
                f"{_align(node.name, 15)} "
                f"[{self._debug_string_inputs(node.input, node.output, 6)}] "
                f"{node.op_type}{ext}:{node.input}->{node.output}"
            )
            if node.op_type in {"Loop", "Scan", "If"}:
                for att in node.attribute:
                    if att.type == AttributeProto.GRAPH:
                        rows.append(
                            f"[GraphBuilder-{hs}.4.make_node] -- subgraph {att.name!r}..."
                        )
                        rows.append(textwrap.indent(pretty_onnx(att.g), "    "))
                        rows.append(
                            f"[GraphBuilder-{hs}.4.make_node] -- subgraph {att.name!r} -- done"
                        )

            if len(rows) > limit:
                rows.append("...")
                break

        rows.append(f"-- {len(self.outputs)} OUTPUTS")
        for io in self.outputs:
            shh = _nice_shape(io.type.tensor_type.shape)
            rows.append(
                f"[GraphBuilder-{hs}.1.make_tensor_output] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
            )

        rows.append(
            f"[GraphBuilder-{hs}] Message completed, there are "
            f"{len(self.initializers_dict)} initializers, "
            f"{len(self.nodes)} nodes, {len(self.inputs)} inputs, "
            f"{len(self.inputs)} outputs."
        )

        if self._parent:
            rows.extend(["############"] * 3)
            rows.append("### PARENT ###")
            rows.extend(["############"] * 3)
            rows.append(self._parent.get_debug_msg())
        res = "\n".join(rows)
        if os.environ.get("DUMPMSG", ""):
            if "process.graph_module" in self._debug_msg:
                with open(f"{os.environ.get('DUMPMSG')}.module.txt", "w") as f:
                    f.write(str(self._debug_msg["process.graph_module"]))
            if "process.graph_module.graph" in self._debug_msg:
                with open(f"{os.environ.get('DUMPMSG')}.graph.txt", "w") as f:
                    f.write(str(self._debug_msg["process.graph_module.graph"]))
        return res

    def process(
        self,
        graph_module: "torch.fx.GraphModule",  # noqa: F821
        interpreter: "DynamoInterpreter",  # noqa: F821
    ):
        """
        Environment variable ``ONNX_BUILDER_PROGRESS=1`` can be used to show
        a progress bar on big models.
        """
        self._debug_msg["process.graph_module"] = graph_module
        self._debug_msg["process.graph_module.graph"] = graph_module.graph

        # looks into output marked as "alias_of_input"
        # see https://pytorch.org/functorch/main/_modules/torch/_functorch/aot_autograd.html
        # in that case, gen_alias_from_base is mixing the input data and the output stride
        # places = []
        # for node in graph_module.graph.nodes:
        #     if node.op == "placeholder":
        #         places.append(node)
        # for node in places:
        #     with graph_module.graph.inserting_after(node):
        #         cloned_node = graph_module.graph.call_method("clone", args=(node.target,))
        #         node.replace_all_uses_with(cloned_node)
        # graph_module.recompile()
        interpreter.start_graph(graph_module.graph)

        inputs = []
        for n in graph_module.graph.nodes:
            if n.op == "placeholder":
                # Not tensor constant are not captured by the exporter.
                inputs.append(
                    (n, bool(n.users), type(n.meta["val"]) if "val" in n.meta else None)
                )
        inputs_to_remove = []
        for n, has_users, type_value in inputs[::-1]:
            if has_users or type_value not in {int, bool, float, type(None)}:
                break
            inputs_to_remove.append(n.name)

        if int(os.environ.get("ONNX_BUILDER_PROGRESS", "0")) or (
            self.verbose and len(graph_module.graph.nodes) > 100
        ):
            try:
                import tqdm

                loop = tqdm.tqdm(list(enumerate(graph_module.graph.nodes)))
            except ImportError:
                loop = enumerate(graph_module.graph.nodes)
        else:
            loop = enumerate(graph_module.graph.nodes)

        inputs_to_remove = set(inputs_to_remove)
        self._debug_msg["process.inputs_to_remove"] = inputs_to_remove
        for i, node in loop:
            if node.op == "placeholder" and node.name in inputs_to_remove and not node.users:
                continue
            self._debug_msg["process.progress"] = (
                f"node {i}/{len(graph_module.graph.nodes)} target={node.target}"
            )
            interpreter.run_node(node)
        interpreter.end_graph(graph_module.graph)

    def _extend_local_function_inputs(self) -> Tuple[Tuple[str, Any], Set[Tuple[str, str]]]:
        """
        All the initializers may not be used in the main function.
        The function filter out all the unused initializers.
        The functions also filters out the unused local functions.

        :param proto: function to modify, modified inplace
        :param functions: other local functions
        :param initializers_dict: initializers
        :return: the new proto, the local functions names, and the used initializers
        """
        # Let's sort the additional inputs by size, bigger is first.
        used_initializers = self._get_used_initializers()
        inits = [
            (self.get_initializer_size(k), k, v)
            for k, v in self.initializers_dict.items()
            if k in used_initializers
        ]
        inits.sort(reverse=True)
        inits = [_[1:] for _ in inits]
        inputs_to_add = [_[0] for _ in inits]
        used_functions = self._get_used_local_functions()
        return inputs_to_add, used_functions

    def _check_constant(self, node: NodeProto, prefix: str):
        assert isinstance(node, NodeProto), f"Unexpected type {type(node)} for node"
        if node.op_type != "Constant":
            return
        assert (
            len(node.attribute) == 1
        ), f"{prefix}: unexpected number of attribute in node {node}"
        assert (
            node.attribute[0].type != AttributeProto.GRAPH
        ), f"{prefix}: wrong attribute type in node {node}"

    def _check_constants(self, prefix="before-inline", add: Optional[Any] = None):
        for v in self.constants_node_.values():
            self._check_constant(v, prefix)
        for v in self.nodes:
            self._check_constant(v, prefix)
        for k, v in self.functions.items():
            for node in v.node:
                self._check_constant(node, f"{prefix}-[{k}]")
        if add is not None:
            assert isinstance(add, FunctionProto), f"Not implemented for type {type(add)}"
            for node in add.node:
                self._check_constant(node, f"{prefix}-[add]")

    def _to_onnx_function(self, function_options, opsets, mask_outputs):
        if self._debug_local_function:
            print(f"[GraphBuilder-{self._hash()}.to_onnx] export_as_function {function_options}")
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.to_onnx] make_function "
                f"{len(self.initializers_dict)} inits "
                f"{len(self._parameter_renaming)} params"
            )
        assert (
            function_options.name not in FunctionOptions.empty_names
            and function_options.domain not in FunctionOptions.empty_names
        ), (
            f"Function name={function_options.name!r} cannot be empty and "
            f"Function domain={function_options.domain!r} cannot be empty."
        )
        key = function_options.domain, function_options.name
        assert (
            key not in self.functions
        ), f"The given name {key} is already taken by a local function{self.pretty_text()}"
        if function_options.move_initializer_to_constant:
            if self._debug_local_function:
                print(
                    f"[GraphBuilder-{self._hash()}.to_onnx] move_initializers_to_constant "
                    f"{len(self.initializers_dict)}"
                )
            self.move_initializers_to_constant(
                full_parameter_name=False,
                threshold=function_options.external_threshold,
                verbose=max(0, self.verbose - 1),
            )
            if self._debug_local_function:
                print(
                    f"[GraphBuilder-{self._hash()}.to_onnx] remaining_initializers "
                    f"{len(self.initializers_dict)}-{sorted(self.initializers_dict)}"
                )
        # if self._parameter_renaming: we don't necessarily need to rename here.
        # We better not if we want to make this function equivalent to it.
        proto = oh.make_function(
            function_options.domain,
            function_options.name,
            [i.name for i in self.inputs],
            [o.name for mask, o in zip(mask_outputs, self.outputs) if mask],
            self.nodes,
            opsets,
        )

        if not function_options.return_initializer:
            return proto

        if len(self.initializers_dict) == 0 and len(self.functions) == 0:
            res = dict(proto=proto)
            if self.functions:
                used_functions = self._get_used_local_functions()
                if used_functions:
                    res["functions"] = used_functions
            return res

        # We need to move the initializers as inputs, we sort them by decresing size
        inits, functions = self._extend_local_function_inputs()
        proto.input.extend(inits)
        res = dict(
            proto=proto,
            initializers_name=inits,
            initializers_dict={
                self._parameter_renaming.get(k, k): v
                for k, v in self.initializers_dict.items()
                if k in set(inits)
            },
            initializers_renaming={
                k: self._parameter_renaming.get(k, k)
                for k, v in self.initializers_dict.items()
                if k in set(inits)
            },
        )
        if functions:
            res["functions"] = [v for k, v in self.functions.items() if k in functions]
        return res

    def _update_model_with_parameter_renaming(self, model: ModelProto):
        assert self.initializers_dict, (
            f"Some parameters are renamed {self._parameter_renaming} "
            f"but there is no initializer{self.get_debug_msg()}"
        )
        # We rename.
        nodes_add = []
        setp = set(self._parameter_renaming)
        for node in self.nodes:

            needs_rewrite = False
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    hidden = self._get_hidden_inputs(att.g)
                    if hidden & setp:
                        needs_rewrite = True
                        # needs rewrite
                        new_g = self._rename_inputs_in_subgraph(att.g, self._parameter_renaming)

            if not needs_rewrite:
                seti = set(node.input)
                if not (seti & setp):
                    nodes_add.append(node)
                    continue

            node2 = NodeProto()
            node2.doc_string = node.doc_string
            node2.name = node.name
            node2.op_type = node.op_type
            node2.domain = node.domain
            node2.input.extend([self._parameter_renaming.get(i, i) for i in node.input])
            node2.output.extend(node.output)

            atts = []
            for att in node.attribute:
                if att.type != AttributeProto.GRAPH:
                    atts.append(att)
                    continue

                new_g = self._rename_inputs_in_subgraph(att.g, self._parameter_renaming)
                atts.append(oh.make_attribute(att.name, new_g))

            node2.attribute.extend(atts)
            nodes_add.append(node2)

        model.graph.node.extend(nodes_add)

        seti = set(i.name for i in self.inputs)
        if seti & setp:
            new_inputs = []
            for i in self.inputs:
                if i.name in self.setp:
                    v = ValueInfoProto()
                    v.ParseFromString(v.SerializeToString())
                    v.name = self._parameter_renaming[i.name]
                    new_inputs.append(v)
                else:
                    new_inputs.append(i)
            model.graph.input.extend(new_inputs)
        else:
            model.graph.input.extend(self.inputs)

    def _update_metadata_props(self, model: ModelProto, **kwargs):
        values = (
            self.existing_metadata_props.copy()
            if hasattr(self, "existing_metadata_props")
            else {}
        )
        values.update({k: str(v) for k, v in kwargs.items()})
        values.update(
            {
                "input_args": string_type(self.input_args, with_shape=True, with_min_max=True),
                "input_kwargs": string_type(
                    self.input_kwargs, with_shape=True, with_min_max=True
                ),
                "optimizations": str(self.optimization_options),
            }
        )
        if self.dynamic_shapes:
            values["dynamic_shapes"] = pprint.pformat(self.dynamic_shapes)
        if self.output_dynamic_shapes:
            values["output_dynamic_shapes"] = pprint.pformat(self.output_dynamic_shapes)
        if self.constraints_:
            values["_discovered_shape_constraints"] = pprint.pformat(self.constraints_)
        if self._known_value_shape:
            filtered = {
                k: v for k, v in self._known_value_shape.items() if not k.startswith("init")
            }
            if filtered:
                values["_known_value_shape"] = self._format_dict(filtered)
        if values:
            oh.set_metadata_props(model, values)

    def _get_size(self, t: Union["torch.Tensor", np.ndarray, TensorProto]) -> int:  # noqa: F821
        if hasattr(t, "shape"):
            return np.prod(t.shape) * size_type(t.dtype)
        assert isinstance(t, TensorProto), f"Unexpected type {type(t)}"
        shape = tuple(t.dims)
        dtype = t.data_type
        return np.prod(shape) * size_type(dtype)

    def to_onnx(
        self,
        optimize: bool = True,
        large_model: bool = False,
        external_threshold: int = 1024,
        return_optimize_report: bool = False,
        inline: bool = True,
        function_options: Optional[FunctionOptions] = None,
        mask_outputs: Optional[List[bool]] = None,
        as_graph_proto: bool = False,
    ) -> Union[FunctionProto, ModelProto, GraphProto, TorchModelContainer, Dict[str, Any]]:
        """
        Conversion to onnx. Only then the initializers are converted into TensorProto.

        :param optimize: disable or enable the optimization,
            the optimization are set when the class constructor is called
        :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
            it lets the user to decide later if the weights should be part of the model
            or saved as external weights
        :param external_threshold: if large_model is True, every tensor above this limit
            is stored as external
        :param return_optimize_report: return statistics about the optimization as well
        :param inline: inline local functions, this is done before
            any optimization takes place
        :param function_options: to be set to export as a function
        :param mask_outputs: to filter out some outputs if not None
        :param as_graph_proto: return a GraphProto with no initializers,
            they are returned as well.
        :return: the proto
        """
        assert self.nodes, f"No node to convert{self.get_debug_msg()}"
        if function_options is None:
            function_options = FunctionOptions()
        if len(self.nodes) == 0:
            raise RuntimeError(f"The onnx model is empty (no node).\n{self.get_debug_msg()}")

        if inline:
            self._check([], "before-inline")
            self._check_constants("before-inline")
            stats = self.inline_functions(verbose=self.verbose)
            self._check_constants("after-inline")
            self._check([], "after-inline")
        else:
            stats = None

        if optimize:
            statso = self.optimize()
            if stats:
                stats.extend(statso)
            else:
                stats = statso
            if self._parameter_renaming:
                # Adding the true names back.
                update = {}
                update_source = {}
                for k, v in self.initializers_dict.items():
                    if k in self._parameter_renaming:
                        update[self._parameter_renaming[k]] = v
                        update_source[self._parameter_renaming[k]] = (
                            self.initializers_dict_sources[k]
                        )
                self.initializers_dict.update(update)
                self.initializers_dict_sources.update(update_source)

        assert (
            len(self.nodes) > 0
        ), f"The onnx model is empty after optimization (no node).{self.get_debug_msg()}"

        # adding to the model everything we know
        self._add_shape_information_node_metadata()
        opsets = [oh.make_opsetid(*o) for o in self.opsets.items()]
        if mask_outputs is None:
            mask_outputs = [True for o in self.outputs]
        else:
            assert len(mask_outputs) == len(self.outputs), (
                f"Length mismatch between mask={mask_outputs} and outputs "
                f"{self.outputs}{self.get_debug_msg()}"
            )

        if function_options.export_as_function:
            # export as a function
            return self._to_onnx_function(function_options, opsets, mask_outputs)

        # export as a model
        if self.ir_version:
            ir_version = self.ir_version
        elif "" in self.opsets:
            ir_version = _default_OPSET_TO_IR_VERSION()[self.opsets[""]]

        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.to_onnx] make_model "
                f"{len(self.initializers_dict)} inits "
                f"{len(self._parameter_renaming)} params"
            )
            print(
                f"[GraphBuilder-{self._hash()}.time_evaluation_constants_] "
                f"{self.time_evaluation_constants_}"
            )

        # building the model
        model = ModelProto()
        model.graph.CopyFrom(GraphProto())
        model.graph.name = "experiment"
        model.graph.output.extend(o for mask, o in zip(mask_outputs, self.outputs) if mask)

        # rename parameters following the original model names if possible
        if self._parameter_renaming:
            self._update_model_with_parameter_renaming(model)
        else:
            model.graph.node.extend(self.nodes)
            model.graph.input.extend(self.inputs)

        if as_graph_proto:
            return model.graph

        # initializers
        initializers, large_initializers = self._build_initializers(
            switch_low_high=sys.byteorder != "big",
            large_model=large_model,
            external_threshold=external_threshold,
            full_parameter_name=True,
        )
        assert not self._parameter_renaming or initializers, (
            f"Some parameters are renamed {self._parameter_renaming} "
            f"self.initializers_dict={set(self.initializers_dict)} "
            f"but there is no initializer{self.get_debug_msg()}"
        )
        try:
            model.graph.initializer.extend(
                init for init in initializers if isinstance(init, TensorProto)
            )
        except Exception as e:
            raise RuntimeError(
                "protobuf is limited to 2 Gb, if this fails here, "
                "it probably means the result is beyond that limit. "
                "You should use large_model=True."
            ) from e
        sparse_initializer = [
            init for init in initializers if isinstance(init, SparseTensorProto)
        ]
        if sparse_initializer:
            assert all(
                init.values.name for init in sparse_initializer
            ), f"Missing name for one sparse initializer among {len(sparse_initializer)}"
            try:
                model.graph.sparse_initializer.extend(sparse_initializer)
            except Exception as e:
                raise RuntimeError(
                    "protobuf is limited to 2 Gb, if this fails here, "
                    "it probably means the result is beyond that limit. "
                    "You should use large_model=True."
                ) from e

        # opsets
        model.opset_import.extend(opsets)
        model.functions.extend(self.functions.values())
        model.ir_version = ir_version
        # Rewrites shape informations, operates replacments
        # to use as much as possible the dynamic dimension given
        # by the user.
        self._add_shape_information(model)

        # final doc_string
        self._update_metadata_props(
            model,
            large_model=large_model,
            inline=inline,
            external_threshold=external_threshold,
            function_options=function_options,
            optimize=optimize,
            n_initializers=len(initializers),
            n_large_initializers=len(large_initializers),
            size_initializers=int(
                sum(
                    np.prod(t.dims)
                    * size_type(t.data_type if isinstance(t, TensorProto) else t.values.data_type)
                    for t in initializers
                )
            ),
            size_large_initializers=sum(self._get_size(t) for t in large_initializers.values()),
            n_nodes=len(model.graph.node),
            n_nodes_other_domain=len([n for n in model.graph.node if n.domain != ""]),
            mask_outputs=mask_outputs,
        )
        assert (
            not optimize
            or not self.optimization_options.remove_identity
            or len([n for n in model.graph.node if n.op_type == "Identity"])
            <= len(model.graph.output)
        ), (
            f"The optimization was not applied. There are two many nodes identity"
            f"\n{self.pretty_text()}"
        )
        if self.check_empty_source:
            for init in model.graph.initializer:
                assert init.doc_string, (
                    f"doc_string is missing for initializer {init.name!r}"
                    f"\n{self.pretty_text()}"
                )

        if large_model:
            lm = TorchModelContainer()
            lm.model_proto = model
            if large_initializers:
                lm.set_large_initializers(large_initializers)
                lm.check_large_initializers()
                if self.check_empty_source:
                    for init in lm.model_proto.graph.initializer:
                        assert init.doc_string, (
                            f"doc_string is missing for initializer {init.name!r}"
                            f"\n{self.pretty_text()}"
                        )
            return (lm, stats) if return_optimize_report else lm

        return (model, stats) if return_optimize_report else model

    def _rename_dynamic_dimension_inplace(
        self, obj: ValueInfoProto, replacements: Dict[str, str]
    ) -> ValueInfoProto:
        """Renames dynamic shapes into a proto."""
        tensor_type = obj.type.tensor_type
        for d in tensor_type.shape.dim:
            if not d.dim_param:
                continue
            if d.dim_param in replacements and replacements[d.dim_param] != d.dim_param:
                d.dim_param = replacements[d.dim_param]
            elif "+" in d.dim_param or "*" in d.dim_param or "-" in d.dim_param:
                # Maybe we should handle more expression?
                d.dim_param = rename_dynamic_expression(d.dim_param, replacements).replace(
                    " ", ""
                )
        return obj

    def _improves_dynamic_dimension_naming(self):
        """
        Improves the naming of the dynamic dimnesion based on what
        the user gave.
        """
        if self._debug_dyn_dim:
            print(
                f"[GraphBuilder-{self._hash()}._improves_dynamic_dimension_naming] "
                f"{len(self.constraints_)} constaints"
            )
            for k, v in sorted(self.constraints_.items()):
                if k in self._debug_dyn_dim or v & self._debug_dyn_dim:
                    print(
                        f"[GraphBuilder-{self._hash()}._improves_dynamic_dimension_naming] .. "
                        f"{k} ~ {sorted(v)}"
                    )

        def _update(dd_flat, names):
            if dd_flat:
                # Let's update the constraints.
                for dyn_name, iname in zip(dd_flat, names):
                    # something like
                    if isinstance(dyn_name, str):
                        assert dyn_name == iname, (
                            f"Issue with one input name and its associated dynamic shape "
                            f"dyn_name={dyn_name!r}, input_name={iname!r}, among "
                            f"\ndyn_name={dd_flat}\nnames={names!r},\n"
                            f"You should give the dummy inputs in the same order than "
                            f"the forward signature, even if the inputs are given in "
                            f"a dictionary.{self.get_debug_msg()}"
                        )
                        continue
                    if not self.has_shape(iname):
                        continue
                    dd_shape = self.get_shape(iname)
                    for axis, sh in enumerate(dd_shape):
                        for dim_name, where in self.dynamic_dimensions_source.items():
                            for d in where:
                                if d["axis"] != axis:
                                    continue
                                try:
                                    r = bool(d["input_name"] != dyn_name)
                                except (
                                    self.torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
                                ):
                                    r = True
                                if r:
                                    continue
                                # We add a constraint.
                                self.add_to_constraints(dim_name, sh)
                                self.add_to_constraints(sh, dim_name)

        _update(self.dynamic_dimensions_source_flat, self.input_names)
        _update(self.output_dynamic_dimensions_source_flat, self.output_names)
        self._improve_constraints()

        # Before calling rename_dynamic_dimension, we expand the list.
        expanded_constraints = {}
        for k, v in self.constraints_.items():
            expanded_constraints[k] = v.copy()
            for i in v:
                expanded_constraints[k].add(i)
                if i not in expanded_constraints:
                    expanded_constraints[i] = set()
                expanded_constraints[i] |= v
                expanded_constraints[i] |= {k}

        # let's rename unknown input dimension with basic rules.
        implicit_names = ["batch", "channel"]
        original = set()
        for k, v in self.dynamic_dimensions_source.items():
            if not k.startswith("DYN") and "+" not in k:
                original.add(k)
                continue
            # We replace it with one implicit name.
            axis = set(d["axis"] for d in v)
            if len(axis) == 1:
                a = axis.pop()
                prefix = (
                    "scalar"
                    if a is None
                    else (
                        implicit_names[a]
                        if a < len(implicit_names)
                        else f"D{a - len(implicit_names)}"
                    )
                )
            else:
                prefix = "g0"
                i = 0
                while prefix in original:
                    i += 1
                    prefix = f"g{i}"
            if prefix not in expanded_constraints:
                expanded_constraints[prefix] = {k}
                if k in expanded_constraints:
                    expanded_constraints[k].add(prefix)
                original.add(prefix)
                continue
            n = f"{prefix}_{1}"
            i = 1
            while n in expanded_constraints:
                i += 1
                n = f"{prefix}_{i}"
            expanded_constraints[n] = {k}
            if k in expanded_constraints:
                expanded_constraints[k].add(n)
            original.add(n)

        # some cleaning
        if (
            "channel" in original
            and "channel" not in self.dynamic_dimensions_source
            and ("cache_length" in original or "seq_length" in original)
        ):
            original.remove("channel")

        if self._debug_dyn_dim:
            print(
                f"[GraphBuilder-{self._hash()}._improves_dynamic_dimension_naming] "
                f"original={original}"
            )
        # Let's process the output constraints.
        assert self.output_dynamic_shapes is None or isinstance(
            self.output_dynamic_shapes, dict
        ), (
            f"Not implemented when output_dynamic_shapes is not a dictionary, "
            f"output_dynamic_shapes={self.output_dynamic_shapes}."
        )
        if self.output_dynamic_shapes is not None:
            for k, v in self.output_dynamic_shapes.items():
                if not self.has_shape(k):
                    continue
                shape = self.get_shape(k)
                for axis, dim in v.items():
                    assert isinstance(dim, self.WrapDim), (
                        f"Unexpected type {type(dim)} in output_dynamic_shapes="
                        f"{self.output_dynamic_shapes}"
                    )
                    current_name = shape[axis]
                    new_name = dim.name
                    if current_name == new_name:
                        continue
                    if new_name not in expanded_constraints:
                        expanded_constraints[new_name] = set()
                    expanded_constraints[new_name].add(current_name)
                    if current_name not in expanded_constraints:
                        expanded_constraints[current_name] = set()
                    expanded_constraints[current_name].add(new_name)

        # once everything is defined, the rewriting can begin.
        replacements = rename_dynamic_dimensions(expanded_constraints, original)
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}._add_shape_information] dynamic shapes "
                f"replacements={replacements}"
            )
        if self._debug_dyn_dim:
            print(
                f"[GraphBuilder-{self._hash()}._improves_dynamic_dimension_naming] "
                f"{len(replacements)} replacements"
            )
            for k, v in sorted(replacements.items()):
                if k in self._debug_dyn_dim or v in self._debug_dyn_dim:
                    print(
                        f"[GraphBuilder-{self._hash()}._improves_dynamic_dimension_naming] "
                        f".. {k!r} -> {v!r}"
                    )

        if replacements:
            if not hasattr(self, "replacements_"):
                self.replacements_dimensions_ = {}
                self.replacements_for_replacements_dimensions_ = replacements
            for k, v in self._known_shapes.items():
                if v is None:
                    continue
                self.replacements_dimensions_[k] = tuple(
                    (rename_dynamic_expression(_, replacements) if isinstance(_, str) else _)
                    for _ in v
                )
        return replacements

    def get_shape_renamed(self, name: str) -> DYNAMIC_SHAPE:
        """Returns the shape of a result using user dimension name."""
        if hasattr(self, "replacements_dimensions_") and name in self.replacements_dimensions_:
            return self.replacements_dimensions_[name]
        return self.get_shape(name)

    def _add_shape_information(
        self, model: Union[GraphProto, ModelProto], update_dim_names: bool = True
    ):
        """
        Adds shape information to the model.

        :param model: final onnx model
        :param update_dim_names: the conversion usually detects many dimension equivalent
            among each others, if True, the function renames as much as possible by using
            the names the user provides
        """
        replacements = self._improves_dynamic_dimension_naming()
        if not replacements:
            return

        if len(model.graph.node) == 0:
            raise RuntimeError(
                f"The onnx model is empty after export to onnx (no node)."
                f"\n{self.get_debug_msg()}"
            )

        # restores the existing value_info
        done = (
            set(init.name for init in model.graph.initializer)
            | set(i.name for i in model.graph.input)
            | set(i.name for i in model.graph.output)
        )
        if replacements:
            for i in model.graph.input:
                self._rename_dynamic_dimension_inplace(i, replacements)
            for i in model.graph.output:
                self._rename_dynamic_dimension_inplace(i, replacements)
        for val in self.value_info:
            if self.has_name(val.name):
                model.graph.value_info.append(
                    self._rename_dynamic_dimension_inplace(val, replacements)
                )
                done.add(val.name)

        # adding shape information
        addition = []
        for name in self._known_names:
            if name in done or not self.has_type(name) or self.get_type(name) == 0:
                continue
            if self.has_shape(name):
                addition.append(
                    self._rename_dynamic_dimension_inplace(
                        oh.make_tensor_value_info(
                            name, self.get_type(name), list(self.get_shape(name))
                        ),
                        replacements,
                    )
                )
            elif self.has_rank(name):
                addition.append(
                    self._rename_dynamic_dimension_inplace(
                        oh.make_tensor_value_info(
                            name, self.get_type(name), [None] * self.get_rank(name)
                        ),
                        replacements,
                    )
                )
        if addition:
            model.graph.value_info.extend(addition)

    def _add_shape_information_node_metadata(self):
        for node in self.nodes:
            data = dict(
                intypes=" / ".join(
                    map(
                        str,
                        [
                            (
                                ""
                                if not i
                                else (
                                    onnx_dtype_name(self.get_type(i)) if self.has_type(i) else "?"
                                )
                            )
                            for i in node.input
                        ],
                    )
                ),
                outtypes=" / ".join(
                    map(
                        str,
                        [
                            (
                                ""
                                if not i
                                else (
                                    onnx_dtype_name(self.get_type(i)) if self.has_type(i) else "?"
                                )
                            )
                            for i in node.output
                        ],
                    )
                ),
                inshapes=" / ".join(
                    map(
                        str,
                        [
                            "" if not i else (self.get_shape(i) if self.has_shape(i) else "?")
                            for i in node.input
                        ],
                    )
                ),
                outshapes=" / ".join(
                    map(
                        str,
                        [
                            "" if not i else (self.get_shape(i) if self.has_shape(i) else "?")
                            for i in node.output
                        ],
                    )
                ),
            )
            invalues = [self.value_as_shape(i) for i in node.input]
            outvalues = [self.value_as_shape(i) for i in node.output]
            if any(i is not None for i in invalues):
                data["invalueshape"] = " / ".join(
                    map(str, ["" if i is None else i for i in invalues])
                )
            if any(i is not None for i in outvalues):
                data["outvalueshape"] = " / ".join(
                    map(str, ["" if i is None else i for i in outvalues])
                )
            for k, v in data.items():
                entry = node.metadata_props.add()
                entry.key = k
                entry.value = v

    def _improve_constraints(self):
        """Adds more correspondances deduced from self.constraints_."""

        update = {}
        it = 0
        while (update or it == 0) and it < 10:
            it += 1
            update = {}
            for k, v in self.constraints_.items():
                for vv in v:
                    if not isinstance(k, int) or not isinstance(vv, int):
                        continue
                    simpl = simplify_two_expressions(k, vv)
                    if len(simpl) != 2:
                        continue
                    items = list(simpl.items())
                    if items[0][1] != -items[1][1]:
                        continue
                    # adds an equivalence
                    n1, n2 = items[0][0], items[1][0]
                    if n1 not in update:
                        update[n1] = {n2}
                    else:
                        update[n1].add(n2)
                    if n2 not in update:
                        update[n2] = {n1}
                    else:
                        update[n2].add(n1)
                    k2 = rename_expression(k, {n2: n1})
                    vv2 = rename_expression(v, {n1: n2})
                    assert k != k2 and vv != vv2
                    eq = {k, k2, vv, vv2}
                    for e in eq:
                        if e not in update:
                            update[e] = eq
                        else:
                            update[e] |= eq

            for k, v in update.items():
                if self._debug_dyn_dim and self._debug_dyn_dim & {k}:
                    print(f"[GraphBuilder-{self._hash()}._improve_constraints] {k} -> {v}")
                self.add_to_constraints(k, v)

    def io_names(self):
        """Returns the list of inputs, output for nodes."""
        input_names = {i.name for i in self.inputs}
        init_names = set(self.initializers_dict)
        output_names = {i.name for i in self.outputs}
        rows = [
            "",
            f"I<-[{','.join(assert_sorted(input_names))}]",
            f"C<-[{','.join(assert_sorted(init_names))}]",
        ]
        for node in self.nodes:
            rows.append(
                f"N:{node.op_type}:[{','.join(assert_sorted(node.input))}]"
                f"->[{','.join(assert_sorted(node.output))}]"
            )
        rows.append(f"O->[{','.join(assert_sorted(output_names))}]")
        rows.append("")
        return "\n".join(rows)

    def _check_nodes(
        self, nodes: List[NodeProto], known: Set[str], step: str, root: bool = False
    ):
        for node in nodes:
            assert (
                node.domain in self.opsets
            ), f"Domain {node.domain!r} is not registered in {self.opsets}"
            for i in node.input:
                if i == "":
                    continue
                assert not root or self.has_name(i), (
                    f"Name {i!r} not registered, step {step!r}, node type is "
                    f"{node.op_type!r}, root={root!r}, node name is {node.name!r}, "
                    f"input are {node.input}, i={i!r} in known={i in known}"
                    f"{self.get_debug_msg()}"
                )
                assert i in known, (
                    f"Unknown input {i!r}, step {step!r} in node type "
                    f"{node.op_type}, name is {node.name!r}\n"
                    f"{self.pretty_node(node)}{self.get_debug_msg()}"
                )
            if self._has_subgraph_possibly(node):
                for att in node.attribute:
                    if att.type == AttributeProto.GRAPH:
                        g = att.g
                        k2 = known.copy()
                        k2 |= (
                            set(i.name for i in g.input)
                            | set(i.name for i in g.initializer)
                            | set(i.name for i in g.sparse_initializer)
                        )
                        self._check_nodes(
                            g.node, k2, step=f"{step}-{node.op_type}-{att.name}-{node.name}"
                        )
                        for o in g.output:
                            assert o.name in k2, (
                                f"Unknown output {o.name!r}, step {step!r}, "
                                f"op_type={node.op_type!r}, att.name={att.name!r}, "
                                f"node.name={node.name}"
                            )
            known |= set(node.output)

    def _check(self, stats: List[Dict[str, Any]], step: str, shadowing: bool = False):
        begin = time.perf_counter()
        assert len(self.nodes) > 0, (
            f"The onnx model is empty (step {step!r}, no node, shadowing={shadowing})"
            f"{self.get_debug_msg()}"
        )
        known = set(n.name for n in self.inputs) | set(self.initializers_dict) | self._context
        self._check_nodes(self.nodes, known, step, root=True)
        for o in self.outputs:
            assert o.name in known, f"Unknown output {o.name!r}, step {step!r}"
        if shadowing:
            shadow = self.shadowing_names()
            assert not shadow, f"Shadowing names were found ({shadow})."
        stats.append(dict(pattern=f"check_{step}", time_in=time.perf_counter() - begin))

    def _import_context_from_parent_graph(self, name: str):
        if self.has_type(name) and self.has_shape(name):
            return
        if not self._parent:
            return
        if not self.has_name(name):
            # We cannot overwrite a local result.
            self.set_name(name, marker="optimize_node_subgraphs_inplace")
        g = self._parent
        parent_name, name = name, name
        if self._parent_node:
            # Coming from a Scan, If, Loop, Node
            if name in self.input_names:
                position = self.input_names.index(name)
                assert position < len(self._parent_node.input), (
                    f"No corresponding input name for {name!r} in node\n"
                    f"{pretty_onnx(self._parent_node, with_attributes=True)}\n---\n"
                    f"{self.get_debug_msg()}"
                )
                parent_name = self._parent_node.input[position]

        if g.has_shape(parent_name):
            self.set_shape(name, g.get_shape(parent_name), allow_zero=True)
        elif g.has_rank(parent_name):
            self.set_rank(name, g.get_rank(parent_name))
        if g.has_type(parent_name):
            self.set_type(name, g.get_type(parent_name))
        if g.has_device(parent_name):
            self.set_device(name, g.get_device(parent_name))
        if parent_name in g.constants_:
            self.constants_[name] = g.constants_[parent_name]
        if parent_name in g.constants_node_:
            self.constants_node_[name] = g.constants_node_[parent_name]
        if parent_name in g._known_value_shape:
            self._known_value_shape[name] = g._known_value_shape[parent_name]
        if parent_name in g._parameter_norename:
            self._parameter_norename.add(name)

    def optimize_node_subgraphs_inplace(self, node: NodeProto, context: Set[str]):
        """Optimizes the subgraphs for a node."""
        new_atts = []
        for att in node.attribute:
            if att.type != AttributeProto.GRAPH:
                new_atts.append(att)
                continue
            if self.verbose > 1:
                print(
                    f"[GraphBuilder-{self._hash()}] optimizes attribute "
                    f"{att.name!r} from node {node.op_type!r}, name={node.name!r}"
                )
            g = GraphBuilder(
                att.g,
                optimization_options=self.optimization_options,
                verbose=max(self.verbose - 1, 0),
                _opsets=self.opsets,
                _context=context,
                _parent=(self, node),
            )
            assert not g.functions, f"unexpected functions in a subgraphs{g.get_debug_msg()}"
            # We need to populate whatever exists.
            g.optimize()

            renaming = {}
            for k in g.initializers_dict:
                if self.has_name(k):
                    nk = self.unique_name(k)
                    renaming[k] = nk
            if renaming:
                g.rename_names(renaming)

            new_g = g.to_onnx(optimize=False, as_graph_proto=True)
            assert isinstance(new_g, GraphProto), f"unexpected type {type(new_g)}"
            if self.verbose > 1:
                print(
                    f"[GraphBuilder-{self._hash()}] done optimizing attribute "
                    f"{att.name!r} from node {node.op_type!r}, name={node.name!r}"
                )
            new_atts.append(oh.make_attribute(att.name, new_g))

            # We need to append functions and initiliazers to the main graph.

            for k, v in g.initializers_dict.items():
                assert (
                    k not in self.initializers_dict
                ), f"name {k!r} already present. That should not be the case."
                self.initializers_dict[k] = v
                self.initializers_dict_sources[k] = g.initializers_dict_sources[k]
                self.set_name(
                    k,
                    marker=g._events.get((k, "set_event"), "optimize_node_subgraphs_inplace"),
                )
                self.set_type(k, g.get_type(k))
                self.set_shape(k, g.get_shape(k))
                if g.has_device(k):
                    self.set_device(k, g.get_device(k))
                if k in g.constants_:
                    self.constants_[k] = g.constants_[k]
                if k in g._parameter_renaming:
                    self._parameter_renaming[k] = g._parameter_renaming[k]
                if k in g._parameter_norename:
                    self._parameter_norename.add(k)
                if k in g._known_torch_value:
                    self._known_torch_value[k] = g._known_torch_value[k]

        del node.attribute[:]
        node.attribute.extend(new_atts)

    def optimize(self, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Optimizes a graph.
        Returns the list of applied processes.

        :param recursive: to overwrite the value provided by the options if True.
        """
        statistics = []
        main_begin = time.perf_counter()

        begin = time.perf_counter()
        self._improves_dynamic_dimension_naming()
        self._another_pass_at_shape_inference()
        statistics.append(
            dict(
                pattern="dynamic_dimension_naming",
                removed=0,
                added=0,
                time_in=time.perf_counter() - begin,
            )
        )
        self._check(statistics, "A-dynamic_dimension_naming")

        if self.verbose or self.optimization_options.verbose:
            print(f"[GraphBuilder-{self._hash()}.optimize] start with {len(self.nodes)} nodes")
            if self.verbose > 1:
                print(
                    f"[GraphBuilder-{self._hash()}.optimize] "
                    f"options={self.optimization_options!r}"
                )
            if self.verbose >= 10:
                print("-- GRAPH BEFORE OPTIMIZATON --")
                print(self.pretty_text())
                print("-- END --")
            else:
                n_patterns = (
                    0
                    if self.optimization_options is None
                    or self.optimization_options.patterns is None
                    else len(self.optimization_options.patterns)
                )
                print(f"[GraphBuilder-{self._hash()}.optimize] #patterns={n_patterns}")

        if self.optimization_options.recursive and recursive:
            if self.verbose > 0:
                print(f"[GraphBuilder-{self._hash()}.optimize] start with subgraphs")
            context = (
                set(i.name for i in self.inputs) | set(self.initializers_dict) | self._context
            )
            for node in self.nodes:
                if any(att.type == AttributeProto.GRAPH for att in node.attribute):
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder-{self._hash()}.optimize] "
                            f"optimize {self.pretty_node(node)}"
                        )
                    self.optimize_node_subgraphs_inplace(node, context)
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder-{self._hash()}.optimize] "
                            f"done {self.pretty_node(node)}"
                        )
                context |= set(o for o in node.output if o)
            if self.verbose > 0:
                print(f"[GraphBuilder-{self._hash()}.optimize] done with subgraphs")
            self._check(statistics, "A-opt-sub")
        else:
            self._check(statistics, "A-opt")

        n = len(self.nodes)

        for order, pass_name in enumerate(self.optimization_options.passes):
            self.optimize_pass(pass_name, order=order, statistics=statistics)

        na, nr = (0, n - len(self.nodes)) if len(self.nodes) < n else (len(self.nodes) - n, 0)

        if self.verbose or self.optimization_options.verbose:
            duration = time.perf_counter() - main_begin
            print(
                f"[GraphBuilder-{self._hash()}.optimize] done with "
                f"{len(self.nodes)} nodes in {duration:.3f}"
            )
            if self.verbose > 1:
                print(self._compile_statistics(statistics))
        statistics.append(
            dict(
                pattern="optimization",
                removed=nr,
                added=na,
                time_in=time.perf_counter() - main_begin,
            )
        )
        return statistics

    def optimize_pass(
        self,
        pass_name: str,
        order: Optional[int] = None,
        statistics: Optional[List[Dict[str, Any]]] = None,
    ):
        begin = time.perf_counter()
        nr, na = 0, 0
        local_stats = []
        if pass_name == "remove_identity":
            if self.optimization_options.remove_identity:
                nr, na = self.remove_identity_nodes()
        elif pass_name == "remove_unused":
            if self.optimization_options.remove_unused:
                na = self.remove_unused()
        elif pass_name == "constant_folding":
            if self.optimization_options.constant_folding:
                stats_cf = self.constant_folding(self.optimization_options.constant_folding)
                nr = stats_cf["n"]
                for k, v in stats_cf.items():
                    if k == "n":
                        continue
                    local_stats.append(
                        dict(
                            pattern=f"apply_constant_folding_{k}",
                            value=v,
                            iteration=0,
                        )
                    )
        elif pass_name == "patterns":
            if self.optimization_options.patterns is not None:
                n = len(self.nodes)
                local_stats = self.optimize_with_patterns(recursive=False)
                if len(self.nodes) < n:
                    nr = n - len(self.nodes)
                else:
                    na = len(self.nodes) - n
        elif pass_name == "remove_duplicated_initializer":
            nr, na = self.remove_duplicated_initializer()
        elif pass_name == "order":
            if self.optimization_options.order:
                local_stats = self.optimize_order()
        else:
            raise ValueError(f"Unknown pass name {pass_name!r}")

        if statistics is not None:
            statistics.append(
                dict(
                    pattern=pass_name,
                    removed=nr,
                    added=na,
                    time_in=time.perf_counter() - begin,
                )
            )
            if local_stats:
                if isinstance(local_stats, list):
                    statistics.extend(local_stats)
                else:
                    statistics.append(local_stats)
        self._check(statistics, f"{pass_name}-{order or 0}")
        return statistics

    def _compile_statistics(self, statistics: List[Dict[str, Any]]) -> str:
        stats = {}
        for obs in statistics:
            pattern = obs["pattern"]
            if pattern not in stats:
                stats[pattern] = {
                    "time_in": 0.0,
                    "iteration": [],
                    "match_index": [],
                    "removed": 0,
                    "added": 0,
                    "instances": 0,
                    "repeated": 0,
                    "max_match": 0,
                    "n_nodes": 0,
                }
            o = stats[pattern]
            for k, v in obs.items():
                if k == "pattern":
                    continue
                if k in {"time_in", "removed", "added", "instances", "repeated", "n_nodes"}:
                    o[k] += v
                    continue
                if k in {"changed", "scale"}:
                    if k not in o:
                        o[k] = 0
                    o[k] += v
                    continue
                if k in {"iter", "max_match"}:
                    if k not in o:
                        o[k] = 0
                    o[k] = max(o[k], v)
                    continue
                if k in {"algo", "value", "exit_point"} and k not in o:
                    o[k] = []
                assert k in o, f"Missing k={k!r} from statistics={statistics!r}"
                o[k].append(v)

        rows = []
        for k, v in assert_sorted(stats.items()):
            line = (
                f"    STAT {k} +{v['added']} -{v['removed']} "
                f"#it={len(set(v['iteration']))} "
                f"maxmatch={max(v['match_index']) if v['match_index'] else 0} "
                f"i={v['instances']} - time={v['time_in']}"
            )
            rows.append(line)

        # adding statistics on node type
        rows.append(self._compile_model_statistics(False))
        rows.append(self._compile_model_statistics(True))
        return "\n".join(rows)

    def _compile_model_statistics(self, detailed: bool):
        rows = [
            f"--MODEL: {len(self.nodes)} nodes, {len(self.inputs)} inputs, "
            f"{len(self.outputs)} outputs, "
            f"{len(self.initializers_dict)} initializers--"
            f"{'DETAILED--' if detailed else ''}"
        ]
        if detailed:

            def _shape(name):
                if self.has_shape(name):
                    s = self.get_shape(name)
                    if len(s) == 0:
                        return "1"
                    return "x".join(map(str, s))
                if self.has_rank(name):
                    r = self.get_rank(name)
                    if r == 0:
                        return "1"
                    return "x".join(["?"] * r)
                return "?"

            def _key(name):
                if not name:
                    return ""
                if isinstance(name, str):
                    tt = self.get_type(name) if self.has_type(name) else "?"
                    return f"{tt}t[{_shape(name)}]"
                if name.op_type == "Transpose":
                    perm = ";".join(map(str, self.get_attribute(name, "perm").ints))
                    return f"{_key(name.input[0])}-perm={perm}"
                return ", ".join(map(_key, name.input))

            cc = Counter([_key(i) for i in self.input_names])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"     INPUT: {v:3d} x {k}")
            cc = Counter([_key(i) for i in self.output_names])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"    OUTPUT: {v:3d} x {k}")
            cc = Counter([_key(i) for i in self.initializers_dict])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"      INIT: {v:3d} x {k}")
            op_types = [(n.domain, n.op_type, _key(n)) for n in self.nodes]
            cc = Counter(op_types)
            for k, v in assert_sorted(cc.items()):
                if k[0] == "":
                    rows.append(f"      NODE: {v:3d} x {k[1]} -SIG- {k[2]}")
                else:
                    rows.append(f"      NODE: {v:3d} x {k[0]}.{k[1]} -SIG- {k[2]}")
        else:
            cc = Counter([self.get_type(i) for i in self.input_names if not self.is_sequence(i)])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"         INPUT: {v:3d} x {k}t")
            cc = Counter([self.is_sequence(i) for i in self.input_names])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"     INPUT-SEQ: {v:3d} x {k}t")
            cc = Counter([self.get_type(i) for i in self.output_names if not self.is_sequence(i)])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"        OUTPUT: {v:3d} x {k}t")
            cc = Counter([self.is_sequence(i) for i in self.output_names])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"    OUTPUT-SEQ: {v:3d} x {k}t")
            cc = Counter([self.get_type(i) for i in self.initializers_dict])
            for k, v in assert_sorted(cc.items()):
                rows.append(f"          INIT: {v:3d} x {k}t")
            op_types = [(n.domain, n.op_type) for n in self.nodes]
            cc = Counter(op_types)
            for k, v in assert_sorted(cc.items()):
                if k[0] == "":
                    rows.append(f"          NODE: {v:3d} x {k[1]}")
                else:
                    rows.append(f"          NODE: {v:3d} x {k[0]}.{k[1]}")
        return "\n".join(rows)

    def optimize_with_patterns(self, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Optimizes this graph with patterns.

        :param recursive: overwrite the value given by the option if this one is True
        :return: the method returns informations about the applied processes.
        """
        from ..xoptim import GraphBuilderPatternOptimization

        gro = GraphBuilderPatternOptimization(
            self,
            verbose=max(self.verbose, self.optimization_options.verbose),
            patterns=self.optimization_options.patterns,
            recursive=self.optimization_options.recursive,
            verifies=self.optimization_options.verifies,
            dump_applied_patterns=self.optimization_options.dump_applied_patterns,
            processor=self.optimization_options.processor,
        )
        return gro.optimize(
            max_iter=self.optimization_options.max_iter,
            remove_identity=self.optimization_options.remove_identity,
            remove_duplicated_shape=self.optimization_options.remove_duplicated_shape,
            stop_after=self.optimization_options.stop_after,
            recursive=recursive,
        )

    def optimize_order(self):
        from ..xoptim.order_optim import OrderOptimization

        opt = OrderOptimization(
            self,
            algorithm=self.optimization_options.order,
            verbose=max(self.verbose, self.optimization_options.verbose),
        )
        return opt.optimize()

    @classmethod
    def _get_hidden_inputs(self, graph: GraphProto) -> Set[str]:
        """
        Returns the hidden inputs (inputs coming from an upper context)
        used by a subgraph.
        """
        hidden = set()
        memo = (
            {i.name for i in graph.initializer}
            | {i.values.name for i in graph.sparse_initializer}
            | {i.name for i in graph.input}
        )
        for node in graph.node:
            for i in node.input:
                if i not in memo:
                    hidden.add(i)
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    hid = self._get_hidden_inputs(att.g)
                    less = set(h for h in hid if h not in memo)
                    hidden |= less
            memo |= set(node.output)
        return hidden

    def _get_used_initializers(self) -> Set[str]:
        """Returns the initializers name involved in the graph."""
        hidden = set()
        memo = set(i.name for i in self.inputs)
        for node in self.nodes:
            for i in node.input:
                if i not in memo:
                    hidden.add(i)
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    hid = self._get_hidden_inputs(att.g)
                    less = set(h for h in hid if h not in memo)
                    hidden |= less
            memo |= set(node.output)
        assert all(name in self.initializers_dict for name in hidden if name), (
            f"Some hidden inputs in {assert_sorted(hidden)!r} are not initializers "
            f"{assert_sorted(self.initializers_dict)}. It is unexpected."
        )
        return hidden

    def _get_used_local_functions(
        self, nodes: Optional[Sequence[NodeProto]] = None
    ) -> Set[Tuple[str, str]]:
        """Returns the local functions used in the graph."""
        if nodes is None:
            nodes = self.nodes
        used = set()
        for node in nodes:
            key = node.domain, node.op_type
            if key in self.functions:
                used.add(key)
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    used |= self._get_used_local_functions(att.g.node)
        # Looking into used functions
        done = set()
        stack = list(used)
        while stack:
            key = stack.pop()
            if key not in done:
                f = self.functions[key]
                used |= self._get_used_local_functions(f.node)
                done.add(key)
        return used

    @classmethod
    def _has_subgraph_possibly(cls, node: NodeProto) -> bool:
        return node.op_type[0] in "LSI" and node.op_type in {"Loop", "Scan", "If", "SequenceMap"}

    @classmethod
    def _enumerate_inputs_with_subgraph_plus_hidden(cls, node: NodeProto) -> Iterator[str]:
        """
        Enumerates all inputs from a node including all the hidden inputs
        from subgraphs.
        """
        yield from node.input
        if cls._has_subgraph_possibly(node):
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    hidden_inputs = cls._get_hidden_inputs(att.g)
                    yield from hidden_inputs
                else:
                    assert (
                        att.type != AttributeProto.GRAPHS
                    ), "Not implemented when att.type == AttributeProto.GRAPHS"

    def remove_unused(self) -> int:
        """
        Simple function to remove unused nodes.
        It does not look into subgraphs and assumes there is none.
        Everything is done in one pass.
        Returns the number of removed nodes.
        """
        self._clean_values_cache()
        start = len(self.nodes)
        # mark outputs
        marked = {o.name: set() for o in self.outputs}
        for node in reversed(self.nodes):
            used = False
            node_inputs = list(self._enumerate_inputs_with_subgraph_plus_hidden(node))
            for o in node.output:
                if o in marked:
                    for i in node_inputs:
                        marked[o].add(i)
                        used = True
            if used or self.do_not_remove(node):
                for i in node_inputs:
                    marked[i] = set()

        # removed nodes
        removed = set()
        marked_set = set(marked)
        for ind, node in enumerate(self.nodes):
            if not (set(node.output) & marked_set):
                removed.add(ind)

        if self.optimization_options.verbose > 1:
            n_not_marked = 0
            for i, (k, v) in enumerate(self.initializers_dict.items()):
                if k not in marked:
                    n_not_marked += 1
                    v = self.initializers_dict[k]
                    if hasattr(v, "dtype") and hasattr(v, "shape"):
                        print(
                            f"[GraphBuilder-{self._hash()}.remove_unused] "
                            f"remove_initializer {n_not_marked}:"
                            f"{i}/{len(self.initializers_dict)}:{k}:{v.dtype}[{v.shape}]"
                        )
                    else:
                        print(
                            f"[GraphBuilder-{self._hash()}.remove_unused] "
                            f"remove_initializer {n_not_marked}:"
                            f"{i}/{len(self.initializers_dict)}:{k}"
                        )

        self.initializers_dict = {k: v for k, v in self.initializers_dict.items() if k in marked}
        self.constants_ = {k: v for k, v in self.constants_.items() if k in marked}

        if self.optimization_options.verbose > 2:
            for i in removed:
                node = self.nodes[i]
                print(
                    f"[GraphBuilder-{self._hash()}.remove_unused_node] remove "
                    f"{node.op_type}-{node.name} -> {node.output}"
                )

        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]
        self._refresh_values_cache()
        return start - len(self.nodes)

    def constant_folding(
        self, options: Union[bool, Set[str]], convert_into_initializer: bool = True
    ) -> Dict[str, float]:
        """
        Folds all constants. Constants are marked during the creation of the graph.
        There is no need to propagate this information.

        :param options: a boolean to fold anything
            which can be folded or a list of operator type
        :param convert_into_initializer: moves the constant as an initializer,
            otherwise, just evaluates it
        :return: dictionary of statistics
        """
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}.constant_folding] -- starts with "
                f"{len(self.constants_)} constants and "
                f"{len(self.nodes)} nodes."
            )
            if self.verbose >= 10:
                for name in self._known_names:
                    print(
                        f"[GraphBuilder-{self._hash()}.constant_folding] cst:: "
                        f"{1 if self.is_constant(name) else '.'} :: {name}"
                    )
        stats_cf = {"new_inits": 0}
        start = len(self.nodes)
        updates = {}
        node_to_remove = set()
        for k, v in self.constants_.items():
            if v is None:
                # this is an initiliazer
                if self.verbose > 4:
                    print(f"[GraphBuilder-{self._hash()}.constant_folding] initializer: {k}")
                if self._debug_foldnot and k not in self.initializers_dict:
                    print(
                        f"[GraphBuilder-{self._hash()}.constant_folding.0] "
                        f"unable to fold {k!r} (v is None)"
                    )
                continue
            assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for k={k!r}"
            if self.verbose > 4:
                print(f"[GraphBuilder-{self._hash()}.constant_folding] from: {v.op_type}({k})")
            if (
                isinstance(options, set)
                and v.op_type not in options
                and (v.domain, v.op_type) not in options
            ):
                if self._debug_foldnot:
                    print(
                        f"[GraphBuilder-{self._hash()}.constant_folding.P] unable to fold "
                        f"{v.op_type}{v.input} -> {v.output}  [{v.name}], k={k!r}, "
                        f"options={options}"
                    )
                continue
            # a node
            if all(map(self.is_constant, v.input)):
                # node evaluation
                output, feeds = self.compute_constant(k, exc=False)
                if output is None:
                    # Evaluation failed.
                    if self._debug_foldnot:
                        print(
                            f"[GraphBuilder-{self._hash()}.constant_folding.A] unable to fold "
                            f"{v.op_type}{v.input} -> {v.output}  [{v.name}], k={k!r}"
                        )
                    assert not self._debug_constant_folding, (
                        f"constant folding unable to fold node [{self.pretty_node(v)}], "
                        f"self.compute_constant(k, exc=False)="
                        f"{string_type(self.compute_constant(k, exc=False))}"
                    )
                    continue
                key = f"{v.domain}_{v.op_type}"
                if key not in stats_cf:
                    stats_cf[key] = 1
                else:
                    stats_cf[key] += 1
                if convert_into_initializer:
                    node_to_remove.add(tuple(v.output))
                if not isinstance(output, tuple):
                    output = (output,)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    if convert_into_initializer:
                        assert "FakeTensor" not in str(type(value)), (
                            f"FakeTensor {name!r} cannot be an initializer {type(value)}, "
                            f"v.op_type={v.op_type!r} (input types: "
                            f"{[type(i) for i in feeds.values()]})"
                            f"{self.get_debug_msg()}"
                        )
                        sources = []
                        for k in feeds:
                            sources.append(
                                f"##{k}/"
                                if k not in self.initializers_dict_sources
                                or not self.initializers_dict_sources[k].source
                                else f"##{k}/{self.initializers_dict_sources[k].source}"
                            )
                        text_sources = "".join(sources) if sources else ""
                        self.add_initializer(
                            name,
                            value,
                            existing=None,
                            source=f"GraphBuilder.constant_folding.from/fold"
                            f"({','.join(assert_sorted(feeds))}){text_sources}",
                            allow_empty=True,
                        )
                        stats_cf["new_inits"] += 1
                    else:
                        if self._debug_foldnot:
                            print(
                                f"[GraphBuilder-{self._hash()}.constant_folding.B] unable to "
                                f"fold {v.op_type}{v.input} -> {v.output}  [{v.name}]"
                            )
                        updates[name] = v
                    if self.verbose > 3:
                        print(
                            f"[GraphBuilder-{self._hash()}.constant_folding] fold_constant:"
                            f"{v.op_type}:{name}[{value.dtype}:"
                            f"{value.shape}]:from:{','.join(assert_sorted(feeds))}"
                        )

        for k, v in updates.items():
            self.update_node_constant(k, v)

        new_nodes = []
        for node in self.nodes:
            if self.do_not_remove(node):
                new_nodes.append(node)
                if self._debug_foldnot:
                    print(
                        f"[GraphBuilder-{self._hash()}.constant_folding.C] unable to fold "
                        f"{node.op_type}{node.input} -> {node.output}  [{node.name}]"
                    )
                continue
            if tuple(node.output) in node_to_remove:
                continue
            new_nodes.append(node)
        self.nodes = new_nodes
        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.constant_folding] ends with "
                f"{len(self.constants_)} constants and "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_} seconds"
            )
        stats_cf["n"] = start - len(self.nodes)
        self._refresh_values_cache()
        return stats_cf

    def _clean_values_cache(self):
        """
        Cleans the cache. The cache is used to avoid the creation of new constants
        while creating a graph. It should be removed the graph is modified.
        """
        if self._values:
            self._values.clear()

    def _refresh_values_cache(self):
        """
        Rebuilds the cache. The cache is used to avoid the creation of new constants
        while creating a graph. It should be removed the graph is modified.
        """
        self._clean_values_cache()
        for k, v in self.initializers_dict.items():
            key = self.make_key(v)
            if key not in self._values:
                self._values[key] = k

    def remove_duplicated_initializer(self) -> Tuple[int, int]:
        """Removes duplicated initializers and inserts identity nodes."""
        self._clean_values_cache()
        replacements = {}
        for k, v in self.initializers_dict.items():
            key = self.make_key(v)
            if key not in self._values:
                self._values[key] = k
            elif key is not None:
                assert (
                    k not in replacements
                ), f"Constant name {k!r} is duplicated{self.get_debug_msg()}"
                replacements[k] = self._values[key]
        if replacements:
            self.rename_names(replacements, with_existing=True)
        return len(replacements), 0

    @classmethod
    def _rename_inputs_in_node(
        cls, node: NodeProto, replacements: Dict[str, str], to_rename: Set[str]
    ) -> NodeProto:
        set_inputs = set(cls._enumerate_inputs_with_subgraph_plus_hidden(node))
        if set_inputs & to_rename:
            new_inputs = [replacements.get(i, i) for i in node.input]
            new_outputs = [replacements.get(i, i) for i in node.output]
            assert set(new_inputs) & set(new_outputs) in ({""}, set()), (
                f"Node type {node.op_type}-{node.name} is incorrectly replaced "
                f"{node.input}->{new_inputs} and {node.output}->{new_outputs}\n"
                f"replacements are\n{pprint.pformat(replacements)}"
            )
            new_node = oh.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                domain=node.domain,
                name=node.name,
            )

            if cls._has_subgraph_possibly(node):
                # Hidden inputs must be taken care of.
                node_attributes = []
                for att in node.attribute:
                    node_attributes.append(
                        att
                        if att.type != AttributeProto.GRAPH
                        else oh.make_attribute(
                            att.name,
                            cls._rename_inputs_in_subgraph(att.g, replacements),
                        )
                    )
            else:
                node_attributes = node.attribute
            new_node.attribute.extend(node_attributes)
            return new_node
        return node

    @classmethod
    def _rename_inputs_in_subgraph(
        cls, graph: GraphProto, replacements: Dict[str, str]
    ) -> GraphProto:
        """
        Renames inputs.
        There is a weird case when one of the result defined in the inner context
        is overriden by the subgraph itself. We should assume this case never happens.
        """
        # graph inputs and outputs should not be changed, initializer as well
        to_rename = set(replacements)
        was_copied = False
        if set(i.name for i in graph.input) & to_rename:
            # An input of the graph is overrides one of the replacements.
            # The replacement should noe take place then.
            replacements = replacements.copy()
            for i in graph.input:
                if i.name in to_rename:
                    del replacements[i.name]
            to_rename = set(replacements)
            was_copied = True

        nodes = []
        for node in graph.node:
            nodes.append(cls._rename_inputs_in_node(node, replacements, to_rename))
            if set(node.output) & to_rename:
                # An output overrides a replacement
                if not was_copied:
                    replacements = replacements.copy()
                    was_copied = True
                for i in node.output:
                    if i in to_rename:
                        del replacements[i]
                to_rename = set(replacements)

        return oh.make_graph(
            nodes,
            graph.name,
            graph.input,
            graph.output,
            graph.initializer,
            sparse_initializer=graph.sparse_initializer,
        )

    def remove_identity_nodes(self) -> Tuple[int, int]:
        """
        Removes identity nodes. Returns the number of removed nodes
        and the number of added nodes.

        .. note::

            onnxruntime does not handle well when it is executing from domain
            *'org.pytorch.aten'* (ATen for example) which outputs results
            on CPU where the expected output is on CUDA. An identity node must be
            kept or inserted in that case. In that particular case, a node can be
            marked so that it does not get deleted: its name must start with
            ``'_DONOTREMOVE_'``.

        The default behavior is to keep the name coming from the node producing the result
        unless this name is an output.
        Shadowing is not an issue but there is also post-shadowing, a result is creating
        after a subgraph has created the same name.
        """
        self._clean_values_cache()
        # make_initializer
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}.remove_identity_nodes] "
                f"-- starts with {len(self.nodes)}"
            )
        # first pass: detect replacements
        new_nodes = []
        input_names = set(i.name for i in self.inputs)
        input_names_and_init = set(self.initializers_dict) | input_names
        output_names = set(i.name for i in self.outputs)
        replacements = {}
        replacements_rev = {}
        removed = 0
        for node in self.nodes:
            if node.op_type != "Identity" or self.do_not_remove(node):
                new_nodes.append(node)
                continue

            if node.output[0] not in output_names:
                old_name, new_name = node.output[0], node.input[0]
            elif (
                node.input[0] not in input_names_and_init
                and node.input[0] not in output_names
                and node.input[0] not in replacements
                and node.input[0] not in self._context  # needed for subgraphs
            ):
                old_name, new_name = node.input[0], node.output[0]
            else:
                new_nodes.append(node)
                continue

            # the new name can be set for replacements as well
            if new_name in replacements:
                new_name = replacements[new_name]
                assert new_name not in replacements, (
                    f"Name {old_name!r} still in {replacements}, "
                    f"node.op_type={node.op_type!r}, "
                    f"node.input={node.input}, node.output={node.output}, "
                    f"input_names={input_names}, output_names={output_names}"
                )
            if old_name in replacements_rev:
                old_old_name = replacements_rev[old_name]
                replacements[old_old_name] = new_name
                replacements_rev[new_name] = old_old_name
            if old_name in replacements:
                replacements[replacements[old_name]] = new_name
            assert new_name not in replacements, (
                f"Name {old_name!r} still in {replacements}, "
                f"node.op_type={node.op_type!r}, "
                f"node.input={node.input}, node.output={node.output}, "
                f"input_names={input_names}, output_names={output_names}"
            )

            if old_name in replacements_rev:
                # A tricky case:
                # x -> Identity -> a -> Identity -> b -> Flatten -> output1
                # x -> Identity -> output0
                # How x should be renamed?
                assert new_name in output_names, (
                    f"replacement {old_name}->{new_name} "
                    f"is not possible because of "
                    f"[{replacements_rev[old_name]}->{old_name}] "
                    f"and {new_name!r} is not an output"
                )
                updates = {}
                for k, v in replacements.items():
                    if v == old_name:
                        updates[k] = new_name
                replacements.update(updates)

            replacements[old_name] = new_name
            replacements_rev[new_name] = old_name

            # verification
            if len(replacements) < 500:
                for k, v in replacements.items():
                    assert v not in replacements, (
                        f"replacement {k}->{v} is not possible because of "
                        f"[{v}->{replacements[v]}], old_name={old_name!r}, "
                        f"new_name={new_name!r}\n---\n{pprint.pformat(replacements)}"
                        f"{self.get_debug_msg()}"
                    )
            removed += 1

        # second pass: replacements in initializer
        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.remove_identity_nodes] found "
                f"{len(replacements)} replacements"
            )
        for k, v in replacements.items():
            if k in self.initializers_dict:
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder-{self._hash()}.remove_identity_nodes] "
                        f"rename initializer {k!r} by {v!r}"
                    )
                assert "FakeTensor" not in str(type(self.initializers_dict[k])), (
                    f"FakeTensor {k!r} cannot be an initializer "
                    f"{type(self.initializers_dict[k])}{self.get_debug_msg()}"
                )
                source = (
                    ""
                    if k not in self.initializers_dict_sources
                    or not self.initializers_dict_sources[k].source
                    else f"##{self.initializers_dict_sources[k].source}"
                )
                k_shape = self.get_shape(k)
                self.add_initializer(
                    v,
                    self.initializers_dict[k],
                    itype=self.get_type(k),
                    shape=k_shape,
                    cst=self.constants_[k],
                    existing=None,
                    source=f"GraphBuilder.remove_identity_nodes/from({k}){source}",
                    allow_empty=len(k_shape) > 0 and 0 in k_shape,
                )
                del self.initializers_dict[k]
                del self.constants_[k]

        # third pass: replacements in node
        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.remove_identity_nodes] "
                f"kept {len(new_nodes)} nodes"
            )
        self.nodes = []
        added = 0
        for node in new_nodes:
            repo = {o for o in node.output if o in replacements}
            repi = {
                o
                for o in self._enumerate_inputs_with_subgraph_plus_hidden(node)
                if o in replacements
            }
            if repi or repo:
                new_inputs = [replacements.get(i, i) for i in node.input]
                new_outputs = [replacements.get(i, i) for i in node.output]
                assert set(new_inputs) & set(new_outputs) in ({""}, set()), (
                    f"Node type {node.op_type}-{node.name} is incorrectly replaced "
                    f"{node.input}->{new_inputs} and {node.output}->{new_outputs}\n"
                    f"replacements are\n{pprint.pformat(replacements)}"
                )
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder-{self._hash()}.remove_identity_nodes] "
                        f"node {node.op_type}-{node.name}:"
                        f"{node.input}->{new_inputs}:{node.output}->{new_outputs}"
                    )
                new_node = node
                del node.input[:]
                node.input.extend(new_inputs)
                del node.output[:]
                node.output.extend(new_outputs)

                # oh.make_node(
                #    node.op_type,
                #    new_inputs,
                #    new_outputs,
                #    domain=node.domain,
                #    name=node.name,
                # )

                added += 1
                removed += 1

                if self._has_subgraph_possibly(node):
                    # Hidden inputs must be taken care of.
                    short_replacements = {k: replacements[k] for k in repi if k in replacements}
                    node_attributes = []
                    for att in node.attribute:
                        node_attributes.append(
                            att
                            if att.type != AttributeProto.GRAPH
                            else oh.make_attribute(
                                att.name,
                                self._rename_inputs_in_subgraph(att.g, short_replacements),
                            )
                        )
                    del new_node.attribute[:]
                    new_node.attribute.extend(node_attributes)

                self.nodes.append(new_node)
                if all(self.is_constant(i) for i in new_node.input):
                    for o in new_node.output:
                        self.update_node_constant(o, new_node)
            else:
                self.nodes.append(node)

        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.remove_identity_nodes] ends with "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_} seconds"
            )

        # fourth pass: simplify the graph.
        identity_outputs = {}
        for node in self.nodes:
            if node.op_type != "Identity" or node.domain != "":
                continue
            anc = node.input[0]
            while anc in identity_outputs:
                anc = identity_outputs[anc]
            identity_outputs[node.output[0]] = anc

        for node in self.nodes:
            new_inputs = []
            rename = False
            for i in node.input:
                if i in identity_outputs:
                    new_inputs.append(identity_outputs[i])
                    rename = True
                else:
                    new_inputs.append(i)
            if rename:
                del node.input[:]
                node.input.extend(new_inputs)

        # results
        self._refresh_values_cache()
        return removed, added

    def remove_duplicated_shape_nodes(self):
        """
        Replaces duplicated operator Shape.
        If one of them return the same output, it is replace by an identity node.
        """
        cache = {}
        changes = {}
        for i, node in enumerate(self.nodes):
            if node.op_type == "Shape" and node.domain == "":
                value = self.value_as_shape(node.output[0])
                if value is not None:
                    if value not in cache:
                        cache[value] = node.output[0]
                    else:
                        changes[i] = oh.make_node(
                            "Identity",
                            [cache[value]],
                            [node.output[0]],
                            name="remove_duplicated_shape_nodes",
                            doc_string=node.doc_string,
                        )
        if changes:
            for i, n in changes.items():
                self.nodes[i] = n
        return len(changes), len(changes)

    def _position_msg(
        self, nodes: List[NodeProto], around: Optional[Tuple[int, int]] = None
    ) -> str:
        "Builds an error message."
        pos = {}
        posi = {}
        for i, n in enumerate(self.nodes):
            if n is None:
                continue
            pos[make_idn(n)] = i
            for o in n.output:
                pos[o] = i
            for o in n.input:
                if o not in posi:
                    posi[o] = i

        rows = []
        for node in nodes:
            if node is None:
                continue
            rows.append(
                f"{node.op_type}({', '.join(node.input)}) -> "
                f"[{', '.join(node.output)}]  -- {node.name}"
            )
            for i in node.input:
                rows.append(f"  -> pos({i}) = {pos.get(i, ' -')} -> {posi.get(i, ' -')}")
            for i in node.output:
                rows.append(f"  <- pos({i}) = {pos.get(i, ' -')} -> {posi.get(i, ' -')}")
        if around is None:
            return "\n".join(rows)

        rows.append("---")
        for i in range(max(0, around[0] - 3), min(len(self.nodes), around[1] + 3)):
            n = self.nodes[i]
            if n is None:
                continue
            rows.append(
                f"P{i}: {n.op_type}({', '.join(n.input)}) -> "
                f"[{', '.join(n.output)}]                   -- {n.name}"
            )
        return "\n".join(rows)

    def _needed_at_first_at(self):
        "Needed by insert_and_remove_nodes."
        needed_at = {}
        first_at = {}
        for i, node in enumerate(self.nodes):
            for name in node.input:
                if name and name not in needed_at:
                    needed_at[name] = i
            for name in node.output:
                if name and name not in first_at:
                    first_at[name] = i
        return needed_at, first_at

    def _move_node_position(self, pos: int) -> int:
        """Tries to move a node at position pos closed to the beginning."""
        the_node = self.nodes[pos]
        first_at = {}
        for i, node in enumerate(self.nodes):
            if i > pos:
                break
            for name in node.output:
                if name not in first_at:
                    first_at[name] = i
        can_be = max(first_at.get(i, 0) for i in the_node.input) + 1
        if can_be >= pos:
            return None
        self.nodes[can_be + 1 : pos + 1] = self.nodes[can_be:pos]
        self.nodes[can_be] = the_node
        return can_be

    def insert_and_remove_nodes(
        self,
        insert_at: Optional[int],
        new_nodes: List[NodeProto],
        removed: List[int],
        opsets: Optional[Dict[str, int]] = None,
        debug: Optional[Any] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> List[NodeProto]:
        """
        Inserts new nodes and removes others.

        :param insert_at: insert the new nodes at this position,
            if empty, the function guesses where to add them
        :param new_nodes: list of nodes to insert
        :param removed: list of nodes to removed (based on their positions)
        :param opsets: opsets used
        :param debug: anything added to exception messages
        :param stats: to get information about time processing
        :return: list of removed nodes
        """
        begin = time.perf_counter()
        assert insert_at is None or not removed or min(removed) <= insert_at, (
            f"The position {insert_at} must be higher than the position "
            f"of the removed nodes {removed}"
        )
        remove_constants = set()
        memo = []
        for i in removed:
            assert i < len(self.nodes), (
                f"Unable to remove node position {i}, there are {len(self.nodes)}, "
                f"type is {self.nodes[i].op_type}"
            )
            n = self.nodes[i]
            if not n:
                # already marked as removed
                continue
            assert n and not self.do_not_remove(n), (
                f"Node {n.name!r} marked as 'DONOTREMOVE' cannot be removed "
                f"for node {i} and type {n.op_type}"
            )
            memo.append(n)
            # We need to remove the constant.
            for o in n.output:
                if o in self.constants_:
                    remove_constants.add(o)
            self.nodes[i] = None

        # finding the position and updating the opsets if needed.
        n_existing = []
        for node in new_nodes:
            for i in self._enumerate_inputs_with_subgraph_plus_hidden(node):
                if not i:
                    # optional input
                    continue
                assert self.has_name(i), (
                    f"Input {i!r} does not exist for node {node.op_type!r}, "
                    f"debug={debug}{self.get_debug_msg()}\n---------\n{node}"
                )
            for o in node.output:
                if self.has_name(o):
                    # connecting to existing output
                    n_existing.append(o)
                elif o:
                    self.set_name(o, f"insert_and_remove_nodes_{node.op_type}_{o}")

            node_domain = node.domain or ""
            if node_domain in self.opsets:
                if opsets and node_domain in opsets:
                    assert compatible_opsets(
                        node_domain,
                        node.op_type,
                        current=self.opsets[node_domain],
                        new_version=opsets[node_domain],
                    ), (
                        f"Incompatible opset for node {node_domain!r} "
                        f"from domain {node_domain!r}, "
                        f"current is {self.opsets[node_domain]}, "
                        f"new is {opsets[node_domain]}"
                    )
            else:
                if opsets and node_domain in opsets:
                    self.opsets[node_domain] = opsets[node_domain]
                else:
                    self.opsets[node_domain] = choose_consistent_domain_opset(
                        node_domain,
                        opsets=self.opsets,
                    )

        assert n_existing, "Any output of the new node is connected to existing names."
        if insert_at is not None:
            for i, n in enumerate(new_nodes):
                assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
                self.nodes.insert(insert_at + i, n)
                self._make_node_set_type_shape_constant(n, True)
                self._make_node_set_type_shape(n)
                self.simple_update_value_shape_with_node(n)
                assert (
                    n.domain != ""
                    or self._has_subgraph_possibly(n)
                    or any(self.is_sequence(o) for o in n.input if o)
                    or any(self.is_sequence(o) for o in n.output if o)
                    or any(not self.has_type(o) for o in n.input if o)
                    or all(self.has_type(o) for o in n.output if o)
                ), (
                    f"Missing one output type in node=[{self.pretty_node(n)}], "
                    f"input_has_type={[(self.has_type(o) if o else '-') for o in n.input]}, "
                    f"output_has_type={[(self.has_type(o) if o else '-') for o in n.output]}, "
                    f"io_has_sequence={[(self.is_sequence(o) if o else '-') for o in n.input]}, "
                    f"{[(self.is_sequence(o) if o else '-') for o in n.output]}, "
                    f"{self.get_debug_msg()}"
                )
            self.nodes = [n for n in self.nodes if n is not None]
            if stats:
                stats.append(
                    dict(
                        pattern="insert_and_remove_nodes",
                        time_in=time.perf_counter() - begin,
                        exit_point="insert_at",
                    )
                )
            return memo

        self.nodes = [n for n in self.nodes if n is not None]

        # Needs to insert the nodes at the right location.
        # Let's find out where the best position is.
        needed_at, first_at = self._needed_at_first_at()

        # First loop to check positions are ok otherwise move a node or two.
        needs_sort = False
        N = len(self.nodes)
        inode = 0
        while inode < len(new_nodes):
            node = new_nodes[inode]
            if node.input:
                min_position = max(first_at.get(i, -1) for i in node.input if i) + 1
            else:
                # a constant node
                min_position = 0
            max_position = min(needed_at.get(o, N) for o in node.output)

            if min_position <= max_position:
                inode += 1
                continue

            # trouble, let's assume one move is ok.
            mini = max((first_at.get(i, -1), i) for i in node.input if i)
            pos, name = mini
            assert (
                name in self.nodes[pos].output
            ), f"Name {name!r} should be at node position {pos}"
            new_position = self._move_node_position(pos)
            if not new_position:
                needs_sort = True
                position_to_insert = mini[0]
                if self.verbose > 1:
                    print(
                        f"[GraphBuilder-{self._hash()}.insert_and_remove_nodes] needs full sort "
                        f"to add new nodes at because node at position {pos} "
                        f"cannot be moved.\n----\n{self._position_msg([node])}"
                        f"\n-------\n{self._position_msg(new_nodes)}"
                    )
                break
            needed_at, first_at = self._needed_at_first_at()

        if needs_sort:
            # Requires full sort.
            self.nodes = [node for node in self.nodes if node is not None]
            pos = min(len(self.nodes), position_to_insert)
            self.nodes = [*self.nodes[:pos], *new_nodes, *self.nodes[pos:]]
            self.topological_sort(stats)
            new_nodes_p = [(None, None, n) for n in new_nodes]
        else:
            # guess the position to insert the nodes at
            # the order of the new nodes is consistent but it may have to be changed
            # if it does not fit the existing order
            def err_msg():
                lines = []
                for n in new_nodes:
                    first = " - ".join([str(first_at.get(i, ".")) for i in n.input if i])
                    need = " - ".join([str(needed_at.get(i, ".")) for i in n.output if i])
                    lines.append(f"{first} -> {need} -- {self.pretty_node(n)}")
                return "\n".join(lines)

            insert_needed_at = {}
            insert_first_at = {}
            N = len(self.nodes)
            inserted_at = []
            new_nodes_p = []
            for init, node in enumerate(new_nodes):
                min_position = (
                    (max(first_at.get(i, -1) for i in node.input if i) + 1) if node.input else 0
                )
                max_position = min(needed_at.get(o, N) for o in node.output)

                assert min_position <= max_position, (
                    f"Unable to insert node {self.pretty_node(node)}, "
                    f"min_position={min_position}, max_position={max_position}, "
                    f"len(nodes)={len(self.nodes)}, previous insertions={inserted_at}, "
                    f"insert_needed_at={insert_needed_at}, insert_first_at={insert_first_at}, "
                    f"inserted_at={inserted_at}\n{self._position_msg([node])}"
                    f"\n-------\n{self._position_msg(new_nodes)}"
                )

                local_min_position = (
                    max(insert_first_at.get(i, -1) for i in node.input if i) if node.input else 0
                )
                local_max_position = min(insert_needed_at.get(o, N) for o in node.output)

                assert local_min_position <= local_max_position, (
                    f"Unable to insert node {self.pretty_node(node)}, "
                    f"local_min_position={local_min_position}, "
                    f"local_max_position={local_max_position}, "
                    f"len(nodes)={len(self.nodes)}, previous insertions={inserted_at}, "
                    f"insert_needed_at={insert_needed_at}, insert_first_at={insert_first_at}"
                    f"\n----\n{err_msg()}\n---\n"
                    f"insert_needed_at={insert_needed_at}\n"
                    f"insert_first_at={insert_first_at}"
                )

                insert_position = max(min_position, local_min_position)

                assert insert_position <= max_position, (
                    f"Insertion will fail, for output {o!r} from node\n"
                    f"{self.pretty_node(node)}\nbecause insert_position="
                    f"{insert_position}, local_min_position={local_min_position}, "
                    f"local_max_position={local_max_position}, "
                    f"min_position={min_position}, max_position={max_position}"
                    f"\n----\n{err_msg()}\n---\n"
                    f"insert_needed_at={insert_needed_at}\n"
                    f"insert_first_at={insert_first_at}"
                )

                new_nodes_p.append((insert_position, init, node))
                for i in node.input:
                    if i:
                        insert_needed_at[i] = min(
                            insert_position, insert_needed_at.get(i, insert_position)
                        )
                for i in node.output:
                    if i:
                        insert_first_at[i] = min(
                            insert_position, insert_first_at.get(i, insert_position)
                        )

            assert len(new_nodes) == len(new_nodes_p), (
                f"Length mismatch between len(new_nodes)={len(new_nodes)} == "
                f"len(new_nodes_p)={len(new_nodes_p)}"
            )
            new_nodes_p.sort()

            # do the addition
            for p, _, n in reversed(new_nodes_p):
                assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
                self.nodes.insert(p, n)

        # finalization
        init_nams = {}
        for _, _, n in new_nodes_p:
            if n.output[0] in init_nams:
                continue
            self._make_node_set_type_shape_constant(n, True)
            self._make_node_set_type_shape(n)
            self.simple_update_value_shape_with_node(n)
            assert (
                n.domain != ""
                or any(not self.has_type(o) for o in n.input if o)
                or all(self.has_type(o) for o in n.output if o)
            ), (
                f"Missing one output type in node={self.pretty_node(n)}, "
                f"input_has_type={[self.has_type(o) for o in n.input]}, "
                f"output_has_type={[self.has_type(o) for o in n.output]}, "
                f"{self.get_debug_msg()}"
            )
        # final removal of the unneeded constants
        for o in remove_constants:
            del self.constants_[o]
        self._refresh_values_cache()
        if stats:
            stats.append(
                dict(
                    pattern="insert_and_remove_nodes",
                    time_in=time.perf_counter() - begin,
                    exit_point="topological_sort" if needs_sort else "positions",
                )
            )
        return memo

    def topological_sort(self, stats: Optional[List[Dict[str, Any]]] = None):
        """
        Reorders the nodes in order to make sure no node is using an output
        created by a node after.
        """
        # Gathering IO
        begin = time.perf_counter()
        self.nodes = [node for node in self.nodes if node is not None]
        output_index_node = {name: -1 for name in self.input_names}  # noqa: C420
        output_index_node.update({name: -1 for name in self.initializers_dict})  # noqa: C420
        ios = {}
        for i, node in enumerate(self.nodes):
            unique = set(node.input)
            subgraphs = [att.g for att in node.attribute if att.type == AttributeProto.GRAPH]
            for sub in subgraphs:
                unique |= set(self._get_hidden_inputs(sub))
            ios[i] = unique
            for o in node.output:
                output_index_node[o] = i
        # Begin sorting
        new_order = [0 for i in range(len(self.nodes))]
        it = 0
        while it < len(self.nodes) + 1:
            changes = 0
            for i in range(len(self.nodes)):
                inputs = ios[i]
                if inputs:
                    indices = [output_index_node[o] for o in inputs]
                    minis = [new_order[i] for i in indices if i >= 0]
                    if minis:
                        mini = max(minis) + 1
                        if mini > new_order[i]:
                            changes += 1
                            new_order[i] = mini
            if changes == 0:
                break
            it += 1
        assert it < len(self.nodes), (
            f"{it} (with {len(self.nodes)}) iterations were necessary to sort the graph. "
            f"There may be a cycle.{self.get_debug_msg()}"
        )

        sorted_nodes = [
            node for _, __, node in sorted(zip(new_order, range(len(self.nodes)), self.nodes))
        ]
        self.nodes = sorted_nodes
        if stats:
            stats.append(dict(pattern="topological_sort", time_in=time.perf_counter() - begin))
        self._check(stats or [], "topological_sort")

    @classmethod
    def _clean_shapes(cls, proto: Union[GraphProto, ModelProto]):
        # cleaning unresolved shapes
        if isinstance(proto, ModelProto):
            cls._clean_shapes(proto.graph)
            return
        new_shapes = []
        for sh in proto.value_info:
            if sh.type.tensor_type.elem_type == 0:
                continue
            new_shapes.append(sh)
        del proto.value_info[:]
        proto.value_info.extend(new_shapes)

    def _update_shape_types_with_proto_one_result(self, val: ValueInfoProto):
        itype = val.type.tensor_type.elem_type
        if itype > 0:
            self.set_type(val.name, itype)
        shape = tuple(
            (
                d.dim_param.replace(" ", "")
                if d.dim_param
                else (d.dim_value if d.dim_value else self.unique_dimension_name("NEWDIM"))
            )
            for d in val.type.tensor_type.shape.dim
        )
        if all(isinstance(s, int) for s in shape) and -1 in shape:
            # Some converters uses -1 to specify a dynamic dimension.
            # We replace the value with a string
            new_shape = []
            for index, s in enumerate(shape):
                if s >= 0:
                    new_shape.append(s)
                    continue
                dyn_name = f"{val.name}_{index}"
                new_shape.append(dyn_name)
            shape = tuple(new_shape)
        for i, sh in enumerate(shape):
            if isinstance(sh, int):
                continue
            self.make_dynamic_object(sh, self.torch.SymInt(sh), input_name=val.name, axis=i)
        if (
            0 in shape
            and len(shape) > 1
            and all(isinstance(s, int) for s in shape)
            and min(shape) == max(shape) == 0
        ):
            # something like (0, 0, 0), we skip.
            if self.verbose > 4:
                print(
                    f"[_update_shape_types_with_proto_one_result] cannot "
                    f"update shape({val.name}) with {shape}"
                )
            return
        if self.verbose > 3:
            print(
                f"[_update_shape_types_with_proto_one_result] "
                f"update shape({val.name}) with {shape}"
            )
        self.set_shape(val.name, shape, exc=False, allow_zero=True)

    def _update_shape_types_with_proto(
        self,
        proto: Union[GraphProto, ModelProto],
        infer_shapes_options: InferShapesOptions = InferShapesOptions.NONE,
    ):
        """
        Updates the shapes and types for an existing model.

        :param proto: model proto
        :param infer_shapes_options: infer shapes to fill information about type and shapes
            run shape inference, if the value is `'new'`,
            existing shapes are ignored
        """
        assert isinstance(
            proto, (GraphProto, ModelProto)
        ), f"Unexpected type {type(proto)} for proto"
        proto_graph = proto if isinstance(proto, GraphProto) else proto.graph
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._update_shape_types_with_proto] -- starts with "
                f"{len(self.nodes)} nodes and {len(getattr(proto_graph, 'value_info', 0))} "
                f"shapes."
            )

        # if infer_shapes_options & InferShapesOptions.NEW:
        #     clean_shapes(proto)
        if infer_shapes_options & InferShapesOptions.ONNX:
            if self.verbose > 1:
                print("[GraphBuilder._update_shape_types_with_proto] infer shapes")
            new_proto = onnx_infer_shapes(
                proto, data_prop=bool(infer_shapes_options & InferShapesOptions.DATA_PROP)
            )
            if self.verbose > 1:
                print(
                    "[GraphBuilder._update_shape_types_with_proto] "
                    f"infer shapes done {time.perf_counter() - begin_} seconds"
                )
            self._clean_shapes(new_proto)
            if self.verbose > 1:
                print(
                    "[GraphBuilder._update_shape_types_with_proto] "
                    f"_clean_shapes after {time.perf_counter() - begin_} seconds"
                )
        else:
            new_proto = proto

        new_proto_graph = new_proto if isinstance(new_proto, GraphProto) else new_proto.graph
        if not hasattr(new_proto_graph, "value_info"):
            if self.verbose > 1:
                print(
                    f"[GraphBuilder-{self._hash()}._update_shape_types_with_proto] ends in "
                    f"{time.perf_counter() - begin_} seconds."
                )
            return

        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._update_shape_types_with_proto] "
                f"walk through {len(proto_graph.value_info)} shapes."
            )
        for val in new_proto_graph.value_info:
            self._update_shape_types_with_proto_one_result(val)

        if infer_shapes_options & InferShapesOptions.BUILDER:
            self.infer_shapes()

        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}._update_shape_types_with_proto] ends in "
                f"{time.perf_counter() - begin_} seconds."
            )

    def _make_node_set_type_shape(self, node: NodeProto):
        """Updates shapes for a node."""
        if node.domain == "":
            node.doc_string += "#Io1"
            set_shape_type_op_any(self, node)
        else:
            # Missing type means it is probably coming from an inlined function.
            node.doc_string += (
                "#Io3" if node.input and not self.has_type(node.input[0]) else "#Io2"
            )
            set_shape_type_custom(self, node)

    def infer_shapes(self) -> Dict[str, Tuple[DYNAMIC_SHAPE, DYNAMIC_SHAPE]]:
        """Runs custom shape inference. Returns the updates."""
        if self.verbose > 1:
            begin = time.perf_counter()
            print("[GraphBuilder.infer_shapes]")
        res = {}
        for node in self.nodes:
            old_shapes = [(self.get_shape(o) if self.has_shape(o) else None) for o in node.output]
            self.simple_update_value_shape_with_node(node)
            self._make_node_set_type_shape(node)
            new_shapes = [(self.get_shape(o) if self.has_shape(o) else None) for o in node.output]
            diff = {}
            for n, a, b in zip(node.output, old_shapes, new_shapes):
                if a != b:
                    diff[n] = (a, b)
            if diff and self.verbose > 2:
                print(
                    f"[GraphBuilder-{self._hash()}.infer_shapes] "
                    f"update node {node.op_type!r}, "
                    f"name {node.name!r}, updates={diff}"
                )
            elif self.verbose > 4:
                print(
                    f"[GraphBuilder-{self._hash()}.infer_shapes] node {node.op_type!r}, "
                    f"name {node.name!r}, shape={dict(zip(node.output, new_shapes))}"
                )
            res.update(diff)
        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}.infer_shapes] done in "
                f"{time.perf_counter() - begin} with {len(diff)} changes"
            )
        return res

    def _update_structures_with_proto(
        self, proto: Union[ModelProto, GraphProto], infer_shapes_options: InferShapesOptions
    ):
        """Updates the shapes and types for an existing model."""
        proto_graph = proto.graph if isinstance(proto, ModelProto) else proto
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._update_structures_with_proto] -- starts with "
                f"{len(proto_graph.node)} nodes"
            )
        self.existing_metadata_props = {p.key: p.value for p in proto.metadata_props}
        if isinstance(proto, (FunctionProto, ModelProto)):
            self.opsets = {d.domain: d.version for d in proto.opset_import}
        else:
            assert self.opsets, "opsets must be not None"
        if self.ir_version is None and isinstance(proto, ModelProto):
            self.ir_version = proto.ir_version
        self.nodes = list(proto_graph.node)
        self.functions = {}
        self.functions_builder = {}
        if isinstance(proto_graph, GraphProto):
            for i in proto_graph.initializer:
                self.add_initializer(
                    i.name,
                    i,
                    allow_empty=True,
                    source=f"GraphBuilder._update_structures_with_proto.1/from({i.name})",
                )
            for i in proto_graph.sparse_initializer:
                self.add_initializer(
                    i.values.name,
                    i,
                    allow_empty=True,
                    source=f"GraphBuilder._update_structures_with_proto.2/from({i.name})",
                )
        if isinstance(proto, ModelProto):
            for f in proto.functions:
                self.add_function(f)
        self.value_info = (
            list(proto_graph.value_info)
            if not (infer_shapes_options & InferShapesOptions.NEW)
            else []
        )
        if isinstance(proto_graph, GraphProto):
            self.inputs = list(proto_graph.input)
            self.outputs = list(proto_graph.output)
            self.input_names = [i.name for i in proto_graph.input]
        else:
            assert isinstance(
                proto_graph, FunctionProto
            ), f"Unexpected type {type(proto_graph)} for proto_graph"
            self.inputs = [
                oh.make_tensor_value_info(name, self.TEMPLATE_TYPE, None)
                for name in proto_graph.input
            ]
            self.outputs = [
                oh.make_tensor_value_info(name, self.TEMPLATE_TYPE, None)
                for name in proto_graph.output
            ]
            self.input_names = list(proto_graph.input)

        available_shapes = (
            {v.name: v for v in proto_graph.value_info}
            if hasattr(proto_graph, "value_info")
            else {}
        )

        for i in self.inputs + self.outputs:
            self.set_name(i.name, f"_update_structures_with_proto_{i}")
            self.set_type(i.name, i.type.tensor_type.elem_type)
            if i.type.tensor_type.shape.dim:
                shape = tuple(
                    (
                        d.dim_param.replace(" ", "")
                        if d.dim_param
                        else (
                            d.dim_value
                            if d.dim_value
                            else self.unique_dimension_name("UNKNOWNDIM")
                        )
                    )
                    for d in i.type.tensor_type.shape.dim
                )
                if all(isinstance(s, int) for s in shape) and -1 in shape:
                    # Some converters uses -1 to specify a dynamic dimension.
                    # We replace the value with a string
                    new_shape = []
                    for index, s in enumerate(shape):
                        if s >= 0:
                            new_shape.append(s)
                            continue
                        dyn_name = f"{i.name}_{index}"
                        new_shape.append(dyn_name)
                    shape = tuple(new_shape)
                new_shape = []
                for axis, sh in enumerate(shape):
                    if isinstance(sh, int):
                        if sh != 0:
                            new_shape.append(sh)
                            continue
                        # We replace it with a letter.
                        sh = f"dim_{i.name}_{axis}"
                    new_shape.append(sh)
                    self.make_dynamic_object(
                        sh, self.torch.SymInt(sh), input_name=i.name, axis=axis
                    )
                shape = tuple(new_shape)
                self.set_shape(i.name, shape)
            if (
                self.get_type(i.name) == TensorProto.INT64
                and self.has_shape(i.name)
                and self.get_shape(i.name) in ((1,), tuple)
                and "dim" in i.name
            ):
                self.set_value_shape(i.name, (i.name,))

        need_identity_removal = False
        new_nodes = []
        for node in self.nodes:
            if self._parent:
                # This is a subgraph, let's import information from
                # the parent graph.
                for i in node.input:
                    self._import_context_from_parent_graph(i)

            # ready to add the node
            self._unique_names |= set(node.output)
            shape_set = self.simple_update_value_shape_with_node(node)
            if node.name:
                self._unique_node_names.add(node.name)

            if shape_set:
                self._make_node_set_type_shape_constant(node, {})
                self._make_node_set_type_shape(node)
                for o in node.output:
                    if not self.has_name(o):
                        self.set_name(o, f"_update_structures_with_proto_n_{o}")
                new_nodes.append(node)
                continue

            if node.op_type == "SequenceConstruct":
                dtypes = [(self.get_type(n) if self.has_type(n) else 0) for n in node.input]
                ranks = [(self.get_rank(n) if self.has_rank(n) else -1) for n in node.input]
                unique_dtypes = set(dtypes)
                if 0 in unique_dtypes:
                    unique_dtypes = set(u for u in unique_dtypes if u != 0)
                    if len(unique_dtypes) == 0:
                        if not self.has_name(node.output[0]):
                            self.set_name(
                                node.output[0],
                                f"_update_structures_with_proto_SC1_{node.output[0]}",
                            )
                        self.set_sequence(
                            node.output[0], 0, shapes=None, ranks=None, unknown=True
                        )
                        new_nodes.append(node)
                        continue

                assert len(unique_dtypes) == 1, (
                    f"A sequence has distinct dtype: {dtypes}, "
                    f"(unique_dtypes={unique_dtypes}), node.name={node.name}, "
                    f"node.input={node.input}"
                )
                if not self.has_name(node.output[0]):
                    self.set_name(
                        node.output[0], f"_update_structures_with_proto_SC2_{node.output[0]}"
                    )
                self.set_sequence(
                    node.output[0],
                    next(_ for _ in unique_dtypes),
                    shapes=None,
                    ranks=ranks,
                )
                new_nodes.append(node)
                continue

            if node.op_type == "SequenceAt":
                position = self.get_constant(node.input[1], computed_value=True)
                seq = self.get_sequence(node.input[0])
                dtype = seq["dtype"]
                if isinstance(dtype, tuple):
                    # More than one type is allowed in torch sequences.
                    dtype = dtype[int(position)]
                if not self.has_name(node.output[0]):
                    self.set_name(
                        node.output[0], f"_update_structures_with_proto_SAt_{node.output[0]}"
                    )
                self.set_type(node.output[0], dtype)
                if seq["ranks"] is None:
                    new_nodes.append(node)
                    continue
                rank = seq["ranks"][int(position)]
                self.set_rank(node.output[0], rank)
                new_nodes.append(node)
                continue

            assert node.op_type not in {"SplitToSequence", "SequenceErase"}, (
                f"Sequence operators are not supported yet and op_type={node.op_type!r}"
                f"(name={node.name!r})."
            )

            if node.op_type == "Constant":
                exist = self.is_exact_same_constant(node)
                if exist is not None:
                    node = oh.make_node(
                        "Identity",
                        [exist.output[0]],
                        [node.output[0]],
                        name="._update_structures_with_proto",
                    )
                    replaced = True
                    need_identity_removal = True
                else:
                    self.add_constant_node(node)
                    replaced = False

                self.update_node_constant(node.output[0], node)
                if not self.has_name(node.output[0]):
                    self.set_name(
                        node.output[0], f"_update_structures_with_proto_Cst_{node.output[0]}"
                    )

                if replaced:
                    self.set_shape(node.output[0], self.get_shape(node.input[0]), allow_zero=True)
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                    if self.has_device(node.input[0]):
                        self.set_device(node.output[0], self.get_device(node.input[0]))
                else:
                    self.set_shape(node.output[0], self._get_tensor_shape(node), allow_zero=True)
                    self.set_type(node.output[0], self._get_tensor_type(node))
            elif node.op_type == "ConstantOfShape" and self.is_constant(node.input[0]):
                exist = self.is_exact_same_constant(node)
                if exist is not None:
                    node = oh.make_node(
                        "Identity",
                        [exist.output[0]],
                        [node.output[0]],
                        name="._update_structures_with_proto",
                    )
                    replaced = True
                    need_identity_removal = True
                else:
                    self.add_constant_node(node)
                    replaced = False

                self.update_node_constant(node.output[0], node)
                if not self.has_name(node.output[0]):
                    self.set_name(
                        node.output[0], f"_update_structures_with_proto_CoF_{node.output[0]}"
                    )

                if replaced:
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                    self.set_shape(node.output[0], self.get_shape(node.input[0]))
                    if self.has_device(node.input[0]):
                        self.set_device(node.output[0], self.get_device(node.input[0]))
                else:
                    value = self.get_constant(node.input[0], computed_value=True)
                    shape = tuple(int(i) for i in value)
                    self.set_shape(node.output[0], shape)
                    if len(node.attribute) == 0:
                        self.set_type(node.output[0], TensorProto.FLOAT)
                    else:
                        value = node.attribute[0].t
                        self.set_type(node.output[0], value.data_type)
            else:
                for o in node.output:
                    if o == "":
                        continue
                    if not self.has_name(o):
                        self.set_name(o, f"_update_structures_with_proto_l_{o}")

                self._make_node_set_type_shape_constant(node, {})
                self._make_node_set_type_shape(node)

            new_nodes.append(node)
            for o in node.output:
                if o in available_shapes:
                    self._update_shape_types_with_proto_one_result(available_shapes[o])

            if not infer_shapes_options:
                if any((x not in available_shapes and not self.has_type(x)) for x in node.output):
                    # second try
                    self._make_node_set_type_shape(node)

                # This test should be enabled when shape inference is complete.
                # assert all(
                #     map(
                #         lambda x: x in available_shapes or self.has_type(x), node.output
                #     )
                # ), (
                #     f"One output of node {node.op_type!r} "
                #     f"(name={node.name!r}) has no type: "
                #     f"{', '.join(o + ((':' + str(self.get_type(o))) "
                #     f"if self.has_type(o) else ':0') for o in node.output)}"
                # )

        self.nodes = new_nodes

        if need_identity_removal:
            self.remove_identity_nodes()

        if self.verbose > 1:
            print(
                f"[GraphBuilder-{self._hash()}._update_structures_with_proto] ends with "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_}"
            )

    def parse_dimension_expression(
        self, expr: str, exc: bool = True, allow_new_dynamic_dimension: bool = False
    ) -> Expression:
        """
        Parses an expression involving dimension.

        :param expr: expr
        :param exc: raises an exception if it fails
        :param allow_new_dynamic_dimension: the new dimension is considered as a new one,
            if the dimension is only one variable (:meth:`str.isidentifier`)
        :return: an expression or None if exc is False and the parsing failed
        """
        try:
            return parse_expression(expr, exc=exc, context=self.dynamic_objects)
        except AssertionError as e:
            if allow_new_dynamic_dimension:
                name = parse_expression(expr, exc=False, context=self.dynamic_objects)
                if name.isidentifier():
                    return name
            raise AssertionError(
                f"Unable to parse an expression expr=[{expr!r}], "
                f"allow_new_dynamic_dimension={allow_new_dynamic_dimension}, "
                f"due to {e}, self.dynamic_objects={self.dynamic_objects}, "
                f"set(self.dynamic_objects_rev)={set(self.dynamic_objects_rev)}, "
                f"self.dynamic_shapes={self.dynamic_shapes}, "
                f"self._dynamic_alias={self._dynamic_alias}{self.get_debug_msg()}"
            ) from e

    def _constant_key(self, node: NodeProto) -> Optional[bytes]:
        """
        Builds a unique key for a constant.
        Returns None if the constant if too big.
        """
        if node.op_type == "ConstantOfShape":
            # We assume initializer are fused.
            name = node.input[0]
            while name in self.constants_alias_:
                name = self.constants_alias_[name]
            key = [node.op_type.encode(), name.encode()]
            for att in node.attribute:
                key.append(att.SerializeToString())
            return b"|".join(key)
        if node.op_type == "Constant":
            shape = self._get_tensor_shape(node)
            size = np.prod(shape) if shape else 1
            if size > self.optimization_options.constant_size:
                # It would be too long.
                return None
            key = [node.op_type.encode()]
            for att in node.attribute:
                key.append(att.SerializeToString())
            return b"|".join(key)

        raise RuntimeError(f"Unexpected node type {node.op_type!r}{self.get_debug_msg()}")

    def add_constant_node(self, node: NodeProto) -> Optional[bytes]:
        """
        Adds a constant node. Any constant equivalent to this one
        will be fused.
        `self.optimization_options.constant_fusing` must be True.
        """
        assert isinstance(node, NodeProto), f"Unexpected type {type(node)} for a node"
        if not self.optimization_options.constant_fusing:
            return None
        key = self._constant_key(node)
        assert (
            key not in self.constants_node_
        ), f"A constant with the same key {key!r} was already added{self.get_debug_msg()}"
        self.constants_node_[key] = node
        return key

    def is_exact_same_constant(self, node: NodeProto) -> Optional[NodeProto]:
        """
        Adds a constant node. Any constant equivalent to this one
        will be fused.
        `self.optimization_options.constant_fusing` must be True.
        """
        if not self.optimization_options.constant_fusing:
            return None
        key = self._constant_key(node)
        if key in self.constants_node_:
            return self.constants_node_[key]
        return None

    @classmethod
    def _format_dict(cls, d: Dict[str, Any]) -> str:
        "Formats a dictionay to avoid a too long string."
        rows = ["{"]
        for k, v in sorted(d.items()):
            sv = str(v)
            if "\n" in sv:
                s_ = sv.split("\n")
                sv = f"{s_[0]}...{s_[-1]}"
            if len(sv) > 50:
                sv = f"{sv[:20]}...{sv[-20:]}"
            rows.append(f"  {k}: {sv},")
        rows.append("}")
        return "\n".join(rows)

    def make_local_function(
        self,
        builder: "GraphBuilder",
        function_options: FunctionOptions,
        optimize: bool = False,
        metadata_props: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Tuple[str, str]]:
        """
        Adds a local function to exiting graph.

        :param builder: builder
        :param function_options: to define how to handle weights
        :param optimize: optimize the function
        :param metadata_props: metadata to add to the function
        :return: the list of added initializers if
            *move_initializer_to_constant* is True,
            and the function name (domain, name),
            it can be changed if one is already existing

        Method :meth:`GraphBuilder.inline_functions`,
        :meth:`GraphBuilder.move_initializers_to_constant` are called on
        the builder if *move_initializer_to_constant* is True.
        It modifies the builder inplace.
        """
        name = function_options.name
        domain = function_options.domain
        if self._debug_local_function:
            print(
                f"[GraphBuilder-{self._hash()}.make_local_function] {name}[{domain}]"
                f"({', '.join(_.name for _ in builder.inputs)}) "
                f"-> {', '.join(_.name for _ in builder.outputs)}"
            )
        assert name, f"function_options is wrong {function_options!r}"
        assert (
            function_options.rename_allowed
            or function_options.merge_allowed
            or not self.has_local_function(name=name, domain=domain)
        ), f"Function {name!r}, domain={domain!r} already exists"

        if function_options.move_initializer_to_constant:
            if function_options.inline:
                self._check_constants("before-inline_functions")
                if self._debug_local_function:
                    print(
                        f"[GraphBuilder-{self._hash()}.make_local_function] inline_functions "
                        f"{len(builder.functions)}"
                    )
                builder.inline_functions(verbose=max(0, self.verbose - 1))
                if self._debug_local_function:
                    print(
                        f"[GraphBuilder-{self._hash()}.make_local_function] after inlining "
                        f"{len(builder.functions)}"
                    )
                self._check_constants("after-inline_functions")

        assert (
            not builder.initializers_dict
            or function_options.return_initializer
            or function_options.move_initializer_to_constant
        ), (
            f"incompatible options, return_initializer or "
            f"move_initializer_to_constant must be True, "
            f"move_initializer_to_constant={function_options.move_initializer_to_constant}, "
            f"function_options.inline={function_options.inline}, "
            f"but function_options={function_options!r} with {len(self.initializers_dict)} "
            f"initializers"
        )
        fct = builder.to_onnx(
            function_options=function_options,
            optimize=optimize,
            inline=False,
        )
        assert isinstance(fct, (dict, FunctionProto)), (
            f"Unexpected type {type(fct)}, function_options={function_options}"
            f"{self.get_debug_msg()}"
        )
        onx = fct["proto"] if isinstance(fct, dict) else fct
        if metadata_props:
            oh.set_metadata_props(onx, metadata_props)

        if self._debug_local_function and isinstance(fct, dict):
            print(f"[GraphBuilder-{self._hash()}.make_local_function] keys={', '.join(fct)}")
            if "initializers_name" in fct:
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] initializers_name="
                    f"{fct['initializers_name']}"
                )
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] initializers_dict="
                    f"{list(fct['initializers_dict'])}"
                )
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] initializers_renaming="
                    f"{fct['initializers_renaming']}"
                )
            print(
                f"[GraphBuilder-{self._hash()}.make_local_function] "
                f"create {onx.name}[{onx.domain}]"
                f"({', '.join(onx.input)}) -> {', '.join(onx.output)})"
            )
        doc_string = f"-- function_options={function_options!r}"
        if self.constraints_:
            doc_string += f"\n-- shape-constraints: {pprint.pformat(self.constraints_)}"
        if self._known_value_shape:
            filtered = {
                k: v for k, v in self._known_value_shape.items() if not k.startswith("init")
            }
            if filtered:
                doc_string += f"\n-- known-value-shape: {self._format_dict(filtered)}"
        onx.doc_string += doc_string + (
            f"\n-- optimized:{builder.optimization_options!r}" if optimize else "not-optimized"
        )

        to_rename = {}
        keys = []
        for key, f in builder.functions.items():
            # What if on is already existing?
            old_key = f.domain, f.name
            new_key = self.add_function(
                f,
                rename_allowed=function_options.rename_allowed,
                merge_allowed=function_options.merge_allowed,
                builder=builder.functions_builder.get(key),
            )
            if new_key != old_key:
                to_rename[old_key] = new_key
            keys.append(new_key)

        if self._debug_local_function:
            print(f"[GraphBuilder-{self._hash()}.make_local_function] to_rename={to_rename}")

        if to_rename:
            # We rename the local functions.
            if self._debug_local_function:
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] "
                    f"to rename inputs={onx.input}"
                )
            onx = self.rename_in_local_functions(to_rename, keys, proto=onx)
            if self._debug_local_function:
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] "
                    f"renamed inputs={onx.input}"
                )

        # Let's rename the initializers.
        if isinstance(fct, dict) and "initializers_dict" in fct:
            assert len(fct["initializers_dict"]) == len(fct["initializers_name"]), (
                f"Names mismatch between {fct['initializers_name']} and "
                f"{list(fct['initializers_dict'])}{builder.get_debug_msg()}"
            )
            repl = {}
            for k, v in fct["initializers_dict"].items():
                new_name = self.add_initializer(
                    self.unique_name(k),
                    v,
                    source=f"GraphBuilder.make_local_function/from({k})",
                )
                repl[k] = new_name
            renaming = fct["initializers_renaming"]
            new_inits = []
            for input_name in fct["initializers_name"]:
                init_name = renaming[input_name]
                repl_name = repl[init_name]
                new_inits.append(repl_name)

            if self._debug_local_function:
                print(f"[GraphBuilder-{self._hash()}.make_local_function] new_inits={new_inits}")
        else:
            new_inits = []

        assert isinstance(onx, FunctionProto), (
            f"Unexpected type {type(onx)}, name={name!r}, domain={domain!r}, "
            f"function_options={function_options}"
        )
        assert all(node.op_type != name or node.domain != domain for node in onx.node), (
            f"Recursivity is not allowed in function {name!r}, domain={domain!r}, "
            f"function_options={function_options}\n------ONNX----\n{pretty_onnx(onx)}"
            f"{self.get_debug_msg()}"
        )
        assert (
            not optimize
            or not builder.optimization_options.remove_identity
            or len([n for n in onx.node if n.op_type == "Identity"]) <= len(onx.output)
        ), (
            f"The optimization was not applied. There are two many nodes identity"
            f"\n{builder.pretty_text()}"
        )

        new_domain, new_name = self.add_function(
            onx,
            rename_allowed=function_options.rename_allowed,
            merge_allowed=function_options.merge_allowed,
            builder=builder,
        )
        if new_domain not in self.opsets:
            self.opsets[new_domain] = 1
        locf = self.get_local_function(new_name, new_domain)
        b = self.get_local_function(new_name, new_domain, builder=True)
        if self._debug_local_function:
            print(
                f"[GraphBuilder-{self._hash()}.make_local_function] "
                f"done with {onx.domain}.{onx.name}({', '.join(onx.input)}) -> "
                f"{', '.join(onx.output)}"
            )
            print(
                f"[GraphBuilder-{self._hash()}.make_local_function] "
                f"new_domain={new_domain!r}, new_name={new_name!r}"
            )
            print(
                f"[GraphBuilder-{self._hash()}.make_local_function] "
                f"check proto {locf.domain}.{onx.name}({', '.join(locf.input)}) -> "
                f"{', '.join(locf.output)}"
            )
            if len(b.input_names) < len(locf.input):
                print(
                    f"[GraphBuilder-{self._hash()}.make_local_function] "
                    f"last inputs are constants, builder is "
                    f"({', '.join(b.input_names)}) -> {', '.join(b.output_names)}"
                )
        assert list(locf.input) >= b.input_names, (
            f"Local builder and local function disagrees on the inputs "
            f"{locf.input} != {b.input_names}\n-----{b.get_debug_msg()}\n----"
            f"{self.get_debug_msg()}"
        )
        return new_inits, (new_domain, new_name)

    def rename_in_local_functions(
        self,
        replacements: Dict[Tuple[str, str], Tuple[str, str]],
        list_keys: List[Tuple[str, str]],
        proto: FunctionProto,
    ) -> FunctionProto:
        """
        Renames local function in a given list of local functions.

        :param replacements: replacements to make
        :param list_keys: list of local function to modify
        :param proto: one function to update as well
        :return: the modified proto for proto

        The function does not modify inplace the functions,
        it creates a copy assuming this one is not too big.
        """
        for key in list_keys:
            assert (
                key in self.functions
            ), f"Local function {key!r} is missing from {assert_sorted(self.functions)}"
            new_f = self._rename_op_type_in_local_functions(self.functions[key], replacements)
            self.functions[key] = new_f
        if proto is None:
            return None
        return self._rename_op_type_in_local_functions(proto, replacements)

    @classmethod
    def _detect_op_type_replacements(
        cls,
        proto: Union[FunctionProto, GraphProto],
        replacements: Dict[Tuple[str, str], Tuple[str, str]],
    ) -> bool:
        """Detects a replacements to make in a proto."""
        for node in proto.node:
            if (node.domain, node.op_type) in replacements:
                return True
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    if cls._detect_op_type_replacements(att.g, replacements):
                        return True
        return False

    def _rename_op_type_in_local_functions(
        self,
        proto: Union[FunctionProto, GraphProto],
        replacements: Dict[Tuple[str, str], Tuple[str, str]],
    ) -> Union[FunctionProto, GraphProto]:
        """
        Modifies the function to replace a call by another one.

        :param proto: the proto to modify
        :param replacements: the replacements to do
        :return: the new proto, or the existing one if no replacements was found
        """
        if not self._detect_op_type_replacements(proto, replacements):
            return proto
        if isinstance(proto, FunctionProto):
            new_proto = FunctionProto()
        elif isinstance(proto, GraphProto):
            new_proto = GraphProto()
        else:
            raise AssertionError(f"Unexpected type {type(proto)}")
        new_proto.ParseFromString(proto.SerializeToString())

        self._check_constants("begin-renaming")
        for node in new_proto.node:
            key = node.domain, node.op_type
            if key in replacements:
                node.domain, node.op_type = replacements[key]
            node_attributes = []
            modified = False
            for att in node.attribute:
                if att.type != AttributeProto.GRAPH:
                    node_attributes.append(att)
                    continue
                if not self._detect_op_type_replacements(att.g, replacements):
                    node_attributes.append(att)
                    continue
                modified = True
                node_attributes.append(
                    oh.make_attribute(
                        att.name,
                        self._rename_op_type_in_local_functions(att.g, replacements),
                    )
                )
            if modified:
                del node.attribute[:]
                node.attribute.extend(node_attributes)
        self._check_constants("after-renaming", new_proto)
        return new_proto

    def add_function(
        self,
        f: FunctionProto,
        rename_allowed: bool = False,
        merge_allowed: bool = False,
        builder: Optional["GraphBuilder"] = None,
    ) -> Tuple[str, str]:
        """
        Adds a new local function.

        :param f: new function to register
        :param rename_allowed: the function can be renamed if a function
            with the same name already exists,
            the proto is modified inplace
        :param merge_allowed: the function is not added if another function
            of the same name already exists and is the same
        :param builder: GraphBuilder used to build the local function,
            it contains shape information the function does not have
        :return: function name

        This function does not add the domain to the list of supported opsets.
        You should use method :meth:`make_local_function` for this.
        """
        key = f.domain, f.name
        if merge_allowed and key in self.functions:
            existing = self.functions[key]
            if same_function_proto(existing, f):
                # No need to add it again.
                if self._debug_local_function:
                    print(
                        f"[GraphBuilder-{self._hash()}.add_function]"
                        f" -- existing {f.name}[{f.domain}] "
                        f"({', '.join(f.input)}) -> {', '.join(f.output)}"
                    )
                return f.domain, f.name
        if rename_allowed and key in self.functions:
            i = 2
            new_name = f"{f.name}__v2"
            while (f.domain, new_name) in self.functions:
                i += 1
                new_name = f"{f.name}__v{i}"
            f.name = new_name
        key = f.domain, f.name
        assert key not in self.functions, (
            f"Function {key} was already added, rename_allowed={rename_allowed}, "
            f"merge_allowed={merge_allowed}, same: "
            f"{same_function_proto(self.functions[key], f)}"
            f"\n{self.pretty_text()}"
        )
        if self._debug_local_function:
            print(
                f"[GraphBuilder-{self._hash()}.add_function] ---- adding {f.name}[{f.domain}] "
                f"({', '.join(f.input)}) -> {', '.join(f.output)}"
            )
        self.functions[key] = f
        if builder:
            self.functions_builder[key] = builder
        elif f.domain not in self.opsets:
            self.opsets[f.domain] = 1
        return f.domain, f.name

    def has_local_function(self, name: str, domain: str = "", builder: bool = False) -> bool:
        """Checks if a local function exists."""
        if builder:
            return (domain, name) in self.functions_builder
        return (domain, name) in self.functions

    def get_local_function(
        self, name: str, domain: str = "", builder: bool = False
    ) -> FunctionProto:
        """Returns a local function."""
        if builder:
            return self.functions_builder[domain, name]
        return self.functions[domain, name]

    def get_local_function_outputs(self, name: str, domain: str = "") -> List[str]:
        """Returns the outputs of a local function."""
        return self.functions[domain, name].output

    def add_to_constraints(self, dim_name: str, value: Union[str, int, Set[Union[str, int]]]):
        if dim_name not in self.constraints_:
            self.constraints_[dim_name] = set()
        if isinstance(value, set):
            self.constraints_[dim_name] |= value
        else:
            self.constraints_[dim_name].add(value)

    def register_constraint_dimension(self, dim_name: str, value: Any):
        """
        Registers a constraint on a dimension.

        :param dim_name: dimension name
        :param value: value to register
        """
        if self._debug_dyn_dim and dim_name in self._debug_dyn_dim:
            print(
                f"[GraphBuilder-{self._hash()}.register_constraint_dimension] "
                f"dim_name={dim_name!r}, value={value!r}"
            )
        self.add_to_constraints(dim_name, value)

    def get_registered_constraints(self) -> Dict[str, Set[Union[str, int]]]:
        """Returns the constraints registered so far."""
        return self.constraints_

    def _to_torch_tensor(self, a: Any) -> "torch.Tensor":  # noqa: F821
        """Torch does not convert numpy dtype very well."""
        if isinstance(a, self.torch.Tensor):
            return a
        if isinstance(a, np.ndarray):
            if len(a.shape) == 0:
                # Then torch may consider this as a the creation of empty array.
                tt = self.torch.from_numpy(a.reshape((1,)).copy())
                tt = tt[0]
            else:
                tt = self.make_torch_tensor_from_np_array(a)
            ttype = onnx_dtype_to_torch_dtype(dtype_to_tensor_dtype(a.dtype))
            res = tt.to(ttype)
            assert a.shape == tuple(res.shape), (
                f"Unexpected shape {res.shape}, expecting shape={a.shape}, "
                f"dtype={res.dtype}, expected dtype={a.dtype}"
            )
            return res
        raise AssertionError(f"Unsupported type {type(a)}, unable to convert to a torch.Tensor.")

    def inline_functions(self, verbose: int = 0) -> int:
        """Inlines local functions. Returns the number of inlined nodes."""
        if not self.functions:
            # Nothing to do
            return

        begin0 = time.perf_counter()

        # Checks opsets
        skip_functions = set()
        for v in self.functions.values():
            if v.metadata_props:
                for p in v.metadata_props:
                    if p.key == "inline" and p.value in ("0", 0, "False", "false", False):
                        skip_functions.add((v.domain, v.name))

            for op in v.opset_import:
                if op.domain in self.opsets:
                    assert op.version == self.opsets[op.domain], (
                        f"Opset version mismatch for domain {op.domain!r}, "
                        f"existing version is {self.opsets[op.domain]}, "
                        f"version for function {v.name!r} is {op.version}"
                    )
                else:
                    self.opsets[op.domain] = op.version

        if verbose:
            print(f"[GraphBuilder-{self._hash()}.inline_functions] begin inlining graph")
            print(
                f"[GraphBuilder-{self._hash()}.inline_functions] skip_functions={skip_functions}"
            )
        stats = []
        self._check(stats, step="before inline")
        begin = time.perf_counter()
        inlined = self._inline_functions_iteration(verbose=verbose, skip_functions=skip_functions)
        stats.append(
            dict(
                pattern="inline",
                time_in=time.perf_counter() - begin,
                iteration=0,
                inlined=inlined,
            )
        )
        self._check(stats, step="after inline iteration 0")
        it = 0
        while inlined:
            it += 1
            begin = time.perf_counter()
            inlined = self._inline_functions_iteration(
                verbose=verbose, skip_functions=skip_functions
            )
            stats.append(
                dict(
                    pattern="inline",
                    time_in=time.perf_counter() - begin,
                    iteration=0,
                    inlined=inlined,
                )
            )
            self._check(stats, step=f"after inline iteration {it}")

        # We can remove the local functions now.
        self.functions = {k: v for k, v in self.functions.items() if k in skip_functions}
        if verbose:
            print(
                f"[GraphBuilder-{self._hash()}.inline_functions] done inlining graph "
                f"{id(self)} in {time.perf_counter()-begin0}"
            )
        return stats

    def local_functions_found(self, g: GraphProto) -> bool:
        for node in g.node:
            if node.domain == "":
                continue
            key = node.domain, node.op_type
            if key in self.functions:
                return True
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if att.type == AttributeProto.GRAPH:
                    if self.has_local_function(att.g):
                        return True
        return False

    def _inline_functions_iteration(
        self, skip_functions: Optional[Set[Tuple[str, str]]] = None, verbose: int = 0
    ) -> int:
        """
        Inlines local functions. Returns the number of replacements.

        :param skip_functions: do not inline these functions
        :param verbose: verbosity
        :return: number of replacements
        """
        skip_functions = skip_functions or set()
        n_replacements = 0
        replacements = []
        for pos, node in enumerate(self.nodes):
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if self.local_functions_found(att.g):
                    # A function was detected in a subgraphs.
                    if verbose:
                        print(
                            f"[GraphBuilder-{self._hash()}._inline_functions_iterations] "
                            f"replace local functions in node {node.op_type!r}, "
                            f"name={node.name!r}"
                        )
                    n_replacements += self._inline_functions_subgraph(
                        att.g, verbose=verbose, skip_functions=skip_functions
                    )
                elif att.ref_attr_name:
                    # we need to retrieve the value of the parameter
                    raise NotImplementedError(f"Unable to inline a function with attribute {att}")

            key = node.domain, node.op_type
            if key not in self.functions or key in skip_functions:
                continue

            n_replacements += 1
            if verbose:
                print(
                    f"[GraphBuilder-{self._hash()}._inline_functions_iterations] inline function "
                    f"{self.functions[key].name!r} domain {self.functions[key].domain!r} "
                    f"[n_replacements={n_replacements}]"
                )
            new_nodes = self._convert_function(
                node.input, node.output, self.functions[key], attributes=node.attribute
            )
            if verbose:
                print(
                    f"[GraphBuilder-{self._hash()}._inline_functions_iterations] "
                    f"done with {len(new_nodes)} new nodes "
                    f"for {self.functions[key].name!r}, {self.functions[key].domain!r}"
                )
            replacements.append((pos, node, new_nodes))

        if n_replacements == 0:
            # No replacements to do.
            return n_replacements

        stat = []
        for pos, node, new_nodes in reversed(replacements):
            self.insert_and_remove_nodes(pos, new_nodes, removed=[pos])
            self._check(stat, step=f"after inlining function {node.op_type!r}")

        return n_replacements

    def _inline_functions_subgraph(
        self,
        g: GraphProto,
        skip_functions: Optional[Set[Tuple[str, str]]] = None,
        verbose: int = 0,
    ) -> int:
        """
        Inlines local functions in subgraph (inplace).
        Returns the number of replacements.
        """
        if self.verbose >= 5:
            print(
                f"[GraphBuilder-{self._hash()}._inline_functions_subgraph] "
                f"inline subgraph {g.name!r}"
            )
        stat = []
        self._check(stat, step="before inline")
        inlined = self._inline_functions_subgraph_iteration(
            g, verbose=verbose, skip_functions=skip_functions
        )
        self._check(stat, step="after inline iteration 0")
        total = inlined
        it = 0
        while inlined:
            it += 1
            inlined = self._inline_functions_subgraph_iteration(
                g, verbose=verbose, skip_functions=skip_functions
            )
            total += inlined
            it += 1
            self._check(stat, step=f"after inline iteration {it}")
        if self.verbose >= 5:
            print(
                f"[GraphBuilder-{self._hash()}._inline_functions_subgraph] "
                f"inlined subgraph {g.name!r}, total={total}"
            )
        return total

    def _inline_functions_subgraph_iteration(
        self,
        g: GraphProto,
        skip_functions: Optional[Set[Tuple[str, str]]] = None,
        verbose: int = 0,
    ) -> int:
        skip_functions = skip_functions or set()
        new_nodes = []
        if verbose:
            print(f"[_inline_functions_subgraph_iteration] begin with {make_idg(g)}")
        n_replacements = 0
        for node in g.node:
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if self.local_functions_found(att.g):
                    # A function was detected in a subgraphs.
                    if verbose:
                        print(
                            f"[_inline_functions_subgraph_iteration] replace local "
                            f"functions in node {node.op_type!r}, name={node.name!r}"
                        )
                    n_replacements += self._inline_functions_subgraph(
                        att.g, verbose=verbose, skip_functions=skip_functions
                    )
                assert (
                    not att.ref_attr_name
                ), f"Missing information to replace {att} in node {self.pretty_node(node)}"

            key = node.domain, node.op_type
            if key not in self.functions or key in skip_functions:
                new_nodes.append(node)
                continue

            n_replacements += 1
            if verbose:
                print(
                    f"[_inline_functions_subgraph_iteration] inline function "
                    f"{self.functions[key].name!r} domain {self.functions[key].domain!r}"
                )
            functions_nodes = self._rename_results(
                self._convert_function(node.input, node.output, self.functions[key]),
                replacements={n: n for n in (set(node.input) | set(node.output))},
            )
            new_nodes.extend(functions_nodes)
            if verbose:
                print(
                    f"[_inline_functions_subgraph_iteration] {len(new_nodes)} new nodes "
                    f"for {self.functions[key].name!r}, {self.functions[key].domain!r}"
                )

        if n_replacements > 0:
            del g.node[:]
            g.node.extend(new_nodes)
        if verbose:
            print(
                f"[_inline_functions_subgraph_iteration] done with "
                f"{make_idg(g)} and {n_replacements} replacements"
            )
        return n_replacements

    def _rename_results(
        self, nodes: List[NodeProto], replacements: Dict[str, str]
    ) -> List[NodeProto]:
        new_nodes = []
        for node in nodes:
            assert all(i in replacements for i in node.input), (
                f"An input from {node.input} is unknown in node "
                f"{node.op_type!r}, name={node.name!r}"
            )
            new_outputs = []
            for o in node.output:
                if o in replacements:
                    assert (
                        replacements[o] == o
                    ), f"o={o!r} must be an output in {assert_sorted(replacements)!r}"
                    new_outputs.append(o)
                    continue
                new_name = self.unique_name(o)
                replacements[o] = new_name
                new_outputs.append(new_name)

            new_node = oh.make_node(
                node.op_type,
                [replacements[i] for i in node.input],
                new_outputs,
                name=self.unique_node_name(node.name),
                domain=node.domain,
            )
            new_atts = []
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                new_atts.append(
                    oh.make_attribute(
                        att.name,
                        self._rename_results_in_subgraph(
                            att.g, replacements=replacements.copy(), check_ref_attr=True
                        ),
                    )
                    if att.type == AttributeProto.GRAPH
                    else att
                )
            new_node.attribute.extend(new_atts)
            new_nodes.append(new_node)
        return new_nodes

    def _rename_results_in_subgraph(
        self,
        g: GraphProto,
        replacements: Dict[str, str],
        check_ref_attr: bool = False,
        attributes: Optional[Sequence[AttributeProto]] = None,
    ) -> GraphProto:
        if self.verbose >= 5:

            def _(d):
                return {k: v for k, v in d.items() if k != v}

            assert g.name
            print(f"[GraphBuilder-{self._hash()}._rename_results_in_subgraph] name={g.name!r}")
            print(
                f"[GraphBuilder-{self._hash()}._rename_results_in_subgraph] "
                f"replacements={_(replacements)}"
            )

        set_rep = set({k: v for k, v in replacements.items() if k != v})
        new_nodes = []
        do = False
        for node in g.node:
            diff = bool(set(node.input) & set_rep)
            new_inputs = [replacements.get(i, i) for i in node.input]
            new_atts = []
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if att.type == AttributeProto.GRAPH:
                    new_g = self._rename_results_in_subgraph(
                        att.g,
                        replacements=replacements.copy(),
                        check_ref_attr=check_ref_attr,
                        attributes=attributes,
                    )
                    if make_idg(new_g) != make_idg(g):
                        diff = True
                        new_atts.append(
                            oh.make_attribute(att.name, new_g)
                            if att.type == AttributeProto.GRAPH
                            else att
                        )
                    else:
                        new_atts.append(att)
                elif attributes and att.ref_attr_name:
                    find = [a for a in attributes if a.name == att.ref_attr_name]
                    assert not check_ref_attr or len(find) == 1, (
                        f"Missing replacement for {att} in {self.pretty_node(node)}, "
                        f"attributes={attributes}"
                    )
                    new_att = AttributeProto()
                    new_att.ParseFromString(find[0].SerializeToString())
                    new_att.name = att.name
                    new_atts.append(new_att)
                    diff = True
                else:
                    new_atts.append(att)

            if diff:
                do = True
                new_node = oh.make_node(
                    node.op_type, new_inputs, node.output, domain=node.domain, name=node.name
                )
                new_node.attribute.extend(new_atts)
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)

            if set(node.output) & set_rep:
                # This is shadowing. We can't continue with replacement.
                set_rep -= set(node.output)
        if not do:
            if self.verbose >= 5:
                print(
                    f"[GraphBuilder-{self._hash()}._rename_results_in_subgraph] "
                    f"no replacement for {g.name!r}"
                )
            return g
        g2 = oh.make_graph(
            new_nodes, g.name, g.input, g.output, g.initializer, g.sparse_initializer
        )
        if self.verbose >= 5:
            print(f"[GraphBuilder-{self._hash()}._rename_results_in_subgraph] done {g2.name!r}")
        return g2

    def _convert_function(
        self,
        inputs: List[str],
        outputs: List[str],
        proto: FunctionProto,
        attributes: Optional[Sequence[AttributeProto]] = None,
    ) -> List[NodeProto]:
        """
        Converts a function into a list of nodes.

        :param inputs: inputs in the calling nodes
        :param outputs: outputs in the calling nodes
        :param proto: function proto
        :param attributes: attribute of the node, needed if the function
            has arguments
        :return: list of nodes
        """
        mapatt = {}
        if attributes:
            for att in attributes:
                mapatt[att.name] = att
        renamed = dict(zip(proto.input, inputs))
        renamed.update(dict(zip(proto.output, outputs)))
        if self.verbose >= 5:

            def _(d):
                return {k: v for k, v in d.items() if k != v}

            print(f"[GraphBuilder-{self._hash()}._convert_function] {proto.domain}.{proto.name}")
            print(f"[GraphBuilder-{self._hash()}._convert_function] renamed={_(renamed)}")
        new_nodes = []
        for node in proto.node:
            new_inputs = []
            for name in node.input:
                if not name:
                    new_inputs.append("")
                    continue
                assert name in renamed, f"Unable to find {name!r} in renamed={renamed!r}"
                new_inputs.append(renamed[name])
            new_outputs = []
            for name in node.output:
                if not name:
                    new_outputs.append("")
                    continue
                if name in renamed:
                    new_outputs.append(renamed[name])
                else:
                    new_name = self.unique_name(name)
                    renamed[name] = new_name
                    new_outputs.append(new_name)

            new_node = oh.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                name=self.unique_node_name(node.name),
                domain=node.domain,
            )
            new_attributes = []
            for att in node.attribute:
                if att.ref_attr_name:
                    assert att.ref_attr_name in mapatt, (
                        f"The node relies on one attribute {att.name!r} with a reference "
                        f"to a function attribute but none is provided in the available "
                        f"set {sorted(mapatt)}."
                    )
                    known_att = mapatt[att.ref_attr_name]
                    assert known_att.type == att.type, (
                        f"Type mimatch netween the attribute {att.name!r} "
                        f"type {att.type} of the node "
                        f"inside the function and the one "
                        f"{known_att.type} outside the function (name {known_att.name!r})"
                    )
                    new_att = AttributeProto()
                    new_att.ParseFromString(known_att.SerializeToString())
                    new_att.name = att.name
                    new_attributes.append(new_att)
                elif att.type == AttributeProto.GRAPH:
                    new_g = self._rename_results_in_subgraph(
                        att.g,
                        replacements=renamed.copy(),
                        check_ref_attr=True,
                        attributes=attributes,
                    )
                    if make_idg(new_g) == make_idg(att.g):
                        new_attributes.append(att)
                    else:
                        new_attributes.append(oh.make_attribute(att.name, new_g))
                else:
                    new_attributes.append(att)
            new_node.attribute.extend(new_attributes)
            new_nodes.append(new_node)

        if self.verbose >= 5:
            print(
                f"[GraphBuilder-{self._hash()}._convert_function] done "
                f"{proto.domain}.{proto.name}"
            )
        return new_nodes

    def move_initializers_to_constant(
        self, full_parameter_name, threshold: Optional[int] = None, verbose: int = 0
    ) -> int:
        """
        Moves initializers as constant nodes.

        :param full_parameter_name: keeps the local name or the full name for the parameters
        :param threshold: only move intializers to constant if their size is below this limit
        :param verbose: verbosity
        :return: number of moved initializers
        """
        self._do_not_turn_constant_initializers = True
        if not self.initializers_dict:
            return

        initializers, _ = self._build_initializers(
            switch_low_high=sys.byteorder != "big",
            large_model=True,
            external_threshold=threshold or 2**30,
            full_parameter_name=full_parameter_name,
        )

        to_remove = []
        cst_nodes = []
        for proto in initializers:
            if proto.external_data:
                # external tensor
                continue
            to_remove.append(proto.name)
            if self.verbose:
                print(
                    f"[move_initializers_to_constant] convert "
                    f"{proto.name!r} into a node 'Constant'"
                )
            cst = oh.make_node(
                "Constant",
                [],
                [proto.name],
                value=proto,
                name=self.unique_node_name("init2cst"),
                doc_string=f"move_initializers_to_constant/{proto.doc_string}",
            )
            cst_nodes.append(cst)
            self.add_constant_node(cst)
            self.update_node_constant(proto.name, cst)
            assert self.has_type(proto.name), f"Type is missing for initializer {proto.name!r}"
            assert self.has_shape(proto.name), f"Shape is missing for initializer {proto.name!r}"

        if to_remove:
            remove = set(to_remove)
            self.initializers_dict = {
                k: v for k, v in self.initializers_dict.items() if k not in remove
            }
            self.nodes = [*cst_nodes, *self.nodes]

    def get_input_dynamic_shape(
        self,
        name: str,
        input_index: int,
        example_shape: STATIC_SHAPE,
        dynamic_shapes: Optional[Any] = None,
        example_value: Optional[Any] = None,
    ) -> DYNAMIC_SHAPE:
        """
        Updates the shape based on the available information.

        :param name: input name
        :param input_index: input index
        :param example_shape: the shape of the given input
        :param dynamic_shapes: used to handle nested dynamic shapes
        :param example_value: one example of the value
        :return: dynamic shape
        """
        if dynamic_shapes is None:
            dynamic_shapes = self.dynamic_shapes
        if dynamic_shapes is None and example_shape is not None:
            if is_static_shape(example_shape):
                return tuple(example_shape)
            # Should we convert SymInt to str.
            return tuple(example_shape)

        if isinstance(example_value, list):
            if dynamic_shapes is None:
                # No info, we use the example values
                return [tuple(map(self._torch_sym_int_to_str, example_value[0].shape))]
            elif isinstance(dynamic_shapes, tuple):
                info = dynamic_shapes[input_index]
            elif isinstance(dynamic_shapes, dict):
                info = dynamic_shapes.get(name, None)
            else:
                raise NotImplementedError(
                    f"Unexpected type for dynamic_shapes={dynamic_shapes}, "
                    f"example_value={string_type(example_value)}"
                    f"{self.get_debug_msg()}"
                )
            return [
                self.get_input_dynamic_shape(
                    None, 0, example_value[0].shape, info if info is None else tuple(info)
                )
            ]

        if example_shape is None and example_shape is None:
            if self.input_args is not None and input_index < len(self.input_args):
                v = self.input_args[input_index]
                if isinstance(v, VirtualTensor):
                    example_shape = v.shape
                else:
                    raise NotImplementedError(
                        f"example_shape is None, example_value as well, "
                        f"type input_arg={type(self.input_args[input_index])}, "
                        f"dynamic_shapes={dynamic_shapes}, input_index={input_index}, "
                        f"self.input_args={self.input_args}, "
                        f"as_function={self.as_function}{self.get_debug_msg()}"
                    )
                assert example_shape is not None, (
                    f"example_shape is None, example_value is None, the input "
                    f"VirtualTensor has no shape as well, "
                    f"type input_arg={type(self.input_args[input_index])}, "
                    f"dynamic_shapes={dynamic_shapes}, input_index={input_index}, "
                    f"self.input_args={self.input_args}, "
                    f"as_function={self.as_function}{self.get_debug_msg()}"
                )
            elif self.as_function:
                # We don't need the shape for function.
                return None
            else:
                raise NotImplementedError(
                    f"example_shape is None, example_value as well, there is no input_args, "
                    f"input_index={input_index}, "
                    f"input_args={string_type(self.input_args, with_shape=True)}, "
                    f"dynamic_shapes={dynamic_shapes}, input_index={input_index}, "
                    f"self.input_args={self.input_args}, "
                    f"as_function={self.as_function}{self.get_debug_msg()}"
                )
        assert example_value is None and example_shape is not None, (
            f"At this stage, a tensor is expected but example_value="
            f"{string_type(example_value)}, example_shape={example_shape}, "
            f"dynamic_shapes={dynamic_shapes}, input_index={input_index}, "
            f"self.input_args={self.input_args}, "
            f"as_function={self.as_function}{self.get_debug_msg()}"
        )
        if isinstance(dynamic_shapes, tuple):
            info = dynamic_shapes[input_index] if input_index < len(dynamic_shapes) else None
        elif isinstance(dynamic_shapes, dict):
            info = dynamic_shapes.get(name, None)
        elif dynamic_shapes is None:
            info = None
        else:
            raise NotImplementedError(
                f"Unexpected type for dynamic_shapes={string_type(dynamic_shapes)}, "
                f"self.dynamic_shapes={string_type(self.dynamic_shapes)}, "
                f"example_value={string_type(example_value)}, name={name!r}, "
                f"example_shape={example_shape}{self.get_debug_msg()}"
            )

        # We could return example_shape.shape (s0, ...) when info is (batch, ...)
        # In case example_shape is missing, then dynamic_shape should prevail.
        if example_shape is not None:
            if info is None:
                return tuple(example_shape)

            # In that case, we need to make sure that dynamic dimmensions
            # appears at the same position.
            ret_shape = list(example_shape)
            if isinstance(info, dict):
                for k, v in info.items():
                    assert k < len(ret_shape), (
                        f"name={name!r}, input_index={input_index}, info={info}, "
                        f"dimension {k} does not extist in shape {ret_shape!r}"
                        f"{self.get_debug_msg()}"
                    )

                    if isinstance(ret_shape[k], self.torch.SymInt):
                        # We let it, set_shape will replace it
                        # by the dynamic dimension name and register an alias.
                        continue
                    assert isinstance(ret_shape[k], int), (
                        f"Incompatible types between example_shape={example_shape}, k={k!r}, "
                        f"{string_type(example_shape)}, info={info}, "
                        f"name={name!r}, input_index={input_index!r}, dynamic_shapes="
                        f"{dynamic_shapes}, example_value={string_type(example_value)}"
                        f"{self.get_debug_msg()}"
                    )
                    # example_shape[k] is int but dynamic_shape says otherwise,
                    # we trust dynamic shape
                    ret_shape[k] = v.name if isinstance(v, self.WrapDim) else v.__name__
            else:
                for i, v in enumerate(info):
                    if i >= len(ret_shape):
                        # torch.export.export flattens everything
                        continue
                    if isinstance(ret_shape[i], self.torch.SymInt):
                        # We let it, set_shape will replace it
                        # by the dynamic dimension name and register an alias.
                        continue
                    if v and not isinstance(v, (dict, tuple, list)) and hasattr(v, "__name__"):
                        # it should be (self.torch.export.Dim,
                        #               self.torch.export.dynamic_shapes._DerivedDim)):
                        assert isinstance(ret_shape[i], int), (
                            f"Incompatible types between example_shape={example_shape}, "
                            f"i={i!r}, {string_type(example_shape)}, info={info}, "
                            f"name={name!r}, input_index={input_index!r}, dynamic_shapes="
                            f"{dynamic_shapes}, example_value={string_type(example_value)}"
                            f"{self.get_debug_msg()}"
                        )
                        # example_shape[i] is int but dynamic_shape says otherwise,
                        # we truct dynamic shape
                        ret_shape[i] = v.__name__
                    # v should be None or a dictionary but the signature forward(*args)
                    # is confusing sometimes.
            return tuple(ret_shape)

        raise NotImplementedError(
            f"unable to return shape, example_shape={example_shape}, info={info}, "
            f"name={name!r}, input_index={input_index!r}, dynamic_shapes="
            f"{dynamic_shapes}, example_value={string_type(example_value)}"
            f"{self.get_debug_msg()}"
        )

    def make_new_dynamic_shape(
        self, rank: int, prefix: str = "d"
    ) -> Tuple["torch.SymInt", ...]:  # noqa: F821
        """
        Creates a dynamic shape of a known rank with new dynamic dimension.
        """

        def _new_name(d):
            if d not in self.dynamic_objects:
                return d
            i = 2
            n = f"{d}_{i}"
            while n in self.dynamic_objects:
                i += 1
                n = f"{n}_{i}"
            return n

        return tuple(self.torch.SymInt(_new_name(f"{prefix}_d{i}")) for i in range(rank))

    def make_torch_tensor_from_np_array(
        self, np_array: np.ndarray
    ) -> "torch.Tensor":  # noqa: F821
        """Converts a numpy array into a :class:`torch.Tensor`."""
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None
        if ml_dtypes is not None:
            dt = np_array.dtype
            if dt == ml_dtypes.bfloat16:
                return self.torch.tensor(np_array.astype(np.float32)).to(self.torch.bfloat16)
        return self.torch.tensor(np_array)

    def rename_names(self, name_to_rename: Dict[str, str], with_existing: bool = False):
        """Renames some names in the graph and its subgraphs."""
        set_inputs = set(_.name for _ in self.inputs)
        set_outputs = set(_.name for _ in self.outputs)
        for k, v in name_to_rename.items():
            assert self.has_name(k), f"Name {k!r} not found{self.get_debug_msg()}"
            assert with_existing or not self.has_name(
                v
            ), f"Name {v!r} already taken{self.get_debug_msg()}"
            assert not self.is_sequence(
                k
            ), f"Renaming not implemented for sequence {k!r}{self.get_debug_msg()}"
            assert (
                "NONE" in k or k not in set_outputs
            ), f"Renaming not implemented for output {k!r} yet{self.get_debug_msg()}"
            if not with_existing:
                if self.has_type(k):
                    self.set_type(v, self.get_type(k))
                if self.has_device(k):
                    self.set_device(v, self.get_device(k))
                if self.has_shape(k):
                    self.set_shape(v, self.get_shape(k))
                elif self.has_rank(k):
                    self.set_shape(v, self.get_shape(k))

                if k in set_inputs:
                    for inp in self.inputs:
                        if inp.name == k:
                            inp.name = v
                            self.set_name(v, marker="rename_names_input")
                            break
                    continue
                elif self.has_name(k):
                    self.set_name(v, marker="GraphBuilder.rename_names")

                if k in self.initializers_dict:
                    self.initializers_dict[v] = self.initializers_dict[k]
                    del self.initializers_dict[k]
                if k in self.initializers_dict_sources:
                    self.initializers_dict_sources[v] = self.initializers_dict_sources[k]
                    del self.initializers_dict_sources[k]
                if k in self.constants_computed_:
                    self.constants_computed_[v] = self.constants_computed_[k]
                    del self.constants_computed_[k]
                if k in self.constants_:
                    self.constants_[v] = self.constants_[k]
                    del self.constants_[k]
                if k in self.constants_node_:
                    self.constants_node_[v] = self.constants_node_[k]
                    del self.constants_node_[k]
                if k in self._known_torch_value:
                    self._known_torch_value[v] = self._known_torch_value[k]
                    del self._known_torch_value[k]
                if k in self._known_sequences:
                    self._known_sequences[v] = self._known_sequences[k]
                    del self._known_sequences[k]
                if k in self._known_value_shape:
                    self._known_value_shape[v] = self._known_value_shape[k]
                    del self._known_value_shape[k]
                if k in self._registered_users:
                    self._registered_users[v] = self._registered_users[k]
                    del self._registered_users[k]
                if k in self._parameter_renaming:
                    self._parameter_renaming[v] = self._parameter_renaming[k]
                    del self._parameter_renaming[k]
                if k in self._parameter_norename:
                    self._parameter_norename.add(v)

        to_rename = set(name_to_rename)

        def _rename_in_nodes(nodes, to_rename):
            for node in nodes:
                if set(node.input) & to_rename:
                    new_input = [name_to_rename.get(i, i) for i in node.input]
                    del node.input[:]
                    node.input.extend(new_input)
                for att in node.attribute:
                    if att.type != AttributeProto.GRAPH:
                        continue
                    _rename_in_nodes(att.g.node, to_rename)
                for o in node.output:
                    if o in to_rename:
                        to_rename -= {o}

        _rename_in_nodes(self.nodes, to_rename)

    def shadowing_names(
        self,
        nodes: Optional[Sequence[NodeProto]] = None,
        existing: Optional[Set[str]] = None,
        shadow_context: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Returns the names being shadowed in one subgraphs."""
        if nodes is None:
            assert (
                existing is None and shadow_context is None
            ), "existing must be None if nodes is None"
            return self.shadowing_names(
                self.nodes,
                set(self.initializers_dict) | set(i.name for i in self.inputs) | self._context,
                set(),
            )

        assert (
            existing is not None and shadow_context is not None
        ), "existing must not be None if nodes is not None"
        shadow = set()
        shadow_context = shadow_context.copy()
        existing = existing.copy()
        for node in nodes:
            not_empty = set(n for n in node.input if n)
            intersection = not_empty & existing
            assert len(intersection) == len(not_empty), (
                f"One input in {not_empty}, node={self.pretty_node(node)} "
                f"was not found in {existing}\n--\n{self.get_debug_msg()}"
            )
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    g = att.g
                    shadow |= set(i.name for i in g.input) & shadow_context
                    shadow |= set(i.name for i in g.initializer) & shadow_context
                    shadow |= set(i.name for i in g.sparse_initializer) & shadow_context
                    shadow |= self.shadowing_names(g.node, existing, existing)

            not_empty = set(n for n in node.output if n)
            shadow |= not_empty & shadow_context
            existing |= not_empty
        return shadow

    def extract_input_names_from_args(self, args: Any) -> List[str]:
        input_names = []
        for a in args:
            if isinstance(a, str) and self.has_name(a):
                if a not in input_names:
                    input_names.append(a)
            elif isinstance(a, (list, tuple)):
                input_names.extend(self.extract_input_names_from_args(a))
            elif isinstance(a, slice):
                input_names.extend(self.extract_input_names_from_args([a.start, a.stop, a.step]))
        dedup_names = []
        unique = set()
        for i in input_names:
            if i not in unique:
                dedup_names.append(i)
                unique.add(i)
        return dedup_names

    def evaluate_dimension_equality_with_constraints(
        self, dimension: Union[str, int], *args
    ) -> bool:
        """
        Checks expressions with dimensions.

        First supported expression:

        * ``f("dim", "d1", "+", "d2")  ->  dimension == args[0] + args[2]``
        """
        if len(args) == 3:
            if args[1] == "+":
                d1, d2 = args[0], args[2]
                if isinstance(dimension, int) and isinstance(d1, int) and isinstance(d2, int):
                    return dimension == d1 + d2
                if isinstance(dimension, str):
                    exp = f"{d1}+{d2}"
                    if dimension == exp or (
                        dimension in self.constraints_ and exp in self.constraints_[dimension]
                    ):
                        return True
        if not simplify_two_expressions(dimension, f"({args[0]}){args[1]}({args[2]})"):
            return True
        raise NotImplementedError(
            f"Unable to evaluate expression with dimension={dimension!r}, "
            f"args={args!r}, constraints={self.constraints_}{self.get_debug_msg()}"
        )
