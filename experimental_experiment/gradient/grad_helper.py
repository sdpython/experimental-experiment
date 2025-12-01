from collections import OrderedDict
from io import BytesIO
from enum import IntFlag
from typing import Dict, List, Optional
import numpy as np
import onnx
from onnx import ModelProto, TensorProto
from onnx.helper import make_model, make_graph, make_node, make_tensor
from onnx_array_api.graph_api import GraphBuilder
from .loss_helper import get_train_initializer


class DerivativeOptions(IntFlag):
    """
    Options defining how to build the onnx graph of the
    gradients.

    * `Zero`: default option, all options are disabled
    * `KeepYieldOp`: keeps the operator *YieldOp* in the graph,
      see @see fn onnx_derivative
    * `KeepOutputs`: keeps the output of the original graph
    * `FillGrad`: does not add any output to specify the gradient
      of the output but assumes it is one
    * `Loss`: the function assumes the loss was added to the graph
    """

    Zero = 0
    KeepYieldOp = 1
    KeepOutputs = 2
    FillGrad = 4
    Loss = 5


def _random_input(typ, shape, batch):
    if typ in ("tensor(double)", TensorProto.DOUBLE):
        dtype = np.float64
    elif typ in ("tensor(float)", TensorProto.FLOAT):
        dtype = np.float32
    else:
        raise NotImplementedError(f"Unable to guess dtype from {typ!r}.")

    if len(shape) <= 1:
        new_shape = shape
    elif shape[0] in (None, 0):
        new_shape = (batch, *shape[1:])
    else:
        new_shape = shape
    return np.random.randn(*new_shape).astype(dtype)


def random_feed(inputs, batch: int = 10, empty_dimension: int = 1) -> Dict[str, np.ndarray]:
    """
    Creates a dictionary of random inputs.

    :param batch: dimension to use as batch dimension if unknown
    :param empty_dimension: if a dimension is null, replaces it by this value
    :return: dictionary
    """
    res = OrderedDict()
    for inp in inputs:
        name = inp.name
        if hasattr(inp.type, "tensor_type"):
            typ = inp.type.tensor_type.elem_type
            shape = tuple(getattr(d, "dim_value", batch) for d in inp.type.tensor_type.shape.dim)
            shape = (shape[0], *[b if b > 0 else empty_dimension for b in shape[1:]])
        else:
            typ = inp.type
            shape = inp.shape
        res[name] = _random_input(typ, shape, batch)
    return res


def onnx_derivative(
    onx: ModelProto,
    weights: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
    options: DerivativeOptions = DerivativeOptions.Zero,
    loss: Optional[str] = None,
    label: Optional[str] = None,
    path_name: Optional[str] = None,
    verbose: int = 0,
) -> ModelProto:
    """
    Builds the gradient for an onnx graph.

    :param onx: onnx graph
    :param weights: gradient against those weights, None for all real weights
    :param inputs: gradient against inputs, None for all real inputs
    :param options: options of type @see cl DerivativeOptions
    :param loss: loss output in case a loss was added in the graph,
        *options* must be equal to `DerivativeOptions.Loss`
    :param label: if *loss* is specified, then the label must be
        specified as well
    :param path_name: if *options* equal to `DerivativeOptions.Loss`,
        the gradient is saved to that path
    :param verbose: verbosity
    :return: onnx graph

    The function calls ``OrtModuleGraphBuilderConfiguration``
    from :epkg:`onnxruntime-training`. This graph is meant to be used
    with `OrtGradientForwardBackward` and includes
    operator `YieldOp`. That's the graph looks this way:

    .. code-block:: python

        import numpy as np
        from onnx.defs import onnx_opset_version
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxAdd
        from experimental_experiment.doc import to_dot
        from experimental_experiment.gradient.grad_helper import (
            onnx_derivative,
            DerivativeOptions,
        )

        opv = onnx_opset_version() - 2

        node = OnnxAdd(
            "X",
            np.array([1], dtype=np.float32),
            op_version=opv,
            output_names=["Y"]
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        try:
            new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepYieldOp)
        except ImportError as e:
            print("onnxruntime-training is not installed", e)
            new_onx = None
        if new_onx:
            print("DOT-SECTION", to_dot(new_onx))

    These operators are the outputs of the
    initial graph and must be replaced by the gradient of these
    outputs to compute the gradient of the weights and the inputs.
    After they are replaced, it looks this way:

    .. code-block:: python

        import numpy as np
        from onnx.defs import onnx_opset_version
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxAdd
        from experimental_experiment.doc import to_dot
        from experimental_experiment.gradient.grad_helper import (
            onnx_derivative,
            DerivativeOptions,
        )

        opv = onnx_opset_version() - 2

        node = OnnxAdd(
            "X",
            np.array([1], dtype=np.float32),
            op_version=opv,
            output_names=["Y"]
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        try:
            new_onx = onnx_derivative(onx, options=DerivativeOptions.Zero)
        except ImportError as e:
            print("onnxruntime-training is not installed", e)
            new_onx = None
        if new_onx:
            print("DOT-SECTION", to_dot(new_onx))

    The user can still compute the outputs.

    .. code-block:: python

        import numpy as np
        from onnx.defs import onnx_opset_version
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxAdd
        from experimental_experiment.doc import to_dot
        from experimental_experiment.gradient.grad_helper import (
            onnx_derivative,
            DerivativeOptions,
        )

        opv = onnx_opset_version() - 2

        node = OnnxAdd(
            "X",
            np.array([1], dtype=np.float32),
            op_version=opv,
            output_names=["Y"]
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        try:
            new_onx = onnx_derivative(onx, options=DerivativeOptions.KeepOutputs)
        except ImportError as e:
            print("onnxruntime-training is not installed", e)
            new_onx = None
        if new_onx:
            print("DOT-SECTION", to_dot(new_onx))

    The input gradient can be filled with a constant matrix
    filled with one and with the expected shape.

    .. code-block:: python

        import numpy as np
        from onnx.defs import onnx_opset_version
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxAdd
        from experimental_experiment.doc import to_dot
        from experimental_experiment.gradient.grad_helper import (
            onnx_derivative,
            DerivativeOptions,
        )

        opv = onnx_opset_version() - 2

        node = OnnxAdd(
            "X",
            np.array([1], dtype=np.float32),
            op_version=opv,
            output_names=["Y"]
        )
        onx = node.to_onnx(
            {"X": FloatTensorType([None, 10])},
            {"Y": FloatTensorType([None, 10])},
            target_opset=opv,
        )
        try:
            new_onx = onnx_derivative(
                onx, options=DerivativeOptions.KeepOutputs | DerivativeOptions.FillGrad
            )
        except ImportError as e:
            print("onnxruntime-training is not installed", e)
            new_onx = None
        if new_onx:
            print("DOT-SECTION", to_dot(new_onx))
    """
    assert isinstance(
        options, DerivativeOptions
    ), f"Options must be from type DerivativeOptions not {type(options)!r}."

    if options == DerivativeOptions.Loss:
        return _onnx_derivative_loss(
            onx,
            weights=weights,
            inputs=inputs,
            options=options,
            loss=loss,
            label=label,
            path_name=path_name,
            verbose=verbose,
        )
    return _onnx_derivative_fw(
        onx, weights=weights, inputs=inputs, options=options, verbose=verbose
    )


def _default_inputs(onx: ModelProto) -> List[str]:
    "Guesses default inputs (float ones) if not specified."
    inputs_name = []
    for i in onx.graph.input:
        try:
            elem_type = i.type.tensor_type.elem_type
        except AttributeError:
            # not a vector
            continue
        if elem_type in (
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ):
            inputs_name.append(i.name)
    return inputs_name


def _onnx_derivative_fw(
    onx: ModelProto,
    weights: List[str],
    inputs: List[str],
    options: DerivativeOptions,
    verbose: int = 0,
) -> ModelProto:
    """Implements a gradient based on class `OrtModuleGraphBuilder`."""
    from onnxruntime.capi._pybind_state import (
        OrtModuleGraphBuilder,
        OrtModuleGraphBuilderConfiguration,
        TrainingGraphTransformerConfiguration,
        Severity,
    )

    if verbose > 0:
        print(f"[_onnx_derivative_fw] weights={weights} inputs={inputs} options={options}")
    if weights is None:
        inits = get_train_initializer(onx)
        weights = list(inits)
        if verbose > 0:
            print(f"[_onnx_derivative_fw] guessed weights={weights}")
    if verbose > 0:
        print("[_onnx_derivative_fw] OrtModuleGraphBuilder")
    builder = OrtModuleGraphBuilder()
    config = OrtModuleGraphBuilderConfiguration()
    config.initializer_names = weights
    config.initializer_names_to_train = weights
    if verbose > 0:
        config.loglevel = Severity.INFO
    if inputs is None:
        inputs_name = _default_inputs(onx)
        if len(inputs_name) > 0:
            config.input_names_require_grad = inputs_name
    config.build_gradient_graph = True

    if verbose > 0:
        print(
            f"[_onnx_derivative_fw] TrainingGraphTransformerConfiguration "
            f"with inputs_name={inputs_name}"
        )
    p = TrainingGraphTransformerConfiguration()
    if verbose > 0:
        print("[_onnx_derivative_fw] builder initialize")
    builder.initialize(onx.SerializeToString(), config)
    if verbose > 0:
        print("[_onnx_derivative_fw] build")
    builder.build(p)
    try:
        train_onnx_model_serialized = builder.get_gradient_model()
    except AttributeError:
        train_onnx_model_serialized = builder.get_model()

    # optimized_pre_grad_model = builder.get_inference_optimized_model()
    grad_yield = onnx.load(BytesIO(train_onnx_model_serialized))
    if options & DerivativeOptions.KeepYieldOp:
        assert (
            options == DerivativeOptions.KeepYieldOp
        ), "Option YieldOd cannot be combined with any other."
        return grad_yield

    yields_op = [
        (index, node)
        for index, node in enumerate(grad_yield.graph.node)
        if node.op_type == "YieldOp"
    ]
    assert len(yields_op) > 0, "No YieldOp was found. The input graph must be wrong."

    other_nodes = [
        (index, node)
        for index, node in enumerate(grad_yield.graph.node)
        if node.op_type != "YieldOp"
    ]

    inputs = list(grad_yield.graph.input)
    if options & DerivativeOptions.KeepOutputs:
        outputs = list(grad_yield.graph.output)
    else:
        original = set(i.name for i in onx.graph.output)
        outputs = [o for o in grad_yield.graph.output if o.name not in original]

    map_out = {o.name: o for o in onx.graph.output}
    set_out = set(map_out)
    for index, yn in yields_op:
        assert len(yn.input) == len(yn.output), (
            f"YieldOp should have the same number of inputs and outputs "
            f"but index={index} and yield op is\n{yn}"
        )
        assert len(set(yn.input) & set_out) == len(
            yn.input
        ), f"Unable to find one output {yn.input!r} in {list(map_out)!r}."
        if not (options & DerivativeOptions.FillGrad):
            for i, inp in enumerate(yn.input):
                out = map_out[inp]
                new_input = onnx.ValueInfoProto()
                new_input.name = yn.output[i]
                new_input.doc_string = "from yieldop"
                new_input.type.CopyFrom(out.type)
                inputs.append(new_input)
        else:
            assert (
                options & DerivativeOptions.KeepOutputs
            ), "FillGrad should be set with KeepOutputs."
            for i, inp in enumerate(yn.input):
                name = f"{inp}_shape"
                node = make_node("Shape", [inp], [name])
                other_nodes.append((index + 0.1, node))
                out = map_out[inp]
                elem_type = out.type.tensor_type.elem_type
                node = make_node(
                    "ConstantOfShape",
                    [name],
                    [yn.output[i]],
                    value=make_tensor("value", elem_type, (1,), [1]),
                )
                other_nodes.append((index + 0.2, node))
        if options & DerivativeOptions.KeepOutputs:
            # Keeps output from the original graph.
            outputs.append(out)

    # Final graph.
    if verbose > 0:
        print("[_onnx_derivative_fw] final graph")
    other_nodes.sort()
    other_nodes = [o[1] for o in other_nodes]
    graph = make_graph(
        other_nodes,
        grad_yield.graph.name,
        inputs,
        outputs,
        list(grad_yield.graph.initializer),
    )
    new_model = make_model(graph)
    new_model.ir_version = grad_yield.ir_version
    new_model.producer_name = grad_yield.producer_name
    new_model.producer_version = grad_yield.producer_version
    new_model.domain = grad_yield.domain
    new_model.model_version = grad_yield.model_version
    new_model.doc_string = grad_yield.doc_string
    new_model.ir_version = onx.ir_version
    if hasattr(onx, "value_info"):
        graph.value_info.extend(grad_yield.value_info)
    del new_model.opset_import[:]
    for oimp in grad_yield.opset_import:
        op_set = new_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if verbose > 0:
        print("[_onnx_derivative_fw] optimize")
    g = GraphBuilder(new_model)
    g.optimize()
    onx_grad = g.to_onnx()
    onx_grad.ir_version = new_model.ir_version
    if verbose > 0:
        print("[_onnx_derivative_fw] done")
    return onx_grad


def _onnx_derivative_loss(
    onx: ModelProto,
    weights: List[str],
    inputs: List[str],
    options: DerivativeOptions,
    loss: str,
    label: str,
    path_name: str,
    verbose: int = 0,
) -> ModelProto:
    """
    Implements a gradient based on class `PyGradientGraphBuilder`.
    """
    from onnxruntime.capi._pybind_state import GradientGraphBuilder

    assert path_name is not None, "path_name must not be None if options is 'Loss'."
    assert weights is None, "weights must be None if options is 'Loss'."
    assert label is not None, "label must not be None if options is 'Loss'."
    assert loss is not None and isinstance(
        loss, str
    ), "loss must not None and a string if options is 'Loss'."
    if isinstance(label, str):
        label = {label}
    else:
        label = set(label)
    if inputs is None:
        inputs_name = _default_inputs(onx)
        inputs = inputs_name
    if isinstance(inputs, str):
        inputs = {inputs}
    else:
        inputs = set(inputs)
    inputs = set(x for x in inputs if x not in label)

    str_onx = onx.SerializeToString()
    if verbose > 0:
        print(f"[_onnx_derivative_loss] label={label!r}, inputs={inputs}, loss={loss}")
    builder = GradientGraphBuilder(str_onx, label, inputs, loss)
    if verbose > 0:
        print(f"[_onnx_derivative_loss] build, onx size={len(str_onx)}")
    builder.build()
    if verbose > 0:
        print(f"[_onnx_derivative_loss] save to {path_name!r}")
    builder.save(path_name)
    if verbose > 0:
        print(f"[_onnx_derivative_loss] load {path_name!r}")
    with open(path_name, "rb") as f:
        grad_onx = onnx.load(f)
        grad_onx.ir_version = onx.ir_version
    if verbose > 0:
        print("[_onnx_derivative_loss] done")
    return grad_onx
