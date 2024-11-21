import ctypes
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import GraphProto, ModelProto, TensorProto
import onnx.helper as oh
import onnx.numpy_helper as onh

STORAGE_TYPE = {
    TensorProto.FLOAT16: np.int16,
    TensorProto.BFLOAT16: np.int16,
}


def _get_type(elem_type: Any, exc: bool = True) -> int:
    if not isinstance(elem_type, int):
        st = str(elem_type)
        if "float32" in st:
            elem_type = TensorProto.FLOAT
        elif "float64" in st:
            elem_type = TensorProto.DOUBLE
        elif "bfloat16" in st:
            elem_type = TensorProto.BFLOAT16
        elif "float16" in st:
            elem_type = TensorProto.FLOAT16
        elif "uint64" in st:
            elem_type = TensorProto.UINT64
        elif "int64" in st:
            elem_type = TensorProto.INT64
        elif "uint32" in st:
            elem_type = TensorProto.UINT32
        elif "int32" in st:
            elem_type = TensorProto.INT32
        elif "uint16" in st:
            elem_type = TensorProto.UINT16
        elif "int16" in st:
            elem_type = TensorProto.INT16
        elif "bool" in st:
            elem_type = TensorProto.BOOL
        elif "uint8" in st:
            elem_type = TensorProto.UINT8
        elif "int8" in st:
            elem_type = TensorProto.INT8
        elif "complex64" in st:
            elem_type = TensorProto.COMPLEX64
        elif "complex128" in st:
            elem_type = TensorProto.COMPLEX128
        elif elem_type is None:
            elem_type = TensorProto.UNDEFINED
        elif exc:
            raise ValueError(f"Unable to interpret elem_type {elem_type!r}.")
    return elem_type


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    """
    Converts a torch dtype into a onnx element type.

    :param to: torch dtype
    :return: onnx type
    """
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.float16:
        return TensorProto.FLOAT16
    if to == torch.bfloat16:
        return TensorProto.BFLOAT16
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.int32:
        return TensorProto.INT32
    if to == torch.bool:
        return TensorProto.BOOL
    if to == torch.SymInt:
        return TensorProto.INT64
    if to == torch.SymFloat:
        return TensorProto.FLOAT
    if to == torch.complex64:
        return TensorProto.COMPLEX64
    if to == torch.complex128:
        return TensorProto.COMPLEX128
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


def dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    """
    Converts a torch dtype or numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return oh.np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError):
        pass
    return torch_dtype_to_onnx_dtype(dt)


def proto_from_array(
    arr: "torch.Tensor",  # noqa: F821
    name: Optional[str] = None,
    verbose: int = 0,
) -> TensorProto:
    """
    Converts a torch Tensor into a TensorProto.

    :param arr: tensor
    :param verbose: display the type and shape
    :return: a TensorProto
    """
    import sys
    import torch

    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Unexpected type {type(arr)}.")
    if arr.is_sparse:
        raise NotImplementedError(
            f"Sparse tensor is not supported yet but initializer {name!r} is."
        )

    # arr.contiguous() is slow after a transpose, maybe there is a way to optimize this.
    if arr.is_contiguous():
        arr_cpu = arr.cpu()
    else:
        arr_cpu = arr.contiguous().cpu()

    numel = torch.numel(arr_cpu)
    element_size = arr_cpu.element_size()

    if arr_cpu.dtype in {torch.bfloat16}:
        np_arr = arr_cpu
    elif arr_cpu.data_ptr() == arr.data_ptr():
        copy = arr_cpu.clone().detach().requires_grad_(False)
        assert arr_cpu.data_ptr() != copy.data_ptr()
        np_arr = np.from_dlpack(copy)
    else:
        np_arr = np.from_dlpack(arr_cpu.detach())

    tensor = TensorProto()
    tensor.dims.extend(arr_cpu.shape)
    tensor.name = name
    itype = _get_type(arr_cpu.dtype)
    assert not hasattr(TensorProto, "INT4") or itype not in {
        TensorProto.INT4,
        TensorProto.UINT4,
    }, f"Type {arr.dtype} is not supported yet for name={name!r}"
    tensor.data_type = itype

    if verbose > 1 and numel > 100:
        print(f"[proto_from_array] {tensor.data_type}[{arr_cpu.shape}]")

    if isinstance(np_arr, torch.Tensor):
        byte_data = (ctypes.c_ubyte * numel * element_size).from_address(np_arr.data_ptr())
        tensor.raw_data = bytes(byte_data)
        if sys.byteorder == "big":
            np_dtype = oh.tensor_dtype_to_np_dtype(STORAGE_TYPE[tensor.data_type])
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
    else:
        tensor.raw_data = np_arr.tobytes()
        if sys.byteorder == "big":
            np_dtype = oh.tensor_dtype_to_np_dtype(tensor.data_type)
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)

    return tensor


class MiniOnnxBuilder:
    """
    Simplified builder to build very simple model.

    :param target_opset: opset to specify
    :param ir_verison: IR version to use
    """

    def __init__(self, target_opset: int = 18, ir_version: int = 10):
        import torch

        self.initializers_dict = {}
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.opsets = {"": target_opset}
        self.ir_version = ir_version
        self.torch = torch

    def append_output_initializer(
        self, name: str, tensor: Union[np.ndarray, "torch.Tensor"]  # noqa: F821
    ):  # noqa: F821
        """
        Adds an initializer as an output.
        The initializer name is prefixed by ``t_``.
        The output name is *name*.
        """
        init_name = f"t_{name}"
        self.initializers_dict[init_name] = tensor
        self.outputs.append(
            oh.make_tensor_value_info(
                name, dtype_to_tensor_dtype(tensor.dtype), tuple(tensor.shape)
            )
        )
        self.nodes.append(oh.make_node("Identity", [init_name], [name]))

    def append_output_sequence(
        self, name: str, tensors: List[Union[np.ndarray, "torch.Tensor"]]  # noqa: F821
    ):  # noqa: F821
        """
        Adds a sequence of initializers as an output.
        The initializers names are prefixed by ``seq_``.
        The output name is ``name``.
        """
        if not tensors:
            # empty list
            self.nodes.append(oh.make_node("SequenceEmpty", [], [name]))
            tensor_type_proto = oh.make_tensor_type_proto(
                elem_type=TensorProto.FLOAT, shape=None
            )
        else:
            assert all(
                isinstance(t, (np.ndarray, self.torch.Tensor)) for t in tensors
            ), f"Nested sequences are not supported, types are {[type(t) for t in tensors]}"
            names = []
            for i, t in enumerate(tensors):
                init_name = f"seq_{name}_{i}"
                self.initializers_dict[init_name] = t
                names.append(init_name)

            self.nodes.append(oh.make_node("SequenceConstruct", names, [name]))
            tensor_type_proto = oh.make_tensor_type_proto(
                elem_type=dtype_to_tensor_dtype(tensors[0].dtype), shape=None
            )

        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        output = oh.make_value_info(name, type_proto=sequence_type_proto)
        self.outputs.append(output)

    def append_output_dict(
        self, name: str, tensors: Dict[str, Union[np.ndarray, "torch.Tensor"]]  # noqa: F821
    ):  # noqa: F821
        """
        Adds two outputs, a string tensors for the keys and a sequence of tensors
        for the values.

        The output name is ``name_keys`` and ``name_values``.
        """
        keys = []
        values = []
        for k, v in tensors.items():
            keys.append(k)
            values.append(v)
        self.append_output_initializer(f"{name}_keys", np.array(keys, dtype=np.str_))
        self.append_output_sequence(f"{name}_values", values)

    def _build_initializers(
        self, switch_low_high: bool
    ) -> Tuple[List[TensorProto], Dict[str, TensorProto]]:
        """
        Builds initializers.

        :param switch_low_high: invert low, high precision
        :return: a list of tensors to stored in the model
        """
        init_dict = self.initializers_dict
        if switch_low_high:
            # Let's try to minimize the time.
            initializer = []
            for k, v in init_dict.items():
                if isinstance(v, TensorProto):
                    initializer.append(v)
                    continue

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
                        t = onh.from_array(v, name=k)
                        initializer.append(t)
                        continue

                    from_np = True
                elif isinstance(v, np.float32):
                    t = onh.from_array(np.array([v], dtype=np.float32), name=k)
                    initializer.append(t)
                    continue
                elif isinstance(v, np.float16):
                    t = onh.from_array(np.array([v], dtype=np.float16), name=k)
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
                    tensor = proto_from_array(v, name=k)

                initializer.append(tensor)

            return initializer

        res = []
        for k, v in init_dict.items():
            if isinstance(v, TensorProto):
                res.append(v)
                continue
            if isinstance(v, self.torch.Tensor):
                # no string tensor
                t = self.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, np.ndarray):
                t = onh.from_array(v, name=k)
                res.append(t)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type "
                f"{type(v)} into a TensorProto."
            )
        return res

    def to_onnx(self) -> ModelProto:
        """
        Conversion to onnx.
        :return: the proto
        """
        opsets = [oh.make_opsetid(*o) for o in self.opsets.items()]
        ir_version = self.ir_version
        model = ModelProto()
        model.graph.CopyFrom(GraphProto())
        model.graph.name = "mini_model"
        model.graph.input.extend(self.inputs)
        model.graph.node.extend(self.nodes)
        model.graph.output.extend(self.outputs)
        initializers = self._build_initializers(switch_low_high=sys.byteorder != "big")
        model.graph.initializer.extend(initializers)
        model.opset_import.extend(opsets)
        model.ir_version = ir_version
        return model


def create_onnx_model_from_input_tensors(
    inputs: Union[Tuple[Any, ...], Dict[str, Any]], switch_low_high: Optional[bool] = None
) -> ModelProto:
    """
    Creates a model proto including all the value as initializers.
    They can be restored by executing the model.
    We assume these inputs are not bigger than 2Gb,
    the limit of protobuf.

    :param inputs: any tuple or dictionary of inputs
    :param switch_low_high: if None, it is equal to ``switch_low_high=sys.byteorder != "big"``
    :return: ModelProto
    """
    import torch

    if switch_low_high is None:
        switch_low_high = sys.byteorder != "big"

    builder = MiniOnnxBuilder()

    if isinstance(inputs, tuple):
        for i, obj in enumerate(inputs):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                builder.append_output_initializer(f"arg_{i}", obj)
            elif isinstance(obj, list):
                builder.append_output_sequence(f"arg_{i}", obj)
            elif isinstance(obj, dict):
                builder.append_output_dict(f"arg_{i}", obj)
            else:
                raise NotImplementedError(f"Not yet implemented for type {type(obj)}, i={i}")
    elif isinstance(inputs, dict):
        for name, obj in inputs.items():
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                builder.append_output_initializer(name, obj)
            elif isinstance(obj, list):
                builder.append_output_sequence(name, obj)
            elif isinstance(obj, dict):
                builder.append_output_dict(name, obj)
            else:
                raise NotImplementedError(
                    f"Not yet implemented for type {type(obj)}, name={name!r}"
                )

    return builder.to_onnx()


def create_input_tensors_from_onnx_model(
    proto: Union[str, ModelProto], tensor_cls: Optional[type] = None
) -> Union[Tuple[Any, ...], Dict[str, Any]]:
    """
    Deserializes tensors stored with function
    :func:`create_onnx_model_from_input_tensors`.
    It relies on :class:`ExtendedReferenceEvaluator
    <experimental_experiment.reference.ExtendedReferenceEvaluator>`
    to restore the tensors.

    :param proto: ModelProto or the file itself
    :param tensor_cls: tensor class, :class:`torch.Tensor` or :class:`numpy.ndarray`.
    :return: ModelProto
    """
    from .reference import ExtendedReferenceEvaluator

    sess = ExtendedReferenceEvaluator(proto)
    got = sess.run(None, {})
    names = sess.output_names
    res = []
    current_keys = None
    for name, g in zip(names, got):
        if name.endswith("_keys"):
            current_keys = g.tolist()
        elif name.endswith("_values"):
            assert current_keys is not None, f"Issue when deserialize output {name!r}"
            if tensor_cls is not None:
                g = [tensor_cls(t) for t in g]
            res.append(dict(zip(current_keys, g)))
            current_keys = None
        elif isinstance(g, list):
            res.append([t if tensor_cls is None else tensor_cls(t) for t in g])
        else:
            res.append(g if tensor_cls is None else tensor_cls(g))
    return tuple(res)
