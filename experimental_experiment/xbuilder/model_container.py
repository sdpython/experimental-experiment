import os
import sys
from typing import Any, Optional
import numpy as np
import onnx.helper as oh
from onnx import ModelProto, StringStringEntryProto, TensorProto
from onnx.model_container import ModelContainer, _set_external_data
from onnx.external_data_helper import _get_all_tensors, uses_external_data


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
        elif elem_type is None:
            elem_type = TensorProto.UNDEFINED
        elif exc:
            raise ValueError(f"Unable to interpret elem_type {elem_type!r}.")
    return elem_type


def proto_from_array(
    arr: "torch.Tensor",  # noqa: F821
    name: Optional[str] = None,
    verbose: int = 0,  # noqa: F821
) -> TensorProto:
    """
    Converts a torch Tensor into a TensorProto.
    """
    import sys
    import torch

    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Unexpected type {type(arr)}.")
    if arr.is_sparse:
        raise NotImplementedError(
            f"Sparse tensor is not supported yet but initializer {name!r} is."
        )

    arr_cont = arr.contiguous() if not arr.is_contiguous() else arr
    arr_cpu = arr_cont.cpu()
    if arr_cpu.data_ptr() == arr.data_ptr():
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

    if verbose > 1 and np.prod(arr_cpu.shape) > 100:
        print(f"[proto_from_array] {tensor.data_type}[{arr_cpu.shape}]")

    if sys.byteorder == "big":
        tensor.raw_data = np_arr.tobytes()
        np_dtype = oh.tensor_dtype_to_np_dtype(tensor.data_type)
        np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
    else:
        tensor.raw_data = np_arr.tobytes()

    return tensor


class TorchModelContainer(ModelContainer):
    """
    Overwrites :class:`onnx.model_container.ModelContainer`
    to support torch tensors.
    """

    def _save_external(
        self, file_path: str, all_tensors_to_one_file: bool
    ) -> ModelProto:
        """Save the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: all tensors in one file

        Returns:
            modified main model proto
        """

        def _clean_name(prefix: str, name: str, unique_names: dict[str, int]) -> str:
            if prefix:
                name = f"{prefix}-{name}"
            for c in ":/\\;,!#":
                name = name.replace(c, "")
            base_name = name
            if name in unique_names:
                i = unique_names[name] + 1
                unique_names[name] = i
                return f"{base_name}_{i}"
            unique_names[name] = 1
            return name

        unique_names: dict[str, int] = {}
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder!r} does not exist.")
        proto = self.model_proto.SerializeToString()
        copy = ModelProto()
        copy.ParseFromString(proto)
        prefix = os.path.splitext(os.path.split(file_path)[-1])[0]

        if all_tensors_to_one_file:
            file_weight = f"{os.path.split(file_path)[1]}.weight"
            full_file_weight = f"{file_path}.weight"
            offset = 0
            with open(full_file_weight, "wb") as f:
                pass

        for tensor in _get_all_tensors(copy):
            if not uses_external_data(tensor):
                continue
            prop: Optional[StringStringEntryProto] = None
            for ext in tensor.external_data:  # type: ignore[assignment]
                if ext.key == "location":  # type: ignore[attr-defined]
                    prop = ext  # type: ignore[assignment]
            if prop is None:
                raise RuntimeError(
                    f"No location found for tensor name {tensor.name!r}."
                )
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )
            np_tensor = self.large_initializers[prop.value]

            if sys.byteorder == "big":
                # Convert endian from little to big
                tensor_bytes = np_tensor.byteswap().tobytes()
            elif isinstance(np_tensor, np.ndarray):
                tensor_bytes = np_tensor.tobytes()
            else:
                import torch

                if isinstance(np_tensor, torch.nn.Parameter):
                    pt = np_tensor.data
                elif isinstance(np_tensor, torch.Tensor):
                    pt = np_tensor
                else:
                    raise NotImplementedError(
                        f"Handling of type {type(np_tensor)} as large initializer "
                        f"is not implemented yet."
                    )

                proto = proto_from_array(pt, name="dummy")
                tensor_bytes = proto.raw_data
                assert (
                    pt.dtype != torch.float32
                    or len(tensor_bytes) == np.prod(pt.shape) * 4
                ), (
                    f"Unexpected size mismatch, buffer size is {len(tensor_bytes)}, "
                    f"but tensor size={np.prod(pt.shape) * 4}, "
                    f"shape={pt.shape}, dtype={pt.dtype}"
                )

            if all_tensors_to_one_file:
                _set_external_data(
                    tensor,
                    location=file_weight,
                    offset=offset,
                    length=len(tensor_bytes),
                )
                offset += len(tensor_bytes)
                with open(full_file_weight, "ab") as f:
                    f.write(tensor_bytes)
            else:
                name = f"{_clean_name(prefix, prop.value, unique_names)}.weight"
                _set_external_data(tensor, location=name)
                full_name = os.path.join(folder, name)
                prop.value = name
                with open(full_name, "wb") as f:
                    f.write(tensor_bytes)

        with open(file_path, "wb") as f:
            f.write(copy.SerializeToString())
        return copy
