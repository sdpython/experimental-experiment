import os
import time
import sys
from typing import Any, Optional
import numpy as np
from onnx import ModelProto, StringStringEntryProto, TensorProto
from onnx.model_container import ModelContainer, _set_external_data
from onnx.external_data_helper import _get_all_tensors, uses_external_data
from onnx.inliner import inline_local_functions
from ..mini_onnx_builder import proto_from_array


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


class TorchModelContainer(ModelContainer):
    """
    Overwrites :class:`onnx.model_container.ModelContainer`
    to support torch tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = {
            "time_export_write_model": 0,
            "time_export_byteswap_tobytes": 0,
            "time_export_tobytes": 0,
            "time_export_proto_from_array": 0,
            "time_export_write_tensor_bytes": 0,
            "time_export_inline_model": 0,
        }
        self.inline = False

    def _save_external(
        self,
        file_path: str,
        all_tensors_to_one_file: bool,
    ) -> ModelProto:
        """Save the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: all tensors in one file
            stats: saves time if not Nones

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
        if folder and not os.path.exists(folder):
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
                raise RuntimeError(f"No location found for tensor name {tensor.name!r}.")
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )
            np_tensor = self.large_initializers[prop.value]

            if sys.byteorder == "big":
                # Convert endian from little to big
                begin = time.perf_counter()
                tensor_bytes = np_tensor.byteswap().tobytes()
                self._stats["time_export_byteswap_tobytes"] += time.perf_counter() - begin
            elif isinstance(np_tensor, np.ndarray):
                begin = time.perf_counter()
                tensor_bytes = np_tensor.tobytes()
                self._stats["time_export_tobytes"] += time.perf_counter() - begin
            elif isinstance(np_tensor, TensorProto):
                tensor_bytes = np_tensor.raw_data
                assert len(tensor_bytes) > 0, f"One tensor is null, np_tensor={np_tensor}."
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

                begin = time.perf_counter()
                proto = proto_from_array(pt, name="dummy")
                self._stats["time_export_proto_from_array"] += time.perf_counter() - begin
                tensor_bytes = proto.raw_data
                assert (
                    pt.dtype != torch.float32 or len(tensor_bytes) == np.prod(pt.shape) * 4
                ), (
                    f"Unexpected size mismatch, buffer size is {len(tensor_bytes)}, "
                    f"but tensor size={np.prod(pt.shape) * 4}, "
                    f"shape={pt.shape}, dtype={pt.dtype}"
                )

            begin = time.perf_counter()
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
            self._stats["time_export_write_tensor_bytes"] += time.perf_counter() - begin

        if self.inline:
            begin = time.perf_counter()
            copy = inline_local_functions(copy)
            self._stats["time_export_inline_model"] += time.perf_counter() - begin

        begin = time.perf_counter()
        with open(file_path, "wb") as f:
            f.write(copy.SerializeToString())

        self._stats["time_export_write_model"] += time.perf_counter() - begin
        return copy
