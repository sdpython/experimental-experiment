import contextlib
from typing import Dict, Generator, List, Tuple
import numpy as np
from onnx import NodeProto
from ..helpers import tensor_dtype_to_np_dtype
from ._shape_helper import DYNAMIC_SHAPE, STATIC_SHAPE, all_int, all_int_or_str
from ._dtype_helper import dtype_to_tensor_dtype, onnx_dtype_to_torch_dtype


@contextlib.contextmanager
def _unset_fake_temporarily() -> Generator:
    import torch

    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


class _GraphBuilderRuntime:

    def _apply_slice_to_shape(
        self,
        shape: STATIC_SHAPE,
        indices: List[slice],
        axes: List[int],
        expand_axes: List[int],
    ) -> STATIC_SHAPE:
        assert isinstance(shape, tuple), f"Unexpected type {type(shape)} for shape: {shape}"
        assert isinstance(
            indices, list
        ), f"Unexpected type {type(indices)} for index: {indices}"
        assert isinstance(axes, list), f"Unexpected type {type(axes)} for index: {axes}"
        assert len(axes) in (
            1,
            len(indices),
        ), f"Mismatch lengths {len(indices)} != {len(axes)}"

        if all(isinstance(i, slice) for i in indices):
            new_shape = []
            for index, axis_ in zip(indices, axes):
                axis = axis_ if axis_ >= 0 else (axis_ + len(shape)) % len(shape)
                while len(new_shape) < axis:
                    assert shape[len(new_shape)] >= 0, (
                        f"Negative value in shape {shape}, indices={indices}, "
                        f"axes={axes}, expand_axes={expand_axes}"
                    )
                    new_shape.append(shape[len(new_shape)])
                assert axis < len(shape), (
                    f"axis={axis} is out of order (shape={shape}, "
                    f"indices={indices}, axes={axes}){self.get_debug_msg()}"
                )
                n = shape[axis]
                start = index.start or 0
                end = index.stop or n
                diff = end - start
                dim = diff // index.step if index.step else diff
                dim = max(dim, 0)
                assert dim >= 0, (
                    f"Negative dim={dim}, axis={axis}, shape={shape}, indices={indices}, "
                    f"axes={axes}, expand_axes={expand_axes}"
                )
                new_shape.append(dim)
        elif all_int(indices):
            assert len(axes) == 1, (
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
            new_shape = [len(indices), *shape[1:]]
        else:
            raise RuntimeError(
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
        for a in shape[len(new_shape) :]:
            assert a >= 0, (
                f"Negative value in shape {shape}, indices={indices}, "
                f"axes={axes}, expand_axes={expand_axes}"
            )
            new_shape.append(a)
        for e in expand_axes:
            new_shape.insert(e, 1)
        return tuple(new_shape)

    def _apply_reshape_to_shape(
        self, input_shape: DYNAMIC_SHAPE, new_shape: STATIC_SHAPE
    ) -> DYNAMIC_SHAPE:
        """
        Returns the shape of the output of a node Reshape.
        """
        assert isinstance(
            input_shape, tuple
        ), f"unexpected type {type(input_shape)} for input_shape."
        assert isinstance(
            new_shape, tuple
        ), f"unexpected type {type(new_shape)} for input_shape."
        assert all_int(new_shape), f"unexpected type for a dimension in {new_shape}"

        # handling zeros --> keeps the original dimension
        new_new_shape = []
        for i, sh in enumerate(new_shape):
            if sh == 0:
                assert i < len(
                    input_shape
                ), f"Unable to apply reshape {new_shape} to input shape {input_shape}"
                new_new_shape.append(input_shape[i])
                continue
            new_new_shape.append(sh)
        new_shape = tuple(new_new_shape)

        if -1 not in new_shape:
            return new_shape
        if all_int(input_shape):
            size = int(np.prod(input_shape))
            div = np.prod([i for i in new_shape if i != -1])
            if div == 0:
                return tuple((int(i) if i >= 0 else 0) for i in new_shape)
            return tuple((int(i) if i >= 0 else int(size // div)) for i in new_shape)
        if all_int_or_str(input_shape):
            if new_shape == (1, -1):
                # common case
                return (1, "*".join(map(str, input_shape)))

        if len(input_shape) == len(new_shape):
            # It is easier to handle.
            res = []
            i_1 = None
            a_int = True
            b_int = True
            for a, b in zip(input_shape, new_shape):
                if not isinstance(a, int):
                    a_int = False
                if isinstance(b, int):
                    if b >= 0:
                        res.append(b)
                    else:
                        i_1 = len(res)
                        res.append(None)
                else:
                    res.append(b)
                    b_int = False
            if i_1 is not None:
                if a_int:
                    size = int(np.prod(input_shape))
                    if b_int:
                        nz = -int(np.prod(new_shape)) // size
                        res[i_1] = nz
                    else:
                        name = "*".join([str(x) for x in res if x is not None])
                        res[i_1] = f"{name}/{size}"
                else:
                    an = "*".join(map(str, input_shape))
                    name = "*".join([str(x) for x in res if x is not None])
                    res[i_1] = f"{an}/({name})"
            return tuple(res)

        # The shape is dynamic and cannot be set.
        return None

    def _apply_transpose(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        x = feeds[node.input[0]]
        if isinstance(x, np.ndarray):
            # Type conversion between numpy and torch is not robust.
            itype = dtype_to_tensor_dtype(x.dtype)
            ttype = onnx_dtype_to_torch_dtype(itype)
            x = self.torch.from_numpy(x.copy()).to(ttype)
        return [self.torch.permute(x, perm).to(x.dtype)]

    def _apply_expand(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        new_shape = feeds[node.input[1]]
        if isinstance(x, self.torch.Tensor):
            try:
                return [x.expand(tuple(int(i) for i in new_shape))]
            except RuntimeError as e:
                raise RuntimeError(
                    f"Unable to compute the constant, new_shape={new_shape}, "
                    f"x.shape={x.shape}, node={node}\n{self.pretty_text()}"
                ) from e
        ones = np.ones(tuple(int(i) for i in new_shape), dtype=x.dtype)
        return [(x * ones).astype(x.dtype)]

    def _apply_squeeze(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        axis = feeds[node.input[1]]
        if len(axis.shape) == 0:
            return [np.squeeze(x, (int(axis),))]
        return [x.squeeze(tuple(int(i) for i in axis))]

    def _apply_unsqueeze(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        axis = feeds[node.input[1]]
        if isinstance(x, np.ndarray):
            if len(axis.shape) == 0:
                return [np.expand_dims(x, (int(axis),))]
            return [np.expand_dims(x, tuple(int(i) for i in axis))]
        if isinstance(axis, np.ndarray):
            axis = [int(axis)] if axis.shape == tuple() else axis.tolist()
        if len(axis) == 1:
            return [x.unsqueeze(int(axis[0]))]
        assert len(axis) > 0, f"axis={axis} is null"
        for a in axis:
            x = x.unsqueeze(int(a))
        return [x]

    def _apply_cast(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        from . import str_tensor_proto_type

        x = feeds[node.input[0]]
        if not isinstance(x, (np.ndarray, self.torch.Tensor)):
            # Maybe a float, then we process it as a float, tensor.to only works
            # on tensors.
            assert isinstance(
                x, (float, int, np.float32, np.float64, np.float16, np.int32, np.int64)
            ), f"Unexpected type {type(x)} for {node.input[0]!r} (node.name={node.name!r})"
            res = self._apply_cast(node, {node.input[0]([x])})
            return [res[0]]
        to, saturate = None, 1
        for att in node.attribute:
            if att.name == "to":
                to = att.i
                break
            if att.name == "saturate":
                saturate = att.i
                break
        assert to, f"to not here in node {node}"
        assert (
            to != 8 and to < 17
        ), f"Cast not implemented for to={to}, {str_tensor_proto_type()}"
        del saturate
        if isinstance(x, np.ndarray):
            # Type conversion between numpy and torch is not robust.
            itype = dtype_to_tensor_dtype(x.dtype)
            ttype = onnx_dtype_to_torch_dtype(itype)
            x = self.torch.Tensor(x).to(ttype)
            assert "FakeTensor" not in str(type(x)), (
                f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                f"{self.get_debug_msg()}"
            )
        assert isinstance(x, self.torch.Tensor), (
            f"Unexpected type {type(x)} for x for node type {node.op_type}, "
            f"name={node.name}, inputs={node.input}, outputs={node.output}"
        )
        ttype = onnx_dtype_to_torch_dtype(to)
        return [x.to(ttype)]

    def _apply_unary_function(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        itype = dtype_to_tensor_dtype(x.dtype)
        if isinstance(x, np.ndarray):
            ttype = tensor_dtype_to_np_dtype(itype)
            if node.op_type == "Sqrt":
                return [np.sqrt(x).astype(ttype)]
            raise AssertionError(
                f"Not implemented for op_type={node.op_type!r}, node={node}, feeds={feeds}"
            )

        ttype = onnx_dtype_to_torch_dtype(itype)
        if node.op_type == "Sqrt":
            return [self.torch.sqrt(x).to(ttype)]
        raise AssertionError(
            f"Not implemented for op_type={node.op_type!r}, node={node}, feeds={feeds}"
        )

    def _apply_trilu(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        upper = True
        for att in node.attribute:
            if att.name == "upper":
                upper = att.i
                break
        assert len(node.input) in (1, 2), (
            f"Unexpected number of inputs (inputs={node.input}) "
            f"for Trilu{self.get_debug_msg()}"
        )
        x = feeds[node.input[0]]
        k = feeds[node.input[1]] if len(node.input) > 1 else np.array(0, dtype=np.int64)
        assert len(x.shape) > 0, (
            f"x cannot be empty but shape is {x.shape}, execution of Trilu "
            f"failed{self.get_debug_msg()}"
        )
        if isinstance(x, self.torch.Tensor):
            assert isinstance(k, self.torch.Tensor), (
                f"Expecting a tensor for {node.input[1]!r} but got "
                f"{type(k)}{self.get_debug_msg()}"
            )
            ak = k.detach().cpu()
            iak = int(ak) if len(ak.shape) == 0 else int(ak[0])
            assert iak <= 1, f"Unexpected value for k={k}{self.get_debug_msg()}"
            return [self.torch.triu(x, k == 0) if upper else self.torch.tril(x, k == 0)]

        assert isinstance(k, np.ndarray), (
            f"Expecting a tensor for {node.input[1]!r} but got "
            f"{type(k)}{self.get_debug_msg()}"
        )
        iak = int(k) if len(k.shape) == 0 else int(k[0])
        return [np.triu(x, iak) if upper else np.tril(x, iak)]

    def _apply_binary_op(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        a, b = feeds[node.input[0]], feeds[node.input[1]]
        if a.dtype != b.dtype:
            a = self._to_torch_tensor(a)
            b = self._to_torch_tensor(b)
        try:
            if node.op_type == "Add":
                return [a + b]
            if node.op_type == "Mul":
                return [a * b]
            if node.op_type == "Sub":
                return [a - b]
            if node.op_type == "Div":
                return [a / b]
            raise AssertionError(f"{node.op_type!r} not implemented")
        except RuntimeError as e:
            raise AssertionError(
                f"Unable to multiply two objects of dtype {a.dtype}, {b.dtype} and "
                f"shapes {a.shape}, {b.shape}, node.op_type={node.op_type!r}, "
                f"node.name={node.name!r}, inputs={node.input}, outputs={node.output}"
            ) from e

    def _apply_where(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        new_feeds = {}
        for k, v in feeds.items():
            if isinstance(v, np.ndarray):
                # Type conversion between numpy and torch is not robust.
                itype = dtype_to_tensor_dtype(v.dtype)
                ttype = onnx_dtype_to_torch_dtype(itype)
                x = self.torch.Tensor(v.copy()).to(ttype)
                assert "FakeTensor" not in str(type(x)), (
                    f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                    f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                    f"{self.get_debug_msg()}"
                )
                new_feeds[k] = x
            else:
                new_feeds[k] = v
        y = self.torch.where(*[new_feeds[k] for k in node.input])
        return [y]

    def _apply_slice(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        new_feeds = {}
        for k, v in feeds.items():
            if isinstance(v, np.ndarray):
                # Type conversion between numpy and torch is not robust.
                itype = dtype_to_tensor_dtype(v.dtype)
                ttype = onnx_dtype_to_torch_dtype(itype)
                x = self.torch.from_numpy(v)
                assert x.dtype == ttype, (
                    f"Unexpected conversion from numpy {v.dtype} to "
                    f"{x.dtype} != {ttype}{self.get_debug_msg()}"
                )

                assert "FakeTensor" not in str(type(x)), (
                    f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                    f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                    f"{self.get_debug_msg()}"
                )
                new_feeds[k] = x
            else:
                new_feeds[k] = v
        assert len(node.input) >= 3, (
            f"Node {node.op_type} (name={node.name!r}) has not enough "
            f"inputs {node.input}\n{self.pretty_text()}"
        )
        data, starts, ends = [new_feeds[k] for k in node.input[:3]]
        axes = new_feeds[node.input[3]] if len(node.input) > 3 and node.input[3] else None
        steps = new_feeds[node.input[4]] if len(node.input) > 4 and node.input[4] else None

        if axes is None:
            if steps is None:
                slices = [slice(s, e) for s, e in zip(starts, ends)]
            else:
                slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
        else:
            if steps is None:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a in zip(starts, ends, axes):
                    slices[a] = slice(s, e)
            else:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a, d in zip(starts, ends, axes, steps):
                    slices[a] = slice(s, e, d)
        res = data[tuple(slices)]
        assert len(res.shape) == 0 or min(res.shape) > 0, (
            f"Empty shape found {res.shape} after Slice when x.shape={data.shape}, "
            f"starts={starts}, ends={ends}, axes={axes}, steps={steps}, "
            f"node.name={node.name!r}, input names={node.input}, "
            f"slices={slices}"
        )
        return [res]

    def _apply_shape_on_shape(
        self, node: NodeProto, shape: Tuple[int, ...]
    ) -> "torch.Tensor":  # noqa: F821
        if node.attribute:
            start = 0
            end = None
            for att in node.attribute:
                if att.name == "start":
                    start = att.i
                elif att.name == "end":
                    end = att.i
            shape = shape[start:] if end is None else shape[start:end]
        return [self.torch.from_numpy(np.array(shape, dtype=np.int64))]

    def _apply_shape(
        self,
        node: NodeProto,
        feeds: Dict[str, "torch.Tensor"],  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        shape = tuple(map(int, feeds[node.input[0]].shape))
        return self._apply_shape_on_shape(node, shape)
