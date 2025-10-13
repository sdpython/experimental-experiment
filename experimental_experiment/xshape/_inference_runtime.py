import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import onnx
from onnx_diagnostic.helpers import string_type
from ..helpers import tensor_dtype_to_np_dtype, onnx_dtype_to_torch_dtype
from ..reference import ExtendedReferenceEvaluator
from ._shape_helper import (
    all_int,
    _reshape_shape,
    is_static_shape,
    reshape_implementation_with_zero,
)
from .shape_type_compute import set_shape_type_op_any, set_shape_type_custom


class _InferenceRuntime:
    """Sets shape and type."""

    def _make_node_set_type_shape(self, node: onnx.NodeProto):
        """Updates shapes for a node."""
        update = self._make_node_set_type_shape_constant(node, {})
        if update is None:
            if node.domain == "":
                node.doc_string += "#Io1"
                update = set_shape_type_op_any(self, node)
            else:
                # Missing type means it is probably coming from an inlined function.
                node.doc_string += (
                    "#Io3" if node.input and not self.has_type(node.input[0]) else "#Io2"
                )
                update = set_shape_type_custom(self, node)
        if update:
            self._calls.append(
                (node.name, node.domain, node.op_type, node.input, node.output, update)
            )
        assert update is not None or not self._debug_shape_missing, (
            f"Shape missing for node type {node.op_type!r}, inputs={node.input}, "
            f"outputs={node.output}\n----\n{node}\n{self.get_debug_msg()}"
        )

    def update_node_constant(self, name: str, node: onnx.NodeProto) -> bool:
        """Updates a constant NodeProto."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name"
        assert node is None or isinstance(
            node, onnx.NodeProto
        ), f"Unexpected type {type(node)} for name={name!r}"
        if self.verbose > 2:
            print(
                f"[GraphBuilder-{self._hash()}.update_node_constant] new constant "
                f"{name!r}, node={None if node is None else node.op_type}"
            )
        assert (
            node is None
            or node.op_type == "Shape"
            or all(self.is_constant(i) for i in node.input if i not in {"", None, "None"})
        ), (
            f"Output {name!r} is constant (node={self.pretty_node(node)}) "
            f"only if every input from {node.input} is constant "
            f"but constants={[self.is_constant(i) for i in node.input]}{self.get_debug_msg()}"
        )
        self.constants_[name] = node
        return True

    def _make_node_set_type_shape_constant(
        self, node: onnx.NodeProto, sts: Optional[Dict[str, Any]]
    ):
        if node.domain != "":
            return

        if all(self.is_constant(i) for i in node.input):
            for o in node.output:
                self.update_node_constant(o, node)

        if node.op_type == "Constant":
            assert (
                len(node.attribute) == 0
                or node.attribute[0].name != "value"
                or node.attribute[0].type != onnx.AttributeProto.GRAPH
            ), f"{node}"
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                size = np.prod(node.attribute[0].t.dims)
            else:
                size = len(node.SerializeToString())
            assert size < self.optimization_options.constant_size, (
                f"A node Constant holds a tensor bigger than "
                f"the constant: {size} >= {self.optimization_options.constant_size}."
            )
            k = node.output[0]
            self.update_node_constant(k, node)
            node.doc_string += ":constant-3:"
            shape = self._get_tensor_shape(node)
            dtype = self._get_tensor_type(node)
            self.set_shape(k, shape)
            self.set_type(k, dtype)
            if self.verbose > 2 or np.prod(shape) > 100:
                print(f"[GraphBuilder-{self._hash()}.make_node] {k}[{dtype}: {shape}]")
            return shape
        elif node.op_type == "ConstantOfShape":
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                itype = node.attribute[0].t.data_type
            else:
                itype = onnx.TensorProto.FLOAT
            self.set_type(node.output[0], itype)
            if self.is_constant(node.input[0]):
                value = self.get_constant(
                    node.input[0], computed_value=True, as_shape=True, exc=False
                )
                if value is not None:
                    # This is needed when concatenating caches.
                    self.set_shape(node.output[0], value, allow_zero=True)
                    node.doc_string += ":constant-9:"
                    return value
            vs = self.value_as_shape(node.input[0])
            if vs is not None:
                self.set_shape(node.output[0], vs, allow_zero=True)
                return vs
            if self.has_shape(node.input[0]):
                shape = self.get_shape(node.input[0])
                if is_static_shape(shape):
                    self.set_rank(node.output[0], shape[0])
                    return True
        elif node.op_type == "Identity":
            shape = None
            if self.has_shape(node.input[0]):
                # allow_zero is True but if it fails here, it means it did not fail
                # before when it should be.
                shape = self.get_shape(node.input[0])
                self.set_shape(node.output[0], shape, allow_zero=True)
            elif self.has_rank(node.input[0]):
                self.set_rank(node.output[0], self.get_rank(node.input[0]))
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[0]):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-4:"
            return shape
        elif node.op_type == "Expand":
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if (
                self.has_shape(node.input[0])
                and is_static_shape(self.get_shape(node.input[0]))
                and self.is_constant(node.input[1])
            ):
                cst, _ = self.compute_constant(node.input[1], exc=False, only_array=True)
                if cst is not None:
                    assert not isinstance(cst, self.torch._subclasses.fake_tensor.FakeTensor), (
                        f"self.compute_constant returns a FakeTensor for {node.input[1]!r}"
                        f"\n{self.pretty_text()}"
                    )
                    assert (
                        not self.has_rank(node.input[1]) or self.get_rank(node.input[1]) == 1
                    ), (
                        f"Unexpected rank {self.get_rank(node.input[1])} for {node.input[1]!r}"
                        f"{self.get_debug_msg()}"
                    )
                    with self.maybe_disable_fake_tensor_mode():
                        assert len(cst.shape) == 1 and cst.min() > 0, (
                            f"Unexpected shape {cst.shape} "
                            f"for computed constant {node.input[1]!r}, "
                            f"cst={cst}{self.get_debug_msg()}"
                        )
                        shape = self.get_shape(node.input[0])
                        new_shape = tuple(int(i) for i in cst)
                    if len(shape) < len(new_shape):
                        shape = (1,) * (len(new_shape) - len(shape)) + shape
                    new_shape = tuple(max(i, j) for i, j in zip(shape, new_shape))
                    self.set_shape(node.output[0], new_shape)
                    return new_shape
        elif node.op_type == "Reshape":
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[1]):
                cst, _ = self.compute_constant(
                    node.input[1], exc=False, only_array=True, allow_empty=True
                )
                if cst is not None:
                    shape_cst = tuple(int(i) for i in cst)
                    if 0 in shape_cst:
                        if self.has_shape(node.input[0]):
                            sh = self.get_shape(node.input[0])
                            shape_cst_last_zero = shape_cst[
                                : len(shape_cst) - 1 - shape_cst[::-1].index(0) + 1
                            ]
                            assert len(sh) >= len(shape_cst_last_zero), (
                                f"Shape discrepancies for name={node.input[0]!r} "
                                f"node.name={node.name!r} "
                                f"between sh={sh} and shape_cst={shape_cst}, "
                                f"shape_cst_last_zero={shape_cst_last_zero}"
                                f"\ncst={cst}{self.get_debug_msg()}"
                            )
                            shape_cst = tuple(
                                [
                                    shape_cst[i] if shape_cst[i] != 0 else sh[i]
                                    for i in range(len(shape_cst))
                                ]
                            )
                        else:
                            shape_cst = None
                    if shape_cst is not None:
                        if -1 in shape_cst:
                            if self.has_shape(node.input[0]):
                                sh = self.get_shape(node.input[0])
                                if is_static_shape(sh):
                                    new_shape = _reshape_shape(sh, shape_cst)
                                    self.set_shape(node.output[0], new_shape)
                                    node.doc_string += ":constant-7a:"
                                    return new_shape
                        else:
                            self.set_shape(node.output[0], shape_cst)
                            node.doc_string += ":constant-7b:"
                            return shape_cst
        elif node.op_type == "Shape":
            ret_shape = None
            self.set_type(node.output[0], onnx.TensorProto.INT64)
            if self.has_rank(node.input[0]):
                rk = self.get_rank(node.input[0])
                if len(node.attribute) == 0:
                    self.set_shape(node.output[0], (rk,))
                else:
                    start = self.get_attribute_with_default(node, "start", 0)
                    if start < 0:
                        start += rk
                    end = self.get_attribute_with_default(node, "end", rk)
                    if end < 0:
                        end += rk
                    self.set_shape(node.output[0], (end - start,))
                    ret_shape = (end - start,)
            elif node.attribute:
                start = self.get_attribute_with_default(node, "start", 0)
                end = self.get_attribute_with_default(node, "end", None)
                if end is not None and end - start > 0:
                    self.set_shape(node.output[0], (end - start,))
                else:
                    self.set_rank(node.output[0], 1)
                    assert not self._debug_shape_missing, (
                        f"Unable to compute the shape of this shape: "
                        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
                    )
            else:
                self.set_rank(node.output[0], 1)
                assert not self._debug_shape_missing, (
                    f"Unable to compute the shape of this shape: "
                    f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
                )
            if self.is_constant(node.input[0]) or (
                self.has_shape(node.input[0]) and all_int(self.get_shape(node.input[0]))
            ):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2:"
            return ret_shape
        elif node.op_type == "Size":
            self.set_type(node.output[0], onnx.TensorProto.INT64)
            self.set_shape(node.output[0], tuple())
            if self.is_constant(node.input[0]):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2s:"
            return tuple()
        elif not sts:
            if node.op_type == "GatherElements":
                if self.has_type(node.input[0]):
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                if self.has_shape(node.input[1]):
                    self.set_shape(node.output[0], self.get_shape(node.input[1]))
                    return self.get_shape(node.input[1])
                elif self.has_rank(node.input[1]):
                    self.set_rank(node.output[0], self.get_rank(node.input[1]))

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
        v = self.constants_[name]
        # It should not be None but a node as it is not an initializer.
        if isinstance(v, onnx.TensorProto):
            return self.get_constant(name, computed_value=True, exc=exc), None

        assert isinstance(
            v, onnx.NodeProto
        ), f"Unexpected type {type(v)} for constant name={name!r}"
        if self._debug_get_constant:
            print(f"[GraphBuilder-{self._hash()}.compute_constant] {self.pretty_node(v)}")

        if v.op_type == "Shape":
            if not self.has_shape(v.input[0]):
                # We stop.
                assert not self._debug_constant_folding, (
                    f"Unable to compute constant because {v.input[0]!r} has no shape"
                    f"in node {self.pretty_node(v)}{self.get_debug_msg()}"
                )
                return None, None
            shape = self.get_shape(v.input[0])
            if is_static_shape(shape):
                if v.attribute:
                    start = 0
                    end = None
                    for att in v.attribute:
                        if att.name == "start":
                            start = att.i
                        elif att.name == "end":
                            end = att.i
                    shape = shape[start:] if end is None else shape[start:end]
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.compute_constant]     - SHAPE "
                            f"{name}: {shape}? start={start}, end={end}"
                        )
                elif self._debug_get_constant:
                    print(
                        f"[GraphBuilder-{self._hash()}.compute_constant] "
                        f"    - SHAPE {name}: {shape}?"
                    )
                return np.array(shape, dtype=np.int64), {
                    v.input[0]: self.ShapeConstant(v.input[0], shape, v)
                }

            if not self.is_constant(v.input[0]):
                # One exception here as the input maybe not
                # be constant but the shape may be known.
                assert all_int(shape), (
                    f"Shape must be static ({shape}) if shape is constant in {v} in "
                    f"{self.pretty_node(v)}{self.get_debug_msg()}"
                )
                with self.maybe_disable_fake_tensor_mode():
                    output = self._apply_shape_on_shape(v, shape)
                    if isinstance(output[0], self.torch.Tensor):
                        # We convert the tensor into numpy array,
                        # it is a small shape anyway so the FakeMode
                        # does not come up as an issue.
                        output = [output[0].detach().cpu().numpy()]
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.compute_constant]     - A "
                            f"{name}: {self.pretty_tensor(output[0])}"
                        )
                    return output[0], {v.input[0]: self.ShapeConstant(v.input[0], shape, v)}
            assert not self._debug_constant_folding, (
                f"Unable to compute constant for node {self.pretty_node(v)}"
                f"{self.get_debug_msg()}"
            )
            return None, None

        feeds = {i: self.get_constant(i, exc=exc, computed_value=True) for i in v.input}
        for kval, val in feeds.items():
            if not exc and "FakeTensor" in str(type(val)):
                assert not self._debug_constant_folding, (
                    f"Unable to compute constant for node {self.pretty_node(v)}"
                    f"because a FakeTensor appeared{self.get_debug_msg()}"
                )
                return None, None
            assert "FakeTensor" not in str(type(val)), (
                f"FakeTensor {kval!r} cannot be an initializer {type(val)}, "
                f"v.op_type={v.op_type!r}"
                f"{self.get_debug_msg()}"
            )
            if val is None:
                assert not self._debug_constant_folding, (
                    f"Unable to compute constant for node {self.pretty_node(v)}"
                    f"because val=None{self.get_debug_msg()}"
                )
                return None, None

        with self.maybe_disable_fake_tensor_mode():
            if v.op_type == "Identity":
                # much faster this way
                output = [feeds[v.input[0]]]
            elif v.op_type == "Reshape":
                # much faster this way
                output = [
                    reshape_implementation_with_zero(feeds[v.input[0]], tuple(feeds[v.input[1]]))
                ]
            elif v.op_type in {
                "Add",
                "Div",
                "Mul",
                "Sub",
            }:
                # bypassing onnx.numpy_helper.from_array, too slow
                output = self._apply_binary_op(v, feeds)
            elif (
                v.op_type == "Pow"
                and self.has_type(v.input[0])
                and self.has_type(v.input[1])
                and self.get_type(v.input[0]) == self.get_type(v.input[1])
            ):
                output = self._apply_binary_op(v, feeds)
            elif v.op_type in {"Exp", "Reciprocal", "Sqrt"}:
                # bypassing onnx.numpy_helper.from_array, too slow
                output = self._apply_unary_function(v, feeds)
            elif hasattr(self, f"_apply_{v.op_type.lower()}"):
                output = getattr(self, f"_apply_{v.op_type.lower()}")(v, feeds)
            elif all(isinstance(v, np.ndarray) for v in feeds.values()):
                if v.op_type not in {"Constant", "ConstantOfShape"} and self.main_opset < 18:
                    # This functionality is not enabled before that opset.
                    if self._debug_get_constant:
                        print(
                            f"[GraphBuilder-{self._hash()}.compute_constant] fails "
                            f"because opset={self.main_opset} for name={name!r}, "
                            f"node={self.pretty_node(v)}"
                        )
                    assert not self._debug_constant_folding, (
                        f"Unable to compute constant opset={self.main_opset}<18"
                        f"for name={name!r}{self.get_debug_msg()}"
                    )
                    return None, None

                # Let's avoid big computation on CPU.
                max_dim = 0
                for _v in feeds.values():
                    max_dim = max(max_dim, np.prod(_v.shape))
                if max_dim >= 2**22:
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder-{self._hash()}.compute_constant] stop computing a "
                            f"constant as it may be too big, shapes are "
                            f"{[_.shape for _ in feeds.values()]}"
                        )
                    assert not self._debug_constant_folding, (
                        f"Unable to compute constant for node {self.pretty_node(v)}"
                        f"because max_dim={max_dim} (shape={_v.shape}){self.get_debug_msg()}"
                    )
                    return None, None

                begin = time.perf_counter()
                ref = ExtendedReferenceEvaluator(v)
                try:
                    output = ref.run(None, feeds)
                except (ValueError, TypeError) as e:
                    sf = ", ".join(f"{k}:{v.dtype}:{v.shape}" for k, v in feeds.items())
                    if "warnings" not in self._debug_msg:
                        self._debug_msg["warnings"] = []
                    sv = str(v).replace("\n", " ")
                    self._debug_msg["warnings"].append(f"Issue with v={sv}, feeds={sf}, e={e}")
                    self.time_evaluation_constants_ += time.perf_counter() - begin
                    assert not self._debug_constant_folding, (
                        f"Unable to compute constant for node {self.pretty_node(v)}"
                        f"due to {e}{self.get_debug_msg()}"
                    )
                    return None, None

                self.time_evaluation_constants_ += time.perf_counter() - begin
            else:
                assert not self._debug_constant_folding, (
                    f"Unable to compute constant for node {self.pretty_node(v)}, "
                    f"feeds={string_type(feeds, with_shape=True, with_min_max=True, limit=20)}"
                    f"{self.get_debug_msg()}"
                )
                return None, None

            cst = None
            for n, val in zip(v.output, output):
                assert not isinstance(val, tuple), f"Unexpected type {type(val)} for n={n!r}"
                assert "FakeTensor" not in str(type(val)), (
                    f"FakeTensor detected {type(val)} in constant {name!r}, "
                    f"v.op_type={v.op_type!r}{self.get_debug_msg()}"
                )
                if self.has_type(n):
                    # numpy changes the expected type sometimes
                    # (like transpose(x: float36) --> float32)
                    itype = self.get_type(n)
                    if hasattr(val, "detach"):
                        val = val.to(onnx_dtype_to_torch_dtype(itype))
                    else:
                        val = val.astype(tensor_dtype_to_np_dtype(itype))
                self.constants_computed_[n] = val
                if name == n:
                    cst = val

        assert (
            len(cst.shape) == 0
            or min(cst.shape) > 0
            or (v.op_type in {"ConstantOfShape", "Cast", "Identity", "Constant"})
        ), (
            f"Output has empty shape {cst.shape}, name={name!r} "
            f"v.op_type={v.op_type!r}, v.name={v.name!r}{self.get_debug_msg()}"
        )
        assert cst is not None, f"Constant {name!r} was not found in {v.output}"
        if isinstance(cst, self.torch._subclasses.fake_tensor.FakeTensor):
            assert not self._debug_constant_folding, (
                f"Unable to compute constant for node {self.pretty_node(v)}"
                f"because a FakeTensor appeared{self.get_debug_msg()}"
            )
            return None, None
        if self._debug_get_constant:
            print(
                f"[GraphBuilder-{self._hash()}.compute_constant] "
                f"    - A {name}: {self.pretty_tensor(cst)}"
            )
        assert (
            not self._debug_constant_folding or cst is not None
        ), f"Unable to compute constant for node {self.pretty_node(v)}{self.get_debug_msg()}"
        return cst, feeds
