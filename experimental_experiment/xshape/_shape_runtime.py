import numpy as np
import onnx
from ._shape_helper import all_int


class _ShapeRuntime:
    """Runs through a few nodes often used to deal with shapes."""

    def simple_update_value_shape_with_node(self, node: onnx.NodeProto) -> bool:
        """Updates ``_known`_value_shape`` for a particular node."""
        if node.domain != "" or node.op_type not in {
            "Abs",
            "Add",
            "Concat",
            "Div",
            "Gather",
            "Identity",
            "Mod",
            "Mul",
            "Range",
            "Scatter",
            "Shape",
            "Slice",
            "Squeeze",
            "Sub",
            "Unsqueeze",
        }:
            return False

        # Constant can be considered as possible shape.
        for i in node.input:
            known = self.value_as_shape(i)
            if known is not None:
                continue
            if not self.is_constant(i):
                continue
            if not self.has_type(i) or self.get_type(i) != onnx.TensorProto.INT64:
                # No chance for this to be used a shape computation.
                continue
            cst = self.get_constant(i, exc=False, computed_value=True)
            if cst is None or len(cst.shape) > 1:
                continue
            with self.maybe_disable_fake_tensor_mode():
                tu = tuple(map(int, cst)) if len(cst.shape) > 0 else int(cst)
            self.set_value_shape(i, tu)

        if node.op_type in {"Identity", "Abs"}:
            value = self.value_as_shape(node.input[0])
            if value is not None:
                node.doc_string += "#SV-Id1"
                self.set_value_shape(
                    node.output[0], value, equal_to=(node.input[0], node.output[0])
                )
                return True
            node.doc_string += "#SV-Id/2"
            return False

        if node.op_type == "Shape":
            if len(node.attribute) == 0:
                if self.has_shape(node.input[0]):
                    node.doc_string += "#SV-Sh1"
                    shape = self.get_shape(node.input[0])
                    self.set_value_shape(node.output[0], shape)
                    if all_int(shape):
                        self.update_node_constant(node.output[0], node)
                    self.set_shape(node.output[0], (len(shape),))
                    return True
                node.doc_string += "#SV-Sh/1"
                return False

            start = self.get_attribute(node, "start", exc=False)
            end = self.get_attribute(node, "end", exc=False)
            if end is None:
                if self.has_rank(node.input[0]):
                    end = self.get_rank(node.input[0])
            if self.has_shape(node.input[0]):
                shape = self.get_shape(node.input[0])
                assert start is None or start.i < len(shape), (
                    f"Shape mismatch, start={0 if start is None else start.i}, "
                    f"shape of {node.input[0]!r} "
                    f"is {shape}{self.get_debug_msg()}"
                )
                if end is None:
                    n_shape = shape[0 if start is None else start.i :]
                    self.set_value_shape(node.output[0], n_shape)
                    if all_int(shape):
                        self.update_node_constant(node.output[0], node)
                    self.set_shape(node.output[0], (len(n_shape),))
                    node.doc_string += "#SV-Sh4"
                    return True
                assert getattr(end, "i", end) <= len(shape), (
                    f"Shape mismatch, end={getattr(end, 'i', end)}, "
                    f"shape of {node.input[0]!r} "
                    f"is {shape}{self.get_debug_msg()}"
                )
                n_shape = shape[0 if start is None else start.i : getattr(end, "i", end)]
                if all_int(shape):
                    self.update_node_constant(node.output[0], node)
                self.set_value_shape(node.output[0], n_shape)
                self.set_shape(node.output[0], (len(n_shape),))
                node.doc_string += "#SV-Sh6"
                return True

            if end is None:
                self.set_value_shape(node.output[0], f"{node.input[0]}[{start.i}:]")
                node.doc_string += "#SV-Sh/6"
                return False

            start = start.i
            end = getattr(end, "i", end)
            if isinstance(start, int) and isinstance(end, int):
                self.set_value_shape(
                    node.output[0], tuple(f"{node.input[0]}[{i}]" for i in range(start, end))
                )
                node.doc_string += "#SV-Sh7"
            else:
                self.set_value_shape(node.output[0], f"{node.input[0]}[{start}:{end}]")
                node.doc_string += "#SV-Sh7"
            return True

        if node.op_type == "Gather":
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[1]):
                y = self.value_as_shape(node.input[0])
                if y is None:
                    node.doc_string += "#SV-Ga/2"
                    return False
                i = self.get_constant(node.input[1], computed_value=True, exc=True)
                if i is None:
                    node.doc_string += "#SV-Ga/3"
                    return False
                if isinstance(y, str) and isinstance(i, int):
                    self.set_value_shape(node.output[0], f"{y}[{i}]")
                    node.doc_string += "#SV-Ga3"
                    self.set_shape(node.output[0], tuple())
                    return True
                if (
                    isinstance(y, str)
                    and isinstance(i, np.ndarray)
                    and i.dtype == np.int64
                    and i.shape in ((1,), tuple())
                ):
                    ii = int(i[0]) if i.shape == (1,) else int(i)
                    self.set_value_shape(node.output[0], f"{y}[{ii}]")
                    node.doc_string += "#SV-Ga4"
                    self.set_shape(node.output[0], (1,) if i.shape == (1,) else tuple())
                    return True
                if isinstance(y, tuple) and isinstance(i, int):
                    self.set_value_shape(node.output[0], y[i])
                    node.doc_string += "#SV-Ga5"
                    self.set_shape(node.output[0], tuple())
                    return True
                if isinstance(y, tuple) and isinstance(i, tuple) and all_int(i):
                    self.set_value_shape(node.output[0], tuple(y[_] for _ in i))
                    self.set_shape(node.output[0], (len(i),))
                    node.doc_string += "#SV-Ga6"
                    return True
                if (
                    isinstance(y, tuple)
                    and isinstance(i, (self.torch.Tensor, np.ndarray))
                    and i.dtype in (np.int64, self.torch.int64)
                    and tuple(i.shape) in ((1,), tuple())
                ):
                    ishape = tuple(i.shape)
                    ii = int(i[0]) if ishape == (1,) else int(i)
                    if self._debug_quiet and ii >= len(y):
                        node.doc_string += "#SV-Ga/77"
                        return False
                    assert ii < len(y), (
                        f"Unexpected value for y={y!r}, i={i!r} in node Gather "
                        f"with inputs={node.input}{self.get_debug_msg()}"
                    )
                    self.set_value_shape(node.output[0], (y[ii],) if i.shape == (1,) else y[ii])
                    self.set_shape(node.output[0], (1,) if i.shape == (1,) else tuple())
                    node.doc_string += "#SV-Ga7"
                    return True
                raise RuntimeError(
                    f"Not implemented when node Gather(x,i) with inputs={node.input}, "
                    f"shape(x)={y!r}, i={i!r}, i.dtype={i.dtype if i is not None else '?'}"
                    f"{self.get_debug_msg()}"
                )
            node.doc_string += "#SV-Ga/7"
            return False

        if node.op_type == "Squeeze":
            if self.is_constant_or_attribute(node, 1, "axes"):
                y = self.value_as_shape(node.input[0])
                if y is None:
                    node.doc_string += "#SV-Sq/3"
                    return False
                i = self.get_constant_or_attribute(node, 1, "axes")
                if isinstance(i, int):
                    ii = i
                elif (
                    isinstance(i, np.ndarray)
                    and i.dtype == np.int64
                    and i.shape in ((1,), tuple())
                ):
                    ii = int(i[0]) if i.shape == (1,) else int(i)
                elif i is None and isinstance(y, tuple) and len(y) == 1:
                    # A dimension a tensor of 1 element turned into a scalar
                    node.doc_string += "#SV-SqDim"
                    self.set_value_shape(node.output[0], y[0])
                    return True
                else:
                    raise RuntimeError(
                        f"Not implemented when node Squeeze with inputs={node.input}, "
                        f"y={y!r}, i={i!r}{self.get_debug_msg()}"
                    )
                assert (
                    ii == 0
                ), f"A shape should only have one axis i={i}, y={y}{self.get_debug_msg()}"
                if isinstance(y, str):
                    node.doc_string += "#SV-Sq1"
                    self.set_value_shape(node.output[0], f"squeeze({y})")
                    return True
                if isinstance(y, int):
                    node.doc_string += "#SV-Sq2"
                    self.set_value_shape(node.output[0], y)
                    return True
                assert isinstance(
                    y, tuple
                ), f"Unexpected type {type(y)} for y={y} and i={i}{self.get_debug_msg()}"
                node.doc_string += "#SV-Sq3"
                self.set_value_shape(node.output[0], y[0])
                return True
            node.doc_string += "#SV-Sq/2"
            return False

        if node.op_type == "Unsqueeze":
            values_0 = self.value_as_shape(node.input[0])
            if isinstance(values_0, tuple) and len(values_0) > 1:
                # This cannot be a shape anymore after this operation
                node.doc_string += "#SV-Unsq/1"
                return False
            if self.has_rank(node.input[0]) and self.get_rank(node.input[0]) > 0:
                # This cannot be a shape anymore.
                node.doc_string += "#SV-Unsq/2"
                return False
            if not self.has_rank(node.input[0]) and values_0 is None:
                node.doc_string += "#SV-Unsq/3"
                return False
            assert self.has_rank(node.input[0]), (
                f"Rank of {node.input[0]!r} is unknown but "
                f"its value is {values_0!r}{self.get_debug_msg()}"
            )
            if len(node.input) > 1:
                cst = self.get_constant(node.input[1], exc=False, computed_value=True)
                cst = tuple() if not cst.shape else tuple(cst)
            else:
                cst = tuple(self.get_attribute(node, "axes").ints)
                assert cst, f"Value={cst!r} is wrong for {node.input[0]}{self.get_debug_msg()}"
            if cst is not None and len(cst) == 1 and self.get_rank(node.input[0]) == 0:
                node.doc_string += "#SV-Unsq4"
                self.set_value_shape(
                    node.output[0], (node.input[0],) if values_0 is None else (values_0,)
                )
                return True

        # after this point, it is all about operators between shapes.
        values = [self.value_as_shape(x) for x in node.input]
        if any(x is None for x in values):
            # it is not a shape
            node.doc_string += "#SV-All/0"
            return False

        if node.op_type == "Concat":
            node.doc_string += "#SV-Co1"
            concatenated = []
            for v in values:
                concatenated.extend(v if isinstance(v, tuple) else (v,))
            self.set_value_shape(node.output[0], tuple(concatenated))
            return True

        if node.op_type == "Range":
            if len(values) == 3:
                args = []
                for v in values:
                    if isinstance(v, int):
                        args.append(v)
                    elif len(v) == 1:
                        # Should not happen.
                        args.append(v[0])
                    else:
                        node.doc_string += "#SV-Ra/1"
                        return False
                if not all_int(args):
                    node.doc_string += "#SV-Ra/2"
                    return False
            node.doc_string += "#SV-Ra"
            self.set_value_shape(node.output[0], tuple(range(*args)))
            return True

        if node.op_type in {"Mul", "Add", "Div", "Sub", "Mod"}:
            fct, symbol = {
                "Add": ((lambda x, y: x + y), "+"),
                "Div": ((lambda x, y: x // y), "/"),
                "Mul": ((lambda x, y: x * y), "*"),
                "Sub": ((lambda x, y: x - y), "-"),
                "Mod": ((lambda x, y: x % y), "%"),
            }[node.op_type]
            m1 = values[0]
            m2 = values[1]
            if isinstance(m1, int) and isinstance(m2, int):
                node.doc_string += f"#SV-{node.op_type}1"
                self.set_value_shape(node.output[0], fct(m1, m2))
                return True
            if isinstance(m1, (int, str)) and isinstance(m2, (int, str)):
                node.doc_string += f"#SV-{node.op_type}2"
                self.set_value_shape(node.output[0], f"{m1}{symbol}{m2}")
                return True

            # One of them is a tuple.
            if not isinstance(m1, tuple):
                m1 = (m1,)
            if not isinstance(m2, tuple):
                m2 = (m2,)
            if len(m1) == len(m2):
                res = []
                for s1, s2 in zip(m1, m2):
                    res.append(
                        fct(s1, s2)
                        if isinstance(s1, int) and isinstance(s2, int)
                        else f"{s1}{symbol}{s2}"
                    )
                self.set_value_shape(node.output[0], tuple(res))
                node.doc_string += f"#SV-{node.op_type}3"
                return True

            if len(m1) == 1:
                res = []
                for s2 in m2:
                    res.append(
                        fct(m1[0], s2)
                        if isinstance(m1[0], int) and isinstance(s2, int)
                        else f"{m1[0]}{symbol}{s2}"
                    )
                self.set_value_shape(node.output[0], tuple(res))
                node.doc_string += f"#SV-{node.op_type}4"
                return True
            if len(m2) == 1:
                res = []
                for s1 in m1:
                    res.append(
                        fct(s1, m2[0])
                        if isinstance(s1, int) and isinstance(m2[0], int)
                        else f"{s1}{symbol}{m2[0]}"
                    )
                self.set_value_shape(node.output[0], tuple(res))
                node.doc_string += f"#SV-{node.op_type}4"
                return True

            # This cannot be a shape anymore.
            node.doc_string += f"#SV-{node.op_type}/0"
            return False

        if node.op_type == "Gather":
            if isinstance(values[1], tuple) and all_int(values[1]):
                shape = (values[0],) if not isinstance(values[0], tuple) else values[0]
                node.doc_string += "#SV-Ga1"
                assert max(values[1]) < len(shape), (
                    f"Unable to compute new value shape when values={values}"
                    f"{self.get_debug_msg()}"
                )
                self.set_value_shape(node.output[0], tuple(shape[i] for i in values[1]))
                return True

        if node.op_type == "Slice":
            if len(values) >= 3 and values[1] == (0,) and values[2] == (9223372036854775807,):
                node.doc_string += "#SV-Sl1"
                self.set_value_shape(node.output[0], values[0])
                return True
            if len(values) < 4 or values[3] != (0,):
                # Not a shape.
                node.doc_string += "#SV-Sl/2"
                return False
            if len(values) == 4 and all_int(values[1]) and all_int(values[2]):
                assert len(values[1]) == len(values[2]) == 1, (
                    f"Unexpected values {values} to compute a shape from node "
                    f"{self.pretty_node(node)}{self.get_debug_msg()}"
                )
                node.doc_string += "#SV-Sl3"
                self.set_value_shape(node.output[0], values[0][values[1][0] : values[2][0]])
                return True
            if (
                len(values) == 4
                and values[1] == (0,)
                and isinstance(values[2][0], str)
                and isinstance(values[3][0], int)
            ):
                # Maybe a shape but probably not.
                node.doc_string += "#SV-Sl/3"
                return False
        raise RuntimeError(
            f"Unable to compute a shape for node {self.pretty_node(node)} "
            f"with values={values}{self.get_debug_msg()}"
        )
