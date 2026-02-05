import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xshape._shape_helper import is_static_shape
from ..patterns_api import MatchResult, PatternOptimization


class TransposeTransposePattern(PatternOptimization):
    """
    Removes two consecutive transpose if the second one put the tensor in origin shape.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("xs", onnx.TensorProto.FLOAT, shape=(1, 1, 32, 128))
        )
        nodes.append(oh.make_node("Transpose", ["xs"], ["r1"], perm=[1, 0, 3, 2]))
        nodes.append(oh.make_node("Transpose", ["r1"], ["xm1"], perm=[0, 1, 3, 2]))
        outputs.append(
            oh.make_tensor_value_info("xm1", onnx.TensorProto.FLOAT, shape=(1, 1, 32, 128))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("xs", onnx.TensorProto.FLOAT, shape=(1, 1, 32, 128))
        )
        nodes.append(oh.make_node("Transpose", ["xs"], ["xm1"], perm=[1, 0, 2, 3]))
        outputs.append(
            oh.make_tensor_value_info("xm1", onnx.TensorProto.FLOAT, shape=(1, 1, 32, 128))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @classmethod
    def apply_transpose(
        cls, perm: Tuple[int, ...], on: List[Union[int, str]]
    ) -> List[Union[int, str]]:
        assert len(perm) == len(on), "length mismatch"
        res = [None for i in on]
        for i, p in enumerate(perm):
            res[i] = on[p]
        return res

    @classmethod
    def apply_transposes(
        cls, perms: List[Tuple[int, ...]], on: Optional[List[Union[int, str]]] = None
    ) -> List[Union[int, str]]:
        if on is None:
            on = list(range(len(perms[0])))
        for p in perms:
            on = cls.apply_transpose(p, on)
        return on

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()
        next_nodes = g.next_nodes(node.output[0])
        next_node = None
        for n in next_nodes:
            if n.op_type == "Transpose":
                next_node = n
        if next_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Three consecutive transpose are not expected but let's continue
        # as if it could be possible.
        nodes = [node, next_node]
        perms = [tuple(g.get_attribute(n, "perm").ints) for n in nodes]
        lens = [len(p) for p in perms]
        assert min(lens) == max(lens), (
            f"Consecutive Transpose should apply on tensors with "
            f"the same rank but perms={perms}."
        )
        first = list(range(lens[0]))
        last = self.apply_transposes(perms)
        if last != first and g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        perms = [tuple(g.get_attribute(n, "perm").ints) for n in [node, next_node]]
        first = list(range(len(perms[0])))
        last = self.apply_transposes(perms)
        if first == last:
            new_node = g.make_node(
                "Identity",
                [node.input[0]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        else:
            new_node = g.make_node(
                "Transpose",
                [node.input[0]],
                next_node.output,
                perm=last,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        new_nodes = [new_node]
        if g.is_used_more_than_once(node.output[0]):
            new_nodes.append(node)
        return new_nodes


class TransposeReshapeTransposePattern(PatternOptimization):
    """
    Swaps Reshape and Transpose in a sequence such as this one:

    ::

        input is 32x4x14x4x14x128

        Transpose(., perm=[0, 1, 3, 2, 4, 5])
        Reshape(., 32x56x56x128)
        Transpose(., perm=[0, 3, 1, 2])

    By:

    ::

        Transpose(., perm=[0, 1, 3, 2, 4, 5])
        Transpose(., perm=[0, 5, 1, 2, 3, 4])
        Reshape(., 32x128x56x56)

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "xts", onnx.TensorProto.FLOAT, shape=(32, 2, 14, 2, 13, 256)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(32, 256, 28, 26))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["shape"],
                value=onh.from_array(
                    np.array([32, 2, 14, 2, 13, 256], dtype=np.int64), name="value"
                ),
            )
        )
        nodes.append(oh.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 3, 1]))
        nodes.append(oh.make_node("Reshape", ["xt", "shape"], ["xts"]))
        nodes.append(oh.make_node("Transpose", ["xts"], ["Y"], perm=[0, 1, 3, 2, 4, 5]))
        outputs.append(
            oh.make_tensor_value_info(
                "xts", onnx.TensorProto.FLOAT, shape=(32, 2, 14, 2, 13, 256)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=(32, 2, 2, 14, 13, 256)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "xts", onnx.TensorProto.FLOAT, shape=(32, 2, 14, 2, 13, 256)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(32, 256, 28, 26))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s6_"],
                value=onh.from_array(
                    np.array([32, 256, 2, 14, 2, 13], dtype=np.int64), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape", ["X", "init7_s6_"], ["TransposeReshapeTransposePattern_xt"]
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["TransposeReshapeTransposePattern_xt"],
                ["xts"],
                perm=[0, 2, 3, 4, 5, 1],
            )
        )
        nodes.append(oh.make_node("Transpose", ["xts"], ["Y"], perm=[0, 1, 3, 2, 4, 5]))
        outputs.append(
            oh.make_tensor_value_info(
                "xts", onnx.TensorProto.FLOAT, shape=(32, 2, 14, 2, 13, 256)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=(32, 2, 2, 14, 13, 256)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        reshape = next_nodes[0]
        if reshape.op_type != "Reshape" or reshape.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(reshape.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        transpose = next_nodes[0]
        if transpose.op_type != "Transpose" or transpose.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        resh_tr = self._new_shape_perm(g, node, reshape, transpose)

        if resh_tr is None:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, reshape, transpose], self.apply)

    def _align_shape(
        self, shape: Tuple[int, ...], new_shape: Tuple[int, ...]
    ) -> Optional[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        mapped: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        i, j = 0, 0
        while i < len(shape) and j < len(new_shape):
            if shape[i] == new_shape[j]:
                mapped.append(((i,), (j,)))
                i += 1
                j += 1
                continue

            ii, jj = [i], [j]
            s1 = shape[i]
            s2 = new_shape[j]
            while s1 != s2 and i < len(shape) and j < len(new_shape):
                if s1 < s2:
                    i += 1
                    assert i < len(shape), f"Unxpected index i={i}, shape={shape}"
                    s1 *= shape[i]
                    ii.append(i)
                else:
                    j += 1
                    assert j < len(new_shape), f"Unxpected index i={j}, shape={new_shape}"
                    s2 *= new_shape[j]
                    jj.append(j)

            if min(len(ii), len(jj)) != 1:
                return None

            mapped.append((tuple(ii), tuple(jj)))
            i += 1
            j += 1

        if i != len(shape) or j != len(new_shape):
            return None
        return mapped

    def _new_shape_perm(
        self,
        g: "GraphBulder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> Optional[Tuple[Tuple[int, ...], List[int], bool]]:
        p1 = list(g.get_attribute(t1_node, "perm").ints)
        p2 = list(g.get_attribute(t2_node, "perm").ints)
        new_shape = g.get_computed_constant(reshape_node.input[1]).tolist()
        if not is_static_shape(new_shape):
            return None
        if -1 in new_shape:
            return None
        if not g.has_shape(reshape_node.input[0]):
            return None
        shape = g.get_shape(reshape_node.input[0])
        mapped = self._align_shape(shape, new_shape)
        if mapped is None:
            return None

        if len(p2) <= len(p1):
            # move the reshape after the next transpose
            if len(mapped) != len(p2):
                return None

            # mapping is done, build new permutation
            new_perm = []
            for p in p2:
                new_perm.extend(mapped[p][0])

            new_reshape = [0 for s in p2]
            for i, p in enumerate(p2):
                new_reshape[i] = new_shape[p]

            return new_perm, new_reshape, True

        # move the reshape before the previous transpose
        if len(mapped) != len(p1):
            return None

        # mapping is done, build new permutation and shape
        rev_p1 = [0 for _ in p1]
        for i, p in enumerate(p1):
            rev_p1[p] = i
        indices = []
        for p in rev_p1:
            indices.extend(mapped[p][1])
        new_reshape = [new_shape[i] for i in indices]
        rev_indices = [0 for _ in indices]
        for i, p in enumerate(indices):
            rev_indices[p] = i
        new_perm = rev_indices

        return new_perm, new_reshape, False

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> List[NodeProto]:
        new_perm, new_shape, after = self._new_shape_perm(g, t1_node, reshape_node, t2_node)
        new_name = g.unique_name(f"{self.__class__.__name__}_{t1_node.output[0]}")
        new_shape_name = g.make_initializer(
            "",
            np.array(new_shape, dtype=np.int64),
            source="TransposeReshapeTransposePattern.apply.new_shape_name",
        )
        if after:
            return [
                t1_node,
                g.make_node(
                    "Transpose",
                    [t1_node.output[0]],
                    [new_name],
                    perm=new_perm,
                    name=f"{self.__class__.__name__}--C--{t2_node.name}",
                    doc_string=t2_node.doc_string,
                ),
                g.make_node(
                    "Reshape",
                    [new_name, new_shape_name],
                    t2_node.output,
                    name=f"{self.__class__.__name__}--D--{reshape_node.name}",
                    doc_string=reshape_node.doc_string,
                ),
            ]

        return [
            g.make_node(
                "Reshape",
                [t1_node.input[0], new_shape_name],
                [new_name],
                name=f"{self.__class__.__name__}--A--{reshape_node.name}",
                doc_string=reshape_node.doc_string,
            ),
            g.make_node(
                "Transpose",
                [new_name],
                [t2_node.input[0]],
                perm=new_perm,
                name=f"{self.__class__.__name__}--B--{t1_node.name}",
                doc_string=t1_node.doc_string,
            ),
            t2_node,
        ]


class TransposeEqualReshapePattern(PatternOptimization):
    """
    Replaces a Transpose by a Reshape when switched dimensions are
    all equal to 1 but one.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(3, 2, 1, 5))
        )
        nodes.append(oh.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1, 3]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(3, 2, 1, 5))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s4_0_1_-1_0"],
                value=onh.from_array(np.array([0, 1, -1, 0], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Reshape", ["X", "init7_s4_0_1_-1_0"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        perms = list(enumerate(g.get_attribute(node, "perm").ints))
        first = None
        for i, p in perms:
            if i != p:
                break
            first = i
        last = None
        for i, p in reversed(perms):
            if i != p:
                break
            last = i
        begin = first + 1 if first is not None else 0
        end = last if last is not None else len(perms)
        shape = g.get_shape(node.input[0])
        not_one = 0
        for i in range(begin, end):
            if shape[i] != 1:
                not_one += 1
        if not_one > 1:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def _make_new_shape(self, input_shape, perm):
        new_shape = []
        for i, p in perm:
            if i == p:
                new_shape.append(0)
            elif input_shape[p] == 1:
                new_shape.append(1)
            elif isinstance(input_shape[p], int):
                new_shape.append(input_shape[p])
            else:
                new_shape.append(-1)
        return new_shape

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        transpose_node: NodeProto,
    ) -> List[NodeProto]:
        new_shape = self._make_new_shape(
            g.get_shape(transpose_node.input[0]),
            list(enumerate(g.get_attribute(transpose_node, "perm").ints)),
        )
        return [
            g.make_node(
                "Reshape",
                [
                    transpose_node.input[0],
                    g.make_initializer(
                        "",
                        np.array(new_shape, dtype=np.int64),
                        source="TransposeEqualReshapePattern.apply.new_shape",
                    ),
                ],
                transpose_node.output,
                name=f"{self.__class__.__name__}--B--{transpose_node.name}",
            )
        ]


class TransposeGatherPattern(PatternOptimization):
    """
    Removes one unnecessary transpose followed by Gather with only one index.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", 16, 80))
        )
        inputs.append(oh.make_tensor_value_info("ind", onnx.TensorProto.INT64, shape=[]))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["ind"],
                value=onh.from_array(np.array(1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Transpose", ["X"], ["xt"], perm=[1, 0, 2, 3]))
        nodes.append(oh.make_node("Gather", ["xt", "ind"], ["Y"], axis=0))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 16, 80))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", 16, 80))
        )
        inputs.append(oh.make_tensor_value_info("ind", onnx.TensorProto.INT64, shape=[]))
        nodes.append(oh.make_node("Gather", ["X", "ind"], ["Y"], axis=1))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 16, 80))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()
        tr_node = g.node_before(node.input[0])
        if not tr_node:
            return self.none(node, inspect.currentframe().f_lineno)
        if tr_node.op_type != "Transpose" or tr_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        rank = g.get_rank(node.input[1])
        if rank != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        perm = tr_node.attribute[0].ints
        axis = node.attribute[0].i if node.attribute else 0
        perm_less = [p for i, p in enumerate(perm) if i != axis]
        if sorted(perm_less) != perm_less:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [tr_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        transpose_node: NodeProto,
        gather_node: NodeProto,
    ) -> List[NodeProto]:
        perm = transpose_node.attribute[0].ints
        axis = gather_node.attribute[0].i if gather_node.attribute else 0
        new_axis = perm[axis]
        new_node = g.make_node(
            "Gather",
            [transpose_node.input[0], gather_node.input[1]],
            gather_node.output,
            axis=new_axis,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )
        if g.is_used_more_than_once(transpose_node.output[0]):
            return [transpose_node, new_node]
        return [new_node]


class SwapUnsqueezeTransposePattern(PatternOptimization):
    """
    Swaps Unsqueeze and Transpose.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        inputs.append(oh.make_tensor_value_info("axes", onnx.TensorProto.INT64, shape=(2,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["axes"],
                value=onh.from_array(np.array([1, 2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Unsqueeze", ["X", "axes"], ["xu"]))
        nodes.append(make_node_extended("Transpose", ["xu"], ["Y"], perm=[0, 2, 1, 4, 3]))
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("e", "f", "g", "h", "i")
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b", "c"))
        )
        inputs.append(oh.make_tensor_value_info("axes", onnx.TensorProto.INT64, shape=(2,)))
        nodes.append(
            make_node_extended(
                "Transpose", ["X"], ["SwapUnsqueezeTransposePattern_Y"], perm=[0, 2, 1]
            )
        )
        nodes.append(
            make_node_extended("Unsqueeze", ["SwapUnsqueezeTransposePattern_Y", "axes"], ["Y"])
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("e", "f", "g", "h", "i")
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_node = g.node_before(node.input[0])
        if not unsq_node:
            return self.none(node, inspect.currentframe().f_lineno)
        if unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(unsq_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(unsq_node.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [unsq_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        unsqueeze_node: NodeProto,
        transpose_node: NodeProto,
    ) -> List[NodeProto]:
        axes = g.get_computed_constant(unsqueeze_node.input[1])
        perm = transpose_node.attribute[0].ints
        if axes.min() < 0:
            axes = (axes + len(perm)) % len(perm)
        set_axes = {int(i) for i in axes}
        permf = [p for p in perm if p not in set_axes]
        iperm = [(p, i) for i, p in enumerate(permf)]
        iperm.sort()
        nperm = [(i, ni) for ni, (_p, i) in enumerate(iperm)]
        nperm.sort()
        new_perm = [_[1] for _ in nperm]

        new_name = g.unique_name(f"{self.__class__.__name__}_{transpose_node.output[0]}")
        new_axes = g.make_initializer(
            "",
            np.array(sorted([perm[a] for a in axes]), dtype=np.int64),
            source=f"{self.__class__.__name__}.apply.new_shape",
        )

        return [
            g.make_node(
                "Transpose",
                [unsqueeze_node.input[0]],
                [new_name],
                perm=new_perm,
                name=f"{self.__class__.__name__}--{transpose_node.name}",
                doc_string=transpose_node.doc_string,
            ),
            g.make_node(
                "Unsqueeze",
                [new_name, new_axes],
                [transpose_node.output[0]],
                name=f"{self.__class__.__name__}--{unsqueeze_node.name}",
                doc_string=unsqueeze_node.doc_string,
            ),
        ]
