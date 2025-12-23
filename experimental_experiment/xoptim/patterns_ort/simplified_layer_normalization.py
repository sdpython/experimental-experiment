import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers import tensor_dtype_to_np_dtype, from_array_extended
from ..patterns_api import MatchResult, PatternOptimization


class SimplifiedLayerNormalizationPattern(PatternOptimization):
    """
    Fuses the nodes equivalent to SimplifiedLayerNormalization.

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
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "D")))
        inputs.append(oh.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["exp"],
                value=onh.from_array(np.array([2.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["axis"],
                value=onh.from_array(np.array([-1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["eps"],
                value=onh.from_array(
                    np.array([9.999999974752427e-07], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([1.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Pow", ["X", "exp"], ["x2"]))
        nodes.append(oh.make_node("ReduceMean", ["x2", "axis"], ["xr"]))
        nodes.append(oh.make_node("Add", ["xr", "eps"], ["xa"]))
        nodes.append(oh.make_node("Sqrt", ["xa"], ["xq"]))
        nodes.append(oh.make_node("Div", ["one", "xq"], ["Z"]))
        nodes.append(oh.make_node("Mul", ["Z", "X"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", 1)))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "D")))
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
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "D")))
        inputs.append(oh.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(oh.make_node("Shape", ["X"], ["shape-X"]))
        nodes.append(oh.make_node("Gather", ["shape-X", "axis"], ["gather-shape-X"]))
        nodes.append(
            oh.make_node(
                "ConstantOfShape",
                ["gather-shape-X"],
                ["constantofshape-gather-shape-X"],
                value=onh.from_array(np.array([1.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "SimplifiedLayerNormalization",
                ["X", "constantofshape-gather-shape-X"],
                ["Y", "Z"],
                axis=-1,
                epsilon=9.999999974752427e-07,
                stash_type=1,
            )
        )
        outputs.append(oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=("a", 1)))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "D")))
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
        if node.op_type != "ReduceMean" or node.domain != "":
            return self.none()
        if len(node.input) < 2:
            return self.none(node, inspect.currentframe().f_lineno)

        axis = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        assert isinstance(axis, tuple), f"unexpected type {type(axis)} for axis"
        if len(axis) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        node_pow = g.node_before(node.input[0])
        if node_pow is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if node_pow.op_type != "Pow" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node_pow.input[1], 2):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes_add = g.next_nodes(node.output[0])
        if len(nodes_add) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        node_add = nodes_add[0]
        if node_add.op_type != "Add" or node_add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(node_add.input[0]) and not g.is_constant_scalar(
            node_add.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_sqrt = g.next_node(node_add.output[0])
        if node_sqrt.op_type != "Sqrt" or node_sqrt.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        node_reciprocal = g.next_node(node_sqrt.output[0])
        if node_reciprocal.op_type not in ("Reciprocal", "Div") or node_reciprocal.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if node_reciprocal.op_type == "Div":
            if node_reciprocal.input[1] != node_sqrt.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant_scalar(node_reciprocal.input[0], 1):
                return self.none(node, inspect.currentframe().f_lineno)

        node_mul = g.next_node(node_reciprocal.output[0])
        if node_mul.op_type != "Mul" or node_mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            g.is_used_more_than_once(node_pow.output[0])
            or g.is_used_more_than_once(node.output[0])
            or g.is_used_more_than_once(node_add.output[0])
            or g.is_used_more_than_once(node_sqrt.output[0])
        ):
            # intermediate results are used
            return self.none(node, inspect.currentframe().f_lineno)

        mul_i = set(node_mul.input)
        cmp = {node_pow.input[0], node_reciprocal.output[0]}
        if mul_i != cmp:
            # We check the multiplication node takes the output of the div node
            # and the input of the pow node.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node_pow, node, node_add, node_sqrt, node_reciprocal, node_mul]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_pow: NodeProto,
        node_reduce: NodeProto,
        node_add: NodeProto,
        node_sqrt: NodeProto,
        node_reciprocal: NodeProto,
        node_mul: NodeProto,
    ) -> List[NodeProto]:
        nname = node_reduce.name
        nodes = []
        epsilon = g.get_computed_constant(node_add.input[1])
        shape = g.get_shape(node_reduce.input[0]) if g.has_shape(node_reduce.input[0]) else None
        axis = g.get_constant_or_attribute(node_reduce, "axes", input_index=1)[0]
        assert shape is None or axis < len(
            shape
        ), f"axis={axis} and shape={shape} don't match for {node_reduce.input[0]!r}"
        stash_type = g.get_type(node_reduce.input[0])
        dtype = tensor_dtype_to_np_dtype(stash_type)
        if shape is not None and isinstance(shape[axis], int):
            # a constant
            scale = g.make_initializer(
                f"ONES{shape[axis]}",
                np.ones((shape[axis],), dtype=dtype),
                source="SimplifiedLayerNormalizationPattern.apply.scale.1",
            )
        else:
            sh = g.make_node(
                "Shape", [node_pow.input[0]], name=f"{self.__class__.__name__}--{nname}"
            )
            axis_name = g.make_initializer(
                "",
                np.array([axis], dtype=np.int64),
                source="SimplifiedLayerNormalizationPattern.apply.axis",
            )
            ga = g.make_node(
                "Gather",
                [sh.output[0], axis_name],
                name=f"{self.__class__.__name__}--{nname}",
            )
            # sc = g.make_node_check_opset(
            #    "Unsqueeze", [ga.output[0]], axes=[0],
            #       name=f"{self.__class__.__name__}--{nname}"
            # )
            cc = g.make_node(
                "ConstantOfShape",
                [ga.output[0]],
                value=from_array_extended(np.array([1], dtype=dtype)),
                name=f"{self.__class__.__name__}--{nname}",
            )
            scale = cc.output[0]
            nodes.extend([sh, ga, cc])

        layer = g.make_node(
            "SimplifiedLayerNormalization",
            [node_pow.input[0], scale],
            [node_mul.output[0], node_reciprocal.output[0]],
            epsilon=float(epsilon[0] if epsilon.shape else epsilon),
            axis=int(axis),
            stash_type=stash_type,
            name=f"{self.__class__.__name__}--{nname}",
        )

        nodes.append(layer)
        return nodes


class SkipLayerNormalizationPattern(PatternOptimization):
    """
    Replaces the sequence Add + LayerNormalization into SkipLayerNormalization.

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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X2", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        inputs.append(
            oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        inputs.append(
            oh.make_tensor_value_info("scale", onnx.TensorProto.FLOAT16, shape=("c",))
        )
        inputs.append(oh.make_tensor_value_info("bias", onnx.TensorProto.FLOAT16, shape=("c",)))
        nodes.append(oh.make_node("Add", ["X1", "X2"], ["add"]))
        nodes.append(
            oh.make_node("LayerNormalization", ["add", "scale", "bias"], ["Y"], axis=-1)
        )
        outputs.append(
            oh.make_tensor_value_info("add", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X2", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        inputs.append(
            oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        inputs.append(
            oh.make_tensor_value_info("scale", onnx.TensorProto.FLOAT16, shape=("c",))
        )
        inputs.append(oh.make_tensor_value_info("bias", onnx.TensorProto.FLOAT16, shape=("c",)))
        nodes.append(
            oh.make_node(
                "SkipLayerNormalization",
                ["X1", "X2", "scale", "bias"],
                ["Y", "unused", "unused2", "add"],
                domain="com.microsoft",
            )
        )
        outputs.append(
            oh.make_tensor_value_info("add", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, shape=("a", "b", "c"))
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
        if node.op_type != "LayerNormalization" or node.domain != "":
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) not in (2, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) > 1 and (
            not g.has_rank(node.input[1]) or g.get_rank(node.input[1]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) > 2 and (
            not g.has_rank(node.input[2]) or g.get_rank(node.input[2]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(node, "axis", exc=False)
        axis = 0 if axis is None else axis.i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before is None or before.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(before.input[1]) or g.get_rank(before.input[1]) not in (2, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        nodes = [before, node]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        norm_node: NodeProto,
    ) -> List[NodeProto]:
        atts = []
        epsilon = g.get_attribute(norm_node, "epsilon", exc=False)
        if epsilon:
            atts.append(epsilon)
        u1 = (
            g.unique_name("unused")
            if len(norm_node.output) < 2 or not norm_node.output[1]
            else norm_node.output[1]
        )
        u2 = (
            g.unique_name("unused")
            if len(norm_node.output) < 3 or not norm_node.output[2]
            else norm_node.output[2]
        )
        layer = g.make_node(
            "SkipLayerNormalization",
            [*add_node.input, *norm_node.input[1:]],
            [norm_node.output[0], u1, u2, add_node.output[0]],
            name=f"{self.__class__.__name__}--{norm_node.name}",
            domain="com.microsoft",
        )
        if atts:
            layer.attribute.extend(atts)
        return [layer]


class SkipSimplifiedLayerNormalizationPattern(PatternOptimization):
    """
    Replaces the sequence Add + SimplifiedLayerNormalization
    by SkipSimplifiedLayerNormalization.

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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("scale", onnx.TensorProto.FLOAT, shape=(192,)))
        inputs.append(
            oh.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["scale"],
                value=onh.from_array(
                    np.array(
                        [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(oh.make_node("Add", ["X", "skip"], ["xs"]))
        nodes.append(
            oh.make_node(
                "SimplifiedLayerNormalization",
                ["xs", "scale"],
                ["ym"],
                axis=-1,
                epsilon=0.10000000149011612,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "ym", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
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
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("scale", onnx.TensorProto.FLOAT, shape=(192,)))
        inputs.append(
            oh.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        nodes.append(
            oh.make_node(
                "SkipSimplifiedLayerNormalization",
                ["X", "skip", "scale"],
                ["ym", "", "", "xs"],
                domain="com.microsoft",
                epsilon=0.10000000149011612,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "ym", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "SimplifiedLayerNormalization" or node.domain != "":
            return self.none()
        if len(node.output) > 1 and (len(node.output) != 2 or g.is_used(node.output[1])):
            # second output is used
            return self.none(node, inspect.currentframe().f_lineno)

        # axis
        axis = g.get_attribute(node, "axis", exc=False)
        axis = -1 if axis is None else axis.i
        if axis != -1 and (
            not g.has_rank(node.input[0]) or axis != g.get_rank(node.input[0]) - 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # stash_type
        stash_type = g.get_attribute(node, "stash_type", exc=False)
        stash_type = TensorProto.FLOAT if stash_type is None else stash_type.i
        if stash_type != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        add = g.node_before(node.input[0])
        if add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.has_shape(add.input[0])
            or not g.has_shape(add.input[1])
            or g.get_shape(add.input[0]) != g.get_shape(add.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [add, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_add: NodeProto,
        node_simplified: NodeProto,
    ) -> List[NodeProto]:
        layer = g.make_node(
            "SkipSimplifiedLayerNormalization",
            [*node_add.input, *node_simplified.input[1:]],
            [node_simplified.output[0], "", "", *node_add.output],
            name=f"{self.__class__.__name__}--{node_simplified.name}",
            domain="com.microsoft",
        )
        layer.attribute.extend(
            att for att in node_simplified.attribute if att.name not in {"axis", "stash_type"}
        )
        return [layer]


class SkipSimplifiedLayerNormalizationMulPattern(PatternOptimization):
    """
    Replaces the sequence SkipSimplifiedLayerNormalization + Mul
    by SkipSimplifiedLayerNormalization.

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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, shape=(192,))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["scale"],
                value=onh.from_array(
                    np.array(
                        [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["weights"],
                value=onh.from_array(
                    np.array(
                        [
                            1000.0,
                            1000.0051879882812,
                            1000.0104370117188,
                            1000.015625,
                            1000.0208129882812,
                            1000.0260620117188,
                            1000.03125,
                            1000.0364379882812,
                            1000.0416870117188,
                            1000.046875,
                            1000.0520629882812,
                            1000.0573120117188,
                            1000.0625,
                            1000.0676879882812,
                            1000.0729370117188,
                            1000.078125,
                            1000.0833129882812,
                            1000.0885620117188,
                            1000.09375,
                            1000.0989379882812,
                            1000.1041870117188,
                            1000.109375,
                            1000.1145629882812,
                            1000.1198120117188,
                            1000.125,
                            1000.1301879882812,
                            1000.1354370117188,
                            1000.140625,
                            1000.1458129882812,
                            1000.1510620117188,
                            1000.15625,
                            1000.1614379882812,
                            1000.1666870117188,
                            1000.171875,
                            1000.1770629882812,
                            1000.1823120117188,
                            1000.1875,
                            1000.1926879882812,
                            1000.1979370117188,
                            1000.203125,
                            1000.2083129882812,
                            1000.2135620117188,
                            1000.21875,
                            1000.2239379882812,
                            1000.2291870117188,
                            1000.234375,
                            1000.2395629882812,
                            1000.2448120117188,
                            1000.25,
                            1000.2551879882812,
                            1000.2604370117188,
                            1000.265625,
                            1000.2708129882812,
                            1000.2760620117188,
                            1000.28125,
                            1000.2864379882812,
                            1000.2916870117188,
                            1000.296875,
                            1000.3020629882812,
                            1000.3073120117188,
                            1000.3125,
                            1000.3176879882812,
                            1000.3229370117188,
                            1000.328125,
                            1000.3333129882812,
                            1000.3385620117188,
                            1000.34375,
                            1000.3489379882812,
                            1000.3541870117188,
                            1000.359375,
                            1000.3645629882812,
                            1000.3698120117188,
                            1000.375,
                            1000.3801879882812,
                            1000.3854370117188,
                            1000.390625,
                            1000.3958129882812,
                            1000.4010620117188,
                            1000.40625,
                            1000.4114379882812,
                            1000.4166870117188,
                            1000.421875,
                            1000.4270629882812,
                            1000.4323120117188,
                            1000.4375,
                            1000.4426879882812,
                            1000.4479370117188,
                            1000.453125,
                            1000.4583129882812,
                            1000.4635620117188,
                            1000.46875,
                            1000.4739379882812,
                            1000.4791870117188,
                            1000.484375,
                            1000.4895629882812,
                            1000.4948120117188,
                            1000.5,
                            1000.5051879882812,
                            1000.5104370117188,
                            1000.515625,
                            1000.5208129882812,
                            1000.5260620117188,
                            1000.53125,
                            1000.5364379882812,
                            1000.5416870117188,
                            1000.546875,
                            1000.5520629882812,
                            1000.5573120117188,
                            1000.5625,
                            1000.5676879882812,
                            1000.5729370117188,
                            1000.578125,
                            1000.5833129882812,
                            1000.5885620117188,
                            1000.59375,
                            1000.5989379882812,
                            1000.6041870117188,
                            1000.609375,
                            1000.6145629882812,
                            1000.6198120117188,
                            1000.625,
                            1000.6301879882812,
                            1000.6354370117188,
                            1000.640625,
                            1000.6458129882812,
                            1000.6510620117188,
                            1000.65625,
                            1000.6614379882812,
                            1000.6666870117188,
                            1000.671875,
                            1000.6770629882812,
                            1000.6823120117188,
                            1000.6875,
                            1000.6926879882812,
                            1000.6979370117188,
                            1000.703125,
                            1000.7083129882812,
                            1000.7135620117188,
                            1000.71875,
                            1000.7239379882812,
                            1000.7291870117188,
                            1000.734375,
                            1000.7395629882812,
                            1000.7448120117188,
                            1000.75,
                            1000.7551879882812,
                            1000.7604370117188,
                            1000.765625,
                            1000.7708129882812,
                            1000.7760620117188,
                            1000.78125,
                            1000.7864379882812,
                            1000.7916870117188,
                            1000.796875,
                            1000.8020629882812,
                            1000.8073120117188,
                            1000.8125,
                            1000.8176879882812,
                            1000.8229370117188,
                            1000.828125,
                            1000.8333129882812,
                            1000.8385620117188,
                            1000.84375,
                            1000.8489379882812,
                            1000.8541870117188,
                            1000.859375,
                            1000.8645629882812,
                            1000.8698120117188,
                            1000.875,
                            1000.8801879882812,
                            1000.8854370117188,
                            1000.890625,
                            1000.8958129882812,
                            1000.9010620117188,
                            1000.90625,
                            1000.9114379882812,
                            1000.9166870117188,
                            1000.921875,
                            1000.9270629882812,
                            1000.9323120117188,
                            1000.9375,
                            1000.9426879882812,
                            1000.9479370117188,
                            1000.953125,
                            1000.9583129882812,
                            1000.9635620117188,
                            1000.96875,
                            1000.9739379882812,
                            1000.9791870117188,
                            1000.984375,
                            1000.9895629882812,
                            1000.9948120117188,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "SkipSimplifiedLayerNormalization",
                ["X", "skip", "scale"],
                ["ym", "", "", "xs"],
                domain="com.microsoft",
                epsilon=0.10000000149011612,
            )
        )
        nodes.append(oh.make_node("Mul", ["ym", "weights"], ["a"]))
        outputs.append(oh.make_tensor_value_info("", onnx.TensorProto.UNDEFINED, []))
        outputs.append(
            oh.make_tensor_value_info(
                "a", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
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
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, shape=(192,))
        )
        nodes.append(
            oh.make_node(
                "SkipSimplifiedLayerNormalization",
                ["X", "skip", "weights"],
                ["a", "", "", "xs"],
                domain="com.microsoft",
                epsilon=0.10000000149011612,
            )
        )
        outputs.append(oh.make_tensor_value_info("", onnx.TensorProto.UNDEFINED, []))
        outputs.append(
            oh.make_tensor_value_info(
                "a", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type, node.domain) != ("SkipSimplifiedLayerNormalization", "com.microsoft"):
            return self.none()
        if (len(node.output) > 1 and node.output[1]) or (len(node.output) > 2 and node.output[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        mul_nodes = g.next_nodes(node.output[0])
        if len(mul_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = mul_nodes[0]
        if mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        index_cst = 1 if mul_node.input[0] == node.output[0] else 0
        if not g.has_shape(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape_renamed(node.input[2]) != g.get_shape_renamed(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(mul_node.input[index_cst]):
            # not supported yet
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        skip_simp_node: NodeProto,
        mul_node: NodeProto,
    ) -> List[NodeProto]:
        index_cst = 1 if mul_node.input[0] == skip_simp_node.output[0] else 0
        cst_skip = g.get_computed_constant(skip_simp_node.input[2])
        if cst_skip.min() == cst_skip.max() == 1:
            cst_name = mul_node.input[index_cst]
        else:
            cst2 = g.get_computed_constant(mul_node.input[index_cst])
            if cst2.min() == cst2.max() == 1:
                cst_name = skip_simp_node.input[2]
            else:
                cst2 = g.get_computed_constant(mul_node.input[index_cst])
                new_cst = cst_skip * cst2
                cst_name = g.make_initializer(
                    f"{skip_simp_node.input[2]}__{mul_node.input[index_cst]}",
                    new_cst,
                    source=f"{self.__class__.__name__}.gamma",
                )

        new_node = g.make_node(
            skip_simp_node.op_type,
            [*skip_simp_node.input[:2], cst_name],
            [mul_node.output[0], *skip_simp_node.output[1:]],
            name=f"{self.__class__.__name__}--{skip_simp_node.name}",
            domain=skip_simp_node.domain,
        )
        if skip_simp_node.attribute:
            new_node.attribute.extend(skip_simp_node.attribute)
        return [new_node]


class SimplifiedLayerNormalizationMulPattern(PatternOptimization):
    """
    Replaces the sequence SimplifiedLayerNormalization + Mul
    by SimplifiedLayerNormalization.

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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, shape=(192,))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["scale"],
                value=onh.from_array(
                    np.array(
                        [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["weights"],
                value=onh.from_array(
                    np.array(
                        [
                            1000.0,
                            1000.0051879882812,
                            1000.0104370117188,
                            1000.015625,
                            1000.0208129882812,
                            1000.0260620117188,
                            1000.03125,
                            1000.0364379882812,
                            1000.0416870117188,
                            1000.046875,
                            1000.0520629882812,
                            1000.0573120117188,
                            1000.0625,
                            1000.0676879882812,
                            1000.0729370117188,
                            1000.078125,
                            1000.0833129882812,
                            1000.0885620117188,
                            1000.09375,
                            1000.0989379882812,
                            1000.1041870117188,
                            1000.109375,
                            1000.1145629882812,
                            1000.1198120117188,
                            1000.125,
                            1000.1301879882812,
                            1000.1354370117188,
                            1000.140625,
                            1000.1458129882812,
                            1000.1510620117188,
                            1000.15625,
                            1000.1614379882812,
                            1000.1666870117188,
                            1000.171875,
                            1000.1770629882812,
                            1000.1823120117188,
                            1000.1875,
                            1000.1926879882812,
                            1000.1979370117188,
                            1000.203125,
                            1000.2083129882812,
                            1000.2135620117188,
                            1000.21875,
                            1000.2239379882812,
                            1000.2291870117188,
                            1000.234375,
                            1000.2395629882812,
                            1000.2448120117188,
                            1000.25,
                            1000.2551879882812,
                            1000.2604370117188,
                            1000.265625,
                            1000.2708129882812,
                            1000.2760620117188,
                            1000.28125,
                            1000.2864379882812,
                            1000.2916870117188,
                            1000.296875,
                            1000.3020629882812,
                            1000.3073120117188,
                            1000.3125,
                            1000.3176879882812,
                            1000.3229370117188,
                            1000.328125,
                            1000.3333129882812,
                            1000.3385620117188,
                            1000.34375,
                            1000.3489379882812,
                            1000.3541870117188,
                            1000.359375,
                            1000.3645629882812,
                            1000.3698120117188,
                            1000.375,
                            1000.3801879882812,
                            1000.3854370117188,
                            1000.390625,
                            1000.3958129882812,
                            1000.4010620117188,
                            1000.40625,
                            1000.4114379882812,
                            1000.4166870117188,
                            1000.421875,
                            1000.4270629882812,
                            1000.4323120117188,
                            1000.4375,
                            1000.4426879882812,
                            1000.4479370117188,
                            1000.453125,
                            1000.4583129882812,
                            1000.4635620117188,
                            1000.46875,
                            1000.4739379882812,
                            1000.4791870117188,
                            1000.484375,
                            1000.4895629882812,
                            1000.4948120117188,
                            1000.5,
                            1000.5051879882812,
                            1000.5104370117188,
                            1000.515625,
                            1000.5208129882812,
                            1000.5260620117188,
                            1000.53125,
                            1000.5364379882812,
                            1000.5416870117188,
                            1000.546875,
                            1000.5520629882812,
                            1000.5573120117188,
                            1000.5625,
                            1000.5676879882812,
                            1000.5729370117188,
                            1000.578125,
                            1000.5833129882812,
                            1000.5885620117188,
                            1000.59375,
                            1000.5989379882812,
                            1000.6041870117188,
                            1000.609375,
                            1000.6145629882812,
                            1000.6198120117188,
                            1000.625,
                            1000.6301879882812,
                            1000.6354370117188,
                            1000.640625,
                            1000.6458129882812,
                            1000.6510620117188,
                            1000.65625,
                            1000.6614379882812,
                            1000.6666870117188,
                            1000.671875,
                            1000.6770629882812,
                            1000.6823120117188,
                            1000.6875,
                            1000.6926879882812,
                            1000.6979370117188,
                            1000.703125,
                            1000.7083129882812,
                            1000.7135620117188,
                            1000.71875,
                            1000.7239379882812,
                            1000.7291870117188,
                            1000.734375,
                            1000.7395629882812,
                            1000.7448120117188,
                            1000.75,
                            1000.7551879882812,
                            1000.7604370117188,
                            1000.765625,
                            1000.7708129882812,
                            1000.7760620117188,
                            1000.78125,
                            1000.7864379882812,
                            1000.7916870117188,
                            1000.796875,
                            1000.8020629882812,
                            1000.8073120117188,
                            1000.8125,
                            1000.8176879882812,
                            1000.8229370117188,
                            1000.828125,
                            1000.8333129882812,
                            1000.8385620117188,
                            1000.84375,
                            1000.8489379882812,
                            1000.8541870117188,
                            1000.859375,
                            1000.8645629882812,
                            1000.8698120117188,
                            1000.875,
                            1000.8801879882812,
                            1000.8854370117188,
                            1000.890625,
                            1000.8958129882812,
                            1000.9010620117188,
                            1000.90625,
                            1000.9114379882812,
                            1000.9166870117188,
                            1000.921875,
                            1000.9270629882812,
                            1000.9323120117188,
                            1000.9375,
                            1000.9426879882812,
                            1000.9479370117188,
                            1000.953125,
                            1000.9583129882812,
                            1000.9635620117188,
                            1000.96875,
                            1000.9739379882812,
                            1000.9791870117188,
                            1000.984375,
                            1000.9895629882812,
                            1000.9948120117188,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "SimplifiedLayerNormalization",
                ["xs", "scale"],
                ["ym"],
                axis=-1,
                epsilon=0.10000000149011612,
            )
        )
        nodes.append(oh.make_node("Mul", ["ym", "weights"], ["a"]))
        outputs.append(oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, shape=[]))
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
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "xs", onnx.TensorProto.FLOAT, shape=("batch", "cache", 192)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, shape=(192,))
        )
        nodes.append(
            oh.make_node(
                "SimplifiedLayerNormalization",
                ["xs", "weights"],
                ["a"],
                axis=-1,
                epsilon=0.10000000149011612,
            )
        )
        outputs.append(oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, shape=[]))
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
        if (node.op_type, node.domain) != ("SimplifiedLayerNormalization", ""):
            return self.none()
        if len(node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst_skip = g.get_computed_constant(node.input[1])
        if cst_skip is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        mul_nodes = g.next_nodes(node.output[0])
        if len(mul_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = mul_nodes[0]
        if mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        index_cst = 1 if mul_node.input[0] == node.output[0] else 0
        if not g.has_shape(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape_renamed(node.input[1]) != g.get_shape_renamed(mul_node.input[index_cst]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(mul_node.input[index_cst]):
            # not supported yet
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        simp_node: NodeProto,
        mul_node: NodeProto,
    ) -> List[NodeProto]:
        index_cst = 1 if mul_node.input[0] == simp_node.output[0] else 0
        cst_skip = g.get_computed_constant(simp_node.input[1])
        if cst_skip.min() == cst_skip.max() == 1:
            cst_name = mul_node.input[index_cst]
        else:
            cst2 = g.get_computed_constant(mul_node.input[index_cst])
            if cst2.min() == cst2.max() == 1:
                cst_name = simp_node.input[1]
            else:
                cst2 = g.get_computed_constant(mul_node.input[index_cst])
                new_cst = cst_skip * cst2
                cst_name = g.make_initializer(
                    f"{simp_node.input[1]}__{mul_node.input[index_cst]}",
                    new_cst,
                    source=f"{self.__class__.__name__}.gamma",
                )

        new_node = g.make_node(
            simp_node.op_type,
            [simp_node.input[0], cst_name],
            [mul_node.output[0], *simp_node.output[1:]],
            name=f"{self.__class__.__name__}--{simp_node.name}",
            domain=simp_node.domain,
        )
        if simp_node.attribute:
            new_node.attribute.extend(simp_node.attribute)
        return [new_node]
