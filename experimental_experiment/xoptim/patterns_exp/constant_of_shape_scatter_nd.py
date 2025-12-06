import inspect
from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from ..patterns_api import MatchResult, PatternOptimization


class ConstantOfShapeScatterNDPattern(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape.

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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "indices", onnx.TensorProto.INT64, shape=("UNKNOWNDIM1", "UNKNOWNDIM2", 1)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=("UNKNOWNDIM",))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "masked_updates",
                onnx.TensorProto.FLOAT,
                shape=("UNKNOWNDIM1^UNKNOWNDIM3", "UNKNOWNDIM2^UNKNOWNDIM4", "UNKNOWNDIM5"),
            )
        )
        nodes.append(
            make_node_extended(
                "ConstantOfShape",
                ["shape"],
                ["data"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "ScatterND", ["data", "indices", "masked_updates"], ["y"], reduction="add"
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM6", "UNKNOWNDIM7")
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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "indices", onnx.TensorProto.INT64, shape=("UNKNOWNDIM1", "UNKNOWNDIM2", 1)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=("UNKNOWNDIM",))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "masked_updates",
                onnx.TensorProto.FLOAT,
                shape=("UNKNOWNDIM1^UNKNOWNDIM3", "UNKNOWNDIM2^UNKNOWNDIM4", "UNKNOWNDIM5"),
            )
        )
        nodes.append(
            make_node_extended(
                "ScatterNDOfShape",
                ["shape", "indices", "masked_updates"],
                ["y"],
                domain="onnx_extended.ortops.optim.cuda",
                strategy="optimize",
                reduction="add",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM6", "UNKNOWNDIM7")
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
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "ScatterND" or node.domain != "":
            return self.none()

        reduction = g.get_attribute(node, "reduction", exc=False)
        if reduction is None or reduction.s == b"none":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_type(node.input[2]):
            itype = g.try_infer_type(node.input[2])
            if itype == 0:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            itype = g.get_type(node.input[2])

        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "ConstantOfShape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        att = g.get_attribute(node_before, "value", False)
        if att is not None:
            arr = to_array(att.t)
            if arr[0] != 0:
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        reduction = g.get_attribute(node, "reduction")
        new_node = g.make_node(
            "ScatterNDOfShape",
            [node_before.input[0], *node.input[1:]],
            node.output,
            strategy="optimize",
            name=f"{self.__class__.__name__}--{node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        new_node.attribute.append(reduction)
        return [new_node]


class MaskedShapeScatterNDPattern(PatternOptimization):
    """
    Replaces Equal, Where, ScatterNDOfShape by MaskedScatterNDOfShape.

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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "updates",
                onnx.TensorProto.FLOAT,
                shape=("UNKNOWNDIM3", "UNKNOWNDIM4", "UNKNOWNDIM5"),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "indices", onnx.TensorProto.INT64, shape=("UNKNOWNDIM1", "UNKNOWNDIM2", 1)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=("UNKNOWNDIM",))
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["mone"],
                value=onh.from_array(np.array([-1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "ScatterNDOfShape",
                ["shape", "indices", "masked_updates"],
                ["y"],
                domain="onnx_extended.ortops.optim.cuda",
                strategy="optimize",
                reduction="add",
            )
        )
        nodes.append(
            make_node_extended(
                "Where", ["masked_indices", "zero", "updates"], ["masked_updates"]
            )
        )
        nodes.append(make_node_extended("Equal", ["indices", "mone"], ["masked_indices"]))
        outputs.append(
            oh.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM6", "UNKNOWNDIM7")
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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "updates",
                onnx.TensorProto.FLOAT,
                shape=("UNKNOWNDIM3", "UNKNOWNDIM4", "UNKNOWNDIM5"),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "indices", onnx.TensorProto.INT64, shape=("UNKNOWNDIM1", "UNKNOWNDIM2", 1)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=("UNKNOWNDIM",))
        )
        nodes.append(
            make_node_extended(
                "MaskedScatterNDOfShape",
                ["shape", "indices", "updates"],
                ["y"],
                domain="onnx_extended.ortops.optim.cuda",
                maskedValue=-1,
                reduction="add",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM6", "UNKNOWNDIM7")
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
        if node.op_type != "ScatterNDOfShape":
            return self.none()

        reduction = g.get_attribute(node, "reduction", exc=False)
        if reduction is None or reduction.s != b"add":
            self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)

        indices = node.input[1]

        where_node = g.node_before(node.input[2])
        if where_node.op_type != "Where" or where_node.domain != "":
            self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(where_node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(where_node.input[1])
        if cst != 0:
            self.none(node, inspect.currentframe().f_lineno)

        equal_node = g.node_before(where_node.input[0])
        if equal_node.op_type != "Equal" or equal_node.domain != "":
            self.none(node, inspect.currentframe().f_lineno)

        indices_again = equal_node.input[0]
        if indices_again != indices:
            self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(equal_node.output[0]):
            self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(equal_node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)

        rank = g.get_rank(indices)
        if rank != 3:
            self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(indices):
            self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(indices)
        if shape[-1] != -1:
            self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        shape_shape = g.get_shape(node.input[0])
        if shape_shape != (2,):
            self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, where_node, equal_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        scatter_node: NodeProto,
        where_node: NodeProto,
        equal_node: NodeProto,
    ) -> List[NodeProto]:
        mask = g.get_constant_scalar(equal_node.input[1])
        assert isinstance(mask, int), f"Unexpected type {type(mask)} for {equal_node.input[1]!r}"
        new_node = g.make_node(
            "MaskedScatterNDOfShape",
            [scatter_node.input[0], scatter_node.input[1], where_node.input[2]],
            scatter_node.output,
            reduction="add",
            maskedValue=int(mask),
            name=f"{self.__class__.__name__}--{scatter_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
