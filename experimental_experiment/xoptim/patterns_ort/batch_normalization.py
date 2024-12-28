import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization


class BatchNormalizationTrainingPattern(PatternOptimization):
    """
    onnxruntime does not support batch normalization with training=1.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none(node, inspect.currentframe().f_lineno)
        if node.op_type != "BatchNormalization" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        training_mode = g.get_attribute(node, "training_mode", exc=False)
        if training_mode is None or training_mode.i == 0:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        bn_node: NodeProto,
    ) -> List[NodeProto]:
        rk = g.get_rank(bn_node.input[0])
        axes = [i for i in range(rk) if i != 1]
        axes_name = g.make_initializer(np.array(axes, dtype=np.int64))
        current_mean = g.unique_name(f"{bn_node.input[0]}_mean")
        mean_node = g.make_node(
            "ReduceMean",
            [bn_node.input[0], axes_name],
            [current_mean],
            name=f"{self.__class__.__name__}--{bn_node.name}",
            keepdim=1,
        )
        centered = g.unique_name(f"{bn_node.input[0]}_center")
        diff_node = g.make_node(
            "Sub",
            [bn_node.input[0], current_mean],
            [centered],
            name=f"{self.__class__.__name__}--{bn_node.name}",
        )
        current_var = g.unique_name(f"{bn_node.input[0]}_var")
        var_node = g.make_node(
            "ReduceMeanSquare",
            [bn_node.input[0], axes_name],
            [current_var],
            name=f"{self.__class__.__name__}--{bn_node.name}",
            keepdim=1,
        )
        atts = g.get_attributes_with_default(bn_node, epsilon=1e-5, momentum=0.9)
        new_nb_node = g.make_node(
            "BatchNormalization",
            [*bn_node.input[:3], current_mean, current_var],
            [bn_node.output[0]],
            training_model=0,
            name=f"{self.__class__.__name__}--{bn_node.name}",
            **atts,
        )

        # running_mean, running_var
        ns = []
        if bn_node.output[1] not in ("", None) and bn_node.output[2] not in ("", None):
            momentum = atts["momentum"]
            dtype = tensor_dtype_to_np_dtype(g.get_type(bn_node.input[0]))
            mom_name = g.make_initializer(np.array([momentum], dtype=dtype))
            mom_1_name = g.make_initializer(np.array([1 - momentum], dtype=dtype))
            p1_mean = g.unique_name(f"{bn_node.output[1]}_m1")
            p1_var = g.unique_name(f"{bn_node.output[2]}_m1")
            p2_mean = g.unique_name(f"{bn_node.output[1]}_m2")
            p2_var = g.unique_name(f"{bn_node.output[2]}_m2")
            ns.extend(
                [
                    g.make_node(
                        "Mul",
                        [bn_node.input[3], mom_name],
                        [p1_mean],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Mul",
                        [bn_node.input[4], mom_name],
                        [p1_var],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Mul",
                        [bn_node.input[3], mom_1_name],
                        [p2_mean],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Mul",
                        [bn_node.input[4], mom_1_name],
                        [p2_var],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Add",
                        [p1_mean, p2_mean],
                        [bn_node.output[1]],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Add",
                        [p1_var, p2_var],
                        [bn_node.output[2]],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                ]
            )

        return [mean_node, diff_node, var_node, new_nb_node, *ns]
