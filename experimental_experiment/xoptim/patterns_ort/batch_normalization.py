import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization


class OrtBatchNormalizationTrainingPattern(PatternOptimization):
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
        axes_name = g.make_initializer(
            "", np.array(axes, dtype=np.int64), source="BatchNormalizationTrainingPattern.1"
        )
        current_mean_axis = g.unique_name(f"{bn_node.input[0]}_mean")
        current_mean = g.unique_name(f"{bn_node.input[0]}_mean")
        mean_node = g.make_node(
            "ReduceMean",
            [bn_node.input[0], axes_name],
            [current_mean_axis],
            name=f"{self.__class__.__name__}--{bn_node.name}",
            keepdims=1,
        )
        mean_node2 = g.make_node(
            "Squeeze",
            [current_mean_axis, axes_name],
            [current_mean],
            name=f"{self.__class__.__name__}--{bn_node.name}",
        )
        centered = g.unique_name(f"{bn_node.input[0]}_center")
        diff_node = g.make_node(
            "Sub",
            [bn_node.input[0], current_mean_axis],
            [centered],
            name=f"{self.__class__.__name__}--{bn_node.name}",
        )
        current_var = g.unique_name(f"{bn_node.input[0]}_var")
        x2 = g.unique_name(f"{bn_node.input[0]}_var2")
        var2_node = g.make_node(
            "Mul",
            [centered, centered],
            [x2],
            name=f"{self.__class__.__name__}--{bn_node.name}",
        )
        var_node = g.make_node(
            "ReduceMean",
            [x2, axes_name],
            [current_var],
            name=f"{self.__class__.__name__}--{bn_node.name}",
            keepdims=0,
        )
        atts = g.get_attributes_with_default(bn_node, epsilon=None, momentum=None)
        new_nb_node = g.make_node(
            "BatchNormalization",
            [*bn_node.input[:3], current_mean, current_var],
            [bn_node.output[0]],
            training_mode=0,
            name=f"{self.__class__.__name__}--{bn_node.name}",
            **atts,
        )

        # running_mean, running_var
        ns = []
        if bn_node.output[1] not in ("", None) and bn_node.output[2] not in ("", None):
            momentum = atts.get(
                "momentum", 0.9
            )  # this value is defined by onnx specifications
            dtype = tensor_dtype_to_np_dtype(g.get_type(bn_node.input[0]))
            mom_name = g.make_initializer(
                "",
                np.array([momentum], dtype=dtype),
                source="BatchNormalizationTrainingPattern.2",
            )
            mom_1_name = g.make_initializer(
                "",
                np.array([1 - momentum], dtype=dtype),
                source="BatchNormalizationTrainingPattern.3",
            )
            p1_mean = g.unique_name(f"{bn_node.output[1]}_m1")
            p1_var = g.unique_name(f"{bn_node.output[2]}_m1")
            p2_mean = g.unique_name(f"{bn_node.output[1]}_m2")
            p2_var = g.unique_name(f"{bn_node.output[2]}_m2")
            same_type = g.get_type(bn_node.output[1]) == g.get_type(bn_node.output[0])
            mean_name, var_name = (
                bn_node.output[1:]
                if same_type
                else (
                    g.unique_name(f"{bn_node.output[1]}_m3"),
                    g.unique_name(f"{bn_node.output[2]}_m3"),
                )
            )
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
                        [current_mean, mom_1_name],
                        [p2_mean],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Mul",
                        [current_var, mom_1_name],
                        [p2_var],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Add",
                        [p1_mean, p2_mean],
                        [mean_name],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                    g.make_node(
                        "Add",
                        [p1_var, p2_var],
                        [var_name],
                        name=f"{self.__class__.__name__}--{bn_node.name}",
                    ),
                ]
            )
            if not same_type:
                itype = g.get_type(bn_node.output[1])
                ns.extend(
                    [
                        g.make_node(
                            "Cast",
                            [mean_name],
                            [bn_node.output[1]],
                            to=itype,
                            name=f"{self.__class__.__name__}--{bn_node.name}",
                        ),
                        g.make_node(
                            "Cast",
                            [var_name],
                            [bn_node.output[2]],
                            to=itype,
                            name=f"{self.__class__.__name__}--{bn_node.name}",
                        ),
                    ]
                )

        return [mean_node, mean_node2, diff_node, var2_node, var_node, new_nb_node, *ns]
