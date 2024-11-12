import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SliceSlicePattern(PatternOptimization):
    """
    Merges consecutive slices if axis are disjoints.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Slice" or node.domain != "":
            return self.none()
        before = g.node_before(node.input[0])
        if (
            before is None
            or g.is_used_more_than_once(node.input[0])
            or before.op_type != "Slice"
            or before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        axis2 = None if len(node.input) < 3 else node.input[3]
        axis1 = None if len(before.input) < 3 else before.input[3]
        if axis1 is None or axis2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(axis1) or not g.is_constant(axis2):
            return self.none(node, inspect.currentframe().f_lineno)

        cst1 = g.get_computed_constant(axis1)
        cst2 = g.get_computed_constant(axis2)
        if cst1 is None or cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        set1 = set(map(int, cst1))
        set2 = set(map(int, cst2))
        if set1 & set2:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        before: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        # merges slices

        new_start = g.unique_name(f"{self.__class__.__name__}_{node.input[1]}_start")
        new_end = g.unique_name(f"{self.__class__.__name__}_{node.input[2]}_end")
        new_axis = g.unique_name(f"{self.__class__.__name__}_{node.input[3]}_axis")
        conc = [
            g.make_node(
                "Concat",
                [before.input[1], node.input[1]],
                [new_start],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-start",
            ),
            g.make_node(
                "Concat",
                [before.input[2], node.input[2]],
                [new_end],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-end",
            ),
            g.make_node(
                "Concat",
                [before.input[3], node.input[3]],
                [new_axis],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-axis",
            ),
        ]
        inputs = [before.input[0], new_start, new_end, new_axis]
        if len(node.input) > 4 and len(before.input) > 4:
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [before.input[4], node.input[4]],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)
        elif len(node.input) > 4:
            one = g.make_initializer(
                "", np.array([1], dtype=np.int64), source="SliceSlicePattern.apply.step.1"
            )
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [one, node.input[4]],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)
        elif len(before.input) > 4:
            one = g.make_initializer(
                "", np.array([1], dtype=np.int64), source="SliceSlicePattern.apply.step.2"
            )
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [before.input[4], one],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)

        node = g.make_node(
            "Slice",
            inputs,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [*conc, node]
