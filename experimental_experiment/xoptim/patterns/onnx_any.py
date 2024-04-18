import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SameChildrenPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.
    """

    @classmethod
    def _cmp(cls, n1: NodeProto, n2: NodeProto) -> bool:
        "Compares two nodes and say if they are the same."
        assert id(n1) != id(n2), f"Two nodes are the same not identical copies {n1}"
        if n1.input != n2.input:
            return False
        if len(n1.output) != len(n2.output):
            return False
        if len(n1.attribute) != len(n2.attribute):
            return False
        for att1, att2 in zip(n1.attribute, n2.attribute):
            if att1.name != att2.name:
                return False
            if att1.type != att2.type:
                return False
            if att1.SerializeToString() != att2.SerializeToString():
                return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) <= 1:
            return self.none()

        if len(next_nodes) == 2:
            n1, n2 = next_nodes
            if n1.op_type != n2.op_type:
                return self.none()
            if not self._cmp(n1, n2):
                return self.none(node, inspect.currentframe().f_lineno)
            nodes = [n1, n2]
        else:
            cp = {}
            for n in next_nodes:
                if n.op_type in cp:
                    cp[n.op_type].append(n)
                else:
                    cp[n.op_type] = [n]
            nodes = []
            for k, v in cp.items():
                if len(v) <= 1:
                    continue
                if len(v) == 2:
                    n1, n2 = v
                    if not self._cmp(n1, n2):
                        continue
                    nodes.extend([n1, n2])
                    continue
                enough = False
                for i in range(0, len(v) - 1):
                    for j in range(i + 1, len(v)):
                        if self._cmp(v[i], v[j]):
                            nodes.extend([v[i], v[j]])
                            enough = True
                            break
                    if enough:
                        break
            if len(nodes) == 0:
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self, g: "GraphBuilder", *nodes: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        """
        The function receives pairs of nodes. We replace every odd node
        by an identity node.
        """
        assert (
            len(nodes) % 2 == 0
        ), f"Expecting an even number of nodes but len(nodes) == {len(nodes)}"
        new_nodes = []
        for i in range(0, len(nodes), 2):
            new_nodes.append(nodes[i])

            for o1, o2 in zip(nodes[i].output, nodes[i + 1].output):
                new_nodes.append(
                    g.make_node(
                        "Identity",
                        [o1],
                        [o2],
                        name=f"{self.__class__.__name__}--{nodes[i+1].name}",
                        doc_string=nodes[i + 1].doc_string,
                    )
                )
        return new_nodes
