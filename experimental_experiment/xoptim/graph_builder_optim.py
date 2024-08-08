import os
import pprint
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from onnx import AttributeProto, NodeProto, TensorProto
from onnx.shape_inference import infer_shapes
import onnx.helper as oh
import onnx.numpy_helper as onh
from ..xbuilder._onnx_helper import enumerate_subgraphs
from ..xbuilder.type_inference import infer_types
from .patterns_api import MatchResult, PatternOptimization
from .patterns import get_default_patterns


def _count(matches):
    stats = {}
    for n in matches:
        cl = n[0].__class__.__name__
        if cl in stats:
            stats[cl] += 1
        else:
            stats[cl] = 1
    return ", ".join([f"{v}*{k}" for k, v in stats.items()])


class GraphBuilderPatternOptimization:
    """
    Implements optimization after the conversion is done.
    The differences between the two models can be display with a
    command line such as:

    ::

        python -m onnx_array_api compare -m1 <model.onnx> -m2 <optimized.onnx> -m nodes -c 80

    This class assumes a pattern cannot reuse an existing name.

    :param builder: GraphBuilder
    :param patterns: list of patterns to apply
    :param recursive: goes through subgraphs
    :param verifies: verifies the model but it takes time
    :param verbose: verbosity
    :param dump_applied_patterns: dump applied patterns in a folder,
        the users can check every pattern dumped as a :epkg:`FunctionProto`
    :param processor: optimization should be made for this processor
        or this list of processors (comma separated value)
    """

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        patterns: Optional[List[PatternOptimization]] = None,
        recursive: bool = False,
        verifies: bool = False,
        verbose: int = 0,
        dump_applied_patterns: Optional[str] = None,
        processor: str = "CPU",
    ):
        assert processor in {
            "CUDA",
            "CPU",
            "CPU,CUDA",
        }, (
            f"Unknown processor {processor!r}, "
            f"if should be string with comma separated value"
        )
        self.builder = builder
        self.verbose = max(verbose, int(os.environ.get("LOG_PATTERN_OPTIMIZE", "0")))
        self.patterns = patterns or get_default_patterns(self.verbose)
        self.recursive = recursive
        self.verifies = verifies
        self.dump_applied_patterns = dump_applied_patterns
        self.processor = processor
        self._build()
        # This assume a name is given once and
        # no constant can replace an existing one.
        # _build method should not change it.
        self._cache_computed_constant = {}

    def has_processor(self, processor: str) -> bool:
        """
        Checks the process is on the list of used processors.
        """
        return processor in self.processor

    @property
    def nodes(self) -> List[NodeProto]:
        "property"
        return self.builder.nodes

    @property
    def input_names(self) -> List[str]:
        "property"
        return self.builder.input_names

    @property
    def inputs(self) -> List[Any]:
        "property"
        return self.builder.inputs

    @property
    def output_names(self) -> List[str]:
        "property"
        return self.builder.output_names

    @property
    def outputs(self) -> List[Any]:
        "property"
        return self.builder.outputs

    @property
    def opsets(self):
        "property"
        return self.builder.opsets

    def iter_nodes(self) -> Iterator:
        "iterator"
        for node in self.builder.nodes:
            yield node

    def _build(self):
        """
        Builds successor and predecessor.
        """
        self.positions_ = {}
        self.nodes_ = {}
        self.outputs_ = {o.name for o in self.builder.outputs}
        for i, node in enumerate(self.builder.nodes):
            key = id(node)
            self.nodes_[key] = node
            self.positions_[key] = i

        self.set_output_names_ = set(self.builder.output_names)
        self.predecessors_ = {}
        self.successors_ = {}
        successors_id = {}
        self.used_ = set()
        for k, v in self.nodes_.items():
            assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for node {k}"
            for o in v.output:
                self.predecessors_[o] = k
            for i in v.input:
                if i not in self.successors_:
                    self.successors_[i] = []
                    successors_id[i] = set()
                if id(k) not in successors_id[i]:
                    # This test avoids the same successor to appear twice if one node
                    # consumes twice the same node.
                    self.successors_[i].append(k)
                    successors_id[i].add(id(k))

            for sub in enumerate_subgraphs(v):
                g = sub[-1]
                sub_knowns = set()
                for n in g.input:
                    sub_knowns.add(n.name)
                for n in g.initializer:
                    sub_knowns.add(n.name)
                for n in g.sparse_initializer:
                    sub_knowns.add(n.name)
                for n in g.node:
                    for i in n.input:
                        if i not in sub_knowns:
                            # an input coming from the parent
                            self.used_.add(i)
                    for i in n.output:
                        sub_knowns.add(i)

    def get_position(self, node: NodeProto) -> int:
        return self.positions_[id(node)]

    def is_used_by_subgraph(self, name: str) -> bool:
        """
        Tells if a result is used by a subgraphs.
        """
        return name in self.used_

    def is_output(self, name: str) -> bool:
        """
        Tells if a result is an output.
        """
        return name in self.outputs_

    def is_used(self, name: str) -> bool:
        """
        Tells if a result is used or not,
        including as an output of the graph.
        """
        if name in self.successors_:
            return True
        if name in name in self.set_output_names_:
            return True
        if name in self.used_:
            return True
        return False

    def is_used_more_than_once(self, name: str) -> bool:
        """
        Tells if a result is used more than once in the current graph or in a subgraph
        or if it is an output.
        """
        if self.is_used_by_subgraph(name):
            return True
        if self.is_output(name):
            return True
        suc = self.successors_[name]
        return len(suc) > 1

    def is_used_only_by(self, name, *nodes: List[NodeProto]) -> bool:
        """
        Tells if a result is only used by a specific set of nodes.
        """
        next_nodes = self.next_nodes(name)
        allowed = set(id(n) for n in nodes)
        return all(map(lambda n: id(n) in allowed, next_nodes))

    def is_constant(self, name: str) -> bool:
        """
        Tells if a result is a constant.
        """
        return self.builder.is_constant(name)

    def is_constant_scalar(self, name: str, value: Optional[Any] = None) -> bool:
        """
        Tells if a constant is a scalar

        :param name: name
        :param value: value to compare to if specified
        :return: boolean
        """
        if not self.is_constant(name):
            return False
        cst_shape = self.get_constant_shape(name, exc=False)
        if cst_shape is None or cst_shape not in (tuple(), (1,)):
            return False
        cst = self.get_computed_constant(name)
        if hasattr(cst, "numpy"):
            # This could fail xith bfloat16, ...
            cst = cst.detach().cpu().numpy()
        assert isinstance(
            cst, np.ndarray
        ), f"Unexpected type for constant {name}!r, type is {type(cst)}"
        shape = cst.shape
        if shape not in (tuple(), (1,)):
            return False
        if value is None:
            return True
        if cst.shape == (1,):
            return all(cst == value)
        return cst == value

    def get_constant_shape(
        self, name: str, exc: bool = True
    ) -> Optional[Tuple[int, ...]]:
        """
        Returns the shape of a constant.

        :param name: name
        :param exc: raises an exception is not possible
        :return: shape
        """
        if name in self._cache_computed_constant:
            return self._cache_computed_constant[name].shape
        if name in self.builder.initializers_dict:
            proto = self.builder.initializers_dict[name]
        elif name in self.builder.constants_:
            proto = self.builder.constants_[name]
        elif self.is_constant(name):
            cst = self.get_computed_constant(name)
            return cst.shape
        else:
            if exc:
                raise AssertionError(
                    f"Unable to retrieve initializer or constant for {name!r}, "
                    f"is_constant={self.is_constant(name)}"
                )
            return None

        if isinstance(proto, TensorProto):
            return tuple(proto.dims)
        if isinstance(proto, NodeProto) and proto.domain == "":
            if proto.op_type == "Cast":
                if self.is_constant(proto.output[0]) and not self.is_constant(
                    proto.input[0]
                ):
                    if exc:
                        raise AssertionError(
                            f"Incompatibilities, output is constant "
                            f"when input is not in node {proto}."
                        )
                    return None
                return self.get_constant_shape(proto.input[0], exc=exc)
            if proto.op_type == "Constant":
                assert (
                    len(proto.attribute) == 1
                ), f"Unexpected number of attribute for node={proto}"
                for att in proto.attribute:
                    if att.name == "value":
                        return tuple(proto.value.dims)
                    if att.name in {"value_float", "value_int"}:
                        return tuple()
                raise AssertionError(
                    f"Unable to retrieve shape for name={name!r} (type is NodeProto), "
                    f"node.op_type={proto.op_type!r}, "
                    f"attributes={[att.name for att in proto.attribute]}."
                )
            if self.is_constant(name):
                cst = self.get_computed_constant(name)
                return None if cst is None else cst.shape
            if exc:
                raise AssertionError(
                    f"Unable to retrieve shape for name={name!r} "
                    f"bash and node {proto.op_type!r}"
                    # f"{self.builder.get_debug_msg()}"
                )
            return None
        if hasattr(proto, "shape"):
            return proto.shape
        if exc:
            raise AssertionError(
                f"Unable to retrieve shape for name={name!r} and type {type(proto)}"
            )
        return None

    def get_constant_scalar(self, name: str) -> Union[int, float]:
        """
        Returns a scalar as a constant.

        :param name: name
        :return: int or float
        """
        cst = self.get_computed_constant(name)
        if hasattr(cst, "numpy"):
            # This could fail xith bfloat16, ...
            cst = cst.detach().cpu().numpy()
        assert isinstance(
            cst, np.ndarray
        ), f"Unexpected type for constant {name}!r, type is {type(cst)}"
        assert cst.shape in (
            tuple(),
            (1,),
        ), f"Unexpected shape {cst.shape} for constant {name!r}"
        shape = cst.shape
        value = cst[0] if shape == (1,) else cst
        if value.dtype in {
            np.float32,
            np.float16,
            np.float64,
            np.dtype("float32"),
            np.dtype("float16"),
            np.dtype("float64"),
        }:
            return float(value)
        return int(value)

    def get_computed_constant(
        self, name: str, statistics: Optional[List[str]] = None
    ) -> Any:
        """
        Returns the value for the constant `name`.
        """
        if name in self._cache_computed_constant:
            value = self._cache_computed_constant[name]
        else:
            value = self.builder.get_constant(name, computed_value=True)
            if value is not None:
                self._cache_computed_constant[name] = value
        if statistics is None:
            return value
        stats = []
        for st in statistics:
            key = name, st
            if key in self._cache_computed_constant:
                stat = self._cache_computed_constant[key]
            else:
                if st == "min":
                    stat = value.min()
                elif st == "max":
                    stat = value.max()
                else:
                    raise RuntimeError(f"Unknown statistics {st!r} for {name!r}.")
                self._cache_computed_constant[key] = stat
            stats.append(stat)
        return stats

    def get_attribute(
        self, node: NodeProto, att_name: str, exc: bool = True
    ) -> Optional[AttributeProto]:
        """
        Returns an attribute for a node.
        """
        return self.builder.get_attribute(node, att_name, exc=exc)

    def get_attributes_with_default(
        self, node: NodeProto, **default_values
    ) -> Dict[str, Any]:
        """
        Returns integer or float values for attributes.
        """
        return self.builder.get_attributes_with_default(node, **default_values)

    def get_axis(self, node: NodeProto, default_axis: Optional[int] = None) -> int:
        """
        Retrieves the axis for many operators.
        """
        att = self.get_attribute(node, "axis", exc=False)
        if att is None:
            assert (
                default_axis is not None
            ), f"Node {node.op_type} has no axis and no default value."
            return default_axis
        return att.i

    def get_constant_or_attribute(
        self,
        node: NodeProto,
        attribute: str,
        input_index: int,
        cvt: Optional[Callable] = None,
    ) -> Any:
        """
        Returns an input or the value of an attribute.
        Some attributes became inputs in more recent opsets.
        The function checks both.

        :param node: node
        :param attribute: attribute name
        :param input_index: input index
        :param cvt: if not None, called this conversion function before
            returning the result
        :return: value
        """
        found = None
        for att in node.attribute:
            if att.name == attribute:
                found = att
        assert found is None, (
            f"get_constant_or_attribute not implemented "
            f"for attribute={attribute!r} and node={node}."
        )
        assert input_index < len(
            node.input
        ), f"Input {input_index} does not exist in node {node}."
        val = self.get_computed_constant(node.input[input_index])
        if cvt is None:
            return val
        try:
            return cvt(val)
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Unable to convert val={val} with cvt={cvt}") from e

    def has_type(self, name: str) -> bool:
        """
        Tells if a result has a type.
        """
        return self.builder.has_type(name)

    def get_type(self, name: str) -> int:
        """
        Returns the type of a result.
        """
        return self.builder.get_type(name)

    def has_rank(self, name: str) -> int:
        """
        Tells if a result has a rank.
        """
        return self.builder.has_rank(name)

    def get_rank(self, name: str) -> int:
        """
        Returns the rank of a result.
        """
        return self.builder.get_rank(name)

    def has_shape(self, name: str) -> bool:
        """
        Tells if a result has a shape.
        """
        return self.builder.has_shape(name)

    def get_shape(self, name: str) -> int:
        """
        Returns the shape of a result.
        """
        return self.builder.get_shape(name)

    def node_before(self, name: str) -> NodeProto:
        """
        Returns the node producing this output.
        Returns None if it is an input or an initializer.
        """
        if name not in self.predecessors_:
            return None
        predecessor = self.predecessors_[name]
        return self.nodes_[predecessor]

    def next_node(self, name: str) -> NodeProto:
        """
        Returns the next node if it is unique, otherwise fails.
        """
        res = self.next_nodes(name)
        assert len(res) == 1, f"Unexpected number of successors {len(res)} for {name!r}"
        return res[0]

    def next_nodes(self, name: str) -> List[NodeProto]:
        """
        Returns the node consuming the given results.
        """
        if name not in self.successors_:
            return []
        return [self.nodes_[i] for i in self.successors_[name]]

    def try_infer_type(self, name: str, exc: bool = False) -> int:
        """
        Tries to infer the type of a result.

        :param name: name of the result for which to infer the type
        :param exc: if True, raises an exception if something goes wrong
        :return: type
        """
        if self.has_type(name):
            it = self.get_type(name)
            if exc and it == 0:
                raise RuntimeError(
                    f"Unable to guess type for {name!r}, "
                    f"knowns types are {pprint.pformat(self.builder._known_types)}"
                )
            return it

        assert (
            name not in self.builder.initializers_dict
        ), f"name {name!r} has no type but it is an initializer"
        assert name not in self.builder.input_names, (
            f"name {name!r} has no type but it is an input, "
            f"known_types={pprint.pformat(self.builder._known_types)}"
        )
        node = self.node_before(name)
        input_types = [
            (self.get_type(i) if self.has_type(i) else 0) for i in node.input
        ]
        output_type = infer_types(node, input_types, name, exc=exc)
        if output_type > 0:
            return output_type

        # second try with more depth
        input_types = [self.try_infer_type(i, exc=exc) for i in node.input]
        output_type = infer_types(node, input_types, name, exc=exc)
        if output_type > 0:
            return output_type

        # no luck
        if exc:
            raise RuntimeError(
                f"Unable to guess type for {name!r}, "
                f"knowns types are {pprint.pformat(self.builder._known_types)}"
            )
        return 0

    def try_infer_shape(self, name: str, exc: bool = False) -> int:
        """
        Tries to infer the type of a result.

        :param name: name of the result for which to infer the type
        :param exc: if True, raises an exception if something goes wrong
        :return: type
        """
        if self.has_shape(name):
            return self.get_shape(name)
        if exc:
            raise RuntimeError(
                f"Unable to guess shape for {name!r}, "
                f"knowns shapes are {pprint.pformat(self.builder._known_shapes)}"
            )
        return None

    @property
    def main_opset(self):
        "Returns the opset for the main domain (assuming it is used)."
        return self.builder.opsets[""]

    def make_initializer(
        self, name: str, value: Any, external: bool = False, msg: str = ""
    ) -> str:
        new_name = self.builder.make_initializer(
            name, value, external=external, msg=msg
        )
        return new_name

    def unique_name(self, prefix: str) -> str:
        "Returns a unique name."
        return self.builder.unique_name(prefix)

    def make_node_check_opset(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates a node without adding it to the graph but
        adapt for some known operators changing over
        multiple opets.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: outputs names, if one integer, creates n unique names,
            if str, creates one unique names, if a list, use the name
        :param domain: node domain
        :param attributes: list of attributes
        :param name: node name
        :param kwargs: other attributes
        :return: a node
        """
        assert domain == "", f"The method only supports the main domain not {domain!r}"
        if op_type in {"Squeeze", "Unsqueeze"}:
            if self.builder.main_opset < 13:
                assert (
                    len(inputs) == 1
                ), f"axis must be given as an attribute for {op_type!r}"
                return self.make_node(
                    op_type,
                    inputs,
                    outputs,
                    domain=domain,
                    attributes=attributes,
                    name=name,
                    **kwargs,
                )
            if len(inputs) == 1 and "axes" in kwargs:
                axes = kwargs["axes"]
                axes_name = self.make_initializer("", np.array([axes], dtype=np.int64))
                inputs.append(axes_name)
                del kwargs["axes"]
            return self.make_node(
                op_type,
                inputs,
                outputs,
                domain=domain,
                attributes=attributes,
                name=name,
                **kwargs,
            )

        raise RuntimeError(f"Operator {op_type!r} not supported yet.")

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> NodeProto:
        """
        Creates a node without adding it to the graph.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: outputs names, if one integer, creates n unique names,
            if str, creates one unique names, if a list, use the name
        :param domain: node domain
        :param attributes: list of attributes
        :param name: node name
        :param kwargs: other attributes
        :return: a node
        """
        assert name is not None and not name.startswith("None"), (
            f"It is good practice to give every node a name so that is "
            f"easier to see where this node is created but name={name!r} "
            f"and op_type={op_type!r}."
        )
        name = self.builder.unique_node_name(name)
        if isinstance(outputs, int):
            if outputs == 1:
                outputs = [self.unique_name(f"{op_type.lower()}-{inputs[0]}")]
            else:
                outputs = [
                    self.unique_name(f"{op_type.lower()}-{inputs[0]}-{i}")
                    for i in range(outputs)
                ]
        elif isinstance(outputs, str):
            outputs = [self.unique_name(outputs)]
        proto = oh.make_node(
            op_type,
            inputs,
            outputs,
            domain=domain,
            name=name,
            **kwargs,
        )

        if all(self.is_constant(i) for i in inputs):
            for o in outputs:
                self.builder.constants_[o] = proto
            proto.doc_string += ":constant-5:"

        assert len(outputs) == len(set(outputs)) or "" in outputs, (
            f"Repeated outputs for node {op_type}({', '.join(inputs)}) -> "
            f"{', '.join(outputs)}"
        )

        if attributes:
            proto.attribute.extend(attributes)
        return proto

    def apply_match(self, match: MatchResult) -> List[NodeProto]:
        """
        Applies one match.
        Returns the new nodes.
        """
        idn = [id(n) for n in match.nodes if n is not None]

        assert all(
            map(lambda i: i in self.nodes_, idn)
        ), f"One node in {idn} is not referenced"

        positions = {id(n): i for i, n in enumerate(self.builder.nodes)}

        assert all(
            map(lambda i: i in positions, idn)
        ), f"One node in {idn} is not referenced"

        removed = [positions[i] for i in idn]
        position_insert = (
            None if match.insert_at is None else positions[id(match.insert_at)]
        )
        new_nodes = match.apply(self, *match.nodes)

        if self.verbose >= 10:
            print(f"[GraphBuilderPatternOptimization.apply_match] {match}")
            for node in match.nodes:
                if node is None:
                    continue
                print(f"  - {node.op_type}: {node.input} -> {node.output}")
            for node in new_nodes:
                if node is None:
                    continue
                print(f"  + {node.op_type}: {node.input} -> {node.output}")

        self.builder.insert_and_remove_nodes(position_insert, new_nodes, removed)
        if self.verbose >= 10:
            print(f"[GraphBuilderPatternOptimization.apply_match] {match} applied.")
        if self.dump_applied_patterns:
            self._save_pattern_as_proto(self.dump_applied_patterns, match, new_nodes)
        return new_nodes

    def _to_cstop(self, init: Any, name: Optional[str] = None) -> NodeProto:
        if isinstance(init, NodeProto):
            assert (
                name is None or init.output[0] == name
            ), f"Name mismatch {name!r} != {init.output[0]!r}"
            return init
        if isinstance(init, TensorProto):
            assert (
                name is None or init.name == name
            ), f"Name mismatch {name!r} != {init.name!r}"
            return oh.make_node("Constant", [], [init.name], value=init)
        if isinstance(init, np.ndarray):
            return self._to_cstop(onh.from_array(init, name=name))
        import torch

        if isinstance(init, torch.Tensor):
            return self._to_cstop(init.detach().cpu().numpy(), name=name)
        raise AssertionError(f"Unexpected type {type(init)}")

    def _save_pattern_as_proto(
        self, folder: str, match: MatchResult, new_nodes: List[NodeProto]
    ):
        assert isinstance(folder, str), f"Unexpected type {type(folder)} for folder."
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        name = f"{match.pattern.__class__.__name__}_0.onnx"
        fullname = os.path.join(folder, name)
        n = 0
        while os.path.exists(fullname):
            n += 1
            name = f"{match.pattern.__class__.__name__}_{n}.onnx"
            fullname = os.path.join(folder, name)

        if self.verbose >= 10:
            print(
                f"[GraphBuilderPatternOptimization._save_pattern_as_proto] save {fullname!r}"
            )

        unique_names = set()
        for node in match.nodes:
            if node is None:
                continue
            unique_names |= set(node.input)
            unique_names |= set(node.output)

        new_initializers = {}
        input_names = set()
        output_names = set()
        for node in new_nodes:
            if node is None:
                continue
            for i in node.input:
                if i in unique_names:
                    input_names.add(i)
                elif i in self.builder.initializers_dict:
                    new_initializers[i] = self.builder.initializers_dict[i]
            for o in node.output:
                if o in unique_names:
                    output_names.add(o)

        old_initializers = {}
        for node in match.nodes:
            if node is None:
                continue
            for i in node.input:
                if i in self.builder.initializers_dict:
                    old_initializers[i] = self.builder.initializers_dict[i]

        new_init_nodes = [
            self._to_cstop(v, name=k) for k, v in new_initializers.items()
        ]
        old_init_nodes = [
            self._to_cstop(v, name=k) for k, v in old_initializers.items()
        ]

        fproto = oh.make_function(
            domain="pattern",
            fname=match.pattern.__class__.__name__,
            inputs=list(input_names),
            outputs=list(output_names),
            nodes=old_init_nodes + [n for n in match.nodes if n is not None],
            opset_imports=[
                oh.make_opsetid(k, v) for k, v in self.builder.opsets.items()
            ],
        )

        fproto_apply = oh.make_function(
            "pattern",
            match.pattern.__class__.__name__,
            list(input_names),
            list(output_names),
            new_init_nodes + [n for n in new_nodes if n is not None],
            opset_imports=[
                oh.make_opsetid(k, v) for k, v in self.builder.opsets.items()
            ],
        )

        def _sh(n):
            if self.builder.has_shape(n):
                return self.builder.get_shape(n)
            if self.builder.has_rank(n):
                return [None] * self.builder.get_rank(n)
            return None

        inputs = [
            oh.make_tensor_value_info(n, self.builder.get_type(n), _sh(n))
            for n in fproto.input
        ]
        outputs = [
            oh.make_tensor_value_info(n, self.builder.get_type(n), _sh(n))
            for n in fproto.output
        ]

        model = oh.make_model(
            oh.make_graph(fproto.node, "pattern", inputs, outputs),
            opset_imports=fproto.opset_import,
        )

        model_apply = oh.make_model(
            oh.make_graph(fproto_apply.node, "pattern", inputs, outputs),
            opset_imports=fproto_apply.opset_import,
        )
        if self.builder.ir_version:
            model.ir_version = self.builder.ir_version
            model_apply.ir_version = self.builder.ir_version

        with open(fullname, "wb") as f:
            f.write(model.SerializeToString())
        if self.verbose >= 10:
            print(
                f"[GraphBuilderPatternOptimization._save_pattern_as_proto] "
                f"saved {fullname!r}"
            )

        name = f"{match.pattern.__class__.__name__}_{n}_apply.onnx"
        fullname = os.path.join(folder, name)
        with open(fullname, "wb") as f:
            f.write(model_apply.SerializeToString())
        if self.verbose >= 10:
            print(
                f"[GraphBuilderPatternOptimization._save_pattern_as_proto] "
                f"saved {fullname!r}"
            )

    def _check_graph(self, statistics, step, iteration, code, verifies):
        begin = time.perf_counter()
        assert (
            len(self.builder.nodes) > 0
        ), f"The onnx model is empty (step {step}, no node)"
        known = set(n.name for n in self.builder.inputs)
        known |= set(self.builder.initializers_dict)
        for p, node in enumerate(self.builder.nodes):
            assert (
                node.domain in self.opsets
            ), f"domain {node.domain!r} is not registered in {self.opsets}"
            for i in node.input:
                if i == "":
                    continue
                assert i in known, (
                    f"Unknown input {i!r}, step {step!r} at position {p} "
                    f"in node {node.op_type} "
                    f"[{node.name}]: {node.input} -> {node.output}"
                )
            known |= set(node.output)

            if verifies:
                if (
                    node.op_type in {"MatMul", "Gemm", "FusedMatMul"}
                    and self.builder.has_shape(node.input[0])
                    and self.builder.has_shape(node.input[1])
                ):
                    sh1 = self.builder.get_shape(node.input[0])[-2:]
                    sh2 = self.builder.get_shape(node.input[1])[-2:]
                    tA = self.builder.get_attribute(node, "transA", exc=False)
                    tB = self.builder.get_attribute(node, "transB", exc=False)
                    tA = 0 if tA is None or tA.i == 0 else 1
                    tB = 0 if tB is None or tB.i == 0 else 1
                    if tA:
                        sh1 = (sh1[1], sh1[0])
                    if tB:
                        sh2 = (sh2[1], sh2[0])
                    assert (
                        type(sh1[-1]) != type(sh2[0]) or sh1[-1] == sh2[0]  # noqa: E721
                    ), (
                        f"Node {node.op_type!r}, inputs={node.input}, "
                        f"shape1={self.builder.get_shape(node.input[0])}, "
                        f"shape2={self.builder.get_shape(node.input[1])}, "
                        f"tA={tA}, tB={tB}."
                    )

        for o in self.builder.outputs:
            assert o.name in known, f"Unknown output {o.name!r}, step {step!r}"
        statistics.append(
            dict(
                pattern=f"check_pattern_{code}",
                time_in=time.perf_counter() - begin,
                iteration=iteration,
            )
        )
        if verifies:
            onx = self.builder.to_onnx(optimize=False)
            new_shapes = infer_shapes(onx)
            for val in new_shapes.graph.value_info:
                itype = val.type.tensor_type.elem_type
                shape = tuple(
                    d.dim_param if d.dim_param else d.dim_value
                    for d in val.type.tensor_type.shape.dim
                )
                assert self.builder.has_name(val.name), f"name {val.name!r} is missing"
                assert (
                    not self.builder.has_type(val.name)
                    or self.builder.get_type(val.name) == itype
                ), (
                    f"Result {val.name!r} has type {itype} but the builder "
                    f"assumes it is {self.builder.get_type(val.name)}"
                )
                assert (
                    not self.builder.has_shape(val.name)
                    or self.builder.get_shape(val.name) == shape
                ), (
                    f"Result {val.name!r} has shape {shape} but the builder "
                    f"assumes it is {self.builder.get_shape(val.name)}"
                )

            # from onnxruntime import InferenceSession
            # InferenceSession(
            #     onx.SerializeToString(),
            #     providers=["CPUExecutionProvider"],
            # )

    def do_not_remove(self, node: NodeProto) -> bool:
        """Tells if a node can be removed."""
        return self.builder.do_not_remove(node)

    def optimize(
        self, max_iter=-1, remove_identity: bool = True, stop_after: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Optimizes the based on the given list of patterns.

        :param max_iter: maximum number of iterations
        :param remove_identity: remove identity nodes, it is better to keep it True,
            not doing it might prevent other patterns to find a set of nodes to optimize
        :param sopt_after: stop after this number of replacements (to debug),
            -1 not to stop
        :return: the method returns informations about the applied processes.

        The algorithm runs multiple iteration until the graph is not evolving
        or `max_iter` is reached. By default, it is equal to the number of nodes.
        An iteration is:

        ::

            matches = []

            builds all successors and predecessors

            for all patterns P:

                for all nodes n:

                    r = p.match(n)
                    if r:
                        if no node already scheduled to be rewritten by another match:
                            matches.append(r)

                for all matches r:
                    apply the match r

        This algorithm may apply more than one rewriting at each iteration
        but it guarantees the local structure when applying the rewriting was
        not altered by another one.
        """
        assert (
            not self.recursive
        ), "GraphBuilderPatternOptimization.optimize does not implement recursivity"
        continue_optimization = True
        if max_iter == -1:
            max_iter = len(self.builder.nodes)
        priorities = list(sorted(set(p.priority for p in self.patterns)))
        assert priorities, "list of priority is null."
        if self.verbose > 0:
            print(
                f"[GraphBuilderPatternOptimization.optimize] start with "
                f"{len(self.builder.nodes)} nodes and {len(self.patterns)} patterns, "
                f"priorities={priorities}"
            )
            for i, (pp, _, pattern) in enumerate(
                sorted((p.priority, repr(p), p) for p in self.patterns)
            ):
                print(
                    f"[GraphBuilderPatternOptimization.optimize] "
                    f"use pattern {i+1:3d}/{len(self.patterns)} - P{pp} - {pattern!r}"
                )

        begin_all = time.perf_counter()
        statistics = []

        n_applied = 0
        last_it = 0
        current_priority_index = 0
        for it in range(max_iter):
            if not continue_optimization:
                break
            if self.verbose > 0:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] iteration {it}: "
                    f"{len(self.builder.nodes)} nodes, "
                    f"priority={priorities[current_priority_index]}"
                )

            # detects patterns

            found = False
            marked = set()
            matches = []
            durations = {}
            for pattern in self.patterns:
                if not continue_optimization:
                    break
                if pattern.priority > priorities[current_priority_index]:
                    # skipping that pattern
                    continue
                begin = time.perf_counter()
                before = len(matches)

                # loop over the nodes
                for match in pattern.enumerate_matches(self):

                    # bypass this node if the name contains some specific name
                    fail_match = False
                    for n in match.nodes:
                        if n and self.do_not_remove(n):
                            fail_match = True
                            break
                    if fail_match:
                        continue

                    # checks that a node is not already part of another pattern
                    bypass = False
                    for n in match.nodes:
                        if n is None:
                            continue
                        if id(n) in marked:
                            # a node is already marked for replacements
                            bypass = True
                            break
                    if bypass:
                        if self.verbose >= 9:
                            print(
                                f"[{self.__class__.__name__}.match] OVERLAP "
                                f"match={match} #marked: {len(marked)})"
                            )
                        continue
                    for n in match.nodes:
                        if n is None:
                            continue
                        marked.add(id(n))
                    found = True
                    if self.verbose > 2:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] "
                            f"match={match}"
                        )
                    matches.append((pattern, match))
                    if stop_after > 0 and len(matches) + n_applied >= stop_after:
                        continue_optimization = False
                        if self.verbose > 0:
                            print(
                                f"[GraphBuilderPatternOptimization.optimize] "
                                f"stop after with "
                                f"{len(matches)} as stop_after={stop_after} "
                                f"and n_applied={n_applied}"
                            )
                        break

                d = time.perf_counter() - begin
                statistics.append(
                    dict(
                        pattern=f"match_{pattern}",
                        iteration=it,
                        instances=len(matches) - before,
                        time_in=d,
                        match_index=len(matches),
                    )
                )
                durations[pattern.__class__.__name__] = (
                    durations.get(pattern.__class__.__name__, 0) + d
                )

            if self.verbose > 0 and matches:
                if durations:
                    rev = max([(v, k) for k, v in durations.items()])
                    revs = f"{rev[-1]}:{rev[0]:.3f}"
                    if len(matches) == 1:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] applies "
                            f"{len(matches)} matches, [0]={str(matches[0][-1])} - "
                            f"time={sum(durations.values()):.3f} | max_time={revs}"
                        )
                    else:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] applies "
                            f"{len(matches)} matches, {_count(matches)} - "
                            f"time={sum(durations.values()):.3f} | max_time={revs}"
                        )
                elif len(matches) == 1:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] applies "
                        f"{len(matches)} matches, [0]={str(matches[0][-1])}"
                    )
                else:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] applies "
                        f"{len(matches)} matches, {_count(matches)}"
                    )

            # applies patterns (they must be disjoined)

            added_types = set()
            n_added = 0
            n_removed = 0
            for im, (pattern, match) in enumerate(matches):

                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] "
                        f"apply {match.to_string(short=False)}"
                    )

                begin = time.perf_counter()
                added_nodes = self.apply_match(match)
                added_types |= set(n.op_type for n in added_nodes)

                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] - add "
                        f"{[n.op_type for n in added_nodes]}"
                    )
                add = len(added_nodes)
                added_outputs = set()
                for n in added_nodes:
                    added_outputs |= set(n.output)

                rem = len([n for n in match.nodes if n is not None])
                removed_outputs = set()
                for n in match.nodes:
                    if n is None:
                        continue
                    removed_outputs |= set(n.output)

                full_removed = set(i for i in removed_outputs if i not in added_outputs)
                for i in full_removed:
                    assert not self.is_output(i), (
                        f"Output {i!r} must not be removed, added_outputs={added_outputs},"
                        f"removed_outputs={removed_outputs}"
                    )

                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] done "
                        f"{match}: -{rem} +{add} nodes"
                    )
                    if full_removed and self.verbose > 4:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] "
                            f"removed outputs {full_removed}"
                        )

                obs = dict(
                    pattern=f"apply_{pattern}",
                    added=add,
                    removed=rem,
                    iteration=it,
                    match_index=im,
                    instances=1,
                    time_in=time.perf_counter() - begin,
                )
                statistics.append(obs)
                self._check_graph(statistics, str(match), it, "A", self.verifies)

                n_added += add
                n_removed += rem
                n_applied += 1

            if self.verbose > 2:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] done all: "
                    f"-{n_removed} +{n_added} nodes"
                )

            if remove_identity and (it < 3 or "Identity" in added_types):
                # remove unnecessary identity nodes
                begin = time.perf_counter()
                id_removed, id_added = self.builder.remove_identity_nodes()
                statistics.append(
                    dict(
                        pattern="remove_identity_nodes",
                        iteration=it,
                        added=id_added,
                        removed=id_removed,
                        time_in=time.perf_counter() - begin,
                    )
                )
                self._check_graph(statistics, "remove_identity", it, "B", self.verifies)

            # rebuild the graph structure

            begin = time.perf_counter()
            self._build()
            statistics.append(
                dict(
                    pattern="build_for_pattern",
                    iteration=it,
                    time_in=time.perf_counter() - begin,
                )
            )

            # next iteration

            last_it = it + 1
            if not found:
                # No match, increase the priority.
                current_priority_index += 1
                if current_priority_index >= len(priorities):
                    # There is priority left to explore.
                    break
                if self.verbose > 0:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] increase priority "
                        f"to {priorities[current_priority_index]}"
                    )

        if self.verbose > 0:
            duration = time.perf_counter() - begin_all
            print(
                f"[GraphBuilderPatternOptimization.optimize] "
                f"done after {last_it} iterations with "
                f"{len(self.builder.nodes)} nodes in {duration:.3f}"
            )
            if self.verbose > 1:
                msg = self.builder._compile_statistics(statistics)
                print(msg)

        return statistics
