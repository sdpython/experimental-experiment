import inspect
import os
import pprint
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..helpers import string_type, string_sig
from ._torch_helper import make_copy
from ._doc_ import TorchOpOverload


class ExportOptions:
    """
    Gathers altogether all the options defining the way to export a model into a graph
    (not onnx).

    :param strict: strict export or not
    :param fallback: fallback to jit
    :param decomposition_table: decomposition_table, a string as well such as default
        to use the default decomposition table returned by
        :func:`get_decomposition_table
        <experimental_experiment.torch_dynamo.get_decomposition_table>`,
        it can ``'all'``, ``'default'`` or a decomposition list
    :param dynamo: to use ``torch._dynamo.export`` instead of :func:`torch.export.export`
    :param tracing: use symbolic tracing
    :param jit: use jit to get a graph then converts it into a fx graph
    :param strategy: to overwrite all the previous parameters with just a value
    :param remove_inplace: remove inplace nodes
    :param aten_as_function: keeps aten function as local function to keep a faithful
        translation of the fx graph.
    :param allow_untyped_output: allows output with no shape and/or no type

    The fallback strategy tries the following in order:

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_interpreter import ExportOptions

        print("-- default fallback")
        pprint.pprint(ExportOptions().get_fallback_options())
        print("-- default fallback with decomposition")
        pprint.pprint(
            ExportOptions(decomposition_table="default").get_fallback_options()
        )

    Most of the models works with strict=True or False and no decompositions.
    But if it contains control flows (test or loop), inplace modifications,
    it may be useful to try different values for strict and to apply decompositions
    ``decomposition_table='default'``. The decompositions removes unused results
    coming from inplace modifications.

    A graph is considered as invalid if decompositions were not run and
    there is one node with no user. This usually indicates one inplace
    operation is still part of the graph.
    """

    _allowed = {
        None: {},
        "none": {},
        "strict": {"strict": True},
        "strict-dec": {"strict": True, "decomposition_table": "default"},
        "strict-decall": {"strict": True, "decomposition_table": "all"},
        "tracing": {"tracing": True},
        "nostrict": {"strict": False},
        "nostrict-dec": {"strict": False, "decomposition_table": "default"},
        "nostrict-decall": {"strict": False, "decomposition_table": "all"},
        "jit": {"jit": True},
        "jit-dec": {"jit": True, "decomposition_table": "default"},
        "jit-decall": {"jit": True, "decomposition_table": "all"},
        "fallback": {"fallback": True},
        "fallback-dec": {"fallback": True, "decomposition_table": "default"},
        "fallback-decall": {"fallback": True, "decomposition_table": "all"},
        "dec": {"decomposition_table": "default"},
        "decall": {"decomposition_table": "all"},
    }

    def __init__(
        self,
        strict: bool = True,
        fallback: bool = False,
        tracing: bool = False,
        jit: bool = False,
        decomposition_table: Optional[
            Union[str, Dict[TorchOpOverload, Callable[..., Any]]]  # noqa: F821
        ] = None,
        strategy: Optional[str] = None,
        dynamo: bool = False,
        aten_as_function: bool = False,
        remove_inplace: bool = True,
        allow_untyped_output: bool = False,
    ):
        self.strict = strict
        self.fallback = fallback
        self.tracing = tracing
        self.decomposition_table = (
            None if decomposition_table in ("none", None) else decomposition_table
        )
        self.dynamo = dynamo
        self.strategy = strategy
        self.jit = jit
        self.aten_as_function = aten_as_function
        self.remove_inplace = remove_inplace
        self.allow_untyped_output = allow_untyped_output

        if strategy is not None:
            assert strategy in self._allowed, (
                f"Unexpected value for strategy={strategy!r}, "
                f"it should be in {sorted(k for k in self._allowed if k is not None)}"
            )
            kwargs = self._allowed[strategy]
            for k, v in kwargs.items():
                setattr(self, k, v)

        assert (
            not self.dynamo or not self.jit
        ), "jit and dynamo cannot be true at the same time"
        assert self.strict or not self.jit, "jit and strict cannot be true at the same time"
        assert (
            self.strict or not self.dynamo
        ), "strict and dynamo cannot be true at the same time"
        assert (
            not tracing or not dynamo
        ), f"Both tracing and dynamo are incompatible options in {self!r}"
        assert (
            not tracing or strict
        ), f"Both tracing and strict=False are incompatible options in {self!r}"

    def __repr__(self) -> str:
        return string_sig(self)

    def get_decomposition_table(
        self,
    ) -> Dict[TorchOpOverload, Callable[..., Any]]:  # noqa: F821
        "Returns the decompisitions table."
        if self.decomposition_table is None:
            return None
        if isinstance(self.decomposition_table, str):
            from ..torch_dynamo import get_decomposition_table_by_name

            return get_decomposition_table_by_name(self.decomposition_table)
        assert isinstance(
            self.decomposition_table, dict
        ), f"Unexpected type {type(self.decomposition_table)} for decomposition_table"
        return self.decomposition_table

    def get_fallback_options(self, kind: Optional[str] = None) -> List["ExportOptions"]:
        """Returns the fallback scenario."""
        if kind is None or kind in ("fallback", "fallback-dec", "fallback-decall"):
            other_dec = None if self.decomposition_table else "default"
            return [
                ExportOptions(strict=True, decomposition_table=self.decomposition_table),
                ExportOptions(strict=False, decomposition_table=self.decomposition_table),
                ExportOptions(strict=True, decomposition_table=other_dec),
                ExportOptions(strict=False, decomposition_table=other_dec),
                ExportOptions(dynamo=True, decomposition_table=self.decomposition_table),
                ExportOptions(dynamo=True, decomposition_table=other_dec),
                ExportOptions(jit=True, decomposition_table=self.decomposition_table),
            ]
        if kind == "strict":
            return [ExportOptions(strict=True), ExportOptions(strict=False)]
        if kind == "nostrict":
            return [ExportOptions(strict=False), ExportOptions(strict=True)]
        if kind in ("jit"):
            return [
                ExportOptions(strict=True),
                ExportOptions(jit=True, decomposition_table=self.decomposition_table),
            ]
        raise AssertionError(f"Unable to return fallback strategy with kind={kind!r}")

    def post_process_exported_program(
        self,
        exported_program: "torch.export.ExportedProgram",  # noqa: F821
        verbose: int = 0,
        print_exported_program: bool = False,
    ) -> "torch.export.ExportedProgram":  # noqa: F821
        """
        Run decompositions, remove inplace operations.
        The graph is modified inplace.
        """
        if self.decomposition_table:
            begin = time.perf_counter()
            dec = apply_decompositions(exported_program, self.decomposition_table)
            if verbose:
                print(
                    f"[ExportOptions.export] done after decomposition "
                    f"in {time.perf_counter() - begin}"
                )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER DECOMPOSITION -- ")
                print(dec)
                print("-- DONE -- ")
            return dec

        if self.remove_inplace:
            self.remove_inplace_nodes(
                exported_program.graph, exported_program=exported_program, verbose=verbose
            )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER REMOVING INLINE -- ")
                print(exported_program)
                print("-- DONE -- ")
        return exported_program

    def export(
        self,
        mod: Any,
        args: Optional[Tuple[Any, ...]],
        kwargs: Optional[Dict[str, Any]],
        tracing_mode: bool,
        dynamic_shapes: Dict,
        same_signature: bool,
        input_names: Optional[List[str]] = None,
        exc: bool = True,
        verbose: int = 0,
    ) -> Union["torch.export.ExportedProgram", "torch.fx.GraphModule"]:  # noqa: F821
        """Exports the model into an exported program."""
        import torch
        from .tracing import CustomTracer

        print_exported_program = os.environ.get("PRINT_EXPORTED_PROGRAM", "0") in (1, "1")

        if self.fallback or self.strategy in {
            "fallback",
            "fallback-dec",
            "fallback-decomposition",
        }:
            self._last_working = None
            if verbose:
                print("[ExportOptions.export] fallback")
            tries = self.get_fallback_options(self.strategy)
            excs = []
            for ion, opt in enumerate(tries):
                if verbose:
                    print(f"[ExportOptions.export] tries {ion+1}/{len(tries)}: {opt}")
                try:
                    res = opt.export(
                        mod,
                        args,
                        kwargs,
                        tracing_mode=tracing_mode,
                        dynamic_shapes=dynamic_shapes,
                        same_signature=same_signature,
                        input_names=input_names,
                        exc=False,
                        verbose=max(verbose - 1, 0),
                    )
                except Exception as e:
                    excs.append((opt, e))
                    if verbose:
                        se = str(e).split("\n", maxsplit=1)[0]
                        print(f"[ExportOptions.export] fails due to {se}")
                    continue

                if isinstance(res, torch.export.ExportedProgram):
                    inplace_nodes = CustomTracer._inplace_nodes(res.graph)
                    if inplace_nodes:
                        # One node has no users, this usually
                        # indicates an inplace modifications.
                        # This is rejected.
                        excs.append(
                            (
                                opt,
                                f"Probable inplace modifications, "
                                f"there are nodes with no users: {inplace_nodes}.",
                            )
                        )
                        if verbose:
                            print(f"[ExportOptions.export] fails due to {excs[-1][-1]}")

                        if not opt.decomposition_table:
                            # We try with decomposition if possible and to save time.
                            if verbose:
                                print(
                                    f"[ExportOptions.export] current decomposition_table="
                                    f"{opt.decomposition_table}, let's try with 'default'"
                                )
                            res = apply_decompositions(res, "default")
                            inplace_nodes = CustomTracer._inplace_nodes(res.graph)
                            if inplace_nodes:
                                # it fails
                                excs.append(
                                    (
                                        opt,
                                        f"Probable inplace modifications, "
                                        f"even after decomposition. "
                                        f"there are nodes with no users: {inplace_nodes}.",
                                    )
                                )
                                if verbose:
                                    print(
                                        f"[ExportOptions.export] fails again with "
                                        f"{excs[-1][-1]}"
                                    )
                                continue
                            opt.decomposition_table = "default"
                        else:
                            continue

                if verbose:
                    print(f"[ExportOptions.export] winning options {opt}")
                self._last_working = opt
                return res

            if exc:
                raise RuntimeError(
                    f"None of the following options {tries} worked, "
                    f"args={string_type(args)}, kwargs={string_type(kwargs)}, "
                    f"exception=\n-----\n{pprint.pformat(excs)}"
                )
            return None

        if verbose:
            print(
                f"[ExportOptions.export] {self!r} - torch.export.export {type(mod).__name__!r}"
            )
            begin = time.perf_counter()

        if self.dynamo:
            # import torch.utils._pytree as pytree
            # flat_args, orig_in_spec = pytree.tree_flatten((args, ))
            # debug: orig_in_spec, type(flat_args), len(flat_args))
            if verbose:
                print("[ExportOptions.export] torch._dynamo.export")
            res = torch._dynamo.export(
                mod,
                aten_graph=True,
                tracing_mode=tracing_mode,
                dynamic_shapes=dynamic_shapes,
                same_signature=same_signature,
                decomposition_table=self.get_decomposition_table(),
                assume_static_by_default=dynamic_shapes is None,
            )(*(args or tuple()), **(kwargs or {}))

            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return res  # _apply_decompositions(res, self.decomposition_table)

        if self.jit:
            if verbose:
                print("[ExportOptions.export] torch.jit.trace")
            from torch._export.converter import TS2EPConverter

            jit_model = torch.jit.trace(
                mod, example_inputs=args, check_trace=False, strict=False
            )
            res = TS2EPConverter(jit_model, args, kwargs).convert()
            dec = apply_decompositions(res, self.decomposition_table)
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return dec

        if self.tracing:
            from .tracing import CustomTracer

            concrete_args = kwargs.copy() if kwargs else {}
            if args:
                sig = inspect.signature(mod.forward)
                for p, a in zip(sig.parameters, args):
                    if a is not None and p not in concrete_args:
                        if isinstance(a, int):
                            # not traceable otherise
                            concrete_args[p] = torch.tensor(a, dtype=torch.int64)
                        elif isinstance(a, float):
                            # not traceable otherise
                            concrete_args[p] = torch.tensor(a, dtype=torch.float32)
                        else:
                            concrete_args[p] = a

            if verbose:
                print(f"[ExportOptions.export] CustomTracer().trace, verbose={verbose}")
                print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
                print(f"[ExportOptions.export] args={string_type(args)}")
                print(f"[ExportOptions.export] kwargs={string_type(kwargs)}")
                print(f"[ExportOptions.export] concrete_args={string_type(concrete_args)}")

            graph = CustomTracer().trace(mod, concrete_args=concrete_args)
            if self.remove_inplace:
                self.remove_inplace_nodes(graph, verbose=verbose)
            gm = torch.fx.GraphModule(mod, graph)
            return gm

        if verbose:
            print(
                f"[ExportOptions.export] torch.export.export "
                f"strict={self.strict}, verbose={verbose}"
            )
            print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
            print(f"[ExportOptions.export] args={string_type(args)}")
            print(f"[ExportOptions.export] kwargs={string_type(kwargs)}")
        if self.strict:
            # torch.export.export may turn Tensor into FakeTensor.
            # We need to make a copy to avoid getting FakeTensor instead
            args0, kwargs0 = args, kwargs
            args = make_copy(args)
            kwargs = make_copy(kwargs)
        if exc:
            exported_program = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=self.strict
            )
        else:
            try:
                exported_program = torch.export.export(
                    mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=self.strict
                )
            except torch._export.verifier.SpecViolationError:
                # see https://github.com/pytorch/pytorch/issues/128394
                if verbose:
                    print("[ExportOptions.export] torch.export._trace._export")
                exported_program = torch.export._trace._export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=dynamic_shapes,
                    pre_dispatch=False,
                    strict=self.strict,
                )
            except torch._dynamo.exc.UserError as e:
                eee = None
                if verbose:
                    print("[ExportOptions.export] torch.export.export")
                try:
                    exported_program = torch.export.export(
                        mod, args, kwargs, strict=self.strict
                    ).graph
                except torch._export.verifier.SpecViolationError as ee:
                    exported_program = None
                    eee = ee
                raise RuntimeError(
                    f"Unable to convert model {type(mod)}, "
                    f"type(args)={type(args)}, type(args[0])="
                    f"{type(args[0]) if isinstance(args, tuple) and args else '?'}, "
                    f"strict={self.strict}, input_names={input_names}\n--\n"
                    f"dynamic_shapes={dynamic_shapes}\n--\ne={e}\n--\neee={eee}"
                    f"\n---exported-program---\n{exported_program}"
                ) from e

        if self.strict:
            # torch.export.export may turn Tensor into FakeTensor.
            # We need to make a copy to avoid getting FakeTensor instead
            args, kwargs = args0, kwargs0
        if exported_program is None:
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return exported_program

        if print_exported_program:
            print("-- EXPORTED PROGRAM AFTER EXPORT -- ")
            print(exported_program)
            print("-- DONE -- ")

        if self.decomposition_table:
            dec = apply_decompositions(exported_program, self.decomposition_table)
            if verbose:
                print(
                    f"[ExportOptions.export] done after decomposition "
                    f"in {time.perf_counter() - begin}"
                )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER DECOMPOSITION -- ")
                print(dec)
                print("-- DONE -- ")
            return dec

        if self.remove_inplace:
            self.remove_inplace_nodes(
                exported_program.graph, exported_program=exported_program, verbose=verbose
            )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER REMOVING INLINE -- ")
                print(exported_program)
                print("-- DONE -- ")

        if verbose:
            print(
                f"[ExportOptions.export] done with no decomposition "
                f"in {time.perf_counter() - begin}"
            )
        return exported_program

    def remove_inplace_nodes(
        self,
        graph: "torch.fx.Graph",  # noqa: F821
        exported_program: Optional["torch.export.ExportedProgram"] = None,  # noqa: F821
        verbose: int = 0,
    ) -> int:
        """
        Post-processing to remove inplace nodes.

        :param graph: graph to modify
        :param exported_program: if available, it is used in the error message
            to make it easier to trace the code source
        :param verbose: verbosity
        :return: number of inplace nodes removed
        """
        from .tracing import CustomTracer

        removed = CustomTracer.remove_unnecessary_slices(graph)
        if removed:
            if verbose:
                print(f"[ExportOptions.export] slices: {removed} slices nodes were removed")
            graph.lint()
        modified = CustomTracer.remove_inplace(
            graph, exported_program=exported_program, verbose=verbose
        )
        if modified:
            if verbose:
                print(
                    f"[ExportOptions.export] inplaces: "
                    f"{modified} inplaced nodes were removed"
                )
            graph.lint()
        return modified


def apply_decompositions(
    exported_program: "torch.export.ExportedProgram", decomposition_table  # noqa: F821
) -> "torch.export.ExportedProgram":  # noqa: F821
    if decomposition_table == "all":
        exported_program = insert_contiguous_between_transpose_and_view(exported_program)
        exported_program = exported_program.run_decompositions()
        return exported_program

    if isinstance(decomposition_table, str):
        from ..torch_dynamo import get_decomposition_table_by_name

        decomposition_table = get_decomposition_table_by_name(decomposition_table)

    if decomposition_table is not None:
        exported_program = insert_contiguous_between_transpose_and_view(exported_program)
        exported_program = exported_program.run_decompositions(decomposition_table)

    return exported_program


def insert_contiguous_between_transpose_and_view(
    exported_program: "torch.export.ExportedProgram",  # noqa: F821
) -> "torch.export.ExportedProgram":  # noqa: F821
    """
    Modifies the module inplace to insert a node 'contiguous' between a node
    'transpose' followed by a node 'view'.
    The modification takes place inplace.
    See issue https://github.com/pytorch/pytorch/issues/136543.
    """
    modified = False
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        if (node.op != "call_method" or node.target != "transpose") and (
            node.op != "call_function"
            or not hasattr(node.target, "name")
            or node.target.name() != "aten::transpose.int"
        ):
            continue
        insert = False
        for user in node.users:
            if (user.op == "call_method" and user.target == "view") or (
                user.op == "call_function"
                and hasattr(node.target, "name")
                and user.target.name() == "aten::view"
            ):
                insert = True
                break
        if not insert:
            continue

        modified = True
        with graph.inserting_after(node):
            new_node = graph.call_method("contiguous", args=(node,))
            node.replace_all_uses_with(new_node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(0, node)
            node.users = {new_node: None}

    if not modified:
        # no rewrite was done.
        return exported_program

    graph.lint()
    return exported_program
