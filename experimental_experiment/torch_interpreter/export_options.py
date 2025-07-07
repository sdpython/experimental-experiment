import inspect
import os
import pprint
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from ..helpers import string_sig, get_sig_kwargs
from ._torch_helper import make_copy
from ._doc_ import TorchOpOverload


class ExportOptions:
    """
    Gathers altogether all the options defining the way to export a model into a graph
    (not onnx).

    :param strict: strict export or not, it only applies
        if :func:`torch.export.export` is called
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
        translation of the fx graph, it can also be a set of function name the export
        should export as local function such as
        ``torch.ops.aten.scaled_dot_product_attention``
    :param allow_untyped_output: allows output with no shape and/or no type
    :param save_ep: to save the exported program, it True, it will save the
        graph as well ``<save_ep>.graph``, it dumps them as text,
        if decompositions are enabled, the exported program before them will be saved
        as well
    :param validate_ep: validates the exported program with the given inputs,
        by default the tolerance is ``1e-5``, use a float instead of a boolean
        to change that value

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
        strict: bool = False,  # strict=False is the default with torch>=2.7
        fallback: bool = False,
        tracing: bool = False,
        jit: bool = False,
        decomposition_table: Optional[
            Union[str, Dict[TorchOpOverload, Callable[..., Any]]]  # noqa: F821
        ] = None,
        strategy: Optional[str] = None,
        dynamo: bool = False,
        aten_as_function: Union[bool, Set[Any]] = (
            "aten.scaled_dot_product_attention.default",
            "aten.index_put.default",
            "aten.index_copy.default",
        ),
        remove_inplace: bool = True,
        allow_untyped_output: bool = False,
        save_ep: Optional[str] = None,
        validate_ep: Union[float, bool] = False,
    ):
        self.strict = strict
        self.fallback = fallback
        self.tracing = tracing
        self.save_ep = save_ep
        self.decomposition_table = (
            None if decomposition_table in ("none", None) else decomposition_table
        )
        self.dynamo = dynamo
        self.strategy = strategy
        self.jit = jit
        self.aten_as_function = aten_as_function
        self.remove_inplace = remove_inplace
        self.allow_untyped_output = allow_untyped_output
        self.validate_ep = validate_ep

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
        assert (
            not tracing or not dynamo
        ), f"Both tracing and dynamo are incompatible options in {self!r}"

    def export_as_aten_function(self, aten_name: Any) -> bool:
        if not self.aten_as_function:
            return False
        if isinstance(self.aten_as_function, bool):
            return self.aten_as_function
        if isinstance(aten_name, str):
            return aten_name in self.aten_as_function
        return aten_name in self.aten_as_function or str(aten_name) in self.aten_as_function

    def __repr__(self) -> str:
        return string_sig(self)

    def clone(self, **kwargs) -> "ExportOptions":
        """Makes a copy and updates some of the values."""
        kw = get_sig_kwargs(self)
        kw.update(kwargs)
        return ExportOptions(**kwargs)

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
                self.clone(strict=True, decomposition_table=self.decomposition_table),
                self.clone(strict=False, decomposition_table=self.decomposition_table),
                self.clone(strict=True, decomposition_table=other_dec),
                self.clone(strict=False, decomposition_table=other_dec),
                self.clone(dynamo=True, decomposition_table=self.decomposition_table),
                self.clone(dynamo=True, decomposition_table=other_dec),
                self.clone(jit=True, decomposition_table=self.decomposition_table),
            ]
        if kind == "strict":
            return [self.clone(strict=True), self.clone(strict=False)]
        if kind == "nostrict":
            return [self.clone(strict=False), self.clone(strict=True)]
        if kind in ("jit"):
            return [
                self.clone(strict=True),
                self.clone(jit=True, decomposition_table=self.decomposition_table),
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
            if verbose:
                begin = time.perf_counter()
                print(f"[ExportOptions.export] run decomposition {self.decomposition_table!r}")
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
            if verbose:
                begin = time.perf_counter()
                print("[ExportOptions.export] remove inplace nodes")
            modified = self.remove_inplace_nodes(
                exported_program.graph, exported_program=exported_program, verbose=verbose
            )
            if verbose:
                print(
                    f"[ExportOptions.export] done remove inplace in "
                    f"{time.perf_counter() - begin}, modified={modified}"
                )
            if modified <= -1:
                # We need to run decomposition to fully remove all inplace operations.
                if verbose:
                    begin = time.perf_counter()
                    print(
                        "[ExportOptions.export] use decomposition "
                        "to remove inplace nodes left"
                    )
                exported_program = exported_program.run_decompositions({})
                if verbose:
                    print(
                        f"[ExportOptions.export] done in {time.perf_counter() - begin}, "
                        f"modified={modified}"
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
        from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
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
                    f"None of the following options {tries} worked, args="
                    f"{string_type(args, limit=20)}, kwargs={string_type(kwargs, limit=20)}, "
                    f"exception=\n-----\n{pprint.pformat(excs)}"
                )
            return None

        if verbose:
            print(
                f"[ExportOptions.export] {self!r} - torch.export.export {type(mod).__name__!r}"
            )
            print(f"[ExportOptions.export] aten_as_function={self.aten_as_function!r}")
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

            if self.save_ep:
                with open(f"{self.save_ep}.old_dynamo", "w") as f:
                    f.write(str(res))
                torch.export.save(res, f"{self.save_ep}.old_dynamo.pt2")
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
            if self.save_ep:
                with open(f"{self.save_ep}.jit", "w") as f:
                    f.write(str(res))
                torch.export.save(res, f"{self.save_ep}.jit.pt2")
            dec = apply_decompositions(res, self.decomposition_table)
            if self.save_ep:
                with open(f"{self.save_ep}.jit.decomposed", "w") as f:
                    f.write(str(dec))
                torch.export.save(dec, f"{self.save_ep}.jit.decomposed.pt2")
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
                print(f"[ExportOptions.export] args={string_type(args, limit=20)}")
                print(f"[ExportOptions.export] kwargs={string_type(kwargs, limit=20)}")
                print(
                    f"[ExportOptions.export] concrete_args="
                    f"{string_type(concrete_args, limit=20)}"
                )

            graph = CustomTracer().trace(mod, concrete_args=concrete_args)
            if self.remove_inplace:
                if verbose:
                    print("[ExportOptions.export] remove_inplace_nodes")
                modified = self.remove_inplace_nodes(graph, verbose=verbose)
                if verbose:
                    print(f"[ExportOptions.export] done, modified={modified}")
            if self.save_ep:
                with open(f"{self.save_ep}.tracing", "w") as f:
                    f.write(str(graph))
            gm = torch.fx.GraphModule(mod, graph)
            return gm

        if verbose:
            print(
                f"[ExportOptions.export] torch.export.export "
                f"strict={self.strict}, verbose={verbose}"
            )
            print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
            print(f"[ExportOptions.export] args={string_type(args, limit=20)}")
            print(f"[ExportOptions.export] kwargs={string_type(kwargs, limit=20)}")
        if self.strict:
            # torch.export.export may turn Tensor into FakeTensor.
            # We need to make a copy to avoid getting FakeTensor instead
            args0, kwargs0 = args, kwargs
            args = make_copy(args)
            kwargs = make_copy(kwargs)
        if verbose:
            t0 = time.perf_counter()
            print(f"[ExportOptions.export] export start with strict={self.strict}...")
        if exc:
            exported_program = torch.export.export(
                mod,
                args,
                kwargs,
                dynamic_shapes=use_dyn_not_str(dynamic_shapes),
                strict=self.strict,
            )
        else:
            try:
                exported_program = torch.export.export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=use_dyn_not_str(dynamic_shapes),
                    strict=self.strict,
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

        if verbose:
            print(f"[ExportOptions.export] export done in {time.perf_counter() - t0}")

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
        if self.save_ep:
            with open(f"{self.save_ep}.ep", "w") as f:
                f.write(str(exported_program))
            with open(f"{self.save_ep}.ep.graph", "w") as f:
                f.write(str(exported_program.graph))
            torch.export.save(exported_program, f"{self.save_ep}.ep.pt2")
        if isinstance(self.validate_ep, float) or self.validate_ep:
            self.validate_exported_program(
                mod, exported_program, args, kwargs, verbose=verbose
            )

        exported_program = self.post_process_exported_program(
            exported_program, verbose=verbose, print_exported_program=print_exported_program
        )
        if verbose:
            print(
                f"[ExportOptions.export] done with no decomposition "
                f"in {time.perf_counter() - begin}"
            )
        return exported_program

    def validate_exported_program(
        self, model, exported_program, args, kwargs, verbose: int = 0
    ):
        """Validates the exported program by running the model."""
        from onnx_diagnostic.helpers.torch_helper import torch_deepcopy

        (ar, kws) = torch_deepcopy((args, kwargs))
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] run model with "
                f"args={string_type(args, with_shape=True)} and "
                f"kwargs={string_type(kwargs, with_shape=True)}"
            )
        expected = model(*(ar or []), **(kws or {}))
        (ar, kws) = torch_deepcopy((args, kwargs))
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] run exported_program with "
                f"args={string_type(args, with_shape=True)} and "
                f"kwargs={string_type(kwargs, with_shape=True)}"
            )
        got = exported_program.module()(*(ar or []), **(kws or {}))
        diff = max_diff(expected, got)
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] discrepancies: {string_diff(diff)}"
            )
        atol = self.validate_ep if isinstance(self.validate_ep, float) else 1e-5
        assert diff["abs"] <= atol, (
            f"Discrepancies oberseved between the model and the exported program "
            f"(atol={atol}) diff={string_diff(diff)}"
        )

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
        :return: number of inplace nodes removed or -1 if there are any remaining inplace nodes
        """
        from .tracing import CustomTracer

        removed = CustomTracer.remove_last_unused_inputs(graph)
        if removed:
            if verbose:
                print(f"[ExportOptions.export] unused inputs: {removed} last unused inputs")
            graph.lint()
        removed = CustomTracer.remove_unnecessary_slices(graph)
        if removed:
            if verbose:
                print(f"[ExportOptions.export] slices: {removed} slices nodes were removed")
            graph.lint()
        modified = CustomTracer.remove_inplace(
            graph, exported_program=exported_program, verbose=verbose, exc=False
        )
        if modified < 0:
            return modified
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
