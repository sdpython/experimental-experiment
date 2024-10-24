import pprint
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
        <experimental_experiment.torch_dynamo.get_decomposition_table>`
    :param dynamo: to use :func:`torch._dynamo.export` instead of:func:`torch.export.export`
    :param jit: use jit to get a graph then converts it into a fx graph
    :param strategy: to overwrite all the previous parameters with just a value

    The fallback strategy tries the following in order:

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_interpreter import ExportOptions

        pprint.pprint(ExportOptions().get_fallback_options())
        pprint.pprint(
            ExportOptions(decomposition_table="default").get_fallback_options()
        )
    """

    _allowed = {
        None: {},
        "none": {},
        "nostrict": {"strict": False},
        "jit": {"jit": True},
        "fallback": {"fallback": True},
        "fallback-default": {"fallback": True, "decomposition_table": "default"},
        "default": {"decomposition_table": "default"},
    }

    def __init__(
        self,
        strict: bool = True,
        fallback: bool = False,
        jit: bool = False,
        decomposition_table: Optional[
            Union[str, Dict[TorchOpOverload, Callable[..., Any]]]  # noqa: F821
        ] = None,
        strategy: Optional[str] = None,
        dynamo: bool = False,
    ):
        self.strict = strict
        self.fallback = fallback
        self.decomposition_table = (
            None if decomposition_table in ("none", None) else decomposition_table
        )
        self.dynamo = dynamo
        self.strategy = strategy
        self.jit = jit

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

    def __repr__(self) -> str:
        if self.strategy:
            return f"{self.__class__.__name__}(strategy={self.strategy!r})"
        if self.dynamo:
            return (
                f"{self.__class__.__name__}(strict={self.strict!r}, "
                f"fallback={self.fallback}, "
                f"decomposition_table={self.decomposition_table!r}, "
                f"dynamo={self.dynamo})"
            )
        if self.decomposition_table:
            return (
                f"{self.__class__.__name__}(strict={self.strict!r}, "
                f"fallback={self.fallback}, "
                f"decomposition_table={self.decomposition_table}!r)"
            )
        return f"{self.__class__.__name__}(strict={self.strict!r}, fallback={self.fallback})"

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

    def get_fallback_options(self) -> List["ExportOptions"]:
        """Returns the fallback scenario."""
        return [
            ExportOptions(decomposition_table=self.decomposition_table),
            ExportOptions(strict=False, decomposition_table=self.decomposition_table),
            ExportOptions(dynamo=True, decomposition_table=self.decomposition_table),
            ExportOptions(jit=True, decomposition_table=self.decomposition_table),
        ]

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
    ):
        """Exports the model into an exported program."""
        import torch
        from ..torch_test_helper import string_type

        if self.fallback or self.strategy == "fallback":
            if verbose:
                print("[ExportOptions.export] fallback")
            tries = self.get_fallback_options()
            excs = []
            for opt in tries:
                try:
                    return opt.export(
                        mod,
                        args,
                        kwargs,
                        tracing_mode=tracing_mode,
                        dynamic_shapes=dynamic_shapes,
                        same_signature=same_signature,
                        input_names=input_names,
                        exc=False,
                        verbose=verbose,
                    )
                except Exception as e:
                    excs.append(e)

            if exc:
                raise RuntimeError(
                    f"None of the following options {tries} worked, "
                    f"args={string_type(args)}, kwargs={string_type(kwargs)}, "
                    f"exception=\n-----\n{pprint.pformat(excs)}"
                )
            return None

        if verbose:
            print(f"[ExportOptions.export] {self!r} - torch.export.export {type(mod)}")
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

        if verbose:
            print("[ExportOptions.export] torch.export.export")
            print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
            print(f"[ExportOptions.export] strict={self.strict}")
            print(f"[ExportOptions.export] args={string_type(args)}")
            print(f"[ExportOptions.export] kwargs={string_type(kwargs)}")
            print(f"[ExportOptions.export] verbose={verbose}")
        if exc:
            exported_mod = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=self.strict
            )
        else:
            try:
                exported_mod = torch.export.export(
                    mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=self.strict
                )
            except torch._export.verifier.SpecViolationError:
                # see https://github.com/pytorch/pytorch/issues/128394
                if verbose:
                    print("[ExportOptions.export] torch.export._trace._export")
                exported_mod = torch.export._trace._export(
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
                    exported_mod = torch.export.export(
                        mod, args, kwargs, strict=self.strict
                    ).graph
                except torch._export.verifier.SpecViolationError as ee:
                    exported_mod = None
                    eee = ee
                raise RuntimeError(
                    f"Unable to convert model {type(mod)}, "
                    f"type(args)={type(args)}, type(args[0])="
                    f"{type(args[0]) if isinstance(args, tuple) and args else '?'}, "
                    f"strict={self.strict}, input_names={input_names}\n--\n"
                    f"dynamic_shapes={dynamic_shapes}\n--\ne={e}\n--\neee={eee}"
                    f"\n---exported-program---\n{exported_mod}"
                ) from e

        if exported_mod is None:
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return exported_mod

        if self.decomposition_table:
            dec = apply_decompositions(exported_mod, self.decomposition_table)
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return dec
        if verbose:
            print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
        return exported_mod


def apply_decompositions(
    exported_mod: "torch.export.ExportedProgram", decomposition_table  # noqa: F821
) -> "torch.export.ExportedProgram":  # noqa: F821
    if decomposition_table == "all":
        exported_mod = insert_contiguous_between_transpose_and_view(exported_mod)
        exported_mod = exported_mod.run_decompositions()
        return exported_mod

    if isinstance(decomposition_table, str):
        from ..torch_dynamo import get_decomposition_table_by_name

        decomposition_table = get_decomposition_table_by_name(decomposition_table)

    if decomposition_table is not None:
        exported_mod = insert_contiguous_between_transpose_and_view(exported_mod)
        exported_mod = exported_mod.run_decompositions(decomposition_table)

    return exported_mod


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
