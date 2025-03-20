import contextlib
import pprint
from typing import Any, Callable, Dict
from .onnx_export_serialization import (
    flatten_with_keys_dynamic_cache,
    flatten_dynamic_cache,
    unflatten_dynamic_cache,
    unflatten_pached_dynamic_cache,
    flatten_mamba_cache,
    flatten_with_keys_mamba_cache,
    unflatten_mamba_cache,
)


def _register_cache_serialization(verbose: int = 0) -> Dict[str, bool]:
    # Cache serialization: to be moved into appropriate packages
    import torch

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    try:
        from transformers.cache_utils import MambaCache
    except ImportError:
        MambaCache = None

    # MambaCache
    unregistered_mamba_cache = True
    if MambaCache is not None and MambaCache in torch.utils._pytree.SUPPORTED_NODES:
        if verbose > 1:
            print(f"[_register_cache_serialization] {MambaCache} already registered")
        # It is already registered because bypass_export_some_errors was called
        # within a section already calling bypass_export_some_errors or transformers
        # has updated its code to do it.
        # No need to register and unregister then.
        unregistered_mamba_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register MambaCache")
        torch.utils._pytree.register_pytree_node(
            MambaCache,
            flatten_mamba_cache,
            unflatten_mamba_cache,
            serialized_type_name=f"{MambaCache.__module__}.{MambaCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_mamba_cache,
        )

    # DynamicCache
    unregistered_dynamic_cache = True
    if DynamicCache is not None and DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        if verbose > 1:
            print(f"[_register_cache_serialization] {DynamicCache} already registered")
        unregistered_dynamic_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register DynamicCache")
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
        )

        # check
        from ..cache_helpers import make_dynamic_cache

        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        values, spec = torch.utils._pytree.tree_flatten(cache)
        cache2 = torch.utils._pytree.tree_unflatten(values, spec)
        # torch.fx._pytree.tree_flatten(cache)
        assert len(cache2.key_cache) == 1

    # patched_DynamicCache
    from .patches.patch_transformers import patched_DynamicCache

    unregistered_patched_dynamic_cache = True
    if (
        patched_DynamicCache is not None
        and patched_DynamicCache in torch.utils._pytree.SUPPORTED_NODES
    ):
        if verbose > 1:
            print(f"[_register_cache_serialization] {patched_DynamicCache} already registered")
        unregistered_patched_dynamic_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register patched_DynamicCache")

        torch.utils._pytree.register_pytree_node(
            patched_DynamicCache,
            flatten_dynamic_cache,
            unflatten_pached_dynamic_cache,
            serialized_type_name=f"{patched_DynamicCache.__module__}.{patched_DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            patched_DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
        )

    return dict(
        DynamicCache=unregistered_dynamic_cache,
        MambaCache=unregistered_mamba_cache,
        patched_DynamicCache=unregistered_patched_dynamic_cache,
    )


def _unregister(cls: type, verbose: int = 0):
    import optree
    import torch

    # torch.fx._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
    optree.unregister_pytree_node(cls, namespace="torch")
    assert cls not in torch.utils._pytree.SUPPORTED_NODES, (
        f"{cls} was not successfull unregistered "
        f"from torch.utils._pytree.SUPPORTED_NODES="
        f"{pprint.pformat(list(torch.utils._pytree.SUPPORTED_NODES))}"
    )
    if verbose:
        print(f"[_unregister_cache_serialization] unregistered {cls.__name__}")


def _unregister_cache_serialization(undo: Dict[str, bool], verbose: int = 0):

    if undo.get("MambaCache", False):
        from transformers.cache_utils import MambaCache

        _unregister(MambaCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister MambaCache")

    if undo.get("DynamicCache", False):
        from transformers.cache_utils import DynamicCache

        _unregister(DynamicCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister DynamicCache")

    if undo.get("patched_DynamicCache", False):
        from .patches.patch_transformers import patched_DynamicCache

        _unregister(patched_DynamicCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister patched_DynamicCache")


@contextlib.contextmanager
def register_additional_serialization_functions(
    verbose: int = 0, replace_dynamic_cache: bool = False
) -> Callable:
    """The necessary modification to run the fx Graph."""
    fct_callable = replacement_before_exporting if replace_dynamic_cache else (lambda x: x)
    done = _register_cache_serialization(verbose=verbose)
    try:
        yield fct_callable
    finally:
        _unregister_cache_serialization(done, verbose=verbose)


@contextlib.contextmanager
def bypass_export_some_errors(
    patch_sympy: bool = True,
    patch_torch: bool = True,
    patch_transformers: bool = False,
    replace_dynamic_cache: bool = False,
    catch_constraints: bool = True,
    verbose: int = 0,
    patch: bool = True,
) -> Callable:
    """
    Tries to bypass some situations :func:`torch.export.export` does not support.

    :param patch_sympy: fix missing method ``name`` for IntegerConstant
    :param patch_torch: patches :epkg:`torch` with supported implementation
    :param patch_transformers: patches :epkg:`transformers` with supported implementation
    :param replace_dynamic_cache: replaces DynamicCache by a patched class
        avoiding issues with the dynamic shapes inferences,
        it should be True with LLM using that class and only during the export
    :param catch_constraints: catch constraints related to dynamic shapes,
        as a result, some dynamic dimension may turn into static ones,
        the environment variable ``SKIP_SOLVE_CONSTRAINTS=0``
        can be put to stop at that stage.
    :param patch: if False, disable all patches except the registration of
        serialization function

    The list of available patches.

    * ``torch.jit.isinstance``
    * ``torch._dynamo.mark_static_address``
    * ``torch._subclasses.fake_impls.infer_size``
    * fix missing method ``name`` for ``sympy.S.IntegerConstant``
    * ``AttentionMaskConverter._make_causal_mask``
    * Serialialization of ``MambaCache`` (in :epkg:`transformers`)
    * Serialialization of ``DynamicCache`` (in :epkg:`transformers`)
    * reduce errors due to shape inference
    * replaces :class:`transformers.cache_utils.DynamicCache` with
      :class:`patched_DynamicCache
      <experimental_experiment.torch_interpreter.patches.patch_transformers.patched_DynamicCache>`

    Serialization issues happen when a module takes one input or output
    has a type :func:`torch.export.export` cannot serialize.

    Examples:

    ::

        with bypass_export_some_errors(
            patch_transformers=True,
            replace_dynamic_cache=True,
        ) as modificator:
            inputs = modificator(inputs)
            onx = to_onnx(..., inputs, ...)

    ::

        with bypass_export_some_errors(
            patch_transformers=True,
            replace_dynamic_cache=True,
        ) as modificator:
            inputs = modificator(inputs)
            onx = torch.onnx.export(..., inputs, ...)

    It can be used as well to fix the torch export:

    ::

        with bypass_export_some_errors(
            patch_transformers=True,
            replace_dynamic_cache=True,
        ) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When running the model through the exported program, only the
    serialization functions need to be restored:

    ::

        with register_additional_serialization_functions() as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When exporting a model with a cache, the following error message
    may appear ``AssertionError: Mutating module attribute _seen_tokens during export.``.
    It can be avoided by setting ``strict=False`` when call :func:`torch.export.export`.
    """
    if not patch:
        fct_callable = replacement_before_exporting if replace_dynamic_cache else (lambda x: x)
        done = _register_cache_serialization(verbose=verbose)
        try:
            yield fct_callable
        finally:
            _unregister_cache_serialization(done, verbose=verbose)
    else:
        import torch
        import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
        import torch.jit

        if verbose:
            print(
                "[bypass_export_some_errors] replace torch.jit.isinstance, "
                "torch._dynamo.mark_static_address"
            )

        ########
        # caches
        ########

        cache_done = _register_cache_serialization(verbose=verbose)

        #############
        # patch sympy
        #############

        if patch_sympy:
            import sympy

            f_sympy_name = getattr(sympy.core.numbers.IntegerConstant, "name", None)

            if verbose:
                print("[bypass_export_some_errors] patch sympy")

            sympy.core.numbers.IntegerConstant.name = lambda self: f"IntCst{str(self)}"

        ###############
        # patch pytorch
        ###############

        if patch_torch:
            from .patches.patch_torch import (
                patched_infer_size,
                patched__broadcast_shapes,
                _catch_produce_guards_and_solve_constraints,
                patch__check_input_constraints_for_graph,
            )

            if verbose:
                print("[bypass_export_some_errors] patch pytorch")

            # torch.jit.isinstance
            f_jit_isinstance = torch.jit.isinstance
            torch.jit.isinstance = isinstance

            # torch._dynamo.mark_static_address
            f_mark_static_address = torch._dynamo.mark_static_address
            torch._dynamo.mark_static_address = lambda *_, **y_: None

            # torch._subclasses.fake_impls.infer_size
            f_infer_size = torch._subclasses.fake_impls.infer_size
            torch._subclasses.fake_impls.infer_size = patched_infer_size

            # torch._refs._broadcast_shapes
            f__broadcast_shapes = torch._refs._broadcast_shapes
            torch._refs._broadcast_shapes = patched__broadcast_shapes
            torch._meta_registrations._broadcast_shapes = patched__broadcast_shapes

        # torch._export.non_strict_utils.produce_guards_and_solve_constraints
        if catch_constraints:
            if verbose:
                print("[bypass_export_some_errors] modifies shape constraints")
            f_produce_guards_and_solve_constraints = (
                torch._export.non_strict_utils.produce_guards_and_solve_constraints
            )
            f__check_input_constraints_for_graph = (
                torch._export.utils._check_input_constraints_for_graph
            )
            torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
                lambda *args, **kwargs: _catch_produce_guards_and_solve_constraints(
                    f_produce_guards_and_solve_constraints, *args, verbose=verbose, **kwargs
                )
            )
            torch._export.utils._check_input_constraints_for_graph = (
                lambda *args, **kwargs: patch__check_input_constraints_for_graph(
                    f__check_input_constraints_for_graph, *args, verbose=verbose, **kwargs
                )
            )

        ####################
        # patch transformers
        ####################

        if patch_transformers:
            import transformers
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            from .patches.patch_transformers import patched_AttentionMaskConverter

            if verbose:
                print("[bypass_export_some_errors] patch transformers")
            keep__make_causal_mask = AttentionMaskConverter._make_causal_mask
            AttentionMaskConverter._make_causal_mask = (
                patched_AttentionMaskConverter._make_causal_mask
            )

        if replace_dynamic_cache:
            import transformers
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            from experimental_experiment.torch_models.fromhub import (
                modeling_phi3_v as modeling_phi3_v,
            )
            from .patches.patch_transformers import patched_DynamicCache

            def raise_assert():
                raise AssertionError("One replacement of DynamicCache was not patched.")

            if verbose:
                print("[bypass_export_some_errors] replace DynamicCache")
            keep_DynamicCache = transformers.cache_utils.DynamicCache
            keep_DynamicCache_init = keep_DynamicCache.__init__
            keep_DynamicCache.__init__ = lambda *args, **kwargs: raise_assert()

            transformers.cache_utils.DynamicCache = patched_DynamicCache
            transformers.generation.utils.DynamicCache = patched_DynamicCache
            transformers.models.llama.modeling_llama.DynamicCache = patched_DynamicCache
            transformers.models.phi.modeling_phi.DynamicCache = patched_DynamicCache
            transformers.models.phi3.modeling_phi3.DynamicCache = patched_DynamicCache
            modeling_phi3_v.DynamicCache = patched_DynamicCache

        ########
        # export
        ########

        fct_callable = replacement_before_exporting if replace_dynamic_cache else (lambda x: x)

        if verbose:
            print("[bypass_export_some_errors] done patching")

        try:
            yield fct_callable
        finally:
            #######
            # sympy
            #######

            if verbose:
                print("[bypass_export_some_errors] remove patches")

            if patch_sympy:

                # tracked by https://github.com/pytorch/pytorch/issues/143494
                if f_sympy_name:
                    sympy.core.numbers.IntegerConstant.name = f_sympy_name
                else:
                    delattr(sympy.core.numbers.IntegerConstant, "name")

                if verbose:
                    print("[bypass_export_some_errors] restored sympy functions")

            #######
            # torch
            #######

            if patch_torch:
                # this should disappear when torch.jit is removed
                torch.jit.isinstance = f_jit_isinstance
                torch._dynamo.mark_static_address = f_mark_static_address
                # tracked by https://github.com/pytorch/pytorch/issues/143495
                torch._subclasses.fake_impls.infer_size = f_infer_size
                torch._refs._broadcast_shapes = f__broadcast_shapes
                torch._meta_registrations._broadcast_shapes = f__broadcast_shapes

                if verbose:
                    print("[bypass_export_some_errors] restored pytorch functions")

            if catch_constraints:
                # to catch or skip dynamic_shapes issues
                torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
                    f_produce_guards_and_solve_constraints
                )
                torch._export.utils._check_input_constraints_for_graph = (
                    f__check_input_constraints_for_graph
                )
                if verbose:
                    print("[bypass_export_some_errors] restored shape constraints")

            ##############
            # transformers
            ##############

            if patch_transformers:
                AttentionMaskConverter._make_causal_mask = keep__make_causal_mask
                if verbose:
                    print("[bypass_export_some_errors] restored transformer")

            if replace_dynamic_cache:
                keep_DynamicCache.__init__ = keep_DynamicCache_init
                transformers.cache_utils.DynamicCache = keep_DynamicCache
                transformers.generation.utils.DynamicCache = keep_DynamicCache
                transformers.models.llama.modeling_llama.DynamicCache = keep_DynamicCache
                transformers.models.phi.modeling_phi.DynamicCache = keep_DynamicCache
                transformers.models.phi3.modeling_phi3.DynamicCache = keep_DynamicCache
                modeling_phi3_v.DynamicCache = keep_DynamicCache
                if verbose:
                    print("[bypass_export_some_errors] restored DynamicCache")

            ########
            # caches
            ########

            _unregister_cache_serialization(cache_done, verbose=verbose)


def replacement_before_exporting(args: Any) -> Any:
    """
    Does replacements on the given inputs such replacing
    :class:`transformers.cache_utils.DynamicCache` by
    :class:`experimental_experiment.torch_interpreter.patches.patch_transformers.patched_DynamicCache`.
    """
    if args is None:
        return None
    if isinstance(args, (int, float)):
        return args
    if isinstance(args, dict):
        return {k: replacement_before_exporting(v) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(replacement_before_exporting(v) for v in args)
    if isinstance(args, list):
        return [replacement_before_exporting(v) for v in args]

    if args.__class__.__name__ == "DynamicCache":
        # Do not use isinstance, the class may have been replaced.
        from .patches.patch_transformers import patched_DynamicCache

        patched = patched_DynamicCache()
        for k in ["_seen_tokens", "key_cache", "value_cache"]:
            setattr(patched, k, getattr(args, k))
        return patched

    return args
