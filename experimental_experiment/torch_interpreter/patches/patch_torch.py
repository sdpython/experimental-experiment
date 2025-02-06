import inspect
import os
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def _catch_produce_guards_and_solve_constraints(
    previous_function: Callable,
    fake_mode: "FakeTensorMode",  # noqa: F821
    gm: "torch.fx.GraphModule",  # noqa: F821
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    equalities_inputs: "EqualityConstraint",  # noqa: F821
    original_signature: inspect.Signature,
    _is_torch_jit_trace: bool = False,
    verbose: int = 0,
):
    try:
        return previous_function(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_catch_produce_guards_and_solve_constraints] ERROR"
                f"produce_guards_and_solve_constraints failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"fake_mode={fake_mode}\n"
                f"dynamic_shapes={dynamic_shapes}\n"
                f"equalities_inputs={equalities_inputs}\n"
                f"original_signature={original_signature}\n"
                f"_is_torch_jit_trace={_is_torch_jit_trace}\n"
                f"exc={e}\ngm={gm}"
            )


def patch__check_input_constraints_for_graph(
    previous_function: Callable,
    input_placeholders: list[torch.fx.Node],
    flat_args_with_path,
    range_constraints,
    verbose: int = 0,
) -> None:
    try:
        return previous_function(input_placeholders, flat_args_with_path, range_constraints)
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_check_input_constraints_for_graph] ERROR"
                f"_check_input_constraints_for_graph failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"input_placeholders={input_placeholders}\n"
                f"range_constraints={range_constraints}\n"
                f"exc={e}"
            )


def patched_infer_size(a, b):
    """Patches ``torch._subclasses.fake_impls.infer_size``."""
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = guard_size_oblivious(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = guard_size_oblivious(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = guard_size_oblivious(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
        else:
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model SmolLM.
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


def patched__broadcast_shapes(*_shapes):
    """Patches ``torch._refs._broadcast_shapes``."""
    from functools import reduce
    from torch._prims_common import IntLike
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape: List[Union[int, torch.SymInt]] = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])

    return common_shape
