from dataclasses import dataclass
import torch


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
        if (
            guard_size_oblivious(sizeA == 1)
            or guard_size_oblivious(sizeB == 1)
            or sizeA == sizeB
        ):
            expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
        else:
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model SmolLM.
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


@dataclass
class patched_EqualityConstraint:
    """Patches ``torch.fx.experimental.symbolic_shapes.EqualityConstraint._rewrite``."""

    def _rewrite(self, src: "Source") -> "sympy.Expr":  # noqa: F821
        """Patched method."""
        # always represent the given source by the root of its equivalence class
        src = self._find(src)
        if src in self._defs:
            # simply look up the definition if it exists
            # NOTE(avik): This works because definitions are always transitively-closed;
            # otherwise we would have to do recursive rewriting.
            return self._defs[src]
        else:
            import sympy

            # otherwise, create a symbol representing the source
            try:
                name = src.name()
            except AttributeError:
                # A constant has no name.
                # In this case, the current implementation of torch fails (17/12/2024).
                # Try model SmolLM.
                name = str(src)
            return sympy.Symbol(name)
