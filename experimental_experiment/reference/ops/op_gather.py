# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Gather(OpRun):
    def _run(self, x, indices, axis=None):
        if x.size == 0 or indices.size == 0:
            if axis is None:
                new_shape = indices.shape
            else:
                new_shape = (*x.shape[:axis], *indices.shape, *x.shape[axis + 1 :])
            if 0 not in new_shape:
                new_shape = (0, *new_shape[1:])
            return (np.empty(new_shape, dtype=x.dtype),)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        try:
            return (np.take(x, indices, axis=axis),)
        except TypeError:
            # distribution x86 requires int32.
            return (np.take(x, indices.astype(int), axis=axis),)
