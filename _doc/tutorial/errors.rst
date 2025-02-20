=================
Unexpected Errors
=================

.. _l-torch-export-export-issues:

Issues with torch.export.export
===============================

Posted on github.

* `torch.export.export fails when one input is a class inheriting from torch.nn.Module <https://github.com/pytorch/pytorch/issues/147326>`_
* `Unable to print in a branch run by torch.cond <https://github.com/pytorch/pytorch/issues/147115>`_
* `How to export a model using topk with a variable number of neighbour? <https://github.com/pytorch/pytorch/issues/146990>`_
* `Dynamic_shapes with Dim fails when DYNAMIC succeeds <https://github.com/pytorch/pytorch/issues/146315>`_
* `torch.cond + torch.non_zero does not work with torch.export.export <https://github.com/pytorch/pytorch/issues/144691>`_
* `infer_size(a, b) fails when it could return a value <https://github.com/pytorch/pytorch/issues/143495>`_
* `sympy.C.ConstantInteger has no method name <https://github.com/pytorch/pytorch/issues/143494>`_
* `torch.export.export fails to export a model with dynamic shapes for a custom type <https://github.com/pytorch/pytorch/issues/142161>`_

RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
===========================================================================

Unexpectedly, :func:`torch.cuda.is_available` may raise the following error:

::

    RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable

The code is as follows:

.. code-block:: python

    def _nvml_based_avail() -> bool:
        return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"

    def is_available() -> bool:
        r"""Return a bool indicating if CUDA is currently available."""
        if not _is_compiled():
            return False
        if _nvml_based_avail():
            # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
            # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
            # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
            return device_count() > 0
        else:
            # The default availability inspection never throws and returns 0 if the driver is missing or can't
            # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
            # API via `cuInit`
            return torch._C._cuda_getDeviceCount() > 0

It does trust :func:`torch.cuda.device_count` but calls 
``torch._C._cuda_getDeviceCount()`` instead which does not seem to
release CUDA memory. One way to solve this is to set
``PYTORCH_NVML_BASED_CUDA_CHECK=1``.
