============
Weird Errors
============

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
