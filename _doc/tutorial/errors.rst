
.. _l-frequent-exporter-errors:

===============================================
Frequent Exceptions or Errors with the Exporter
===============================================

Unsupported functions or classes
================================

If the converter to onnx fails, function :func:`bypass_export_some_errors
<experimental_experiment.torch_interpreter.onnx_export_errors.bypass_export_some_errors>`
may help solving some of them. The ocumentation of this function
gives the list of issues it can bypass.

::

    from experimental_experiment.torch_interpreter.onnx_export_errors import bypass_export_some_errors
    
    with bypass_export_some_errors():
        onx = to_onnx(...)


torch._dynamo.exc.Unsupported
=============================

**torch._dynamo.exc.Unsupported: call_function BuiltinVariable(NotImplementedError) [ConstantVariable()] {}**

This exception started to show up with transformers==4.38.2
but it does not seem related to it. Wrapping the code with the
following fixes it.

::

    with torch.no_grad():
        # ...

RuntimeError
============

**RuntimeError: Encountered autograd state manager op <built-in function _set_grad_enabled> trying to change global autograd state while exporting.**

Wrapping the code around probably solves this issue.

::

    with torch.no_grad():
        # ...
