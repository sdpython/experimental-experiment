
===================
Frequent Exceptions
===================

torch._dynamo.exc.Unsupported
=============================

torch._dynamo.exc.Unsupported: call_function BuiltinVariable(NotImplementedError) [ConstantVariable()] {}
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This exception started to show up with transformers==4.38.2
but it does not seem related to it. Wrapping the code with the
following fixes it.

::

    with torch.no_grad():
        # ...

RuntimeError
============

RuntimeError: Encountered autograd state manager op <built-in function _set_grad_enabled> trying to change global autograd state while exporting.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Wrapping the code around probably solves this issue.

::

    with torch.no_grad():
        # ...
