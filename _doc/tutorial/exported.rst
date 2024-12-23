
Supported Model Signatures
==========================

The following pages explores many kind of signatures for a *forward* method
and how they translate into ONNX when they can. The results are summarized by
the following pages. It tries model taking tensors, list of tensors,
integers or floats. It also tries tests and loops.

* :ref:`torch.export.export and static shapes <le-summary-exported-program>`
* :ref:`torch.export.export and dynamic shapes <led-summary-exported-program>`
* :ref:`conversion to onnx with static shapes <lo-summary-exported-program>`
* :ref:`conversion to onnx with dynamic shapes shapes <lod-summary-exported-program>`

.. toctree::
    :maxdepth: 1

    exported_program
    exported_program_dynamic
    exported_onnx
    exported_onnx_dynamic

.. _l-scenarios-exported-program-export:

Tested Scenarios
++++++++++++++++

* **jit**: tries to capture the graph by using :mod:`torch.jit`
* **jit-decall**: tries to capture the graph by using :mod:`torch.jit`
  then applies decompositions
* **nostrict**: call :func:`torch.export.export(..., strict=False)`
* **nostrict-decall**: call :func:`torch.export.export(..., strict=False)`,
  then applies decompositions
* **strict**: call :func:`torch.export.export(..., strict=True)`
* **strict-decall**: call :func:`torch.export.export(..., strict=True)`,
  then applies decompositions
* **tracing**: trace the execution of the model, it does flatten list,
  dictionaries or custom classes but the graph does not always produce aten function.

And for ONNX:

* **custom-fallback**: tries to export with different sceanrios
  then to convert into ONNX with the custom exporter
* **custom-tracing**: traces the model and then converts
  into ONNX with the custom exporter
* **dynamo-ir**: calls :func:`torch.onnx.export(..., dynamo=True)` and then
  optimizes the model
* **script**: calls :func:`torch.onnx.export(..., dynamo=False)`
