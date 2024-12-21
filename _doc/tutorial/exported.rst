
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
