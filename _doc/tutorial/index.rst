
========
Tutorial
========

This module was started to experiment around function :func:`torch.export.export`
and see what kind of issues occur when leveraging that function to convert
a :class:`torch.nn.Module` into :epkg:`ONNX`.
The tutorial is a collection of examples or benchmark around that topic.
Section :ref:`l-design` explains how a converter works from torch model
to the onnx graph. The official exporter is implemented in :epkg:`pytorch`
itself through the function :func:`torch.onnx.export`.

.. toctree::

    shape
    to_onnx
    errors
    docker