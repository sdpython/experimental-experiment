Exporter Recipes Gallery
========================

A model can be converted to ONNX if :func:`torch.export.export` is
able to convert that model into a graph. This is not always
possible but usually possible with some code changes.
These changes may not be desired as they may hurt the performance
and make the code more complex than it should be.
The conversion is a necessary step to be able to use
ONNX. Next examples shows some recurrent code patterns and
ways to rewrite them so that the exporter works.

See :ref:`l-exporter-recipes` for an organized version of this gallery.

A couple of examples to illustrate different implementation
of dot product (see also :epkg:`sphinx-gallery`).

Getting started
+++++++++++++++

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.

::

    git clone https://github.com/sdpython/experimental-experiment.git
    pip install onnxruntime-gpu nvidia-ml-py
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>

Common Errors
+++++++++++++

Some of them are exposed in the examples. Others may be found at
:ref:`l-frequent-exporter-errors`.

