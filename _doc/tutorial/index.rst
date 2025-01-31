
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
Next sections show many examples, including how to deal with some possible issues.

.. _l-torch-export-export-ds:

torch.export.export: export to a Graph
======================================

All exporters rely on function :func:`torch.export.export` to convert
a pytorch module into a :class:`torch.fx.Graph`. Only then the conversion
to ONNX starts. Most of the issues come from this first step and it is
convenient to understand what it does. pytorch documentation
already has many examples about it. Here are some corner cases.

**Dynamic Shapes**

* :ref:`l-plot-exporter-dynamic_shapes`
* :ref:`l-plot-exporter-lost_dynamic_dimension`
* :ref:`l-plot-exporter-exporter-infer-ds`

**Custom Types as Inputs**

* :ref:`l-plot-torch-export-with-dynamic-cache-201`
* :ref:`l-plot-exporter-nn_modules_inputs`

**Investigate, Export piece by piece**

* :ref:`l-plot-exporter-exporter-reportability`
* :ref:`l-plot-exporter-exporter-draft_export`
* :ref:`l-plot-exporter-recipes-custom-phi35`

**strict = ?**

The exporter relies on :func:`torch.export.export`. It exposes a parameter called
`strict: bool = True` (true by default).
The behaviour is different in some specific configuration.
Page :ref:`led-summary-exported-program` goes through many
kind of model and tells which one is supported and how it is converted.

**decompositions**

Function :func:`torch.export.export` produces an :class:`torch.export.ExportedProgram`.
This class has a method :meth:`torch.export.ExportedProgram.run_decompositions`
which converts the graph into another, usually longer but using
a reduced set of functions or primitive. The converter to ONNX
has less functions to support to convert this second graph.

.. _l-exporter-recipes:

torch.onnx.export: export to ONNX
=================================

These examples relies on :func:`torch.onnx.export`.

**Simple Case**

:ref:`l-plot-torch-linreg-101-oe`

**Dynamic Shapes**

Dynamic shapes should be utilized to create a model capable of handling
inputs with varying shapes while maintaining the same rank.
Section :ref:`l-torch-export-export-ds provides` a couple of examples
on how to define them, as their definition aligns with those used
in :func:`torch.export.export`.

* :ref:`l-plot-exporter-recipes-onnx-exporter-modules`

**Control Flow**

* :ref:`l-plot-exporter-recipes-onnx-exporter-cond`

**Custom Operators**

* :ref:`l-plot-exporter-recipes-onnx-exporter-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-onnx-exporter-custom-ops-inplace`

**Submodules**

* :ref:`l-plot-exporter-recipes-onnx-exporter-modules`

**Models**

* :ref:`l-plot-exporter-recipes-onnx_exporter-phi2`

**Optimization**

It is recommended to optimize the obtained model by running
method :meth:`torch.onnx.ONNXProgram.optimize`. It removes
many unncessary nodes (Identity, multiplication by 1) and other
patterns. It tries to find patterns it knows how to optimize.
See :epkg:`Pattern-based Rewrite Using Rules With onnxscript`.

**Issues**

You can post issues in `pytorch/issues <https://github.com/pytorch/pytorch/issues>`_
and label it with ``module:onnx`` if you find an issue.

.. _l-frequent-exporter-errors:

Frequent Exceptions or Errors with the Exporter
===============================================

**Unsupported functions or classes**

If the converter to onnx fails, function :func:`bypass_export_some_errors
<experimental_experiment.torch_interpreter.onnx_export_errors.bypass_export_some_errors>`
may help solving some of them. The ocumentation of this function
gives the list of issues it can bypass.

::

    from experimental_experiment.torch_interpreter.onnx_export_errors import (
        bypass_export_some_errors,
    )

    with bypass_export_some_errors():
        # export to onnx with (model, inputs, ...)

If the input contains a cache class, you may need to patch the inputs.

::

    from experimental_experiment.torch_interpreter.onnx_export_errors import (
        bypass_export_some_errors,
    )

    with bypass_export_some_errors(patch_transformers=True) as modificator:
        inputs = modificator(inputs)
        # export to onnx with (model, inputs, ...)

This function is a work in progress as the exporter extends the list
of supported models. A standaline copy of this function can be found at
`phi35 <https://github.com/xadupre/examples/tree/main/c2024/phi35>`_.

**torch._dynamo.exc.Unsupported: call_function BuiltinVariable(NotImplementedError) [ConstantVariable()] {}**

This exception started to show up with transformers==4.38.2
but it does not seem related to it. Wrapping the code with the
following fixes it.

::

    with torch.no_grad():
        # ...

**RuntimeError: Encountered autograd state manager op <built-in function _set_grad_enabled> trying to change global autograd state while exporting.**

Wrapping the code around probably solves this issue.

::

    with torch.no_grad():
        # ...

Play with onnx models and onnxruntime
=====================================

:epkg:`onnxscript` is one way to directly create model or function in ONNX.
The :epkg:`onnxscript Tutorial` explains how it works.
Some other examples follow.

* :ref:`l-plot-onnxscript-102`

An exported model can be slow. It can be profiled on CUDA with 
the native profiling NVIDIA built. It can also be profiled with
the tool implemented in :epkg:`onnxruntime`. Next example shows that
on CPU.

* :ref:`l-plot-profile-existing-onnx-101`

.. _l-pytorch-onnx-examples:

Deeper into pytorch and onnx
============================

**101**

* :ref:`l-plot-torch-linreg-101`
* :ref:`l-plot-custom-backend`
* :ref:`l-plot-optimize-101`
* :ref:`l-plot-profile-existing-onnx-101`
* :ref:`l-plot-rewrite-101`
* :ref:`l-plot-torch-export-101`

**102**

* :ref:`l-plot-onnxscript-102`
* :ref:`l-plot-executorch-102`
* :ref:`l-plot-convolution-matmul-102`
* :ref:`l-plot-custom-backend-llama-102`
* :ref:`l-plot-llama-bench-102`

**201**

* :ref:`l-plot-torch-export-201`
* :ref:`l-plot-torch-dort-201`
* :ref:`l-torch-aot-201`
* :ref:`l-plot-torch-export-with-dynamic-cache-201`

**301**

* :ref:`l-plot-onnxrt-diff`
* :ref:`l-plot-llama-diff-export`


to_onnx: another export to investigate
======================================

:func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>` implements
another exporter to ONNX. It does not support all the cases :func:`torch.onnx.export`.
It fails rather trying different options to recover.
It calls :func:`torch.export.export` but does not alter the graph
(no rewriting, no decomposition) before converting this graph to onnx.
It is used to investigate export issues raised by :func:`torch.export.export`.

**Simple Case**

* :ref:`Export a linear regression <l-plot-torch-linreg-101>`

**Dynamic Shapes**

Dynamic shapes should be utilized to create a model capable of handling
inputs with varying shapes while maintaining the same rank.
Section :ref:`l-torch-export-export-ds provides` a couple of examples
on how to define them, as their definition aligns with those used
in :func:`torch.export.export`.

* :ref:`l-plot-exporter-recipes-custom-modules`

**Control Flow**

* :ref:`l-plot-exporter-recipes-custom-cond`
* :ref:`l-plot-exporter-recipes-custom-pdist`

**Custom Operators**

* :ref:`l-plot-exporter-recipes-custom-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-custom-custom-ops-inplace`

**Submodules**

* :ref:`l-plot-exporter-recipes-custom-modules`

**Models**

* :ref:`l-plot-exporter-recipes-custom-phi2`

**Optimization**

* :ref:`l-plot-optimize-101`

Dockers
=======

Old work used to play with :func:`torch.compile` on a docker.

.. toctree::

    docker