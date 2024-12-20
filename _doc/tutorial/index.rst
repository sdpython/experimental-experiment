
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

torch.export.export: export to a Graph
======================================

All exporters rely on function :func:`torch.export.export` to convert
a pytorch module into a :class:`torch.fx.Graph`. Only then the conversion
to ONNX starts. Most of the issues come from this first step and it is
convenient to understand what it does. pytorch documentation
already has many examples about it. Here are some corner cases.

Dynamic Shapes
++++++++++++++

* :ref:`l-plot-exporter-dynamic_shapes`
* :ref:`l-plot-torch-export-with-dynamic-cache-201`
* :ref:`l-plot-exporter-nn_modules_inputs`

strict = ?
++++++++++

The parameter *strict* of :func:`torch.export.export` usually has no impact
except in some rare cases.

The exporter relies on :func:`torch.export.export`. It exposes a parameter called
`strict: bool = True` (true by default).
The behaviour is different in some specific configuration.

**torch.ops.higher_order.scan**

:func:`torch.ops.higher_order.scan` is a way to export a model with a loop.
Not all signatures work with this mode.
Here is an example with scan.

.. runpython::
    :showcode:

    import torch

    def add(carry: torch.Tensor, y: torch.Tensor):
        next_carry = carry + y
        return [next_carry, next_carry]

    class ScanModel(torch.nn.Module):
        def forward(self, x):
            init = torch.zeros_like(x[0])
            carry, out = torch.ops.higher_order.scan(
                add, [init], [x], dim=0, reverse=False, additional_inputs=[]
            )
            return carry

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    model = ScanModel()
    expected = model(x)
    print("------")
    print(expected, x.sum(axis=0))
    print("------ strict=False")
    print(torch.export.export(model, (x,), strict=False).graph)
    print("------ strict=True")
    print(torch.export.export(model, (x,), strict=True).graph)

**inplace x[..., i] = y**

This expression cannot be captured with ``strict=False``.

.. runpython::
    :showcode:

    import torch

    class UpdateModel(torch.nn.Module):
        def forward(
            self, x: torch.Tensor, update: torch.Tensor, kv_index: torch.LongTensor
        ):
            x = x.clone()
            x[..., kv_index] = update
            return x

    example_inputs = (
        torch.ones((4, 4, 10)).to(torch.float32),
        (torch.arange(2) + 10).to(torch.float32).reshape((1, 1, 2)),
        torch.Tensor([1, 2]).to(torch.int32),
    )

    model = UpdateModel()

    try:
        torch.export.export(model, (x,), strict=False)
    except Exception as e:
        print(e)

torch.onnx.export: export to ONNX
=================================

These examples relies on :func:`torch.onnx.export`.

Simple Case
+++++++++++

:ref:`l-plot-torch-linreg-101-oe`

Control Flow
++++++++++++

* :ref:`l-plot-exporter-recipes-onnx-exporter-cond`

Custom Operators
++++++++++++++++

* :ref:`l-plot-exporter-recipes-onnx-exporter-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-onnx-exporter-custom-ops-inplace`

Submodules
++++++++++

* :ref:`l-plot-exporter-recipes-onnx-exporter-modules`

Models
++++++

* :ref:`l-plot-exporter-recipes-onnx_exporter-phi2`

Optimization
++++++++++++

See :epkg:`Pattern-based Rewrite Using Rules With onnxscript`.

Supported Scenarios
===================

The following pages explores many kind of signatures for a *forward* method
and how they translate into ONNX when they can. The result are summarized by
the following pages. It tries model taking tensors, list of tensors,
integers or floats. It also tries test and loops.

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



.. toctree::
    :maxdepth: 2
    :caption:
    
    exported
    exporter_recipes
    docker


Frequent Exceptions or Errors with the Exporter
===============================================

Unsupported functions or classes
++++++++++++++++++++++++++++++++

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

torch._dynamo.exc.Unsupported
+++++++++++++++++++++++++++++

**torch._dynamo.exc.Unsupported: call_function BuiltinVariable(NotImplementedError) [ConstantVariable()] {}**

This exception started to show up with transformers==4.38.2
but it does not seem related to it. Wrapping the code with the
following fixes it.

::

    with torch.no_grad():
        # ...

RuntimeError
++++++++++++

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

101
+++

* :ref:`l-plot-torch-linreg-101`
* :ref:`l-plot-custom-backend`
* :ref:`l-plot-optimize-101`
* :ref:`l-plot-profile-existing-onnx-101`
* :ref:`l-plot-rewrite-101`
* :ref:`l-plot-torch-export-101`

102
+++

* :ref:`l-plot-onnxscript-102`
* :ref:`l-plot-executorch-102`
* :ref:`l-plot-convolution-matmul-102`
* :ref:`l-plot-custom-backend-llama-102`
* :ref:`l-plot-llama-bench-102`

201
+++

* :ref:`l-plot-torch-export-201`
* :ref:`l-plot-torch-dort-201`
* :ref:`l-torch-aot-201`
* :ref:`l-plot-torch-export-with-dynamic-cache-201`

301
+++

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

Simple Case
+++++++++++

* :ref:`Export a linear regression <l-plot-torch-linreg-101>`

Control Flow
++++++++++++

* :ref:`l-plot-exporter-recipes-custom-cond`
* :ref:`l-plot-exporter-recipes-custom-pdist`

Custom Operators
++++++++++++++++

* :ref:`l-plot-exporter-recipes-custom-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-custom-custom-ops-inplace`

Submodules
++++++++++

* :ref:`l-plot-exporter-recipes-custom-modules`

Model
+++++

* :ref:`l-plot-exporter-recipes-custom-phi2`

Optimization
++++++++++++

* :ref:`l-plot-optimize-101`

Dockers
=======

Old work used to play with :func:`torch.compile` on a docker.

.. toctree::

    docker