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
Section :ref:`l-torch-export-export-ds` provides a couple of examples
on how to define them, as their definition aligns with those used
in :func:`torch.export.export`.

* :ref:`l-plot-exporter-recipes-custom-dynpad`

**Control Flow**

* :ref:`l-plot-exporter-recipes-custom-cond`

**Custom Operators**

* :ref:`l-plot-exporter-recipes-custom-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-custom-custom-ops-inplace`

**Submodules**

* :ref:`l-plot-exporter-recipes-custom-modules`

**Optimization**

* :ref:`l-plot-optimize-101`
