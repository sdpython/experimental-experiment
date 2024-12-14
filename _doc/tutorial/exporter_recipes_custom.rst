
====================
Recipes with to_onnx
====================

These examples relies on :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`,
an exporter implemented to investigate possible ways.
It calls :func:`torch.export.export` but does not alter the graph
(no rewriting, no decomposition) before converting this graph to onnx.

Control Flow
++++++++++++

* :ref:`l-plot-exporter-recipes-custom-cond`
* :ref:`l-plot-exporter-recipes-custom-pdist`

Custom Operators
================

* :ref:`l-plot-exporter-recipes-custom-custom-ops-fct`
* :ref:`l-plot-exporter-recipes-custom-custom-ops-inplace`

Submodules
==========

* :ref:`l-plot-exporter-recipes-custom-modules`

Model
=====

* :ref:`l-plot-exporter-recipes-custom-phi2`
