===================
graph_builder_optim
===================

get_default_patterns
====================

.. autofunction:: experimental_experiment.torch_exp.optimization_patterns.get_default_patterns

get_pattern_list
================

.. autofunction:: experimental_experiment.torch_exp.optimization_patterns.get_pattern_list

Classes
=======

GraphBuilderPatternOptimization
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.graph_builder_optim.GraphBuilderPatternOptimization
    :members:


MatchResult
+++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.MatchResult
    :members:

PatternOptimization
+++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.PatternOptimization
    :members:

.. _l-pattern-optimization-onnx:

Onnx Patterns
=============

CastPattern
+++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.CastPattern
    :members:

ExpandPattern
+++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.ExpandPattern
    :members:

MulMulMulPattern
++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.MulMulMulPattern
    :members:

ReshapeMatMulReshapePattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.ReshapeMatMulReshapePattern
    :members:

ReshapeReshapePattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.ReshapeReshapePattern
    :members:

RotaryConcatPartPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.RotaryConcatPartPattern
    :members:

Sub1MulPattern
++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.Sub1MulPattern
    :members:

TransposeMatMulPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.TransposeMatMulPattern
    :members:

TransposeTransposePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.TransposeTransposePattern
    :members:

UnsqueezeUnsqueezePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.UnsqueezeUnsqueezePattern
    :members:

.. _l-pattern-optimization-ort:

Ort Patterns
============

ConstantOfShapeScatterNDPattern
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.torch_exp.patterns.ConstantOfShapeScatterNDPattern
    :members:
