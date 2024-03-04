===================
graph_builder_optim
===================

get_default_patterns
====================

.. autofunction:: experimental_experiment.xoptim.patterns.get_default_patterns

get_pattern_list
================

.. autofunction:: experimental_experiment.xoptim.patterns.get_pattern_list

Classes
=======

GraphBuilderPatternOptimization
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.graph_builder_optim.GraphBuilderPatternOptimization
    :members:


MatchResult
+++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.MatchResult
    :members:

PatternOptimization
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.PatternOptimization
    :members:

.. _l-pattern-optimization-onnx:

Onnx Patterns
=============

CastPattern
+++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastPattern
    :members:

ExpandPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandPattern
    :members:

MulMulMulScalarPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.MulMulMulScalarPattern
    :members:

ReduceReshapePattern
++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReduceReshapePattern
    :members:

ReshapeMatMulReshapePattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeMatMulReshapePattern
    :members:

ReshapeMatMul2of3Pattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeMatMul2of3Pattern
    :members:

Reshape2Of3Pattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Reshape2Of3Pattern
    :members:

ReshapeReshapePattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapePattern
    :members:

RotaryConcatPartPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.RotaryConcatPartPattern
    :members:

Sub1MulPattern
++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Sub1MulPattern
    :members:

TransposeMatMulPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeMatMulPattern
    :members:

TransposeReshapeMatMulPattern
+++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeReshapeMatMulPattern
    :members:

TransposeTransposePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeTransposePattern
    :members:

UnsqueezeUnsqueezePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.UnsqueezeUnsqueezePattern
    :members:

.. _l-pattern-optimization-ort:

Ort Patterns
============

ConstantOfShapeScatterNDPattern
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.constant_of_shape_scatter_nd.ConstantOfShapeScatterNDPattern
    :members:

FusedMatMulPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulPattern
    :members:
