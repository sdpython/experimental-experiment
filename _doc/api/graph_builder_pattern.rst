===================
graph_builder_optim
===================

get_default_patterns
====================

.. autofunction:: experimental_experiment.xoptim.patterns.get_default_patterns

get_onnxruntime_patterns
========================

.. autofunction:: experimental_experiment.xoptim.patterns_ort.get_onnxruntime_patterns

get_fix_patterns
================

.. autofunction:: experimental_experiment.xoptim.patterns_fix.get_fix_patterns

get_pattern_list
================

.. autofunction:: experimental_experiment.xoptim.patterns.get_pattern_list

Classes
=======

GraphBuilderPatternOptimization
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.GraphBuilderPatternOptimization
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

Onnx (default) Patterns
=======================

CastPattern
+++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastPattern
    :members:

CastCastBinaryPattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastCastBinaryPattern
    :members:

ExpandPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandPattern
    :members:

ExpandBroadcastPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandBroadcastPattern
    :members:

ExpandSwapPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandSwapPattern
    :members:

MatMulReshape2Of3Pattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.MatMulReshape2Of3Pattern
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

Reshape2Of3Pattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Reshape2Of3Pattern
    :members:

ReshapeReshapeBinaryPattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapeBinaryPattern
    :members:

ReshapeReshapePattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapePattern
    :members:

RotaryConcatPartPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.RotaryConcatPartPattern
    :members:

SlicesSplitPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SlicesSplitPattern
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

Fix Patterns
============

AddReductionScatterND
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_fix.add_reduction_scatter_nd.AddReductionScatterND
    :members:

Experimental Patterns
=====================

ConstantOfShapeScatterNDPattern
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constant_of_shape_scatter_nd.ConstantOfShapeScatterNDPattern
    :members:

.. _l-pattern-optimization-ort:

Ort Patterns
============

SimplifiedLayerNormalizationPattern
+++++++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.simplified_layer_normalization.SimplifiedLayerNormalizationPattern
    :members:

FusedMatMulPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulPattern
    :members:

SoftmaxGradPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.activation_grad.SoftmaxGradPattern
    :members:

