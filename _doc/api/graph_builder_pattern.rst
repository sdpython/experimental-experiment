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

get_pattern
===========

.. autofunction:: experimental_experiment.xoptim.get_pattern

get_pattern_list
================

.. autofunction:: experimental_experiment.xoptim.get_pattern_list

Classes
=======

GraphBuilderPatternOptimization
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.GraphBuilderPatternOptimization
    :members:


MatchResult
+++++++++++

.. autoclass:: experimental_experiment.xoptim.MatchResult
    :members:

PatternOptimization
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.PatternOptimization
    :members:

EasyPatternOptimization
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.EasyPatternOptimization
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

SameChildrenPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SameChildrenPattern
    :members:

SlicesSplitPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SlicesSplitPattern
    :members:

Sub1MulPattern
++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Sub1MulPattern
    :members:

SwitchOrderBinaryPattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SwitchOrderBinaryPattern
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

AddAddMulMulPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddAddMulMulPattern
    :members:

AddMulPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulPattern
    :members:

ConstantOfShapeScatterNDPattern
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constant_of_shape_scatter_nd.ConstantOfShapeScatterNDPattern
    :members:

MulSigmoidPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.MulSigmoidPattern
    :members:

NegXplus1Pattern
++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.NegXplus1Pattern
    :members:

ReplaceZeroPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.where_replace.ReplaceZeroPattern
    :members:

SimpleRotaryPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.simple_rotary.SimpleRotaryPattern
    :members:

TriMatrixPattern
++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constants.TriMatrixPattern
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

FusedMatMulPatternx2
++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulPatternx2
    :members:

GatherGrad
++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.gather_grad.GatherGrad
    :members:

SoftmaxGradPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.activation_grad.SoftmaxGradPattern
    :members:
