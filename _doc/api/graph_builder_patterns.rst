=====================
Optimization Patterns
=====================

.. _l-pattern-optimization-onnx:

Onnx (default) Patterns
=======================

CastPattern
+++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastPattern

CastCastBinaryPattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastCastBinaryPattern

CastOpCastPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.CastOpCastPattern

ComputationCastOpCastPattern
++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ComputationCastOpCastPattern

DivByMulScalarPattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.DivByMulScalarPattern

ExpandPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandPattern

ExpandBroadcastPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandBroadcastPattern

ExpandSwapPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandSwapPattern

IdentityPattern
+++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.IdentityPattern

MatMulReshape2Of3Pattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.MatMulReshape2Of3Pattern

MulMulMulScalarPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.MulMulMulScalarPattern

ReduceReshapePattern
++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReduceReshapePattern

ReduceSumNormalizePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReduceSumNormalizePattern

ReshapeMatMulReshapePattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeMatMulReshapePattern

Reshape2Of3Pattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Reshape2Of3Pattern

ReshapeReshapeBinaryPattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapeBinaryPattern

ReshapeReshapePattern
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapePattern

RotaryConcatPartPattern
+++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.RotaryConcatPartPattern

SameChildrenPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SameChildrenPattern

SlicesSplitPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SlicesSplitPattern
    :members:

Sub1MulPattern
++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.Sub1MulPattern

SwitchOrderBinaryPattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.SwitchOrderBinaryPattern

TransposeMatMulPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeMatMulPattern

TransposeReshapeMatMulPattern
+++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeReshapeMatMulPattern

TransposeTransposePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeTransposePattern

UnsqueezeUnsqueezePattern
+++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns.UnsqueezeUnsqueezePattern

Fix Patterns
============

AddReductionScatterND
+++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_fix.add_reduction_scatter_nd.AddReductionScatterND

Experimental Patterns
=====================

AddAddMulMulPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddAddMulMulPattern

AddMulPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulPattern

AddMulBroadcastPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulBroadcastPattern

AddMulSharedInputPattern
++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulSharedInputPattern

AddMulSharedInputBroadcastPattern
+++++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulSharedInputBroadcastPattern

ConstantOfShapeScatterNDPattern
+++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constant_of_shape_scatter_nd.ConstantOfShapeScatterNDPattern

MulSigmoidPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.MulSigmoidPattern

NegXplus1Pattern
++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.NegXplus1Pattern

ReplaceZeroPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.where_replace.ReplaceZeroPattern

SimpleRotaryPattern
+++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.simple_rotary.SimpleRotaryPattern

SubMulPattern
+++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.SubMulPattern

SubMulBroadcastPattern
++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.SubMulBroadcastPattern

TransposeCastPattern
++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.unary_operators.TransposeCastPattern

TriMatrixPattern
++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constants.TriMatrixPattern

.. _l-pattern-optimization-ort:

Ort Patterns
============

SimplifiedLayerNormalizationPattern
+++++++++++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.simplified_layer_normalization.SimplifiedLayerNormalizationPattern

FusedMatMulPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulPattern

FusedMatMulx2Pattern
++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulx2Pattern

FusedMatMulTransposePattern
+++++++++++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulTransposePattern

GatherGradPattern
+++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.gather_grad.GatherGradPattern

SoftmaxGradPattern
++++++++++++++++++

.. autoclass:: experimental_experiment.xoptim.patterns_ort.activation_grad.SoftmaxGradPattern
