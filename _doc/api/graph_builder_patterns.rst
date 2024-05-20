=====================
Optimization Patterns
=====================

.. _l-pattern-optimization-onnx:

Onnx (default) Patterns
=======================

.. autoclass:: experimental_experiment.xoptim.patterns.CastPattern

.. autoclass:: experimental_experiment.xoptim.patterns.CastCastBinaryPattern

.. autoclass:: experimental_experiment.xoptim.patterns.CastOpCastPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ComputationCastOpCastPattern

.. autoclass:: experimental_experiment.xoptim.patterns.DivByMulScalarPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandBroadcastPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ExpandSwapPattern

.. autoclass:: experimental_experiment.xoptim.patterns.IdentityPattern

.. autoclass:: experimental_experiment.xoptim.patterns.MatMulReshape2Of3Pattern

.. autoclass:: experimental_experiment.xoptim.patterns.MulMulMulScalarPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ReduceReshapePattern

.. autoclass:: experimental_experiment.xoptim.patterns.ReduceSumNormalizePattern

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeMatMulReshapePattern

.. autoclass:: experimental_experiment.xoptim.patterns.Reshape2Of3Pattern

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapeBinaryPattern

.. autoclass:: experimental_experiment.xoptim.patterns.ReshapeReshapePattern

.. autoclass:: experimental_experiment.xoptim.patterns.RotaryConcatPartPattern

.. autoclass:: experimental_experiment.xoptim.patterns.SameChildrenPattern

.. autoclass:: experimental_experiment.xoptim.patterns.SlicesSplitPattern

.. autoclass:: experimental_experiment.xoptim.patterns.Sub1MulPattern

.. autoclass:: experimental_experiment.xoptim.patterns.SwitchOrderBinaryPattern

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeMatMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeReshapeMatMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns.TransposeTransposePattern

.. autoclass:: experimental_experiment.xoptim.patterns.UnsqueezeEqualPattern

.. autoclass:: experimental_experiment.xoptim.patterns.UnsqueezeUnsqueezePattern

Custom Op Patterns
==================

.. autoclass:: experimental_experiment.xoptim.patterns_fix.add_reduction_scatter_nd.AddReductionScatterND

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddAddMulMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulBroadcastPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulSharedInputPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulSharedInputBroadcastPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.AddMulTransposePattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constant_of_shape_scatter_nd.ConstantOfShapeScatterNDPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constant_of_shape_scatter_nd.MaskedShapeScatterNDPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.MulSigmoidPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.NegXplus1Pattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.where_replace.ReplaceZeroPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.simple_rotary.SimpleRotaryPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.SubMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.binary_operators.SubMulBroadcastPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.unary_operators.TransposeCastPattern

.. autoclass:: experimental_experiment.xoptim.patterns_exp.constants.TriMatrixPattern

.. _l-pattern-optimization-ort:

Ort Patterns
============

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulPattern

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulx2Pattern

.. autoclass:: experimental_experiment.xoptim.patterns_ort.fused_matmul.FusedMatMulTransposePattern

.. autoclass:: experimental_experiment.xoptim.patterns_ort.gather_grad.GatherGradPattern

.. autoclass:: experimental_experiment.xoptim.patterns_ort.simplified_layer_normalization.SimplifiedLayerNormalizationPattern

.. autoclass:: experimental_experiment.xoptim.patterns_ort.activation_grad.SoftmaxGradPattern
