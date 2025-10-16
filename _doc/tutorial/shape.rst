
============
ShapeBuilder
============

:func:`onnx.shape_inference.infer_shapes` tries to infer
shapes and types based on input shapes. It does not
supports formulas and introduces new symbols.
Examples :ref:`l-plot-shape_inference-201` compares shape inference
from :epkg:`onnx` and the shape inference provided by class
:class:`BasicShapeBuilder <experimental_experiment.xshape.shape_builder_impl.BasicShapeBuilder>`.

This class walks through all nodes and looks into a list of functions
computing the output shapes based on the node type.
It tries as much as possible to express the new shape with formulas
based on the dimensions used to defined the inputs.
The list of functions is available in :mod:`experimental_experiment.xshape.shape_type_compute`
called from class :class:`_InferenceRuntime <experimental_experiment.xshape._inference_runtime._InferenceRuntime>`.

While doing this, every function may try to compute some tiny constants
in :class:`_BuilderRuntime <experimental_experiment.xshape._builder_runtime._BuilderRuntime>`.
This is used by :class:`_ShapeRuntime <experimental_experiment.xshape._shape_runtime._ShapeRuntime>`
to deduce some shapes.

For example, if **X** has shape ``("d1", 2)`` then ``Shape(X, start=1)`` is constant ``[2]``.
This can be later used to infer the shape after a reshape.

After getting an expression, a few postprocessing are applied to reduce
its complexity. This relies on :mod:`ast`. It is done by function
:func:`simplify_expression <experimental_experiment.xshape.simplify_expressions.simplify_expression>`.
``d + f - f`` is replaced by ``d``.
