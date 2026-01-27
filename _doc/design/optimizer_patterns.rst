.. _l-design-pattern-optimizer-patterns:

==================
Available Patterns
==================

Default Patterns
================

.. runpython::
    :showcode:
    :rst:

    from experimental_experiment.xoptim import get_pattern_list
    names = sorted([pat.__class__.__name__ for pat in get_pattern_list("default")])
    for i, name in enumerate(names):
        print(f"* {i+1}: :class:`{name} <experimental_experiment.xoptim.patterns.{name}>")

Patterns specific to onnxruntime
================================

.. runpython::
    :showcode:
    :rst:

    from experimental_experiment.xoptim import get_pattern_list
    names = sorted([pat.__class__.__name__ for pat in get_pattern_list("onnxruntime")])
    for i, name in enumerate(names):
        print(f"* {i+1}: :class:`{name} <experimental_experiment.xoptim.pattern_ort.{name}>")

Patterns specific to ai.onnx.ml
===============================

.. runpython::
    :showcode:
    :rst:

    from experimental_experiment.xoptim import get_pattern_list
    names = sorted([pat.__class__.__name__ for pat in get_pattern_list("ml")])
    for i, name in enumerate(names):
        print(f"* {i+1}: :class:`{name} <experimental_experiment.xoptim.pattern_ml.{name}>")

Experimental Patterns
=====================

This works on CUDA with :epkg:`onnx-extended`.

.. runpython::
    :showcode:
    :rst:

    from experimental_experiment.xoptim import get_pattern_list
    names = sorted([pat.__class__.__name__ for pat in get_pattern_list("experimental")])
    for i, name in enumerate(names):
        print(f"* {i+1}: :class:`{name} <experimental_experiment.xoptim.pattern_exp.{name}>")
