
experimental-experiment: fuzzy work
===================================

.. image:: https://dev.azure.com/xavierdupre3/experimental-experiment/_apis/build/status/sdpython.experimental-experiment
    :target: https://dev.azure.com/xavierdupre3/experimental-experiment/

.. image:: https://badge.fury.io/py/experimental-experiment.svg
    :target: http://badge.fury.io/py/experimental-experiment

.. image:: http://img.shields.io/github/issues/sdpython/experimental-experiment.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/experimental-experiment/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://img.shields.io/github/repo-size/sdpython/experimental-experiment
    :target: https://github.com/sdpython/experimental-experiment/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/sdpython/experimental-experiment/branch/main/graph/badge.svg?token=Wb9ZGDta8J 
    :target: https://codecov.io/gh/sdpython/experimental-experiment

**experimental-experiment** is mostly to experiment ideas.

Source are `sdpython/experimental-experiment
<https://github.com/sdpython/experimental-experiment>`_.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    design/index
    tutorial/index
    tutorial/exported
    api/index
    galleries
    command_lines
    miscellaneous/index

.. toctree::
    :maxdepth: 1
    :caption: More

    CHANGELOGS
    license

The documentation was updated on:

.. runpython::
    
    import datetime
    print(datetime.datetime.now())

With the following versions:

.. runpython::

    import numpy    
    import ml_dtypes
    import sklearn
    import onnx
    import onnxruntime
    import onnxscript
    import torch
    import transformers
    import monai
    import timm

    for m in [
        numpy,
        ml_dtypes,
        sklearn,
        onnx,
        onnxruntime,
        onnxscript,
        torch,
        transformers,
        monai,
        timm,
    ]:
        print(f"{m.__name__}: {m.__version__}")

    from experimental_experiment.ext_test_case import has_onnxruntime_training
    print(f"has_onnxruntime_training: {has_onnxruntime_training()}")

Size of the package:

.. runpython::

    import os
    import pprint
    import pandas
    from experimental_experiment import __file__
    from experimental_experiment.ext_test_case import statistics_on_folder

    df = pandas.DataFrame(statistics_on_folder(os.path.dirname(__file__), aggregation=1))
    gr = df[["dir", "ext", "lines", "chars"]].groupby(["ext", "dir"]).sum()
    print(gr)

Older versions
++++++++++++++

* `0.1.0 <../v0.1.0/index.html>`_
