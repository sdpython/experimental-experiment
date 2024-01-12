
.. image:: https://github.com/sdpython/experimental-experiment/raw/main/_doc/_static/logo.png
    :width: 120

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

Getting started
+++++++++++++++

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.

::

    git clone https://github.com/sdpython/experimental-experiment.git
    pip install onnxruntime-gpu pynvml
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>

Then install *onnx-rewriter*.

Compare torch exporters
+++++++++++++++++++++++

The script evaluates the memory peak, the computation time of the exporters.
It also compares the exported models when run through onnxruntime.
The full script takes around 20 minutes to complete. It stores on disk
all the graphs, the data used to draw them, and the models.

::

    python _doc/examples/plot_torch_export.py -s large
