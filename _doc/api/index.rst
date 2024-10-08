
===
API
===

.. toctree::
    :maxdepth: 1
    :caption: Builder and Optimizer

    onnx_export
    graph_builder
    graph_builder_pattern
    graph_builder_patterns
    order_optimization
    interpreter
    aten_function
    aten_method
    prims_function
    torch_dynamo
    dimension

.. toctree::
    :maxdepth: 1
    :caption: Tools

    args
    bench_run
    model_run
    gradient
    reference
    torch_model_container
    torch_helper
    torch_test
    convert
    misc

**Versions**

The documentation was generated with the following versions.

.. runpython::
    :showcode:

    import onnx
    import onnx_array_api
    import onnxruntime
    import torch
    import transformers
    import onnxscript

    for pck in [onnx, onnx_array_api, onnxruntime, torch, transformers, onnxscript]:
        if pck is None:
            continue
        try:
            print(f"{pck.__name__}: {pck.__version__}")
        except AttributeError as e:
            print(f"{pck.__name__}: {e}")

**Statistics**

.. runpython::
    :showcode:

    import os
    import pandas
    import experimental_experiment
    from experimental_experiment.ext_test_case import (
        statistics_on_file,
        statistics_on_folder,
    )

    root = os.path.dirname(experimental_experiment.__file__)
    stat = statistics_on_folder(
        [
            root,
            os.path.join(root, "..", "_doc"),
            os.path.join(root, "..", "_unittests"),
        ],
        aggregation=2,
    )

    df = pandas.DataFrame(stat)
    gr = df.drop("name", axis=1).groupby(["ext", "dir"]).sum().reset_index()
    gr = gr[gr["dir"] != "_doc/auto_examples"]
    print(gr)
    print("--------------------")
    print("total in experimental_experiment/...")
    total = (
        gr[gr["dir"].str.contains("experimental_experiment/")]
        .drop(["ext", "dir"], axis=1)
        .sum(axis=0)
    )
    print(total)

