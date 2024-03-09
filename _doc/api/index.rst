
===
API
===

.. toctree::
    :maxdepth: 1

    gradient
    reference
    graph_builder
    graph_builder_pattern
    interpreter
    onnx_export
    aten_function
    aten_method
    prims_function
    convert
    torch_helper
    torch_dynamo
    misc

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
        total = (
            gr[gr["dir"].str.contains("experimental_experiment/")]
            .drop(["ext", "dir"], axis=1)
            .sum(axis=0)
        )
        print(total)

