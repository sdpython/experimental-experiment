======================================
Exported into ONNX with Dynamic Shapes
======================================

The following script shows the exported program for many short cases
and various way to retrieve the :class:`torch.fx.Graph` equivalent
to the original model. The tested scenarios are described at
:ref:`l-scenarios-exported-program-export`.

.. runpython::
    :showcode:
    :rst:
    :toggle: code
    :warningout: UserWarning

    import inspect
    import textwrap
    import pandas
    from experimental_experiment.torch_interpreter.eval import discover, run_exporter
    from experimental_experiment.ext_test_case import unit_test_going
    from experimental_experiment.helpers import pretty_onnx

    cases = discover()
    print()
    print(":ref:`Summary <lod-summary-exported-program>`")
    print()
    sorted_cases = sorted(cases.items())
    if unit_test_going():
        sorted_cases = sorted_cases[:3]
    for name, cls_model in sorted_cases:
        print(f"* :ref:`{name} <lod-model-case-export-{name}>`")
    print()

    obs = []
    for name, cls_model in sorted(cases.items()):
        print()
        print(f".. _lod-model-case-export-{name}:")
        print()
        print(name)
        print("=" * len(name))
        print()
        print("forward")
        print("+++++++")
        print()
        print("::")
        print()
        print(textwrap.indent(textwrap.dedent(inspect.getsource(cls_model.forward)), "    "))
        print()
        for exporter in (
            "custom",
            "custom-tracing",
            "dynamo-ir",
        ):
            expname = exporter.replace("export-", "")
            print()
            print(expname)
            print("+" * len(expname))
            print()
            res = run_exporter(exporter, cls_model, True, quiet=True)
            case_ref = f":ref:`{name} <lod-model-case-export-{name}>`"
            if "exported" in res:
                print("::")
                print()
                print(textwrap.indent(pretty_onnx(res["onnx"]), "    "))
                print()
                obs.append(dict(case=case_ref, error="", exporter=exporter))
            else:
                print("**FAILED**")
                print()
                print("::")
                print()
                print(textwrap.indent(str(res["error"]), "    "))
                print()
                obs.append(dict(case=case_ref, error="FAIL", exporter=exporter))

    print()
    print(".. _lod-summary-exported-program:")
    print()
    print("Summary")
    print("+++++++")
    print()
    df = pandas.DataFrame(obs)
    piv = df.pivot(index="case", columns="exporter", values="error")
    print(piv.to_markdown(tablefmt="rst"))
    print()
