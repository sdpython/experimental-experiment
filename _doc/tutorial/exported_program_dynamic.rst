=====================================
Exported Programs with Dynamic Shapes
=====================================

The following script shows the exported program for many short cases
and various way to retrieve the :class:`torch.fx.Graph` equivalent
to the original model.

.. runpython::
    :showcode:
    :rst:

    import inspect
    import textwrap
    import pandas
    from experimental_experiment.torch_interpreter.eval import discover, run_exporter

    cases = discover()
    print()
    print(":ref:`Summary <led-summary-exported-program>`")
    print()
    sorted_cases = sorted(cases.items())
    if unit_test_going():
        sorted_cases = sorted_cases[:3]
    for name, cls_model in sorted_cases:
        print(f"* :ref:`{name} <led-model-case-export-{name}>`")
    print()

    obs = []
    for name, cls_model in sorted(cases.items()):
        print()
        print(f".. _led-model-case-export-{name}:")
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
            "export-strict",
            "export-strict-decall",
            "export-nostrict",
            "export-nostrict-decall",
            "export-jit",
            "export-jit-decall",
            "export-tracing",
        ):
            expname = exporter.replace("export-", "")
            print()
            print(expname)
            print("+" * len(expname))
            print()
            res = run_exporter(exporter, cls_model, True, quiet=True)
            case_ref = f":ref:`{name} <led-model-case-export-{name}>`"
            if "exported" in res:
                print("::")
                print()
                print(textwrap.indent(str(res["exported"].graph), "    "))
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
    print(".. _led-summary-exported-program:")
    print()
    print("Summary")
    print("+++++++")
    print()
    df = pandas.DataFrame(obs)
    piv = df.pivot(index="case", columns="exporter", values="error")
    print(piv.to_markdown(tablefmt="rst"))
    print()
