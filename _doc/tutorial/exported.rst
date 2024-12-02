.. _l-exported-program-cases:

=================
Exported Programs
=================

The following script shows the exported program for many short cases
and various way to retrieve the :class:`torch.fx.Graph` equivalent
to the original model.

.. runpython::
    :showcode:
    :rst:

    import inspect
    import textwrap
    from experimental_experiment.torch_interpreter.eval import discover, run_exporter

    cases = discover()
    print()
    for name, cls_model in sorted(cases.items()):
        print(f"* :ref:`{name} <l-model-case-export-{name}>`")
    print()

    for name, cls_model in sorted(cases.items()):
        print()
        print(f".. _l-model-case-export-{name}:")
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
            "export-strict-decomposition",
            "export-nostrict",
            "export-nostrict-decomposition",
            "export-jit",
            "export-jit-decomposition",
            "export-tracing",
        ):
            expname = exporter.replace("export-", "")
            print()
            print(expname)
            print("+" * len(expname))
            print()
            res = run_exporter(exporter, cls_model, False, quiet=True)
            if "exported" in res:
                print("::")
                print()
                print(textwrap.indent(str(res["exported"].graph), "    "))
                print()
            else:
                print("**FAILED**")
                print()
                print("::")
                print()
                print(textwrap.indent(str(res["error"]), "    "))
                print()
