"""
.. _l-plot-exporter-coverage:

Measures the exporter success on many test cases
================================================

All test cases can be found in module
:mod:`experimental_experiment.torch_interpreter.eval.model_cases`.
Page :ref:`l-export-supported-signatures` shows the exported
program for many of those cases.

"""

from experimental_experiment.args import get_parsed_args

script_args = get_parsed_args(
    "plot_exporter_coverage",
    description=__doc__,
    exporter=("all", "an exporter to rerun"),
    dynamic=("all", "use dyanmic shapes"),
    case=(
        "three",
        "model cases, two for the first two (to test), "
        "all to select all, a name or a regular expression fior a subset",
    ),
    quiet=("1", "0 or 1"),
    verbose=("1", "verbosity"),
    expose="exporter,dyanmic,case,quiet,verbose",
)

exporters = (
    (
        "export-strict",
        "export-strict-dec",
        "export-nostrict",
        "export-nostrict-dec",
        "export-jit",
        "export-tracing",
        "custom-strict",
        "custom-nostrict",
        "custom-strict-dec",
        "custom-nostrict-dec",
        "custom-tracing",
        "dynamo",
        "dynamo-ir",
        "script",
    )
    if script_args.exporter == "all"
    else script_args.exporter.split(",")
)
dynamic = (0, 1) if script_args.dynamic == "all" else (int(script_args.dynamic),)
cases = None if script_args.case == "all" else script_args.case.split(",")
quiet = bool(int(script_args.quiet))
verbose = int(script_args.verbose)

import pandas
from experimental_experiment.torch_interpreter.eval import evaluation

obs = evaluation(
    exporters=exporters, dynamic=dynamic, cases=cases, quiet=quiet, verbose=verbose
)

# %%
# The results

df = pandas.DataFrame(obs).sort_values(["dynamic", "name", "exporter"]).reset_index(drop=True)
df.to_csv("plot-exporter-coverage.csv", index=False)
df.to_excel("plot-exporter-coverage.xlsx")
for c in ["error", "error_step"]:
    if c in df.columns:
        df[c] = df[c].fillna("")
print(df)

# %%
# Errors if any or all successes.

piv = df.pivot(
    index=["dynamic", "name"],
    columns=["exporter"],
    values="error_step" if "error_step" in df.columns else "success",
)

piv.to_excel("plot-exporter-coverage-summary.xlsx")
print(piv)
