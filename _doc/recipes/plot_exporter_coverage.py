"""
.. _l-plot-exporter-coverage:

Measures the exporter success on many test cases
================================================

All test cases can be found in module
:mod:`experimental_experiment.torch_interpreter.eval.model_cases`.

"""

from experimental_experiment.args import get_parsed_args

script_args = get_parsed_args(
    "plot_exporter_coverage",
    description=__doc__,
    exporter=("all", "an exporter to rerun"),
    dynamic=("all", "use dyanmic shapes"),
    case=("all", "model cases"),
    quiet=("1", "0 or 1"),
    verbose=("1", "verbosity"),
    expose="exporter,dyanmic,case,quiet,verbose",
)

exporters = (
    (
        "custom-strict",
        "custom-nostrict",
        "custom-strict-dec",
        "custom-nostrict-dec",
        "custom-tracing",
        "dynamo",
        "dynamo-ir",
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

#####################################
# The results

df = pandas.DataFrame(obs).sort_values(["dynamic", "name", "exporter"]).reset_index(drop=True)
df.to_csv("plot-exporter-coverage.csv", index=False)
df.to_excel("plot-exporter-coverage.xlsx")
print(df)

######################################
# Simple Results

piv = df.pivot(
    index=["dynamic", "name"],
    columns=["exporter"],
    values="error_step" if "error_step" in df.columns else "success",
).fillna("")
piv.to_excel("plot-exporter-coverage-summary.xlsx")
print(piv)
