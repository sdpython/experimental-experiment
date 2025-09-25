import json
import sys
import textwrap
import onnx
from typing import Any, List, Optional
from argparse import ArgumentParser, RawTextHelpFormatter, BooleanOptionalAction
from textwrap import dedent


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="experimental-experiment",
        description="experiment-experiment main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
        Type 'python -m experimental_experiment <cmd> --help'
        to get help for a specific command.

        lighten    - makes an onnx model lighter by removing the weights,
        unlighten  - restores an onnx model produces by the previous experiment
        optimize   - optimizes an onnx model by fusing nodes
        run        - runs a model and measure the inference time
        print      - prints the model on standard output
        find       - find node consuming or producing a result
        """
        ),
    )
    parser.add_argument(
        "cmd",
        choices=["lighten", "unlighten", "optimize", "run", "print", "find"],
        help="Selects a command.",
    )
    return parser


def get_parser_lighten() -> ArgumentParser:
    parser = ArgumentParser(
        prog="lighten",
        description=dedent(
            """
        Removes the weights from a heavy model, stores statistics to restore
        random weights.
        """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "a big onnx file to the repository.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to lighten",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="onnx model to output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_lighten(argv: List[Any]):
    from .onnx_tools import onnx_lighten

    parser = get_parser_lighten()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    new_onx, stats = onnx_lighten(onx, verbose=args.verbose)
    jstats = json.dumps(stats)
    if args.verbose:
        print("save file {args.input!r}")
    if args.verbose:
        print("write file {args.output!r}")
    with open(args.output, "wb") as f:
        f.write(new_onx.SerializeToString())
    name = f"{args.output}.stats"
    with open(name, "w") as f:
        f.write(jstats)
    if args.verbose:
        print("done")


def get_parser_unlighten() -> ArgumentParser:
    parser = ArgumentParser(
        prog="unlighten",
        description=dedent(
            """
        Restores random weights for a model reduces with command lighten,
        the command expects to find a file nearby with extension '.stats'.
        """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "a big onnx file to the repository.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to unlighten",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="onnx model to output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_unlighten(argv: List[Any]):
    from .onnx_tools import onnx_unlighten

    parser = get_parser_lighten()
    args = parser.parse_args(argv[1:])
    new_onx = onnx_unlighten(args.input, verbose=args.verbose)
    if args.verbose:
        print(f"save file {args.output}")
    with open(args.output, "wb") as f:
        f.write(new_onx.SerializeToString())
    if args.verbose:
        print("done")


def get_parser_optimize() -> ArgumentParser:
    parser = ArgumentParser(
        prog="optimize",
        formatter_class=RawTextHelpFormatter,
        description=dedent(
            """
        Optimizes an onnx model by fusing nodes. It looks for patterns in the graphs
        and replaces them by the corresponding nodes. It also does basic optimization
        such as removing identity nodes or unused nodes.
        """
        ),
        epilog=textwrap.dedent(
            """
        The goal is to make the model faster.
        Argument patterns defines the patterns to apply or the set of patterns.
        It defines the following sets of patterns

        - '' or none    : no pattern optimization
        - default       : rewrites standard onnx operators into other standard onnx operators
        - ml            : does the same with operators defined in domain 'ai.onnx.ml'
        - onnxruntime   : introduces fused nodes defined in onnxruntime
        - experimental  : introduces fused nodes defined in module onnx-extended

        Examples of values:

        - none
        - default
        - ml
        - default+ml+onnxruntime+experimental
        - default+ml+onnxruntime+experimental-ReshapeReshapePattern

        The last one applies all patterns but one. The list of patterns can be
        obtained by running:

            python -m experimental_experiment optimize --patterns=list -i '' -o ''
        """
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to optimize",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="onnx model to output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        type=int,
        help="verbosity",
    )
    parser.add_argument(
        "--infer_shapes",
        default=True,
        action=BooleanOptionalAction,
        help="infer shapes before optimizing the model",
    )
    parser.add_argument(
        "--identity",
        default=True,
        action=BooleanOptionalAction,
        help="remove identity nodes",
    )
    parser.add_argument(
        "--folding",
        default=True,
        action=BooleanOptionalAction,
        help="does constant folding",
    )
    parser.add_argument(
        "--processor",
        default="CPU",
        help=textwrap.dedent(
            """
            optimization for a specific processor, CPU, CUDA or both CPU,CUDA,
            some operators are only available in one processor"""
        ).strip("\n"),
    )
    parser.add_argument(
        "--patterns",
        default="default",
        help="patterns optimization to apply, see below",
    )
    parser.add_argument(
        "--max_iter",
        default=-1,
        type=int,
        help="number of iterations for pattern optimization, -1 for many",
    )
    parser.add_argument(
        "--dump_applied_patterns",
        default="",
        help="dumps applied patterns in this folder if specified",
    )
    parser.add_argument(
        "--remove_shape_info",
        default=True,
        action=BooleanOptionalAction,
        help="remove shape information before outputting the model",
    )
    return parser


def _cmd_optimize(argv: List[Any]):
    parser = get_parser_optimize()
    args = parser.parse_args(argv[1:])

    if args.patterns == "list":
        from experimental_experiment.xoptim import get_pattern_list

        if args.verbose:
            print("prints out the list of available patterns")

        for s in ["default", "ml", "onnxruntime", "experimental"]:
            print()
            print(f"-- {s} patterns")
            pats = get_pattern_list(s)
            for p in pats:
                print(p)
        return

    from experimental_experiment.xoptim import get_pattern_list
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    if args.verbose:
        print(f"load file {args.input}")
    onx = onnx.load(args.input, load_external_data=False)
    if args.verbose:
        print(f"load file {args.input}")

    pats = get_pattern_list(args.patterns, verbose=args.verbose)

    if args.verbose:
        print(f"begin optimization with {len(pats)} patterns")
    gr = GraphBuilder(
        onx,
        infer_shapes_options=args.infer_shapes,
        optimization_options=OptimizationOptions(
            patterns=pats,
            verbose=args.verbose,
            remove_unused=True,
            constant_folding=args.folding,
            remove_identity=args.identity,
            max_iter=args.max_iter,
            dump_applied_patterns=(
                None if not args.dump_applied_patterns else args.dump_applied_patterns
            ),
            processor=args.processor,
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)
    if args.remove_shape_info:
        if args.verbose:
            print(f"remove shape information {len(opt_onx.graph.value_info)}")
        del opt_onx.graph.value_info[:]
        if args.verbose:
            print("done")

    if args.verbose:
        print(f"save file {args.output}")
    with open(args.output, "wb") as f:
        f.write(opt_onx.SerializePartialToString())
    if args.verbose:
        print("done")


def get_parser_run() -> ArgumentParser:
    parser = ArgumentParser(
        prog="run",
        formatter_class=RawTextHelpFormatter,
        description=dedent(
            """
        Runs a model with dummy inputs and measures the inference time.
        """
        ),
        epilog=textwrap.dedent(
            """
        It checks a model runs and the inference time on the same inputs.
        """
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model to optimize",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        type=int,
        help="verbosity",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=1,
        required=False,
        type=int,
        help="batch size, if it can be changed",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        default=10,
        required=False,
        type=int,
        help="number of time to repeat the measure",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        default=5,
        required=False,
        type=int,
        help="number of time to warmup the model",
    )
    parser.add_argument(
        "-p",
        "--processor",
        default="CPU",
        help=textwrap.dedent(
            """
            providers to launch, CPU, CUDA, CUDA,CPU.
            """
        ).strip("\n"),
    )
    parser.add_argument("--validate", default="", help="validate the output with another model")
    return parser


def _cmd_run(argv: List[Any]):
    parser = get_parser_run()
    args = parser.parse_args(argv[1:])

    from .model_run import model_run

    stats = model_run(
        model=args.model,
        repeat=args.repeat,
        warmup=args.warmup,
        verbose=args.verbose,
        batch_size=args.batch,
        processor=args.processor,
        validate=args.validate,
    )

    for k, v in sorted(stats.items()):
        print(f":{k},{v};")


def get_parser_print() -> ArgumentParser:
    parser = ArgumentParser(
        prog="print",
        description=dedent(
            """
        Prints the model on the standard output.
        """
        ),
        epilog="To show a model.",
    )
    parser.add_argument(
        "fmt", choices=["pretty", "builder", "raw"], help="Format to use.", default="pretty"
    )
    parser.add_argument("input", type=str, help="onnx model to load")
    return parser


def _cmd_print(argv: List[Any]):
    parser = get_parser_print()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    if args.fmt == "raw":
        print(onx)
    elif args.fmt == "pretty":
        from .helpers import pretty_onnx

        print(pretty_onnx(onx))
    elif args.fmt == "builder":
        from .xbuilder import GraphBuilder

        gr = GraphBuilder(onx)
        print(gr.pretty_text())
    else:
        raise ValueError(f"Unexpected value fmt={args.fmt!r}")


def get_parser_find() -> ArgumentParser:
    parser = ArgumentParser(
        prog="find",
        description=dedent(
            """
        Look into a model and search for a set of names,
        tells which node is consuming or producing it.
        """
        ),
        epilog="Enables Some quick validation.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to unlighten",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        required=False,
        help="names to look at comma separated values",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_find(argv: List[Any]):
    from .onnx_tools import onnx_find

    parser = get_parser_find()
    args = parser.parse_args(argv[1:])
    onnx_find(args.input, verbose=args.verbose, watch=set(args.names.split(",")))


def main(argv: Optional[List[Any]] = None):
    fcts = dict(
        lighten=_cmd_lighten,
        unlighten=_cmd_unlighten,
        optimize=_cmd_optimize,
        run=_cmd_run,
        print=_cmd_print,
        find=_cmd_find,
    )

    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0 or (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                lighten=get_parser_lighten,
                unlighten=get_parser_unlighten,
                optimize=get_parser_optimize,
                run=get_parser_run,
                print=get_parser_print,
                find=get_parser_find,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(f"Unknown command {cmd!r}, use --help to get the list of known command.")
