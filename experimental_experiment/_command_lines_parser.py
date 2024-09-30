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
        """
        ),
    )
    parser.add_argument(
        "cmd", choices=["lighten", "unlighten", "optimize"], help="Selects a command."
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

        The last one applies all patterns but one.
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
        help="number of iterations for pattern optimization, -1 for many",
    )
    parser.add_argument(
        "--dump_applied_patterns",
        default="",
        help="dumps applied patterns in this folder if specified",
    )
    return parser


def _cmd_optimize(argv: List[Any]):
    parser = get_parser_optimize()
    args = parser.parse_args(argv[1:])

    if args.patterns == "list":
        from experimental_experiment.xoptim import get_pattern_list

        if args.verbose:
            print("prints out the list of available patterns")

        print()
        for s in ["default", "ml", "onnxruntime", "experimental"]:
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
        infer_shapes=args.infer_shapes,
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

    if args.verbose:
        print(f"save file {args.output}")
    with open(args.output, "wb") as f:
        f.write(opt_onx.SerializeToStrign())
    if args.verbose:
        print("done")


def main(argv: Optional[List[Any]] = None):
    fcts = dict(lighten=_cmd_lighten, unlighten=_cmd_unlighten, optimize=_cmd_optimize)

    if argv is None:
        argv = sys.argv[1:]
    if (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                lighten=get_parser_lighten,
                unlighten=get_parser_unlighten,
                optimize=get_parser_optimize,
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
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
