import json
import sys
import onnx
from typing import Any, List, Optional
from argparse import ArgumentParser
from textwrap import dedent


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="experiment-experiment",
        description="experiment-experiment main command line.",
        epilog="Type 'python -m experimental_experiment <cmd> --help' "
        "to get help for a specific command.",
    )
    parser.add_argument(
        "cmd",
        choices=["lighten", "unlighten"],
        help=dedent(
            """
        Selects a command.
        
        'lighten' makes a model lighter by removing the weights,
        'unlighten' restore a model produces by the previous experiment
        """
        ),
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


def main(argv: Optional[List[Any]] = None):
    fcts = dict(lighten=_cmd_lighten, unlighten=_cmd_unlighten)

    if argv is None:
        argv = sys.argv[1:]
    if (len(argv) <= 1 and argv[0] not in fcts) or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(lighten=get_parser_lighten, unlighten=get_parser_unlighten)
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
