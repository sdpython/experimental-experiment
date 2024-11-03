"""
Check Model
===========

::

    python -m experimental_experiment.torch_bench.check_model \\
           --test optimizer --model dump3bug.onnx
"""


def main():
    """
    Main function for command line
    ``python -m experimental_experiment.torch_bench.check_model``.
    """
    from experimental_experiment.args import get_parsed_args

    script_args = get_parsed_args(
        "experimental_experiment.torch_bench.check_model",
        test=("optimizer", "the scenario to test, optimizer runs onnxscript optimizations"),
        model=("", "needs to be specified"),
        suffix=(".check", "suffix added to the model"),
        verbose=(0, "verbosity"),
        description=__doc__,
        expose="test,suffix,model",
    )

    import os
    import onnx

    assert os.path.exists(script_args.model), f"Model {script_args.model!r} does not exist."
    name, ext = os.path.splitext(script_args.model)
    output = f"{name}{script_args.suffix}{ext}"

    verbose = script_args.verbose

    if verbose:
        print(f"check model {script_args.model!r}")
        print(f"with command {script_args.test!r}")

    if script_args.test == "optimizer":
        from onnxscript import optimizer

        if verbose:
            print(f"loading model {script_args.model!r}")
        onx = onnx.load(script_args.model)
        if verbose:
            print("optimize model")
        optimized = optimizer.optimize(onx)
        if verbose:
            print("done")
            print(f"save into {output}")
        with open(output, "wb") as f:
            f.write(optimized.SerializeToString())
        if verbose:
            print("done")
    else:
        raise AssertionError(f"Unsupported test {script_args.test!r}")


if __name__ == "__main__":
    main()
