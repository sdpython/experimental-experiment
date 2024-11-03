"""
Profiles an execution
=====================

The datas should be saved with script ``dort_bench.py`` and option ``--export <something>``.

::

    python -m experimental_experiment.torch_bench.dort_bench_profile --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.dort_bench_profile \\
           --model model.onnx --inputs model.onnx.pkl

"""


def main():
    """
    Main function for command line
    ``python -m experimental_experiment.torch_bench.dort_bench_profile``.
    """
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        "experimental_experiment.torch_bench.dort_bench_profile",
        description=__doc__,
        model=("model.onnx", "model to load"),
        inputs=("model.onnx.mkl", "inputs for the model"),
        profile=(0, "runs the profiling"),
        rewrite=(0, "rewrite again"),
        debug=(0, "run a backend to debug"),
        repeat=5,
        warmup=5,
        expose="model,inputs,warmup,repeat,profile,rewrite,debug",
    )

    import pickle
    import time
    import numpy as np
    import onnx
    import matplotlib.pyplot as plt
    import torch
    from torch._C import _from_dlpack
    from onnxruntime import InferenceSession, SessionOptions, RunOptions
    from onnxruntime.capi import _pybind_state as ORTC
    from experimental_experiment.torch_dynamo.fast_backend import (
        _run_onnx_session_with_ortvaluevector,
    )
    from experimental_experiment.convert.convert_helper import optimize_model_proto_oxs
    from experimental_experiment.torch_dynamo.backend_helper import get_dimensions

    print(f"-- loading inputs {args.model}")
    with open(args.inputs, "rb") as f:
        inputs = pickle.load(f)
    print("-- done")

    assert isinstance(inputs, list), f"Unexpected type {type(inputs)} for {args.inputs}"
    assert len(inputs) == 3, f"Unexpected length {len(inputs)} for {args.inputs}"
    print(f"input_names={inputs[0]}")
    print(f"output_names={inputs[2]}")
    max_device = -1
    for i, t in enumerate(inputs[1]):
        if isinstance(t, torch.Tensor):
            print(f"input {i}: device={t.get_device()} dtype={t.dtype} shape={t.shape}")
            max_device = max(t.get_device(), max_device)
        else:
            print(f"input {i}: type={type(t)}")

    providers = (
        [("CUDAExecutionProvider", {"device_id": max_device}), ("CPUExecutionProvider", {})]
        if max_device >= 0
        else ["CPUExecutionProvider"]
    )

    sess_options = SessionOptions()
    if args.profile in (1, "1"):
        sess_options.enable_profiling = True
    run_options = RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    if args.rewrite in (1, "1"):
        model_model = args.model.replace(".onnx", ".rewrite.onnx")
        print(f"-- optimize again into {model_model}")
        proto = onnx.load(args.model)
        new_proto = optimize_model_proto_oxs(proto, verbose=args.verbose)
        onnx.save(new_proto, model_model)
        print("-- done")
    else:
        model_model = args.model

    print(f"-- loading model {model_model}")
    onnx_model = onnx.load(model_model)
    is_dimension_in, is_dimension_out = get_dimensions(onnx_model)
    sess = InferenceSession(model_model, sess_options, providers=providers)
    print("-- done")

    TORCH_DTYPE_TO_NUMPY_DTYPE = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

    DEVICES = {-1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)}
    for i in range(torch.cuda.device_count()):
        DEVICES[i] = ORTC.OrtDevice(ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i)

    input_names, output_names = inputs[0], inputs[2]
    inputs = inputs[1]
    is_cuda = max_device >= 0

    if args.debug:
        print("-- debugging")
        from experimental_experiment.reference import ExtendedReferenceEvaluator

        ref = ExtendedReferenceEvaluator(model_model, verbose=10)
        feeds = dict(
            zip(
                input_names,
                [
                    (
                        t.detach().cpu().numpy()
                        if isinstance(t, torch.Tensor)
                        else np.array([int(t)], dtype=np.int64)
                    )
                    for t in inputs
                ],
            )
        )
        ref.run(None, feeds)
        print("-- end debugging")

    print(f"-- warmup: {args.warmup}")
    begin = time.perf_counter()
    for i in range(args.warmup):
        if is_cuda:
            torch.cuda.synchronize()
        res = _run_onnx_session_with_ortvaluevector(
            ORTC.OrtValueVector,
            _from_dlpack,
            TORCH_DTYPE_TO_NUMPY_DTYPE,
            DEVICES,
            run_options,
            sess,
            input_names,
            inputs,
            output_names,
            is_dimension_in=is_dimension_in,
            is_dimension_out=is_dimension_out,
        )
        if is_cuda:
            torch.cuda.synchronize()
        if i == 0:
            for ti, t in enumerate(res):
                if isinstance(t, torch.Tensor):
                    print(
                        f"  output {ti}: device={t.get_device()} "
                        f"dtype={t.dtype} - shape={t.shape}"
                    )
                elif isinstance(t, torch.SymInt):
                    print(f"  output {ti}: dimension {t}")
                elif isinstance(t, torch.SymFloat):
                    print(f"  output {ti}: dimensiof {t}")

    warmup_time = time.perf_counter() - begin
    print(f"-- done: warmup time {warmup_time}")

    print(f"-- measure: {args.repeat}")
    times = []
    for _ in range(args.repeat):
        if is_cuda:
            torch.cuda.synchronize()
        begin = time.perf_counter()
        res = _run_onnx_session_with_ortvaluevector(
            ORTC.OrtValueVector,
            _from_dlpack,
            TORCH_DTYPE_TO_NUMPY_DTYPE,
            DEVICES,
            run_options,
            sess,
            input_names,
            inputs,
            output_names,
            is_dimension_in=is_dimension_in,
            is_dimension_out=is_dimension_out,
        )
        if is_cuda:
            torch.cuda.synchronize()
        d = time.perf_counter()
        times.append(d - begin)
    print(f"-- times: {np.mean(times)} - {times}")

    if args.profile in (1, "1"):
        from onnx_extended.tools.js_profile import (
            js_profile_to_dataframe,
            plot_ort_profile,
        )
        from onnx_extended.tools.js_profile import plot_ort_profile_timeline

        def _align(s, n):
            if len(s) >= n:
                return s[:n]
            return s + " " * (n - len(s))

        prof = sess.end_profiling()
        print(f"-- profiling name {prof}")
        onx = onnx.load(model_model)
        n_nodes = len(onx.graph.node)
        n_unique_nodes = len(set(n.name for n in onx.graph.node))
        print(
            "\n".join(
                f"{_align(n.op_type, 16)} - {n.input} -> {n.output}" for n in onx.graph.node
            )
        )

        # first graph: aggregated profile
        df = js_profile_to_dataframe(prof, first_it_out=True)
        df.to_csv(f"{model_model}.csv", errors="ignore")
        df.to_excel(f"{model_model}.xlsx")
        assert set(df["it==0"]) == {0, 1}
        for v in set(df["it==0"]):
            dfv = df[df["it==0"] == v]
            vs = "after" if v == 0 else "warmup"
            fig, ax = plt.subplots(1, 2, figsize=(10, max(5, n_unique_nodes // 12)))

            plot_ort_profile(
                dfv, ax[0], ax[1], f"profiling {vs} {n_nodes} nodes\n{model_model}"
            )
            fig.tight_layout()
            fig.savefig(f"{model_model}_{vs}.png")

        # second graph: timeline

        fig, ax = plt.subplots(1, 1, figsize=(5, max(5, n_nodes)))
        iteration = args.repeat - 2
        plot_ort_profile_timeline(
            df,
            ax,
            iteration=iteration,
            title=f"profiling it={iteration} {n_nodes} nodes\n{model_model}",
        )
        fig.tight_layout()
        fig.savefig(f"{model_model}_{iteration}_timeline.png")


if __name__ == "__main__":
    main()
