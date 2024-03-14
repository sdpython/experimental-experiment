import time
import warnings
from typing import List, Union
from onnx import ModelProto
from onnx.inliner import inline_local_functions


def inline_model_proto(model_proto: ModelProto) -> ModelProto:
    """
    Inlines a model.

    :param model_proto: ModelProto
    :return: inlined model
    """
    # model = onnx.load(input_file_name, load_external_data=False)
    return inline_local_functions(model_proto)


def optimize_model_proto(
    model_proto: ModelProto, verbose: int = 0, onnx_shape_inference: bool = False
) -> ModelProto:
    """
    Optimizes a model proto to optimize onnxruntime.

    :param model_proto: ModelProto
    :param verbose: verbosity
    :param onnx_shape_inference: enable shape inference
    :return: optimized model

    You should run that before calling this function

    ::

        onnx_model = exported.to_model_proto(
            opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version
        )

        from experimental_experiment.convert.convert_helper import optimize_model_proto
        onnx_model = optimize_model_proto(onnx_model)
    """
    from onnxrewriter.optimizer import optimize
    from onnxrewriter.rewriter import rewrite

    begin = time.perf_counter()

    if verbose:
        print(
            f"[optimize_model_proto] starts inliner with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    model_proto = inline_model_proto(model_proto)

    if verbose:
        print(
            f"[optimize_model_proto] inliner done in {time.perf_counter() - begin} seconds."
        )
        print(
            f"[optimize_model_proto] starts optimize with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    begin = time.perf_counter()

    model_proto = optimize(
        model_proto,
        num_iterations=2,
        onnx_shape_inference=onnx_shape_inference,
        # function_aware_folding=True,
    )

    if verbose:
        print(
            f"[optimize_model_proto] optimize done in {time.perf_counter() - begin} seconds."
        )
        print(
            f"[optimize_model_proto] starts rewrite with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    begin = time.perf_counter()

    try:
        model_proto = rewrite(model_proto)

        if verbose:
            print(
                f"[optimize_model_proto] rewrite done in {time.perf_counter() - begin} "
                f"seconds with {len(model_proto.graph.node)} nodes and "
                f"{len(model_proto.functions)} local functions"
            )

    except ValueError as e:
        warnings.warn(
            f"onnxrewrite.rewrite failed due to {e}, "
            f"saving the model into 'bug-onnxrewriter.onnx'"
        )
        with open("bug-onnxrewriter.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())
        if verbose:
            print(
                f"[optimize_model_proto] failed in {time.perf_counter() - begin} "
                f"seconds (see 'bug-onnxrewriter.onnx')."
            )

    return model_proto


def ort_optimize(
    onnx_model: Union[str, ModelProto],
    output: str,
    providers: Union[str, List[str]] = "cpu",
    disable_aot: bool = False,
):
    """
    Optimizes the model with onnxruntime.

    :param onnx_model: ModelProto or file path
    :param output: path for the output
    :param providers: providers, cpu, cuda or a list of providers
    :param disable_aot: disable AOT
    """
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = output
    if disable_aot:
        opts.add_session_config_entry("session.disable_aot_function_inlining", "1")

    if providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif providers == "cuda":
        providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    assert isinstance(providers, list), f"Unexpected value for providers={providers!r}"
    onnxruntime.InferenceSession(
        (
            onnx_model.SerializeToString()
            if isinstance(onnx_model, ModelProto)
            else onnx_model
        ),
        opts,
        providers=providers,
    )
