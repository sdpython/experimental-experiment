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


def optimize_model_proto(model_proto: ModelProto) -> ModelProto:
    """
    Optimizes a model proto to optimize onnxruntime.

    :param model_proto: ModelProto
    :return: optimized model

    You should run that before calling this function

    ::

        onnx_model = exported.to_model_proto(
            opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version
        )

        from experimental_experiment.convert.convert_helper import optimize_model_proto
        onnx_model = optimize_model_proto(onnx_model)
    """
    try:
        from onnxrewriter.optimizer import optimize
        from onnxrewriter.rewriter.transformers import rewrite
    except ImportError:
        return model_proto

    # model_proto = inline_model_proto(model_proto)
    model_proto = optimize(
        model_proto,
        num_iterations=2,
        onnx_shape_inference=False,
        function_aware_folding=True,
    )
    model_proto = rewrite(model_proto)
    return model_proto


def ort_optimize(
    onnx_model: Union[str, ModelProto],
    output: str,
    providers: Union[str, List[str]] = "cpu",
):
    """
    Optimizes the model with onnxruntime.

    :param onnx_model: ModelProto or file path
    :param output: path for the output
    :param providers: providers, cpu, cuda or a list of providers
    """
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = output
    if providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif providers == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
