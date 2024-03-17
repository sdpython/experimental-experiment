import contextlib
import io
import warnings
from typing import Any, Dict, List, Optional, Union
from onnx import ModelProto


def check_model_ort(
    onx: ModelProto, providers: Optional[Union[str, List[str]]] = None
) -> "onnxruntime.InferenceSession":  # noqa: F821
    """
    Loads a model with onnxruntime.

    :param onx: ModelProto
    :param providers: list of providers, None fur CPU, cpu for CPU, cuda for CUDA
    :return: InferenceSession
    """
    from onnxruntime import InferenceSession

    if providers is None or providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif providers == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if isinstance(onx, str):
        try:
            return InferenceSession(onx, providers=providers)
        except Exception as e:
            import onnx
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"onnxruntime cannot load the model "
                f"due to {e}\n{onnx_simple_text_plot(onnx.load(onx))}"
            )
        return
    try:
        return InferenceSession(onx.SerializeToString(), providers=providers)
    except Exception as e:
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        raise AssertionError(
            f"onnxruntime cannot load the model"
            f"due to {e}\n{onnx_simple_text_plot(onx)}"
        )


def export_to_onnx(
    model: Any,
    *args: List[Any],
    verbose: int = 0,
    return_builder: bool = False,
    torch_script: bool = True,
    target_opset: int = 18,
    prefix: Optional[str] = None,
    rename_inputs: bool = False,
) -> Dict[str, Union[str, ModelProto, "GraphBuilder"]]:  # noqa: F821
    """
    Exports a model to ONNX.

    :param model: model to export
    :param args: arguments
    :param verbose: verbosity
    :param return_builder: returns the builder
    :param torch_script: export with torch.script as well
    :param target_opset: opset to export into
    :param prefix: prefix to choose to export into
    :param rename_inputs: rename the inputs into ``input_{i}``
    :return: dictionary with ModelProto, builder, filenames
    """
    from .xbuilder import OptimizationOptions
    from .torch_interpreter import to_onnx

    ret = {}
    if torch_script and prefix is not None:
        import torch

        filename = f"{prefix}.onnx"
        with contextlib.redirect_stdout(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(model, args, filename, input_names=["input"])
                ret["torch.script"] = filename

    onx = to_onnx(
        model,
        tuple(args),
        input_names=[f"input{i}" for i in range(len(args))] if rename_inputs else None,
        options=OptimizationOptions(verbose=verbose),
        verbose=verbose,
        return_builder=return_builder,
    )
    ret["proto"] = onx
    if prefix is not None:
        filename = f"{prefix}.custom.onnx"
        with open(filename, "wb") as f:
            f.write((onx[0] if return_builder else onx).SerializeToString())
        ret["custom"] = filename
    return ret
