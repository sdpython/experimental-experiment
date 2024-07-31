import time
from typing import Any, Dict, List, Optional, Union
from onnx import ModelProto, helper as oh, load as onnx_load
from onnx.inliner import inline_local_functions


def inline_model_proto(model_proto: ModelProto) -> ModelProto:
    """
    Inlines a model.

    :param model_proto: ModelProto
    :return: inlined model
    """
    # model = onnx.load(input_file_name, load_external_data=False)
    return inline_local_functions(model_proto)


def _fix_details(model: ModelProto, verbose: int = 0) -> ModelProto:
    # ScatterND + Aten ops
    print("[_fix_details] START")
    for node in model.graph.node:
        if node.op_type == "ScatterND":
            if len(node.attribute) == 0:
                if verbose:
                    print("[_fix_details] ScatterND, add reduction to add")
                node.attribute.append(oh.make_attribute("reduction", "add"))
            else:
                red = node.attribute[0].s
                if red != b"add":
                    if verbose:
                        print("[_fix_details] ScatterND, change reduction to add")
                    del node.attribute[:]
                    node.attribute.append(oh.make_attribute("reduction", "add"))
        elif node.op_type == "ATen":
            fname = None
            for att in node.attribute:
                if att.name == "operator":
                    fname = att.s
            if fname == b"_scaled_dot_product_efficient_attention_backward":
                if verbose:
                    print(
                        "[_fix_details] ATen, delete last output for "
                        "_scaled_dot_product_efficient_attention_backward"
                    )
                outputs = list(node.output)
                del node.output[:]
                outputs[-1] = ""
                node.output.extend(outputs)
    if verbose:
        print("[_fix_details] DONE")
    return model


def optimize_model_proto_oxs(
    model_proto: ModelProto,
    verbose: int = 0,
    onnx_shape_inference: bool = False,
    inplace: bool = True,
    stats: Optional[Dict[str, Any]] = None,
) -> ModelProto:
    """
    Optimizes a model proto to optimize onnxruntime.

    :param model_proto: ModelProto
    :param verbose: verbosity
    :param onnx_shape_inference: enable shape inference
    :param inplace: the function modifies the proto inplace as well
    :param stats: if not empty, stores information
    :return: optimized model

    You should run that before calling this function

    ::

        onnx_model = exported.to_model_proto(
            opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version
        )

        from experimental_experiment.convert.convert_helper import optimize_model_proto_oxs
        onnx_model = optimize_model_proto_oxs(onnx_model)
    """
    from onnxscript.optimizer import optimize
    from onnxscript.rewriter import rewrite

    if verbose:
        print(
            f"[optimize_model_proto_oxs] starts optimize with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    first_model_proto = model_proto

    begin = time.perf_counter()

    model_proto = optimize(
        model_proto,
        num_iterations=2,
        onnx_shape_inference=onnx_shape_inference,
    )

    if stats:
        stats["oxs_optimize_time"] = time.perf_counter() - begin
    if verbose:
        print(
            f"[optimize_model_proto_oxs] optimize done in "
            f"{time.perf_counter() - begin} seconds."
        )
        print(
            f"[optimize_model_proto_oxs] starts rewrite with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    begin = time.perf_counter()

    model_proto = rewrite(model_proto)

    if stats:
        stats["oxs_rewrite_time"] = time.perf_counter() - begin
    if verbose:
        print(
            f"[optimize_model_proto_oxs] rewrite done in {time.perf_counter() - begin} "
            f"seconds with {len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )
        print(
            f"[optimize_model_proto_oxs] starts inlining with "
            f"{len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    begin = time.perf_counter()

    model_proto = inline_local_functions(model_proto)

    if stats:
        stats["oxs_inline_time"] = time.perf_counter() - begin
    if verbose:
        print(
            f"[optimize_model_proto_oxs] inlining done in {time.perf_counter() - begin} "
            f"seconds with {len(model_proto.graph.node)} nodes and "
            f"{len(model_proto.functions)} local functions"
        )

    # _fix_details(model_proto)
    if inplace:
        del first_model_proto.graph.node[:]
        del first_model_proto.functions[:]
        del first_model_proto.graph.initializer[:]
        del first_model_proto.opset_import[:]

        first_model_proto.graph.node.extend(model_proto.graph.node)
        first_model_proto.functions.extend(model_proto.functions)
        first_model_proto.graph.initializer.extend(model_proto.graph.initializer)
        first_model_proto.opset_import.extend(model_proto.opset_import)

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
    from .ort_helper import append_custom_libraries

    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = output
    if disable_aot:
        opts.add_session_config_entry("session.disable_aot_function_inlining", "1")

    if providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif not isinstance(providers, list) and providers.startswith("cuda"):
        device_id = 0 if ":" not in providers else int(providers.split(":")[1])
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            ("CPUExecutionProvider", {}),
        ]
    assert isinstance(providers, list), f"Unexpected value for providers={providers!r}"

    if isinstance(onnx_model, str):
        onnx_model = onnx_load(onnx_model)

    append_custom_libraries(onnx_model, opts)

    onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        opts,
        providers=providers,
    )
