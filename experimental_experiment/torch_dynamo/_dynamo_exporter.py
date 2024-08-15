from typing import Any


def _extract_graph_module_outputs(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
) -> Any:
    """Collect "val" fields from outputs metadata in this torch.fx.GraphModule."""
    for node in graph_module.graph.nodes:
        if node.op == "output":
            # Output node is unique. Let's retrieve output values from
            # this node's input list. And then just return.
            return node.args[0]
    raise ValueError("No output node found in this torch.fx.GraphModule.")


def _maybe_map_to_meta_val(value):
    if hasattr(value, "meta") and "val" in value.meta:
        # Select outputs with "val" information. Without "val",
        # it's not possible access output_arg.meta["val"].device.
        return value.meta["val"]
    else:
        return value


def _dynamo_export(
    graph_module,
    args,
    verbose,
    target_opset,
    dispatcher,
    optimize,
    enable_pattern,
    disable_pattern,
    rename_inputs,
    device,
    **kwargs,
):
    import torch
    from torch.onnx._internal.fx import fx_onnx_interpreter, onnxfunction_dispatcher
    from torch.onnx._internal.fx import diagnostics

    try:
        from torch.onnx._internal._legacy_exporter import OnnxRegistry
    except ImportError:
        from torch.onnx._internal.exporter import OnnxRegistry
    from torch.onnx._internal.diagnostics import infra

    context = diagnostics.DiagnosticContext(
        "_dynamo_export",
        torch.__version__,
        infra.DiagnosticOptions(),
    )
    onnx_registry = OnnxRegistry()

    self_onnxfunction_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
        onnx_registry, context
    )

    graph_module = torch.onnx._internal.fx.passes.MovePlaceholderToFront(
        context, graph_module
    ).run()

    # Create the object to iterate through the nodes in graph one-by-one
    # and calls the corresponding ONNX exporter for each node.
    fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=context)
    # Cast FX variables if they will result schema-mismatch when searching
    # for ONNX operator. E.g., add(double_tensor, int_tensor) is fine in PyTorch,
    # but ONNX expects add(double_tensor, double_tensor).
    graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(
        context, graph_module
    ).run()

    # Start the per-node exporting process. It's conceptually a for loop
    # scanning through the nodes in the graph.
    exported = fx_interpreter.run(
        fx_graph_module=graph_module,
        onnxfunction_dispatcher=self_onnxfunction_dispatcher,
        op_level_debug=False,
    )
    # Convert the exported result to ONNX ModelProto.
    onnx_model = exported.to_model_proto(opset_version=target_opset)

    # Modify ONNX model using pre-registered graph transforms.
    # They are in-place modifications for avoiding unnecessary
    # copy of ONNX initializers.
    if optimize:
        from ..convert.convert_helper import optimize_model_proto_oxs

        onnx_model = optimize_model_proto_oxs(onnx_model)

    return onnx_model, None
