
================================
Use the custom exporter in torch
================================

*Subject to change*

File `onnxruntime.py`
=====================

This change enables the custom rewriter is an environment variable is enabled.
Look for substring ``TODO:``.

.. code-block:: python

    def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
        """This function replaces GraphModule._wrapped_call in compiled model.

        The _wrapped_call is the underlying implementation of forward method. Replacing
        it means we delegate the computation to _ort_acclerated_call and therefore
        onnxruntime.InferenceSession.
        """
        cached_execution_info_per_session = (
            self._all_ort_execution_info.search_reusable_session_execution_info(
                graph_module, *args
            )
        )
        if cached_execution_info_per_session:
            onnx_session = cached_execution_info_per_session.session
            input_names = cached_execution_info_per_session.input_names
            output_names = cached_execution_info_per_session.output_names
            input_value_infos = cached_execution_info_per_session.input_value_infos
            output_value_infos = cached_execution_info_per_session.output_value_infos
            input_devices = cached_execution_info_per_session.input_devices
            output_devices = cached_execution_info_per_session.output_devices
            prim_outputs = cached_execution_info_per_session.example_outputs
        else:
            # It's first time seeing such as graph. Let's make a new session
            # (type: onnxruntime.InferenceSession) for it.
            
            ##########################
            # TODO: Insert these lines
            ##########################

            use_other_rewriter = bool(os.environ.get("ONNXRT_CHANGE_REWRITER", None))
            if use_other_rewriter:
                from experimental_experiment.torch_interpreter import to_onnx
                from experimental_experiment.torch_interpreter._torch_helper import create_input_names
                from experimental_experiment.xbuilder import OptimizationOptions
                
                input_names = input_names = create_input_names(graph_module, args)
                dispatcher = None
                target_opset = self._resolved_onnx_exporter_options.onnx_registry.opset_version
                options = OptimizationOptions(
                    remove_unused=True,
                    constant_folding=False,
                    patterns="default",
                    verbose=1,
                )                
                onnx_model, builder = to_onnx(
                    graph_module,
                    tuple(args),
                    input_names=input_names,
                    options=options,
                    verbose=1,
                    target_opset=target_opset,
                    return_builder=True,
                    dispatcher=dispatcher,
                )

                def maybe_map_to_meta_val(value):
                    if hasattr(value, "meta") and "val" in value.meta:
                        # Select outputs with "val" information. Without "val",
                        # it's not possible access output_arg.meta["val"].device.
                        return value.meta["val"]
                    return value

                extracted_outputs = _extract_graph_module_outputs(graph_module)
                prim_outputs = _pytree.tree_map(maybe_map_to_meta_val, extracted_outputs)

            else:

            ####################################
            # TODO: end of the insertion
            # TODO: indent what follows
            ####################################

                graph_module = torch.onnx._internal.fx.passes.MovePlaceholderToFront(
                    self._resolved_onnx_exporter_options.diagnostic_context,
                    graph_module,
                ).run()
                # Generate reference outputs. They are used to indicate output
                # tensors' types and devices when calling ORT.
                #
                # WARNING: The downstream code should not change prim_outputs and
                # this backend should always produces output with schema identical to prim_outputs'.

                if self._resolved_onnx_exporter_options.dynamic_shapes:
                    # No pre-allocation when dynamic shape is enabled.
                    self.preallocate_output = False
                    extracted_outputs = _extract_graph_module_outputs(graph_module)

                    def maybe_map_to_meta_val(value):
                        if hasattr(value, "meta") and "val" in value.meta:
                            # Select outputs with "val" information. Without "val",
                            # it's not possible access output_arg.meta["val"].device.
                            return value.meta["val"]
                        else:
                            return value

                    prim_outputs = _pytree.tree_map(
                        maybe_map_to_meta_val, extracted_outputs
                    )
                else:
                    try:
                        prim_outputs = FakeTensorProp(graph_module).propagate(
                            *args, **kwargs
                        )
                    except Exception:
                        logger.warning("FakeTensorProb failed for %s", graph_module)
                        # When FakeTensorProp fails, it is not possible to preallocate output buffers
                        # because the output shapes are not inferred.
                        self.preallocate_output = False

                        # rethrow FakeTensorProb failure because it is not yet currently handled.
                        raise

                # Create the object to iterate through the nodes in graph one-by-one
                # and calls the corresponding ONNX exporter for each node.
                fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(
                    diagnostic_context=self._resolved_onnx_exporter_options.diagnostic_context
                )
                # Cast FX variables if they will result schema-mismatch when searching
                # for ONNX operator. E.g., add(double_tensor, int_tensor) is fine in PyTorch,
                # but ONNX expects add(double_tensor, double_tensor).
                graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(
                    self._resolved_onnx_exporter_options.diagnostic_context, graph_module
                ).run()
                # Start the per-node exporting process. It's conceptually a for loop
                # scanning through the nodes in the graph.
                exported = fx_interpreter.run(
                    fx_graph_module=graph_module,
                    onnxfunction_dispatcher=self._resolved_onnx_exporter_options.onnxfunction_dispatcher,
                    op_level_debug=self._resolved_onnx_exporter_options.op_level_debug,
                )
                # Convert the exported result to ONNX ModelProto.
                onnx_model = exported.to_model_proto(
                    opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version,
                )

            ####################################
            # TODO: end of the modification
            ####################################

            # Modify ONNX model using pre-registered graph transforms.
            # They are in-place modifications for avoiding unnecessary
            # copy of ONNX initializers.
            if self._options.pre_ort_model_transforms:
                for transform in self._options.pre_ort_model_transforms:
                    transform(onnx_model)

            onnx_model_bytes = onnx_model.SerializeToString()
            if os.environ.get("ONNXRT_DUMP_PATH", None):
                # If not empty, environment variable ONNXRT_DUMP_PATH defined the path
                # where generated onnx files should be stored.
                # This module keeps a global variables keeping track of the
                # stored models.
                # If ONNXRT_DUMP_PATH="dumped/dumped_model_"
                # The first file name will be 'dumped/dumped_model_0.onnx'.
                # For every dumped model, a text file 'dumped/dumped_model_0.txt'
                # is created as well to contain the string representing the graph_module.
                _dump_onnx_model(onnx_model_bytes, graph_module=graph_module)

            # Initialize a ORT session to execute this ONNX model.
            # Note that TorchDynamo assumes all inputs/outputs are on the
            # same device, but it's subject to change (very likely with
            # dynamic shape support), so we add execution providers
            # based on the logic in _select_eps: (explicitly preferred EPs,
            # EPs inferred from inputs or graph, and the fallback default EP)/
            #
            # TODO(wschin): enable external allocators.
            # See https://github.com/pytorch/pytorch/issues/106867
            onnx_session = onnxruntime.InferenceSession(
                path_or_bytes=onnx_model_bytes,
                sess_options=self._options.ort_session_options,
                providers=self._select_eps(graph_module, *args),
            )

            # Cache ORT session. It's reused for the same "graph_module".
            # Generate ONNX model and extract its input and output names.
            input_names = tuple(input.name for input in onnx_model.graph.input)
            output_names = tuple(output.name for output in onnx_model.graph.output)
            input_devices = _get_onnx_devices(args)
            # Cache devices for inputs and outputs. They are used to invoke
            # ORT session. Output devices indicate where (e.g., GPU or CPU)
            # to store outputs
            if isinstance(prim_outputs, tuple):
                output_devices = _get_onnx_devices(prim_outputs)
            else:
                output_devices = _get_onnx_devices((prim_outputs,))

            input_value_infos = tuple(input for input in onnx_model.graph.input)
            output_value_infos = tuple(output for output in onnx_model.graph.output)

            execution_info_per_session = OrtExecutionInfoPerSession(
                session=onnx_session,
                input_names=input_names,
                input_value_infos=input_value_infos,
                output_names=output_names,
                output_value_infos=output_value_infos,
                input_devices=input_devices,
                output_devices=output_devices,
                example_outputs=prim_outputs,
            )

            self._all_ort_execution_info.cache_session_execution_info(
                graph_module, execution_info_per_session
            )

        self.execution_count += 1

        # ORT always returns a tuple of outputs. If the original output is a tensor,
        # ORT output's first element must be extracted and returned. Otherwise, type
        # mismatch may happen in downstream computation.
        is_single_tensor_output = isinstance(prim_outputs, torch.Tensor)
        normalized_prim_outputs = (
            (prim_outputs,) if is_single_tensor_output else prim_outputs
        )
        assert isinstance(normalized_prim_outputs, tuple)
        assert all(
            isinstance(elem, (torch.Tensor, torch.SymInt, int))
            for elem in normalized_prim_outputs
        )

        _nvtx_range_push("run_onnx_session_with_ortvaluevector")
        onnx_outputs = self.run(
            onnx_session,
            input_names,
            args,
            input_devices,
            output_names,
            normalized_prim_outputs,
            output_devices,
            self._options.preallocate_output,
            input_value_infos,
            normalized_prim_outputs,
        )
        _nvtx_range_pop()

        if self._assert_allclose_to_baseline:
            # Compute baseline.
            baseline_outputs = torch._prims.executor.execute(
                graph_module, *args, executor="aten"
            )
            normalized_baseline_ouptuts = (
                (baseline_outputs,) if is_single_tensor_output else baseline_outputs
            )
            # Ensure every output tensor is close to the corresponding baseline.
            for onnx_output, baseline_output in zip(
                onnx_outputs, normalized_baseline_ouptuts
            ):
                torch.testing.assert_close(onnx_output, baseline_output)
        return onnx_outputs[0] if is_single_tensor_output else onnx_outputs

Examples
========

.. runpython::
    :showcode:

    import os
    import warnings
    import numpy as np
    import onnx

    # from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    import torch
    import torch.onnx
    from experimental_experiment.torch_helper.training_helper import (
        make_aot_ort,
        train_loop,
    )
    from experimental_experiment.torch_helper.dump_helper import dump_onnx

    # from experimental_experiment.torch_interpreter import to_onnx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel


    def ids_tensor(shape, vocab_size):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(np.random.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


    config = LlamaConfig(
        hidden_size=16,
        num_hidden_layers=1,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=1024,
        num_attention_heads=2,
    )
    config._attn_implementation = "eager"

    model = LlamaModel(config)

    batch, seq, vocab_size = 2, 1024, 1024

    input_ids = ids_tensor([batch, seq], vocab_size)
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

    model(input_ids, input_mask)

    os.environ["ONNXRT_CHANGE_REWRITER"] = "1"

    local_aot_ort, _ = make_aot_ort(
        dynamic=True,
        rewrite=True,
        verbose=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
        with dump_onnx("dort-llama-ort", folder="dump_llama", clean=True):
            train_loop(optimized_mod, input_ids, input_mask)

    names = [_ for _ in os.listdir("dump_llama") if _.endswith(".onnx")]
    print("------------------------------------------")
    print(f"exported model: {names}")
    for name in names:
        print()
        print("NODES in {name!r}")
        onx = onnx.load(os.path.join("dump_llama", name))
        for i, node in enumerate(onx.graph.node):
            print(
                f"{i+1}/{len(onx.graph.node)}: {node.op_type} {node.input} -> {node.output}"
            )
