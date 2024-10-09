=========
CodeLlama
=========

.. code-block:: python

    """
    ========================================
    102: Fuse kernels in a small Llama Model
    ========================================

    This example leverages the function :epkg:`torch.compile` and the ability
    to use a custom backend to test the optimization of a model by fusing
    simple element-wise kernels.

    It takes a small Llama model and uses a backend based on :epkg:`onnxruntime`.
    The model is converted into ONNX and then optimized by fusing element-wise
    kernels.

    ::

        python plot_custom_backend_llama --optim default

    The script requires the following packages beside pytorch,
    :epkg:`onnxruntime-training` (for GPU), :epkg:`onnx-extended`
    (compiled for GPU) and :epkg:`transformers`==4.37.2.
    """

    from experimental_experiment.args import get_parsed_args

    script_args = get_parsed_args(
        "plot_custom_backend_llama",
        with_mask=(0, "tries with a mask as a secondary input"),
        optim=("default", "Optimization to apply, empty string for all"),
        tokenizer=("1", "loads the tokenizer or not to reduce the memory foot print"),
        description=__doc__,
        expose="config,num_hidden_layers,with_mask,optim",
    )

    assert script_args.optim, "optim must be specified."
    assert script_args.with_mask in (0, "0"), "with_mask is not implemented."


    print(f"with_mask={script_args.with_mask!r}")
    print(f"optim={script_args.optim!r}")
    print(f"tokenizer={script_args.tokenizer!r}")

    load_tokenizer = script_args in ("1", 1)

    #################################
    # Imports.

    import os
    import time
    import numpy as np
    import pandas
    from tqdm import tqdm
    import torch
    from experimental_experiment.xbuilder import OptimizationOptions
    from experimental_experiment.torch_dynamo import onnx_custom_backend
    from experimental_experiment.bench_run import get_machine

    has_cuda = torch.cuda.is_available()
    machine = get_machine()
    print(f"has_cuda={has_cuda}")
    print(f"processor: {machine['processor_name']}")
    print(f"device: {machine.get('device_name', '?')}")

    ########################################
    # The dummy model
    # ===============

    ######################################
    # The number of time we run the model to measure
    # the inference.
    warmup = 8
    N = 10

    ###########################################
    # Let's create the model.

    # see https://huggingface.co/docs/transformers/en/model_doc/code_llama
    if os.path.exists("CodeLlama-7b-model"):
        print("load the model")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = (
            AutoTokenizer.from_pretrained("./CodeLlama-7b-tokenizer") if load_tokenizer else None
        )
        model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-model")
    else:
        print("retrieve the model")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
            tokenizer.save_pretrained("CodeLlama-7b-tokenizer")
        else:
            tokenizer = None
        model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
        model.save_pretrained("CodeLlama-7b-model")


    def ids_tensor(shape, vocab_size):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(np.random.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


    ##########################################
    # Small example on how to generate an answer.

    processor = "cuda" if has_cuda else "cpu"

    if load_tokenizer:
        PROMPT = '''
        def optimize_model_by_fusing_kernel(
            model_or_filename,
            fused_patterns: Union[str, List[str]] = "default",
            validate_performance: bool = False,
            filename: Optional[str] = None,
        ) -> str:
            """ <FILL_ME> """
            return optimized_model
        '''

        with torch.no_grad():
            print("tokenize the input")
            input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(processor)
            print("run the model")
            model = model.to(processor)
            generated_ids = model.generate(input_ids, max_new_tokens=128).to(processor)
            print("interpret the answer")
            filling = tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            print("---")
            print(PROMPT.replace("<FILL_ME>", filling))
            print("done")
    else:
        input_ids = ids_tensor((1, 128), 32016)

    # Input dimension
    print(f"Input shape: {input_ids.shape}")

    # We use those inputs to benchmark the models.
    inputs = (input_ids,)

    # Just to make sure everything is ok.

    print(f"moving model and inputs to processor={processor!r}")
    model = model.to(processor)
    inputs = tuple(i.to(processor) for i in inputs)

    ##########################################
    # Measure of eager mode
    # =====================


    print("------------------------------------")
    times = []

    with torch.no_grad():

        # warmup
        print("warmup eager")
        for _ in tqdm(range(warmup)):
            model(*inputs, use_cache=False)
            if has_cuda:
                torch.cuda.synchronize()

        # repeat
        print("repeat eager")
        begin = time.perf_counter()
        for _ in tqdm(range(N)):
            model(*inputs, use_cache=False)
            if has_cuda:
                torch.cuda.synchronize()
        d = (time.perf_counter() - begin) / N
        baseline = d
        times.append(dict(optium="eager", processor=processor, avg_time=d, warmup=warmup, N=N))
        print("avg time eager", d)

    ############################################
    # Measure with the custom backend
    # ===============================
    #
    # Three kind of optimization:
    #
    # - **default**: the onnx model is optimized with less onnx operators
    # - **default+onnxruntime**: the onnx model is optimized with fused kernels
    #   implemented by onnxruntime
    # - **default+onnxruntime+experimental**: the onnx model is optimized with fused kernels
    #   implemented by onnxruntime and also custom kernels, this does not work on
    #   CPU.
    #
    # Some links:
    #
    # * :class:`experimental_experiment.xbuilder.OptimizationOptions`:
    #   that class defines the optimizations to apply after the model
    #   is converted to onnx,
    # * :func:`experimental_experiment.torch_dynamo.onnx_custom_backend`:
    #   that function implements the custom backend based on :epkg:`onnxruntime`,
    #   it converts the model into ONNX, optimizes and runs it,
    #   it does not support :epkg:`graph break`,
    #   it does not work well with dynamic shapes yet.
    #
    # The GPU memory is not fully freed before two iterations. Only one scenario
    # should be handled in the same process.
    # Results may be very different with a different chip.

    optimization = [script_args.optim]

    with torch.no_grad():

        for optim in optimization:
            print("----------------------")
            print(f"optim={optim}")

            options = OptimizationOptions(
                constant_folding=True,
                patterns=None if optim == "" else optim,
                verbose=0,
                processor=processor.upper(),
            )

            # The backend used here overwrite some of the parameters provided by
            # function onnx_custom_backend.
            custom_custom_backend = lambda *args, optim=optim, options=options, **kwargs: onnx_custom_backend(  # noqa: E731, E501
                *args,
                target_opset=18,
                verbose=0,
                options=options,
                optimize=optim != "",
                dump_prefix=f"dump_onx_llama_{optim.replace('+', '_')}",
                **kwargs,
            )

            # The function setting the backend.
            compiled_model = torch.compile(
                model, backend=custom_custom_backend, fullgraph=True, dynamic=False
            )

            # warmup
            print("warmup compiled model")
            for _ in tqdm(range(warmup)):
                compiled_model(*inputs, use_cache=False)
                if has_cuda:
                    torch.cuda.synchronize()

            # repeat
            print("repeat compiled_model")
            begin = time.perf_counter()
            for _ in tqdm(range(N)):
                compiled_model(*inputs, use_cache=False)
                if has_cuda:
                    torch.cuda.synchronize()
            d = (time.perf_counter() - begin) / N

            times.append(
                dict(
                    optium=optim,
                    processor=processor,
                    avg_time=d,
                    warmup=warmup,
                    N=N,
                    speedup=baseline / d,
                )
            )
            print(f"avg time custom backend with optimization={optim!r}", d)

    ###############################################
    # Final results
    # =============
    #
    # avg_time, lower is better,
    # speedup compare to eager mode, higher is better.

    df = pandas.DataFrame(times)
    print(df)
