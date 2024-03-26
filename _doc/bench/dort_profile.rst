experimental_experiment.torch_bench.dort_profile
================================================

::

    python -m experimental_experiment.torch_bench.dort_profile \
        -w 3                    \
        -r 5                    \
        --config medium         \
        --mixed 1               \
        --device cuda           \
        --verbose 0             \
        --backend custom        \
        --num_hidden_layers 1   \
        --dynamic 0


::

    llama config={'input_dims': [(2, 1024), (2, 1024), (2, 1024), (2, 1024), (2, 1024), (2, 1024), (2, 1024), (2, 1024)], 'hidden_size': 1024, 'num_hidden_layers': 1, 'vocab_size': 1024, 'intermediate_size': 1024, 'max_position_embeddings': 1024, 'num_attention_heads': 2, '_attn_implementation': 'eager'}
    backend=custom
    verbose=0
    implementation=eager
    mixed=1
    disable_pattern=[]
    enable_pattern=['default']
    CUDA no model: memory allocated=0, reserved=0
    CUDA model loaded: memory allocated=37762048, reserved=44040192
    Build the compile model with backend=custom
    dynamic=False
    warmup on device=cuda
    CUDA memory allocated=37762048, reserved=44040192
    warmup done in 2.9089201999995566s.
    memory allocated=37786624, reserved=65011712
    measures
    measures done.
    dynamic=0
    mixed=1
    backend=custom
    num_hidden_layers=1
    mixed=1
    repeat=5
    device=cuda
    avg=0.05731345999993209
    times=[0.05914619999930437, 0.05698019999999815, 0.05694430000039574, 0.05702159999964351, 0.05647500000031869]
    warmup_times=[2.7931430999997247, 0.05779750000056083, 0.05797959999927116]
    -----------
    :llama,2x1024-1024-1-1024-1024-1024-2-eager;
    :config,medium;
    :mixed,1;
    :dynamic,False;
    :backend,custom;
    :repeat,5;
    :warmup,3;
    :torch,2.3.0.dev20240314+cu118;
    :transformers,4.37.2;
    :patterns,+default-;
    :warmup_time,2.9089201999995566;
    :time,0.05731345999993209;
    --------------------------------------------------------------------------
    g                                                            --    5   15 -- 0.00006 0.01478 -- _aot_autograd/utils.py:88:g (g)
        runtime_wrapper                                          --    5    5 -- 0.00013 0.01476 -- _aot_autograd/runtime_wrappers.py:77:runtime_wrapper (runtime_wrapper)
            call_func_at_runtime_with_args                       --    5    5 -- 0.00007 0.01435 -- _aot_autograd/utils.py:105:call_func_at_runtime_with_args (call_func_at_runtime_with_args) +++
            __init__                                             --    5    5 -- 0.00002 0.00004 -- autograd/grad_mode.py:350:__init__ (__init__)
            __enter__                                            --    5    5 -- 0.00000 0.00000 -- autograd/grad_mode.py:355:__enter__ (__enter__)
            __exit__                                             --    5    5 -- 0.00001 0.00002 -- autograd/grad_mode.py:358:__exit__ (__exit__)
            <built-in method builtins.isinstance>                --   20   20 -- 0.00001 0.00001 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
            <built-in method torch.tensor>                       --    5    5 -- 0.00014 0.00014 -- ~:0:<built-in method torch.tensor> (<built-in method torch.tensor>)
            <method 'detach' of 'torch._C.TensorBase' objects>   --   20   20 -- 0.00007 0.00007 -- ~:0:<method 'detach' of 'torch._C.TensorBase' objects> (<method 'detach' of 'torch._C.TensorBase' objects>)
        apply                                                    --    5    5 -- 0.00006 0.01424 -- autograd/function.py:582:apply (apply)
            unwrap_dead_wrappers                                 --    5    5 -- 0.00006 0.00030 -- _functorch/utils.py:19:unwrap_dead_wrappers (unwrap_dead_wrappers)
                <genexpr>                                        --   80   80 -- 0.00013 0.00023 -- _functorch/utils.py:21:<genexpr> (<genexpr>)
                    <built-in method builtins.isinstance>        --   75   75 -- 0.00004 0.00004 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                    <built-in method torc...orch.unwrap_if_dead> --   75   75 -- 0.00007 0.00007 -- ~:0:<built-in method torch._C._functorch.unwrap_if_dead> (<built-in method torch._C._functorch.unwrap_if_dead>)
            __getattribute__                                     --   10   10 -- 0.00003 0.00003 -- autograd/function.py:346:__getattribute__ (__getattribute__) +++
            <built-in method apply>                              --    5    5 -- 0.00026 0.01385 -- ~:0:<built-in method apply> (<built-in method apply>)
                forward                                          --    5    5 -- 0.00034 0.01356 -- _aot_autograd/jit_compile_runtime_wrappers.py:485:forward (forward)
                    <genexpr>                                    --  190  190 -- 0.00019 0.00030 -- _aot_autograd/jit_compile_runtime_wrappers.py:539:<genexpr> (<genexpr>)
                        <method '_is_view' ...nsorBase' objects> --  185  185 -- 0.00010 0.00010 -- ~:0:<method '_is_view' of 'torch._C.TensorBase' objects> (<method '_is_view' of 'torch._C.TensorBase' objects>)
                    <listcomp>                                   --    5    5 -- 0.00005 0.00005 -- _aot_autograd/jit_compile_runtime_wrappers.py:604:<listcomp> (<listcomp>)
                    <listcomp>                                   --    5    5 -- 0.00002 0.00003 -- _aot_autograd/jit_compile_runtime_wrappers.py:610:<listcomp> (<listcomp>)
                        <built-in method builtins.isinstance>    --   15   15 -- 0.00001 0.00001 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                    functionalized_rng_runtime_epilogue          --    5    5 -- 0.00000 0.00000 -- _aot_autograd/runtime_wrappers.py:287:functionalized_rng_runtime_epilogue (functionalized_rng_runtime_epilogue)
                    tensors_saved_for_backwards_slice            --    5    5 -- 0.00001 0.00001 -- _aot_autograd/schemas.py:410:tensors_saved_for_backwards_slice (tensors_saved_for_backwards_slice)
                    symints_saved_for_backwards_slice            --    5    5 -- 0.00001 0.00001 -- _aot_autograd/schemas.py:418:symints_saved_for_backwards_slice (symints_saved_for_backwards_slice)
                    call_func_at_runtime_with_args               --    5    5 -- 0.00006 0.01219 -- _aot_autograd/utils.py:105:call_func_at_runtime_with_args (call_func_at_runtime_with_args) +++
                    save_for_backward                            --    5    5 -- 0.00001 0.00001 -- autograd/function.py:33:save_for_backward (save_for_backward)
                    mark_non_differentiable                      --    5    5 -- 0.00001 0.00001 -- autograd/function.py:189:mark_non_differentiable (mark_non_differentiable)
                    __getattribute__                             --   80   80 -- 0.00007 0.00007 -- autograd/function.py:346:__getattribute__ (__getattribute__) +++
                    __call__                                     --    5    5 -- 0.00002 0.00018 -- torch/_ops.py:852:__call__ (__call__)
                        <built-in method to...aten._unsafe_view> --    5    5 -- 0.00017 0.00017 -- ~:0:<built-in method torch._ops.aten._unsafe_view> (<built-in method torch._ops.aten._unsafe_view>)
                    <built-in method builtins.all>               --   10   10 -- 0.00010 0.00036 -- ~:0:<built-in method builtins.all> (<built-in method builtins.all>)
                        <genexpr>                                --  190  190 -- 0.00018 0.00026 -- _aot_autograd/jit_compile_runtime_wrappers.py:536:<genexpr> (<genexpr>)
                            <built-in method ...tins.isinstance> --  185  185 -- 0.00008 0.00008 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                        <genexpr>                                --    5    5 -- 0.00000 0.00000 -- _aot_autograd/jit_compile_runtime_wrappers.py:547:<genexpr> (<genexpr>)
                __getattribute__                                 --   15   15 -- 0.00003 0.00003 -- autograd/function.py:346:__getattribute__ (__getattribute__) +++
        __call__                                                 --    5    5 -- 0.00017 0.01209 -- torch_dynamo/fast_backend.py:171:__call__ (__call__)
            _run_onnx_session_with_ortvaluevector                --    5    5 -- 0.00013 0.01182 -- torch_dynamo/fast_backend.py:264:_run_onnx_session_with_ortvaluevector (_run_onnx_session_with_ortvaluevector)
                run_with_ortvaluevector                          --    5    5 -- 0.00914 0.00914 -- capi/onnxruntime_inference_collection.py:339:run_with_ortvaluevector (run_with_ortvaluevector)
                _get_ortvalues_from_torch_tensors                --    5    5 -- 0.00096 0.00154 -- torch_dynamo/fast_backend.py:195:_get_ortvalues_from_torch_tensors (_get_ortvalues_from_torch_tensors)
                    <method 'append' of 'list' objects>          --  575  575 -- 0.00026 0.00026 -- ~:0:<method 'append' of 'list' objects> (<method 'append' of 'list' objects>) +++
                    <built-in method builtins.isinstance>        --   80   80 -- 0.00005 0.00005 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                    <built-in method builtins.max>               --   75   75 -- 0.00006 0.00006 -- ~:0:<built-in method builtins.max> (<built-in method builtins.max>)
                    <method 'data_ptr' of...TensorBase' objects> --   75   75 -- 0.00006 0.00006 -- ~:0:<method 'data_ptr' of 'torch._C.TensorBase' objects> (<method 'data_ptr' of 'torch._C.TensorBase' objects>)
                    <method 'get_device' ...TensorBase' objects> --  150  150 -- 0.00008 0.00008 -- ~:0:<method 'get_device' of 'torch._C.TensorBase' objects> (<method 'get_device' of 'torch._C.TensorBase' objects>)
                    <method 'size' of 'to...TensorBase' objects> --   75   75 -- 0.00007 0.00007 -- ~:0:<method 'size' of 'torch._C.TensorBase' objects> (<method 'size' of 'torch._C.TensorBase' objects>)
                _ortvalues_to_torch_tensor                       --    5    5 -- 0.00036 0.00080 -- torch_dynamo/fast_backend.py:253:_ortvalues_to_torch_tensor (_ortvalues_to_torch_tensor)
                    <genexpr>                                    --  205  205 -- 0.00032 0.00042 -- torch_dynamo/fast_backend.py:260:<genexpr> (<genexpr>)
                        _post_process                            --  200  200 -- 0.00010 0.00010 -- torch_dynamo/fast_backend.py:50:_post_process (_post_process)
                <genexpr>                                        --   80   80 -- 0.00013 0.00021 -- torch_dynamo/fast_backend.py:268:<genexpr> (<genexpr>)
                    <built-in method builtins.isinstance>        --   75   75 -- 0.00004 0.00004 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                    <method 'contiguous' ...TensorBase' objects> --   75   75 -- 0.00005 0.00005 -- ~:0:<method 'contiguous' of 'torch._C.TensorBase' objects> (<method 'contiguous' of 'torch._C.TensorBase' objects>)
            <built-in method builtins.isinstance>                --  200  200 -- 0.00010 0.00010 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
    call_func_at_runtime_with_args                               --    5   10 -- 0.00013 0.01435 -- _aot_autograd/utils.py:105:call_func_at_runtime_with_args (call_func_at_runtime_with_args)
        normalize_as_list                                        --   10   10 -- 0.00002 0.00002 -- _aot_autograd/utils.py:69:normalize_as_list (normalize_as_list)
        g                                                        --   10    5 -- 0.00004 0.01426 -- _aot_autograd/utils.py:88:g (g) +++
    _fn                                                          --    5   10 -- 0.00013 0.01576 -- _dynamo/eval_frame.py:427:_fn (_fn)
        revert                                                   --   10   10 -- 0.00001 0.00001 -- _dynamo/eval_frame.py:148:revert (revert)
        nothing                                                  --    5    5 -- 0.00000 0.00000 -- _dynamo/eval_frame.py:256:nothing (nothing)
        always_false                                             --   10   10 -- 0.00001 0.00001 -- _dynamo/eval_frame.py:260:always_false (always_false)
        <listcomp>                                               --   10   10 -- 0.00004 0.00020 -- _dynamo/eval_frame.py:447:<listcomp> (<listcomp>)
            change                                               --   10   10 -- 0.00005 0.00007 -- _dynamo/eval_frame.py:140:change (change)
            call_on_enter                                        --    5    5 -- 0.00001 0.00005 -- _dynamo/eval_frame.py:317:call_on_enter (call_on_enter)
                on_enter                                         --    5    5 -- 0.00001 0.00004 -- _dynamo/eval_frame.py:524:on_enter (on_enter)
                    install_generation_tagging_init              --    5    5 -- 0.00002 0.00003 -- _dynamo/mutation_guard.py:101:install_generation_tagging_init (install_generation_tagging_init)
            change                                               --    5    5 -- 0.00002 0.00004 -- utils/_config_module.py:289:change (change)
                <dictcomp>                                       --    5    5 -- 0.00001 0.00001 -- utils/_config_module.py:290:<dictcomp> (<dictcomp>)
        inner                                                    --    5    5 -- 0.00002 0.01483 -- _dynamo/external_utils.py:34:inner (inner)
            forward                                              --    5    5 -- 0.00002 0.01482 -- _functorch/aot_autograd.py:913:forward (forward)
                g                                                --    5    5 -- 0.00002 0.01478 -- _aot_autograd/utils.py:88:g (g) +++
        is_fx_tracing                                            --    5    5 -- 0.00001 0.00001 -- fx/_symbolic_trace.py:46:is_fx_tracing (is_fx_tracing)
        _wrapped_call_impl                                       --    5    5 -- 0.00002 0.01547 -- modules/module.py:1523:_wrapped_call_impl (_wrapped_call_impl) +++
        revert                                                   --    5    5 -- 0.00001 0.00002 -- utils/_config_module.py:293:revert (revert)
        <built-in method torch._C._...eval_frame.set_eval_frame> --   20   20 -- 0.00002 0.00002 -- ~:0:<built-in method torch._C._dynamo.eval_frame.set_eval_frame> (<built-in method torch._C._dynamo.eval_frame.set_eval_frame>)
    __getattribute__                                             --  105  105 -- 0.00012 0.00012 -- autograd/function.py:346:__getattribute__ (__getattribute__)
    is_available                                                 --   10   10 -- 0.00008 0.00033 -- cuda/__init__.py:105:is_available (is_available)
        _is_compiled                                             --   10   10 -- 0.00002 0.00003 -- cuda/__init__.py:96:_is_compiled (_is_compiled)
        _nvml_based_avail                                        --   10   10 -- 0.00003 0.00021 -- cuda/__init__.py:101:_nvml_based_avail (_nvml_based_avail)
            getenv                                               --   10   10 -- 0.00003 0.00018 -- python3.10/os.py:772:getenv (getenv)
                get                                              --   10   10 -- 0.00004 0.00015 -- python3.10/_collections_abc.py:821:get (get)
                    __getitem__                                  --   10   10 -- 0.00006 0.00011 -- python3.10/os.py:675:__getitem__ (__getitem__)
                        encode                                   --   10   10 -- 0.00004 0.00005 -- python3.10/os.py:755:encode (encode)
    _lazy_init                                                   --   10   10 -- 0.00004 0.00007 -- cuda/__init__.py:263:_lazy_init (_lazy_init)
        is_initialized                                           --   10   10 -- 0.00002 0.00003 -- cuda/__init__.py:216:is_initialized (is_initialized)
    _wrapped_call_impl                                           --    5   10 -- 0.00004 0.01586 -- modules/module.py:1523:_wrapped_call_impl (_wrapped_call_impl)
        _call_impl                                               --    5   10 -- 0.00014 0.01583 -- modules/module.py:1529:_call_impl (_call_impl)
            guard                                                --    5    5 -- 0.00023 0.00039 -- <string>:2:guard (guard)
                check_current_backend                            --    5    5 -- 0.00001 0.00001 -- _dynamo/eval_frame.py:86:check_current_backend (check_current_backend)
                __getattr__                                      --    5    5 -- 0.00002 0.00002 -- modules/module.py:1691:__getattr__ (__getattr__)
                <method 'keys' of 'coll....OrderedDict' objects> --   20   20 -- 0.00001 0.00001 -- ~:0:<method 'keys' of 'collections.OrderedDict' objects> (<method 'keys' of 'collections.OrderedDict' objects>)
                <built-in method torch.....guards.check_type_id> --   40   40 -- 0.00002 0.00002 -- ~:0:<built-in method torch._C._dynamo.guards.check_type_id> (<built-in method torch._C._dynamo.guards.check_type_id>)
                <built-in method torch....o.guards.check_obj_id> --   75   75 -- 0.00005 0.00005 -- ~:0:<built-in method torch._C._dynamo.guards.check_obj_id> (<built-in method torch._C._dynamo.guards.check_obj_id>)
            _fn                                                  --    5    5 -- 0.00007 0.01576 -- _dynamo/eval_frame.py:427:_fn (_fn) +++
            forward                                              --    5    5 -- 0.00002 0.01496 -- torch_models/llama_helper.py:179:forward (forward)
                _fn                                              --    5    5 -- 0.00005 0.01494 -- _dynamo/eval_frame.py:427:_fn (_fn) +++
    is_scripting                                                 --   25   25 -- 0.00002 0.00002 -- torch/_jit_internal.py:1120:is_scripting (is_scripting)
    main_loop                                                    --    1    1 -- 0.00037 0.28833 -- torch_bench/dort_profile.py:174:main_loop (main_loop)
        loop_iteration                                           --    5    5 -- 0.00014 0.28625 -- torch_bench/dort_profile.py:101:loop_iteration (loop_iteration)
            __init__                                             --    5    5 -- 0.00009 0.00029 -- amp/autocast_mode.py:187:__init__ (__init__)
                amp_definitely_not_available                     --    5    5 -- 0.00001 0.00017 -- amp/common.py:8:amp_definitely_not_available (amp_definitely_not_available)
                    is_available                                 --    5    5 -- 0.00002 0.00016 -- cuda/__init__.py:105:is_available (is_available) +++
                is_scripting                                     --    5    5 -- 0.00000 0.00000 -- torch/_jit_internal.py:1120:is_scripting (is_scripting) +++
            __enter__                                            --    5    5 -- 0.00005 0.00009 -- amp/autocast_mode.py:320:__enter__ (__enter__)
                is_scripting                                     --    5    5 -- 0.00000 0.00000 -- torch/_jit_internal.py:1120:is_scripting (is_scripting) +++
            __exit__                                             --    5    5 -- 0.00005 0.00008 -- amp/autocast_mode.py:370:__exit__ (__exit__)
                is_scripting                                     --    5    5 -- 0.00000 0.00000 -- torch/_jit_internal.py:1120:is_scripting (is_scripting) +++
            synchronize                                          --    5    5 -- 0.00008 0.25052 -- cuda/__init__.py:782:synchronize (synchronize)
                _lazy_init                                       --    5    5 -- 0.00003 0.00006 -- cuda/__init__.py:263:_lazy_init (_lazy_init) +++
                __init__                                         --    5    5 -- 0.00002 0.00044 -- cuda/__init__.py:360:__init__ (__init__)
                    _get_device_index                            --    5    5 -- 0.00004 0.00043 -- cuda/_utils.py:9:_get_device_index (_get_device_index)
                        is_scripting                             --    5    5 -- 0.00001 0.00001 -- torch/_jit_internal.py:1120:is_scripting (is_scripting) +++
                        _get_device_index                        --    5    5 -- 0.00003 0.00037 -- torch/_utils.py:759:_get_device_index (_get_device_index)
                            is_scripting                         --    5    5 -- 0.00000 0.00000 -- torch/_jit_internal.py:1120:is_scripting (is_scripting) +++
                            _get_current_device_index            --    5    5 -- 0.00001 0.00033 -- torch/_utils.py:733:_get_current_device_index (_get_current_device_index)
                                _get_device_attr                 --    5    5 -- 0.00007 0.00032 -- torch/_utils.py:721:_get_device_attr (_get_device_attr)
                                    _get_available_device_type   --    5    5 -- 0.00001 0.00019 -- torch/_utils.py:708:_get_available_device_type (_get_available_device_type)
                                        is_available             --    5    5 -- 0.00006 0.00018 -- cuda/__init__.py:105:is_available (is_available) +++
                                    <lambda>                     --    5    5 -- 0.00001 0.00005 -- torch/_utils.py:735:<lambda> (<lambda>)
                                        current_device           --    5    5 -- 0.00001 0.00004 -- cuda/__init__.py:776:current_device (current_device)
                                            _lazy_init           --    5    5 -- 0.00001 0.00001 -- cuda/__init__.py:263:_lazy_init (_lazy_init) +++
                            <built-in method ...tins.isinstance> --   15   15 -- 0.00001 0.00001 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                        <built-in method builtins.isinstance>    --   20   20 -- 0.00001 0.00001 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>) +++
                __enter__                                        --    5    5 -- 0.00002 0.00002 -- cuda/__init__.py:364:__enter__ (__enter__)
                __exit__                                         --    5    5 -- 0.00003 0.00006 -- cuda/__init__.py:367:__exit__ (__exit__)
                <built-in method torch._C._cuda_synchronize>     --    5    5 -- 0.24986 0.24986 -- ~:0:<built-in method torch._C._cuda_synchronize> (<built-in method torch._C._cuda_synchronize>)
            _wrapped_call_impl                                   --    5    5 -- 0.00002 0.01586 -- modules/module.py:1523:_wrapped_call_impl (_wrapped_call_impl) +++
            backward                                             --    5    5 -- 0.00012 0.01879 -- torch/_tensor.py:466:backward (backward)
                backward                                         --    5    5 -- 0.00010 0.01867 -- autograd/__init__.py:165:backward (backward)
                    _make_grads                                  --    5    5 -- 0.00006 0.00028 -- autograd/__init__.py:60:_make_grads (_make_grads)
                        <built-in method torch.ones_like>        --    5    5 -- 0.00021 0.00021 -- ~:0:<built-in method torch.ones_like> (<built-in method torch.ones_like>)
                    _tensor_or_tensors_to_tuple                  --    5    5 -- 0.00001 0.00001 -- autograd/__init__.py:155:_tensor_or_tensors_to_tuple (_tensor_or_tensors_to_tuple)
                    _engine_run_backward                         --    5    5 -- 0.00007 0.01825 -- autograd/graph.py:739:_engine_run_backward (_engine_run_backward)
                        getEffectiveLevel                        --    5    5 -- 0.00002 0.00002 -- logging/__init__.py:1710:getEffectiveLevel (getEffectiveLevel)
                        <method 'run_backwa...gineBase' objects> --    5    5 -- 0.01816 0.01816 -- ~:0:<method 'run_backward' of 'torch._C._EngineBase' objects> (<method 'run_backward' of 'torch._C._EngineBase' objects>)
            <method 'sum' of 'torch._C.TensorBase' objects>      --    5    5 -- 0.00049 0.00049 -- ~:0:<method 'sum' of 'torch._C.TensorBase' objects> (<method 'sum' of 'torch._C.TensorBase' objects>)
        <listcomp>                                               --    5    5 -- 0.00006 0.00169 -- torch_bench/dort_profile.py:176:<listcomp> (<listcomp>)
            <method 'to' of 'torch._C.TensorBase' objects>       --   10   10 -- 0.00162 0.00162 -- ~:0:<method 'to' of 'torch._C.TensorBase' objects> (<method 'to' of 'torch._C.TensorBase' objects>)
    <method 'append' of 'list' objects>                          --  585  585 -- 0.00027 0.00027 -- ~:0:<method 'append' of 'list' objects> (<method 'append' of 'list' objects>)
    <built-in method builtins.isinstance>                        --  730  730 -- 0.00036 0.00036 -- ~:0:<built-in method builtins.isinstance> (<built-in method builtins.isinstance>)
    <built-in method builtins.len>                               --   25   25 -- 0.00003 0.00003 -- ~:0:<built-in method builtins.len> (<built-in method builtins.len>)
    <built-in method builtins.hasattr>                           --   30   30 -- 0.00003 0.00003 -- ~:0:<built-in method builtins.hasattr> (<built-in method builtins.hasattr>)
    <built-in method builtins.id>                                --   15   15 -- 0.00001 0.00001 -- ~:0:<built-in method builtins.id> (<built-in method builtins.id>)
