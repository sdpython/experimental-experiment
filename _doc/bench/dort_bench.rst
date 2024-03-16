experimental_experiment.torch_bench.dort_bench
==============================================

::

    python -m experimental_experiment.torch_bench.dort_bench \
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
    torch/_functorch/_aot_autograd/utils.py:117: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
    warnings.warn(
    warmup done in 3.2293482000004587s.
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
    avg=0.05740846000007878
    times=[0.05616349999945669, 0.05769109999982902, 0.05737370000042574, 0.058259600000383216, 0.05755440000029921]
    warmup_times=[3.1141115999998874, 0.05787190000046394, 0.057364700000107405]
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
    :warmup_time,3.2293482000004587;
    :time,0.05740846000007878;
