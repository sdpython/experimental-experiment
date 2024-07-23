=============================================================
Measuring the exporters on a short list of HuggingFace models
=============================================================

One Run
=======

The script loads a model, *ElectraForQuestionAnswering* in this case,
warms up 10 times, measure the time to run inference 30 times. Then it converts it
into onnx, and do the same. One example:

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter script --verbose 3 --quiet 0 -w 1 -r 3

::

    [bash_bench_huggingface] start
    device=cpu
    dtype=
    dump_folder=dump_bash_bench
    dynamic=0
    exporter=script
    model=ElectraForQuestionAnswering
    opt_patterns=
    output_data=output_data_bash_bench_huggingface.py.csv
    process=0
    quiet=0
    repeat=3
    target_opset=18
    verbose=3
    warmup=1
    Running model 'ElectraForQuestionAnswering'
    [BenchmarkRunner.benchmark] test model 'ElectraForQuestionAnswering' with exporter='script'
    [BenchmarkRunner.benchmark] load model 'ElectraForQuestionAnswering'
    [benchmarkrunner.benchmark] model wrapped with class <class 'experimental_experiment.torch_bench._bash_bench_model_runner.WrappedModelToTuple'>
    [BenchmarkRunner.benchmark] model size and dtype 13483522, float32
    [BenchmarkRunner.benchmark] warmup model 'ElectraForQuestionAnswering' - 1 times
    [benchmarkrunner.benchmark] output_size=65537.0
    [BenchmarkRunner.benchmark] repeat model 'ElectraForQuestionAnswering' - 3 times
    [BenchmarkRunner.benchmark] export model 'ElectraForQuestionAnswering'
    [BenchmarkRunner.benchmark] inference model 'ElectraForQuestionAnswering'
    [BenchmarkRunner.benchmark] warmup script - 'ElectraForQuestionAnswering'
    [benchmarkrunner.benchmark] no_grad=True torch.is_grad_enabled()=False before warmup
    [benchmarkrunner.benchmark] torch.is_grad_enabled()=False after warmup
    [BenchmarkRunner.benchmark] repeat ort 'ElectraForQuestionAnswering'
    [BenchmarkRunner.benchmark] done model with 46 metrics
    [BenchmarkRunner.benchmark] done model 'ElectraForQuestionAnswering' with exporter='script' in 116.15856191800003
    :_index,ElectraForQuestionAnswering-script;
    :capability,6.1;
    :cpu,8;
    :date_start,2024-07-09;
    :device,cpu;
    :device_name,NVIDIA GeForce GTX 1060;
    :discrepancies_abs,1.3709068298339844e-06;
    :discrepancies_rel,0.03255894407629967;
    :executable,/usr/bin/python;
    :exporter,script;
    :filename,dump_test_models/ElectraForQuestionAnswering-script-cpu-/model.onnx;
    :flag_fake_tensor,False;
    :flag_no_grad,True;
    :flag_training,False;
    :has_cuda,True;
    :input_size,32896;
    :machine,x86_64;
    :model_name,ElectraForQuestionAnswering;
    :onnx_filesize,55613972;
    :onnx_input_names,input.1|onnx::Clip_1|onnx::Clip_2;
    :onnx_model,1;
    :onnx_n_inputs,3;
    :onnx_n_outputs,3;
    :onnx_optimized,0;
    :onnx_output_names,1300|onnx::SoftmaxCrossEntropyLoss_1286|onnx::SoftmaxCrossEntropyLoss_1288;
    :opt_patterns,;
    :output_size,65537.0;
    :params_dtype,float32;
    :params_size,13483522;
    :processor,x86_64;
    :providers,CPUExecutionProvider;
    :repeat,3;
    :speedup,1.3189447836001065;
    :speedup_increase,0.3189447836001065;
    :time_export,19.68962045799981;
    :time_latency,10.412437652000031;
    :time_latency_eager,13.733430325666783;
    :time_load,0.3337397940003939;
    :time_session,0.22385592099999485;
    :time_total,116.15856191800003;
    :time_warmup,10.869273103000069;
    :time_warmup_eager,12.341592189000039;
    :version,3.10.12;
    :version_onnxruntime,1.18.0+cu118;
    :version_torch,2.5.0.dev20240705+cu118;
    :version_transformers,4.42.3;
    :warmup,1;

List of models
==============

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ""

::

     0 - 101Dummy
     1 - 101Dummy16
     2 - 101DummyTuple
     3 - AlbertForMaskedLM
     4 - AlbertForQuestionAnswering
     5 - AllenaiLongformerBase
     6 - BartForCausalLM
     7 - BartForConditionalGeneration
     8 - BertForMaskedLM
     9 - BertForQuestionAnswering
    10 - BlenderbotForCausalLM
    11 - BlenderbotForConditionalGeneration
    12 - BlenderbotSmallForCausalLM
    13 - BlenderbotSmallForConditionalGeneration
    ...

Multiple Runs
=============

``--model all`` runs the same command as above a in new process each time,
``--model All`` runs the same command as above a in new process each time,
``--model Head`` runs the same command as above a in new process each time
with the ten first model of the benchark,
``--model Tail`` runs the same command as above a in new process each time
with the ten first model of the benchark.
Any value with ``,`` means the command line needs to be run multiple times
with multiple values. For example, the following command line:

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter script,dynamo2 --verbose 3 --quiet 1 -w 1 -r 3

Will run:

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter script --verbose 3 --quiet 1 -w 1 -r 3
    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter dynamo2 --verbose 3 --quiet 1 -w 1 -r 3

Multiple fields may have multiple values.
Every run outputs some variable following the format
``:<name>,<value>;``. All of these expressions are collected
and aggregated in a csv file.

Aggregated Report
=================

An aggregated report can be produced by command line:

::

    python -m experimental_experiment.torch_bench.bash_bench_agg summary.xlsx bench1.csv bench2.csv ...
