=========================================================
Measuring the exporters on a short list of sets of models
=========================================================

This benchmark aims measures a couple of exporter or ways to run a pytorch model
and various sets of models to check which one is running or better in some conditions.
It can be triggered on sets or models through a different script for each of them:

* **explicit**: ``python -m experimental_experiment.torch_bench.bash_bench_explicit``
* **huggingface**: ``python -m experimental_experiment.torch_bench.bash_bench_huggingface``
* **huggingface_big**: ``python -m experimental_experiment.torch_bench.bash_bench_huggingface_big``
* **issues**: ``python -m experimental_experiment.torch_bench.bash_bench_issues``
* **timm**: ``python -m experimental_experiment.torch_bench.bash_bench_timm``
* **torchbench**: ``python -m experimental_experiment.torch_bench.bash_bench_torchbench``
* **torchbench_ado**: ``python -m experimental_experiment.torch_bench.bash_bench_torchbench_ado``
* **untrained**: ``python -m experimental_experiment.torch_bench.bash_bench_untrained``

**huggingface** is a set of models coming from :epkg:`transformers`,
**huggingface_big** is a another set of models coming from :epkg:`transformers`, models are bigger,
**timm** is a set of models coming from :epkg:`timm`,
**torchbench** and **torchbench_ado** models come from :epkg:`torchbench`,
**explicit** is a set of custom models,
**issues** is a set of models to track after they failed,
**untrained** is a set similar to *huggingface_big* but it bypasses the downloading
part which can takes several minutes.

These scripts are usually uses in two ways:

* a single run: to investigate a failure or a slow model
* a batch run: to benchmark many models on many exporters

Examples are using with ``bash_bench_huggingface`` but any of the other can be used.

List of models
==============

The list of supported models can be obtained by running:

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

Single Run
==========

The script loads a model, *ElectraForQuestionAnswering* in this case,
warms up 10 times, measure the time to run inference 30 times. Then it converts it
into onnx, and do the same. This script is usually run with ``--quiet=0``
to ensure the script stops as soon as an exception is raised. One example:

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

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter script,onnx_dynamo --verbose 3 --quiet 1 -w 1 -r 3

Will run:

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter script --verbose 3 --quiet 1 -w 1 -r 3
    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ElectraForQuestionAnswering --device cpu --exporter onnx_dynamo --verbose 3 --quiet 1 -w 1 -r 3

Multiple fields may have multiple values.
Every run outputs some variable following the format
``:<name>,<value>;``. All of these expressions are collected
and aggregated in a csv file.

Aggregated Report
=================

An aggregated report can be produced by command line:

::

    python -m experimental_experiment.torch_bench.bash_bench_agg summary.xlsx bench1.csv bench2.csv ...

Other options of this command line allow the user to filter in ir out some data
(see ``--filter_in``, ``--filter_out``). The aggregator assumes every differences
in the version is a tested difference. If not, different versions can be ignored
by using ``--skip_keys=version,version_torch`` or any other key column not meant
to be used in the report.
