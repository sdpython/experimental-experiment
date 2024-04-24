====================================
Interesting scripts or command lines
====================================

Max Size
========

::

    clear
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 10 --backend eager
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 6 --backend dynger
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 10 --backend inductor
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 10 --backend ortmodule
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 7 --backend custom 
    echo "############################################"
    python -m experimental_experiment.torch_bench.dort_bench --device cuda --mixed=1 -w 3 --enable_pattern=default+onnxruntime+experimental --config large  --num_hidden_layer 7 --backend ort+
    echo "############################################"

