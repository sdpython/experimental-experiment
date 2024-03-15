python -m experimental_experiment.torch_bench.dort_bench -w 1 -r 2 --config medium  --mixed 0 --device cuda --backend ort --num_hidden_layers 1 --dynamic 0
python -m experimental_experiment.torch_bench.dort_bench -w 1 -r 2 --config medium  --mixed 1 --device cuda --backend custom --num_hidden_layers 1 --dynamic 0
python -m experimental_experiment.torch_bench.dort_bench -w 1 -r 2 --config medium  --mixed 0 --device cuda --backend ort --num_hidden_layers 1 --dynamic 1
python -m experimental_experiment.torch_bench.dort_bench -w 1 -r 2 --config medium  --mixed 1 --device cuda --backend custom --num_hidden_layers 1 --dynamic 1
