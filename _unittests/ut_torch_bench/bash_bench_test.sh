#############
# HuggingFace
#############

clear

# A single run

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom --model 0 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_one.csv || exit 1

# Two models

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom --model 0,1 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_two.csv || exit 1

# Two optimizers

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom --model 0 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_opt.csv --opt_pattern=,default || exit 1

# Two exporters

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom,script --model 0 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_exporter.csv || exit 1

# Everything

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom,script --model 0,1 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_all.csv --opt_pattern=,default || exit 1

# Split-process

python -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 0 --memory_peak 0 --exporter custom,script --model 0,1 -r 30 -w 10 --verbose 1 --output_data test_bash_bench_split.csv --opt_pattern=,default --split_process 1 || exit 1
