#!/bin/bash

# Initialization
VENV="benchenv"
TRANSFORMERS_VERSION="4.41.2"
PYTORCH_VERSION="nightly"
ORT_VERSION="nightly"
DEVICE="cuda"
SERIES="experiment,onnxscript"
DTYPE="float16"
CUDA_VERSION="11.8"
SKIP_INSTALL="0"
MAMBA="NO"
QUICK=0

# not yet expose
EVAL_MODE="inference"
COMPILER=""


export PYTHONPATH=

# Help
usage() {
    echo "Usage: $0 --transformers-version <version> --pytorch-version <version> --device <cuda> --series <name> --ort-version <version> --dtype <dtype> --cuda-version <version> --skip_install --mamba NO --quick --compiler=dynamo-onnx"
    echo
    echo "Options:"
    echo "  --transformers-version  obvious"
    echo "  --pytorch-version       obvious"
    echo "  --ort-version           obvious"
    echo "  --device                cuda or cpu"
    echo "  --dtypre                float32, float16, ..."
    echo "  --series                name of the series to run"
    echo "  --venv                  name of the virtual environment"
    echo "  --cuda-version          cuda version"
    echo "  --skip-install          skip the installation of the virtual envrionment"
    echo "  --mamba                 version of python to use with minimamba instead of the default python"
    echo "  --quick                 short version"
    echo "  --compiler              compiler, dynamo-onnx, dynamo-onnx-optimize, export, inductor, torchscript-onnx, export, export-aot-inductor, torch-onnx-patch"
    echo
    echo "Example: $0 --transformers-version 4.37.2 --pytorch-version nightly --ort-version nightly --device cuda --series onnxscript --dtype float16 --cuda-version=11.8 --mamba=3.11 --quick --compiler torchscript-onnx"
    echo
    echo "Once the installation is done, you can skip the installation by adding --skip-install to the command line."
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --transformers-version)
            TRANSFORMERS_VERSION="$2"
            shift 2
            ;;
        --pytorch-version)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --ort-version)
            ORT_VERSION="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --series)
            SERIES="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --venv)
            VENV="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL="1"
            shift 1
            ;;
        --mamba)
            MAMBA="$2"
            shift 2
            ;;
        --quick)
            QUICK="1"
            shift 1
            ;;
        --compiler)
            COMPILER="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown options: $1"
            usage
            ;;
    esac
done

# Options
echo "[$0] TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
echo "[$0] PYTORCH_VERSION=${PYTORCH_VERSION}"
echo "[$0] ORT_VERSION=${ORT_VERSION}"
echo "[$0] DEVICE=${DEVICE}"
echo "[$0] SERIES=${SERIES}"
echo "[$0] DTYPE=${DTYPE}"
echo "[$0] VENV=${VENV}"
echo "[$0] PATH=${PATH}"
echo "[$0] CUDA_VERSION=${CUDA_VERSION}"
echo "[$0] MAMBA=${MAMBA}"
echo "[$0] COMPILER=${COMPILER}"
echo "[$0] EVAL_MODE=${EVAL_MODE}"

if [[ $CUDA_VERSION != "11.8" ]];
then
    echo "[$0] not implemented yet for CUDA_VERSION=11.8"
    exit 1
fi

CUDA_VERSION_NO_DOT=118
BENCHNAME="${VENV}${DEVICE}${MAMBA}"
HERE=$(pwd)
export MAMBA_ROOT_PREFIX=${HERE}/mambaroot
export PYTHONPATH=${HERE}/repos/torchbenchmark:${HERE}/repos/pytorch/benchmarks/dynamo

# Prepare the virtual environment
alias python=python3
export | grep PATH

if [[ $SKIP_INSTALL == "0" ]];
then
    if [[ $MAMBA == "NO" ]]
    then
        echo "[$0] create the virtual environement ${BENCHNAME}"
        python3 -m pip install virtualenv || exit 0
        python3 -m venv ${BENCHNAME} || exit 0
        echo "[$0] done venv"

        echo "[$0] Install requirements"
        source "${BENCHNAME}/bin/activate"
    else
        if [[ ! -d "bin/micromamba" ]]
        then 
            echo "[$0] download minimamba"
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba || exit 1
            if [[ -d MAMBA_ROOT_PREFIX ]]
            then
                mkdir $MAMBA_ROOT_PREFIX
            fi
            echo "[$0] done"
            echo "[$0] execute minimamba"
            eval "$(./bin/micromamba shell hook -s posix)" || exit 1
            #  pythonmicromamba activate
            echo "[$0] done minimamba"
        fi
        current_dir=$(pwd)
        #export PATH=$current_dir/minimamba3/bin:$PATH
        #echo "[$0] install python=${MAMBA}"
        #micromamba install -y python=${MAMBA} -c conda-forge || exit 1
        echo "[$0] initialize mamba"
        echo "[$0] done"
        echo "[$0] create the virtual environement ${BENCHNAME}"
        micromamba create -y -n "${BENCHNAME}" -c conda-forge python=${MAMBA} || exit 1
        echo "[$0] done create"
        echo "[$0] activate"
        micromamba activate $BENCHNAME || exit 1
        python --version
    fi

    pip install --upgrade pip
    echo "[$0] Install numpy"
    pip install setuptools wheel --upgrade
    pip install numpy==1.26.4 pandas matplotlib openpyxl sympy flatbuffers h5py packaging onnx cerberus pybind11 cython onnx-array-api boto3
    echo "[$0] done numpy"

    echo "[$0] Install pytorch"
    if [[ $PYTORCH_VERSION == "nightly" ]];
    then
        if [[ $DEVICE == "cuda" ]];
        then
            pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_NO_DOT}
        else
            pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi
    else
        if [[ $DEVICE == "cuda" ]];
        then
            pip install --upgrade --pre torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_NO_DOT}
        else
            pip install --pre torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi
    fi
    echo "[$0] done pytorch"

    echo "[$0] Install onnxruntime"
    if [[ $PYTORCH_VERSION == "nightly" ]];
    then
        if [[ $DEVICE == "cuda" ]];
        then
            pip install --upgrade -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
        else
            pip install --upgrade -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
        fi
    else
        if [[ $DEVICE == "cuda" ]];
        then
            pip install "onnxruntime-training-gpu==${ORT_VERSION}"
        else
            pip install "onnxruntime-training==${ORT_VERSION}"
        fi
    fi
    echo "[$0] done onnxruntime"

    echo "[$0] Install transformers, deepspeed"
    python -m pip install "transformers==${TRANSFORMERS_VERSION}" deepspeed
    echo "[$0] done transformers"
        
    echo "[$0] Install onnxscript from source"
    python -m pip install git+https://github.com/microsoft/onnxscript.git
    echo "[$0] done"

    if [[ ! -d "repos" ]]
    then
        mkdir repos
    fi

    cd repos
    echo "[$0] onnx-extended"
    if [[ ! -d "onnx-extended" ]]
    then
        git clone https://github.com/sdpython/onnx-extended.git
    fi
    cd onnx-extended
    pip install . -v --config-settings="--cuda-version=${CUDA_VERSION} --cuda-link=SHARED"
    cd ..
    echo "[$0] done onnx-extended"

    echo "[$0] pytorch/benchmark"
    if [ ! -d torchbenchmark ]
    then
        git clone https://github.com/pytorch/benchmark.git torchbenchmark --recursive
    fi
    cd torchbenchmark
    git submodule sync
    git submodule update --init --recursive --jobs 0
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    # python install.py --continue_on_fail
    cd ..
    echo "[$0] done pytorch/benchmark"
    cd ..

    echo "[$0] experimental-experiment"
    python -m pip install git+https://github.com/sdpython/experimental-experiment.git
    echo "[$0] done experimental-experiment"

    echo "[$0] clone pytorch/benchmark"
    cd repos
    mkdir pytorch
    cd pytorch
    git init
    git config core.sparsecheckout true
    echo benchmarks >> .git/info/sparse-checkout
    git remote add -f origin https://github.com/pytorch/pytorch.git
    git checkout origin/main -- benchmarks
    git pull origin main
    cd ..
    cd ..
    echo "[$0] done pytorch/benchmark"

    if [[ $MAMBA == "NO" ]]
    then
        deactivate || exit 1
    else
        ./bin/micromamba deactivate || exit 1
    fi
    echo "[$0] done requirements"
fi

echo "[$0] activate environment with MAMBA=${MAMBA}"
echo "[$0] BENCHNAME=${BENCHNAME}"
current_dir=$(pwd)
if [[ $MAMBA == "NO" ]]
then
    source "${BENCHNAME}/bin/activate" || exit 1
else
    eval "$(./bin/micromamba shell hook -s posix)" || exit 1
    micromamba activate $BENCHNAME || exit 1
fi
echo "[$0] done activate"


# VERIFICATIONS
echo "[$0] VERIFICATIONS"

echo "[$0] python version"
python --version

echo "[$0] CUDA AVAILABLE?"
python -c "import torch;print(torch.cuda.is_available())"
echo "[$0] NVIDIA-SMI"
nvidia-smi

echo "[$0] pip freeze"
pip freeze

echo "[$0] DONE VERIFICATIONS"


# BENCHMARK
echo "[$0] BENCHMARK"

if [[ ! -d "results" ]]
then
    mkdir results
fi
HERE=$(pwd)
cd results

current_datetime=$(date "+%Y%m%dH%H%M")
if [[ ! -d "$current_datetime" ]]
then
    mkdir $current_datetime
fi
cd $current_datetime

IFS=','
read -ra KIND <<< "$SERIES"
for name in "${KIND[@]}";
do
    echo "--------------------------------------------------------"
    echo "[$0] START $name"
    if [[ ! -d "$name" ]]
    then
        mkdir $name
    fi
    cd $name

    #########################################################
    #########################################################

    if [[ $name == "experiment" ]];
    then
        echo "not implemented yet"
    fi

    if [[ $name == "onnxscript" ]];
    then
        echo "not implemented yet"
    fi

    if [[ $name == "ado" ]] || [[ $name == "huggingface" ]] || [[ $name == "torchbench" ]] || [[ $name == "timm_models" ]];
    then

        for mode in "performance" "accuracy"; do

            output_file="r_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.csv"
            echo "[$0] output_file=${output_file}"
            echo "" > "log_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.start"

            # dynamic mode is an extra argument to the compiler, parse it from the compiler name
            if [[ $compiler == *"-dynamic" ]]; then
                base_compiler=${COMPILER%-dynamic}
                dynamic=true
            else
                base_compiler=$COMPILER
                dynamic=""
            fi

            export_mode=""
            backend_mode=""
            case "$base_compiler" in
                dynamo-onnx*|torchscript-onnx|export|export-aot-inductor|torch-onnx-patch)
                    export_mode="${base_compiler}"
                    ;;
                dort)
                    backend_mode="onnxrt"
                    ;;
                *)
                    backend_mode="${base_compiler}"
                    ;;
            esac

            output_res=$(pwd)
            bench_args="--${mode} --${DTYPE} -d${DEVICE} --output=${output_file} --output-directory=$output_res"
            bench_args+=" --${EVAL_MODE}"
            bench_args+=${export_mode:+" --$export_mode"}
            bench_args+=${backend_mode:+" --backend=$backend_mode"}
            bench_args+=" --dashboard --timeout 3000 --batch-size 1 --nopython"
            bench_args+=${dynamic:+" --dynamic-shapes"}

            if [[ $quick == "1" ]]
            then
                case "$name" in
                    "torchbench")
                        bench_args+=(-k resnet18)
                        ;;
                    "huggingface")
                        bench_args+=(-k ElectraForQuestionAnswering)
                        ;;
                    "timm_models")
                        bench_args+=(-k lcnet_050)
                        ;;
                    "ado")
                        bench_args+=(-k stable_diffusion_text_encoder)
                        ;;
                    *)
                        echo "Unknown name: ${name}"
                        exit 1
                        ;;
                esac
            fi

            model_skip_list=(
                "fambench_xlmr"
                "pytorch_CycleGAN_and_pix2pix"
                "timm_efficientdet"
                "torchrec_dlrm"
                "clip"
            )
            for model in "${model_skip_list[@]}";
            do
                bench_args+=(-x "$model")
            done

            ado_model_skip_list=(
                "llama_v2_7b_16h"
                "stable_diffusion_xl"
                "stable_diffusion_text_encoder"
                "stable_diffusion_unet"
                "phi_1_5"
                "phi_2"
                "hf_Yi"
                "hf_distil_whisper"
                "mistral_7b_instruct"
                "orca_2"
                "hf_mixtral"
                "codellama"
                "llava"
                "moondream"
            )

            if [ "$EVAL_MODE" == "training" ];
            then
                DTYPE="amp"
            fi
            # export TORCHLIB_EXPERIMENTAL_USE_IR="${TORCHLIB_EXPERIMENTAL_USE_IR}"
            # echo "TORCHLIB_EXPERIMENTAL_USE_IR env var is set to ${TORCHLIB_EXPERIMENTAL_USE_IR}"

            bench_file="${HERE}/repos/pytorch/benchmarks/dynamo/${name}.py"
            if [ "$name" == "ado" ];
            then
                bench_args+=(-n10)  # Number of timed runs. Set to small number due to models being large.
                bench_args+=(--no-skip)
            else
                bench_args+=(-n40)  # Number of timed runs
                for model in "${ado_model_skip_list[@]}";
                do
                    bench_args+=(-x "$model")
                done
            fi

            if [[ ! -f $bench_file ]]
            then
                echo "'${bench_file}' is missing."
            else
                echo "--"
                echo "[$0] RUN python ${bench_file} ${bench_args[@]}"
                python --version
                echo '# eval "$(./bin/micromamba shell hook -s posix)' > "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh"
                echo "# micromamba activate ${BENCHNAME}" >> "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh"
                echo "export PYTHONPATH=${HERE}/repos/torchbenchmark:${HERE}/repos/pytorch/benchmarks/dynamo" >> "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh"
                echo "python ${bench_file} ${bench_args[@]}" >> "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh"
                echo "# micromamba dactivate" >> "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh"
                bash "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.sh" > "cmd_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.log" 2>&1
                echo "[$0] DONE ${output_file}"
            fi

            echo "" > "log_${name}_${COMPILER}_${DTYPE}_${EVAL_MODE}_${DEVICE}_${mode}.end"
        done

    fi

    #########################################################
    #########################################################

    cd ..
    echo "[$0] DONE $name"

done

cd ..
cd ..

if [[ $MAMBA == "NO" ]]
then
    deactivate || exit 1
else
    micromamba deactivate || exit 1
fi
echo "[$0] DONE BENCHMARKS"
