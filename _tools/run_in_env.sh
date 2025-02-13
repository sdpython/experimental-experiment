#!/bin/bash

# Initialization
VENV="benchenv"
TRANSFORMERS_VERSION="4.45.0"
PYTORCH_VERSION="nightly"
ONNXSCRIPT_VERSION="source"
ORT_VERSION="nightly"
DEVICE="cuda"
DTYPE="float16"
CUDA_VERSION="11.8"
SKIP_INSTALL="0"
MAMBA="NO"
TAG=""
QUICK=0


export PYTHONPATH=

# Help
usage() {
    echo "Usage: $0 --transformers-version <version> --pytorch-version <version> --onnxscript-version <version> --ort-version <version> --cuda-version <version> --skip_install --mamba NO ..."
    echo
    echo "Options:"
    echo "  --transformers-version  obvious"
    echo "  --pytorch-version       obvious"
    echo "  --onnxscript-version    obvious"
    echo "  --ort-version           obvious"
    echo "  --venv                  name of the virtual environment"
    echo "  --cuda-version          cuda version"
    echo "  --skip-install          skip the installation of the virtual environment"
    echo "  --mamba                 version of python to use with minimamba instead of the default python"
    echo "  --tag                   tag"
    echo
    echo "Example: $0 --transformers-version 4.37.2 --pytorch-version nightly --ort-version nightly --onnxscript-version nightly --cuda-version 11.8 --mamba 3.11 \\"
    echo "         -m experimental_experiment.torch_bench.bash_bench_huggingface --device cuda --dtype float16 --quiet 1 --memory_peak 1 --exporter script,onnx_dynamo,custom --model All -r 30 -w 10 --verbose 1 --dump_ort 1"
    echo
    echo "Once the installation is done, you can skip the installation by adding --skip-install to the command line."
    exit 1
}

echo "[$0] starts parsing..."
ARGS=()
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
        --onnxscript-version)
            ONNXSCRIPT_VERSION="$2"
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
        --tag)
            TAG="$2"
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
        --help)
            usage
            ;;
        *)
            if echo "$1" | grep -q "="; then
                echo "argument '$1' is invalid."
                exit 1
            fi
            ARGS+=("$1")
            shift 1
            ;;
        esac
done

if [[ ${#ARGS[@]} ]];
then
    # Let's add dummy parameters.
    ARGS=("-c")
    ARGS+=("import sys;print(sys.executable)") 
fi


# Options
echo "[$0] TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
echo "[$0] ONNXSCRIPT_VERSION=${ONNXSCRIPT_VERSION}"
echo "[$0] PYTORCH_VERSION=${PYTORCH_VERSION}"
echo "[$0] ORT_VERSION=${ORT_VERSION}"
echo "[$0] VENV=${VENV}"
echo "[$0] PATH=${PATH}"
echo "[$0] TAG=${TAG}"
echo "[$0] CUDA_VERSION=${CUDA_VERSION}"
echo "[$0] MAMBA=${MAMBA}"
echo "[$0] ARGS=${ARGS[*]}"

if [[ $CUDA_VERSION != "11.8" ]];
then
    echo "[$0] not implemented yet for CUDA_VERSION=11.8"
    exit 1
fi

CUDA_VERSION_NO_DOT=118
BENCHNAME="${VENV}${MAMBA}${TAG}"
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
            # temp_value=$(./bin/micromamba shell hook -s posix)
            # echo "[$0]  -- ${temp_value}"
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
        python --version || exit 1
    fi

    pip install --upgrade pip
    echo "[$0] Install numpy"
    pip install setuptools wheel --upgrade
    pip install numpy==1.26.4 pandas matplotlib openpyxl sympy flatbuffers h5py packaging onnx cerberus pybind11 cython onnx-array-api boto3
    echo "[$0] done numpy"

    echo "[$0] - install iopath..."
    pip install iopath fbgemm_gpu_nightly pyre-extensions opencv-python effdet pyre-extensions
    echo "[$0] done other dependencies"

    if [[ $PYTORCH_VERSION == "nightly" ]];
    then
        if [ $DEVICE == "cuda" ] || [ $DEVICE == "cuda:1" ] || [ $DEVICE == "cuda:2" ];
        then
            echo "[$0] Install nightly pytorch + cuda"
            pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_NO_DOT}
        else
            echo "[$0] Install nightly pytorch + cpu"
            pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi
    else
        if [ $DEVICE == "cuda" ] || [ $DEVICE == "cuda:1" ] || [ $DEVICE == "cuda:2" ];
        then
            echo "[$0] Install pytorch==${PYTORCH_VERSION} + cuda"
            pip install --upgrade --pre torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_NO_DOT}
        else
            echo "[$0] Install pytorch==${PYTORCH_VERSION} + cpu"
            pip install --pre torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi
    fi
    echo "[$0] done pytorch"

    echo "[$0] - install  timme..."
    pip install torchrec ffdet timm
    echo "[$0] done other dependencies"

    if [[ $ORT_VERSION == "nightly" ]];
    then
        if [ $DEVICE == "cuda" ] | [ $DEVICE == "cuda:1" ] | [ $DEVICE == "cuda:2" ];
        then
            echo "[$0] Install nightly onnxruntime + cuda"
            pip install --upgrade -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
        else
            echo "[$0] Install nightly onnxruntime + cpu"
            pip install --upgrade -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
        fi
    else
        if [ $DEVICE == "cuda" ] | [ $DEVICE == "cuda:1" ] | [ $DEVICE == "cuda:2" ];
        then
            echo "[$0] Install onnxruntime==${ORT_VERSION} + cuda"
            pip install "onnxruntime-training-gpu==${ORT_VERSION}"
        else
            echo "[$0] Install onnxruntime==${ORT_VERSION} + cpu"
            pip install "onnxruntime-training==${ORT_VERSION}"
        fi
    fi
    echo "[$0] done onnxruntime"

    echo "[$0] Install transformers==${TRANSFORMERS_VERSION}, deepspeed"
    python -m pip install "transformers==${TRANSFORMERS_VERSION}" deepspeed
    echo "[$0] done transformers"
        
    if [[ $ONNXSCRIPT_VERSION == "source" ]];
    then
        echo "[$0] Install onnxscript from source"
        python -m pip install git+https://github.com/microsoft/onnxscript.git
    else 
        if [[ $ONNXSCRIPT_VERSION == "nightly" ]];
        then
            echo "[$0] Install onnxscript from source"
            python -m pip install onnxscript --upgrade --pre
        else 
            echo "[$0] Install onnxscript==${ONNXSCRIPT_VERSION}"
            python -m pip install onnxscript==${ONNXSCRIPT_VERSION} --upgrade --pre
        fi
    fi
    echo "[$0] done onnxscript"

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

    echo "[$0] MAMBA=${MAMBA}"
    if [[ $MAMBA == "NO" ]]
    then
        echo "[$0] deactivate python env"
        deactivate || exit 1
    else
        echo "[$0] deactivate mamba env"
        micromamba deactivate || exit 1
    fi
    echo "[$0] done requirements"
fi

echo "[$0] activate environment with MAMBA=${MAMBA}"
echo "[$0] BENCHNAME=${BENCHNAME}"
current_dir=$(pwd)
if [[ $MAMBA == "NO" ]]
then
    echo "[$0] activate python env"
    source "${BENCHNAME}/bin/activate" || exit 1
else
    echo "[$0] activate mamba env"
    # temp_value=$(./bin/micromamba shell hook -s posix)
    # echo "[$0] ... ${temp_value}"
    eval "$(./bin/micromamba shell hook -s posix)" || exit 1
    micromamba activate $BENCHNAME || exit 1
fi
echo "[$0] done activate"


# VERIFICATIONS
echo "[$0] VERIFICATIONS"

echo "[$0] python version"
python --version || exit 1

echo "[$0] CUDA AVAILABLE?"
python -c "import torch;print(torch.cuda.device_count() > 0)"
echo "[$0] NVIDIA-SMI"
nvidia-smi

echo "[$0] pip freeze"
pip freeze || exit 1

echo "[$0] DONE VERIFICATIONS"


# BENCHMARK
echo "[$0] RUN python ${ARGS[*]}"

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

echo "--------------------------------------------------------"
echo "[$0] START"
echo "" > "log.begin"

echo "[$0] RUN python ${ARGS[*]}"
python --version
echo '# eval "$(./bin/micromamba shell hook -s posix)' > "cmd.sh"
echo "# micromamba activate ${BENCHNAME}" >> "cmd.sh"
echo "export PYTHONPATH=${HERE}/repos/torchbenchmark:${HERE}/repos/pytorch/benchmarks/dynamo" >> "cmd.sh"
echo "python ${ARGS[*]}" >> "cmd.sh"
echo "# micromamba dactivate" >> "cmd.sh"
bash "cmd.sh" > "cmd.log" 2>&1
echo "[$0] DONE python ${ARGS[*]}"

echo "" > "log.end"

cd ..
cd ..

if [[ $MAMBA == "NO" ]]
then
    deactivate || exit 1
else
    micromamba deactivate || exit 1
fi
echo "[$0] DONE BENCHMARKS"
