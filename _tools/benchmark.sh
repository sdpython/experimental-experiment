#!/bin/bash

# Initialization
VENV="benchenv"
TRANSFORMERS_VERSION="4.41.2"
PYTORCH_VERSION="nightly"
ORT_VERSION="nightly"
DEVICE="cuda"
SERIES="exp"
DTYPE="float16"
CUDA_VERSION="11.8"
SKIP_INSTALL="0"


# Help
usage() {
    echo "Usage: $0 --transformers-version <version> --pytorch-version <version> --device <cuda> --series <name> --ort-version <version> --dtype <dtype> --cuda-version <version> --skip_install"
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
    echo
    echo "Exemple: $0 --transformers-version 4.37.2 --pytorch-version nightly --ort-version nightly --device cuda --series exp --dtype float16 --cuda-version=11.8"
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
        --skip_install)
            SKIP_INSTALL="1"
            shift 1
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

if [[ $CUDA_VERSION != "11.8" ]];
then
    echo "[$0] not implemented yet for CUDA_VERSION=11.8"
    exit 1
fi

CUDA_VERSION_NO_DOT=118
BENCHNAME="${VENV}${DEVICE}"


# Prepare the virtual environment
alias python=python3

if [[ $SKIP_INSTALL == "0" ]];
then
    echo "[$0] Prepare the virtual environement ${BENCHNAME}"
    python3 -m pip install virtualenv || exit 0
    python3 -m venv ${BENCHNAME} || exit 0
    echo "[$0] done venv"


    echo "[$0] Install requirements"
    source "${BENCHNAME}/bin/activate"
    pip install --upgrade pip
    echo "[$0] Install numpy"
    pip install setuptools wheel --upgrade
    pip install numpy==1.26.4 pandas matplotlib openpyxl
    echo "[$0] done"

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
    echo "[$0] done"

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
            pip install onnxruntime-training-gpu==${ORT_VERSION}
        else
            pip install onnxruntime-training==${ORT_VERSION}
        fi
    fi
    echo "[$0] done"

    echo "[$0] Install transformers, deepspeed"
    python -m pip install transformers==${TRANSFORMERS_VERSION} deepspeed
    echo "[$0] done"
        
    echo "[$0] Install onnxscript from source"
    python -m pip install git+https://github.com/microsoft/onnxscript.git
    echo "[$0] done"

    source "${BENCHNAME}/bin/deactivate"
    echo "[$0] done requirements"
fi

echo "[$0] VERIFICATIONS"
source "${BENCHNAME}/bin/activate"

python -c "import torch;print('CUDA AVAILABLE', torch.cuda.is_available())"
nvidia-smi

pip freeze

source "${BENCHNAME}/bin/deactivate"
echo "[$0] DONE"


