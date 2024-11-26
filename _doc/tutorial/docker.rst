
===================
Start from a docker
===================

Benchmark a llama model
=======================

The chosen docker is the following one:
`PyTorch Release 24.04 - NVIDIA Docs <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html>`_.

**Create the docker**

.. code-block:: bash

    sudo docker run --gpus all -it --name docker-dort --rm -v `pwd`:/github -w /github nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
    sudo docker commit docker-dort docker-dort-updated
    sudo docker run  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_ADMIN --gpus all -it --name docker-dort-new --rm -v `pwd`:/github -w /github docker-dort-updated

**Set up the environment**

.. code-block:: bash

    pip install --upgrade numpy scipy jupyter matplotlib black ruff cython pybind11 flatbuffers sympy sphinx furo coloredlogs onnx cerberus pandas scikit-learn parameterized pytest isort pytest-cov openpyxl accelerate
    pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

    # flash attention
    # if this appear: 
    # /usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
    # see https://github.com/oobabooga/text-generation-webui/issues/4182

    pip uninstall -y flash-attn
    FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn

    # transformers
    # inductor is not working with the latest version, onnxruntime is very slow probably due to a wrong sequence of kernels (but nsys profile does not work)
    pip install transformers==4.37.1

    # versions
    pip freeze

::

    Cerberus==1.3.5
    Cython==3.0.10
    flash-attn==2.5.8
    numpy==1.26.4
    nvfuser==0.1.6a0+a684e2a
    nvidia-cublas-cu12==12.4.2.65
    nvidia-cuda-cupti-cu12==12.4.99
    nvidia-cuda-nvrtc-cu12==12.4.99
    nvidia-cuda-runtime-cu12==12.4.99
    nvidia-cudnn-cu12==8.9.7.29
    nvidia-cufft-cu12==11.2.0.44
    nvidia-curand-cu12==10.3.5.119
    nvidia-cusolver-cu12==11.6.0.99
    nvidia-cusparse-cu12==12.3.0.142
    nvidia-dali-cuda120==1.36.0
    nvidia-nccl-cu12==2.20.5
    nvidia-nvimgcodec-cu12==0.2.0.7
    nvidia-nvjitlink-cu12==12.4.99
    nvidia-nvtx-cu12==12.4.99
    nvidia-pyindex==1.0.9
    onnx==1.16.0
    onnx-array-api==0.2.0
    onnxruntime-training==1.19.0+cu124
    pybind11==2.12.0
    torch==2.4.0.dev20240522+cu124
    torch-ort==1.18.0
    torchaudio==2.2.0.dev20240522+cu124
    torchdata @ file:///opt/pytorch/data
    torchtext @ file:///opt/pytorch/text
    torchvision==0.19.0.dev20240522+cu124
    transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@6a9edc38bf9b941b7d369af5103fa8fe0b121d61
    transformers==4.37.1


The docker can be saved with ``sudo docker commit docker-dort-new docker-dort-updated``.

**Compiling onnxruntime**

.. code-block:: bash

    # inside the docker again
    cd github

    # onnxruntime-training: build from source
    git clone https://github.com/microsoft/onnxruntime.git
    cd /github/github/onnxruntime
    git config --global --add safe.directory /github/github/onnxruntime
    git config --global --add safe.directory /github/github/onnxruntime/cmake/external/emsdk
    git config --global --add safe.directory /github/github/onnxruntime/cmake/external/libprotobuf-mutator
    git config --global --add safe.directory /github/github/onnxruntime/cmake/external/onnx

    clear&&CUDA_VERSION=12.4 CUDACXX=/usr/local/cuda-12.4/bin/nvcc python ./tools/ci_build/build.py \
        --config Release --build_wheel --build_dir ./build/linux_cuda --build_shared_lib \
        --use_cuda --cuda_home /usr/local/cuda-12.4/ --cudnn_home /usr/local/cuda-12.4/ \
        --cuda_version=12.4 --enable_training --enable_training_ops  --parallel \
        --skip_tests --enable_nvtx_profile --cmake_extra_defines onnxruntime_ENABLE_ATEN=ON --allow_running_as_root

    # onnxscript
    git clone https://github.com/microsoft/onnxscript.git

    # optional
    git clone https://github.com/onnx/sklearn-onnx.git
    git clone https://github.com/onnx/onnxmltools.git
    git clone https://github.com/microsoft/onnxconverter-common.git

**Install ort extension**

.. code-block:: bash

    export PYTHONPATH=/github/github/onnxruntime/build/linux_cuda/Release
    python install torch_ort
    python -m torch_ort.configure

**Experimental packages**

Mostly made for research until the ideas migrates to an officially supported package.

.. code-block:: bash

    # extra function used to manipulate or display onnx models
    pip install onnx-array-api

    # custom onnxruntime CUDA kernels
    git clone https://github.com/sdpython/onnx-extended.git
    cd /github/github/onnx-extended
    python setup.py build_ext --inplace --cuda-version=12.4 --cuda-link=SHARED

    # experimentation
    git clone https://github.com/sdpython/experimental-experiment.git

**Run DORT on llama on a specific backend**

.. code-block:: bash

    cd /github/github/experimental-experiment
    export PYTHONPATH=/github/github/experimental-experiment/:/github/github/onnx-extended:/github/github/onnxscript:/github/github/onnxruntime/build/linux_cuda/Release:/github/github/sklearn-onnx:/github/github/onnxmltools:/github/github/onnxconverter-common

    # check that dort is working on llama and export the onnx model (flag --help to see other options)
    python -m experimental_experiment.torch_bench.dort_bench --backend ort+ --device cuda --mixed=1 --export model -w 3 -r 5 --enable_pattern=default+onnxruntime+experimental --num_hidden_layers=1

**Test all backend on llama**

.. code-block:: bash

    # full benchmark 
    python _doc/examples/plot_llama_bench_102.py --device=cuda --num_hidden_layers=2 --mixed=1 --backend=eager,dynger,ortmodule,inductor,ort+,custom --config=large --num_hidden_layers=2

**Notes**

Version ``torch==2.4.0.dev20240522`` seems to have a with mixed precision and dynamic shapes.
To replicate:

.. code-block:: bash

    python _unittests/ut_torch_interpreter/test_onnx_export_dynamic_shapes.py
