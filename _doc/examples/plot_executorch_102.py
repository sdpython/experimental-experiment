"""
.. _l-plot-executorch-102:

102: First test with ExecuTorch
===============================

This script demonstrates :epkg:`ExecuTorch` on a very simple example,
see also :epkg:`ExecuTorch Tutorial`,
:epkg:`ExecuTorch Runtime Python API Reference`.

Convert a Model
+++++++++++++++
"""

from pathlib import Path
import torch

if 1:  # try:
    from executorch.exir import (
        EdgeProgramManager,
        to_edge,
        ExecutorchProgramManager,
        ExecutorchBackendConfig,
    )
    from executorch.runtime import Verification, Runtime, Program, Method

    # This line is needed when executing to_backend.
    from executorch.exir.backend.test.backend_with_compiler_demo import (  # noqa
        BackendWithCompilerDemo,
    )

    executorch = True
else:  # except ImportError:
    print("executorch is not installed.")
    executorch = None


class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)


inputs = (torch.randn(1, 5),)
model = Neuron()
expected = model(*inputs)
exported_program = torch.export.export(model, inputs)
print(exported_program.graph)

######################################
# Conversion to an `EdgeProgramManager`.

if executorch:
    edge_program: EdgeProgramManager = to_edge(exported_program)
    print(f"edge_program {edge_program!r}")

######################################
# Serializes.

if executorch:
    save_path = "plot_executorch_101.pte"
    executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
        ExecutorchBackendConfig(
            passes=[],  # User-defined passes
        )
    )

    with open(save_path, "wb") as file:
        file.write(executorch_program.buffer)
    print(f"model saved into {save_path!r}")


########################################
# It can be specialized for a specific backend.
#
# ::
#
#       from executorch.exir.backend.backend_api import LoweredBackendModule, to_backend
#
#       lowered_module: LoweredBackendModule = to_backend(
#           "BackendWithCompilerDemo",
#           to_be_lowered_module,
#           [],
#       )
#       with open(save_path, "wb") as f:
#           f.write(lowered_module.buffer())

######################################
# Execution
# +++++++++

if executorch:
    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(
        Path("plot_executorch_101.pte"), verification=Verification.Minimal
    )

    print("Program methods:", program.method_names)
    forward: Method = program.load_method("forward")

    outputs = forward.execute(inputs)
    print("forward:", forward)

###################
# Let's compare.

if executorch:
    diff = torch.abs(outputs[0] - expected).max()
    print("max discrepancies:", diff)
