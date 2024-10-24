"""
.. _l-plot-torch-export-101:

101: First test with ExecuTorch
===============================

See :epkg:`ExecuTorch`, :epkg:`ExecuTorch Tutorial`,
:epkg:`ExecuTorch Runtime Python API Reference`.

Convert a Model
+++++++++++++++
"""

from pathlib import Path
import torch
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

edge_program: EdgeProgramManager = to_edge(exported_program)

######################################
# Serializes.

save_path = "plot_executorch_101.pte"
executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

with open(save_path, "wb") as file:
    file.write(executorch_program.buffer)


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

et_runtime: Runtime = Runtime.get()
program: Program = et_runtime.load_program(
    Path("plot_executorch_101.pte"), verification=Verification.Minimal
)

print("Program methods:", program.method_names)
forward: Method = program.load_method("forward")

outputs = forward.execute(inputs)

###################
# Let's compare.

diff = torch.abs(outputs[0] - expected).max()
print("max discrepancies:", diff)
