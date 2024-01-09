"""
To ignore
=========
"""
import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F


class MyModelClass(nn.Module):
    def __init__(self):
        super(MyModelClass, self).__init__()
        self.large = False
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model_and_input():
    shape = [1, 1, 16, 16]
    input_tensor = torch.rand(*shape).to(torch.float32)
    model = MyModelClass()
    assert model(input_tensor) is not None
    return model, input_tensor


def get_torch_dort(model, *args):
    optimized_mod = torch.compile(model, backend="onnxrt", fullgraph=True)
    # fails: FAIL : Type Error: Type (tensor(int64)) of output arg (max_pool2d_with_indices_1) of node (_aten_max_pool_with_indices_onnx_16) does not match expected type (tensor(float)).
    optimized_mod(*args)
    return optimized_mod


model, input_tensor = create_model_and_input()
get_torch_dort(model, input_tensor)
