from ..values import Opset
from .opset16 import Opset16


class Opset17(Opset16):
    def __new__(cls):
        return Opset.__new__(cls, "", 17)
