from ..values import Opset
from .opset17 import Opset17


class Opset18(Opset17):
    def __new__(cls):
        return Opset.__new__(cls, "", 18)
