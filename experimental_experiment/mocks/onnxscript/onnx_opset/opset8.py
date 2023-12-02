from ..values import Opset
from .opset7 import Opset7


class Opset8(Opset7):
    def __new__(cls):
        return Opset.__new__(cls, "", 8)

    def __init__(self):
        pass
