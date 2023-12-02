from ..values import Opset
from .opset6 import Opset6


class Opset7(Opset6):
    def __new__(cls):
        return Opset.__new__(cls, "", 7)

    def __init__(self):
        pass
