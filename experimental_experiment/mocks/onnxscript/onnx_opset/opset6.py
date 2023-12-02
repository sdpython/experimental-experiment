from ..values import Opset
from .opset5 import Opset5


class Opset6(Opset5):
    def __new__(cls):
        return Opset.__new__(cls, "", 6)

    def __init__(self):
        pass
