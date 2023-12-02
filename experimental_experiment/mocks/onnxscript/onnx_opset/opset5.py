from ..values import Opset
from .opset4 import Opset4


class Opset5(Opset4):
    def __new__(cls):
        return Opset.__new__(cls, "", 5)

    def __init__(self):
        pass
