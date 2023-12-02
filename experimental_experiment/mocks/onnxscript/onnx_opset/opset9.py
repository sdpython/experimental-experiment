from ..values import Opset
from .opset8 import Opset8


class Opset9(Opset8):
    def __new__(cls):
        return Opset.__new__(cls, "", 9)

    def __init__(self):
        pass
