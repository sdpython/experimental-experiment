from ..values import Opset
from .opset9 import Opset9


class Opset10(Opset9):
    def __new__(cls):
        return Opset.__new__(cls, "", 10)

    def __init__(self):
        pass
