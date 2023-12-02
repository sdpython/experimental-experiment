from ..values import Opset
from .opset1 import Opset1


class Opset2(Opset1):
    def __new__(cls):
        return Opset.__new__(cls, "", 2)

    def __init__(self):
        pass
