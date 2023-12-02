from ..values import Opset
from .opset12 import Opset12


class Opset13(Opset12):
    def __new__(cls):
        return Opset.__new__(cls, "", 13)

    def __init__(self):
        pass
