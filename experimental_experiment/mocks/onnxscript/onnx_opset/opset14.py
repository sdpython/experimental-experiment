from ..values import Opset
from .opset13 import Opset13


class Opset14(Opset13):
    def __new__(cls):
        return Opset.__new__(cls, "", 14)

    def __init__(self):
        pass
