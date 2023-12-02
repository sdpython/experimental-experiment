from ..values import Opset
from .opset11 import Opset11


class Opset12(Opset11):
    def __new__(cls):
        return Opset.__new__(cls, "", 12)

    def __init__(self):
        pass
