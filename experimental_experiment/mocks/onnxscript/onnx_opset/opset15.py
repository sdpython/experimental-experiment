from ..values import Opset
from .opset14 import Opset14


class Opset15(Opset14):
    def __new__(cls):
        return Opset.__new__(cls, "", 15)

    def __init__(self):
        pass
