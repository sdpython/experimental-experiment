from ..values import Opset
from .opset15 import Opset15


class Opset16(Opset15):
    def __new__(cls):
        return Opset.__new__(cls, "", 16)

    def __init__(self):
        pass
