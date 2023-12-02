from ..values import Opset
from .opset2 import Opset2


class Opset3(Opset2):
    def __new__(cls):
        return Opset.__new__(cls, "", 3)

    def __init__(self):
        pass
