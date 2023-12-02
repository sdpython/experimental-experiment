from ..values import Opset
from .opset3 import Opset3


class Opset4(Opset3):
    def __new__(cls):
        return Opset.__new__(cls, "", 4)

    def __init__(self):
        pass
