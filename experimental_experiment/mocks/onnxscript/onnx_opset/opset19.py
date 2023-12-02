from ..values import Opset
from .opset18 import Opset18


class Opset19(Opset18):
    def __new__(cls):
        return Opset.__new__(cls, "", 19)

    def __init__(self):
        pass
