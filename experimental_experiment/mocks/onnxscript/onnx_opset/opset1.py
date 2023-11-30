from ..values import Opset


class Opset1(Opset):
    def __new__(cls):
        return Opset.__new__(cls, "", 1)

    def __init__(self):
        pass
