from .opset1 import Opset1
from .opset2 import Opset2
from .opset3 import Opset3
from .opset4 import Opset4
from .opset5 import Opset5
from .opset6 import Opset6
from .opset7 import Opset7
from .opset8 import Opset8
from .opset9 import Opset9
from .opset10 import Opset10
from .opset11 import Opset11
from .opset12 import Opset12
from .opset13 import Opset13
from .opset14 import Opset14
from .opset15 import Opset15
from .opset16 import Opset16
from .opset17 import Opset17
from .opset18 import Opset18
from .opset19 import Opset19


last_opset = 19

all_opsets = {
    ("", 1): Opset1(),
    ("", 2): Opset2(),
    ("", 3): Opset3(),
    ("", 4): Opset4(),
    ("", 5): Opset5(),
    ("", 6): Opset6(),
    ("", 7): Opset7(),
    ("", 8): Opset8(),
    ("", 9): Opset9(),
    ("", 10): Opset10(),
    ("", 11): Opset11(),
    ("", 12): Opset12(),
    ("", 13): Opset13(),
    ("", 14): Opset14(),
    ("", 15): Opset15(),
    ("", 16): Opset16(),
    ("", 17): Opset17(),
    ("", 18): Opset18(),
    ("", 19): Opset19(),
}


default_opset = all_opsets["", 18]
