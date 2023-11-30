from .opset1 import Opset1
from .opset17 import Opset17
from .opset18 import Opset18

last_opset = 21

all_opsets = {("", 1): Opset1(), ("", 17): Opset17(), ("", 18): Opset18()}
