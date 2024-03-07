from onnx.reference.ops.op_slice import SliceCommon


class Slice_10(SliceCommon):
    def __init__(self, onnx_node, run_params):
        SliceCommon.__init__(self, onnx_node, run_params)


class Slice_1(SliceCommon):
    def __init__(self, onnx_node, run_params):
        print(onnx_node)
        SliceCommon.__init__(self, onnx_node, run_params)
        for f in ["starts", "ends", "steps", "axes"]:
            if not hasattr(self, f):
                continue
            if getattr(self, f) is not None and len(getattr(self, f)) == 0:
                setattr(self, f, None)

    def _run(self, data, axes=None, ends=None, starts=None):
        return SliceCommon._run(self, data, starts, ends, axes)
