from onnx.reference.op_run import OpRun


class Rotary(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, X, splits=None, side=None):
        assert splits is None or (
            splits.shape == (2,) and splits[0] == splits[1]
        ), f"Unexpected split value {splits}"
        last_dim = X.shape[-1] // 2
        cp = X.copy()
        if side == "left":
            cp[..., :last_dim] = X[..., last_dim:]
            cp[..., last_dim:] = -X[..., :last_dim]
        else:
            cp[..., :last_dim] = -X[..., last_dim:]
            cp[..., last_dim:] = X[..., :last_dim]
        return (cp,)
