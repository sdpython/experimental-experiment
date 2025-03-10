import numpy as np
import scipy.special as scipy_special
from onnx.reference.op_run import OpRun


class Attention(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        weights,
        bias,
        mask_index,
        past,
        attention_bias,
        num_heads=None,
    ):
        assert past is None, f"Attention not implemented if past == {past!r}"
        assert (
            num_heads == attention_bias.shape[1]
        ), f"num_heads={num_heads} not in attention_bias.shape={attention_bias.shape}"
        d = weights.shape[1] // 3
        q_weights = weights[:, :d]
        k_weights = weights[:, d : d * 2]
        v_weights = weights[:, d * 2 :]

        d = bias.shape[0] // 3
        q_bias = bias[:d]
        k_bias = bias[d : d * 2]
        v_bias = bias[d * 2 :]

        shape_4d = (*x.shape[:2], num_heads, -1)

        # nodes
        mask_applied = mask_index == 0
        xqb = x @ q_weights + q_bias
        xqb_4d = xqb.reshape(shape_4d)
        xkb = x @ k_weights + k_bias
        xkb_4d = xkb.reshape(shape_4d)
        xvb = x @ v_weights + v_bias
        xvb_4d = xvb.reshape(shape_4d)
        rot_xqb = np.transpose(xqb_4d, axes=(0, 2, 1, 3))
        rot_xkb = np.transpose(xkb_4d, axes=(0, 2, 1, 3))
        matmul = 0.125 * rot_xqb @ np.transpose(rot_xkb, [0, 1, 3, 2])
        transpose_3 = np.transpose(xvb_4d, axes=(0, 2, 1, 3))
        add_322 = matmul + attention_bias
        masked_fill_2 = np.where(mask_applied, -np.inf, add_322)
        softmax = scipy_special.softmax(masked_fill_2, axis=-1)
        masked_fill_3 = np.where(mask_applied, 0, softmax)
        matmul_1 = masked_fill_3 @ transpose_3
        transpose_5 = np.transpose(matmul_1, axes=(0, 2, 1, 3))
        view_3 = transpose_5.reshape(x.shape)
        return (view_3,)
