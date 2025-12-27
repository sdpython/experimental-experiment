import time
import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime

itype = onnx.TensorProto.FLOAT16
dtype = np.float16

print("-- creating the model")
model1 = oh.make_model(
    oh.make_graph(
        [
            oh.make_node(
                "Attention",
                ["Q", "K", "V"],
                ["Y1"],
                kv_num_heads=16,
                q_num_heads=16,
                scale=0.11180339,
                softmax_precision=1,
            ),
        ],
        "test",
        [
            oh.make_tensor_value_info("Q", itype, ["batch", "seq", 1280]),
            oh.make_tensor_value_info("K", itype, ["batch", "seq", 1280]),
            oh.make_tensor_value_info("V", itype, ["batch", "seq", 1280]),
        ],
        [
            oh.make_tensor_value_info("Y1", itype, ["batch", "seq", 1280]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 23), oh.make_opsetid("com.microsoft", 1)],
    ir_version=11,
)
onnx.save(model1, "attention23.onnx")

model2 = oh.make_model(
    oh.make_graph(
        [
            oh.make_node(
                "MultiHeadAttention",
                ["Q", "K", "V"],
                ["Y2"],
                num_heads=16,
                scale=0.11180339,
                domain="com.microsoft",
            ),
        ],
        "test",
        [
            oh.make_tensor_value_info("Q", itype, ["batch", "seq", 1280]),
            oh.make_tensor_value_info("K", itype, ["batch", "seq", 1280]),
            oh.make_tensor_value_info("V", itype, ["batch", "seq", 1280]),
        ],
        [
            oh.make_tensor_value_info("Y2", itype, ["batch", "seq", 1280]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 23), oh.make_opsetid("com.microsoft", 1)],
    ir_version=11,
)
onnx.save(model2, "mha.onnx")

N = 14308 // 4
feeds = dict(
    Q=np.random.rand(1, N, 1280).astype(dtype),
    K=np.random.rand(1, N, 1280).astype(dtype),
    V=np.random.rand(1, N, 1280).astype(dtype),
)

print("-- creating the session")
opts = onnxruntime.SessionOptions()
# opts.enable_profiling = True
# opts.intra_op_num_threads = 1
# opts.inter_op_num_threads = 1
sess1 = onnxruntime.InferenceSession(
    model1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)
sess2 = onnxruntime.InferenceSession(
    model2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)
n = 2
print("-- run Attention")
times = []
for _t in range(n):
    t = time.perf_counter()
    sess1.run(None, feeds)
    b = time.perf_counter() - t
    print(b)
    times.append(b)
print("-- done", times)

print("-- run MHA")
times = []
for _t in range(n):
    t = time.perf_counter()
    sess2.run(None, feeds)
    b = time.perf_counter() - t
    print(b)
    times.append(b)
print("-- done", times)
