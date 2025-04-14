"""
Fuzzy work.
"""

__version__ = "0.1.1"
__author__ = "Xavier Dupré"


def reset_torch_transformers(gallery_conf, fname):
    "Resets torch dynamo for :epkg:`sphinx-gallery`."
    import matplotlib.pyplot as plt
    import torch

    plt.style.use("ggplot")
    torch._dynamo.reset()
