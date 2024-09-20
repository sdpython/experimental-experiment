def str_dtype(dtype: "torch.dtype") -> str:  # noqa: F821
    """Converts a dtype into a string."""
    return str(dtype).replace("torch.", "")
