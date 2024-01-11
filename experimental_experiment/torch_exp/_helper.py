from typing import Any


def make_hash(obj: Any) -> str:
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"
