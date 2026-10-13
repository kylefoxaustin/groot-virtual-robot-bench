"""Stub submodule — see flash_attn/__init__.py. Names resolve; invoking raises."""
from .flash_attn_interface import _LoudCallable
def __getattr__(name):
    if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
    return _LoudCallable(f"bert_padding.{name}")
