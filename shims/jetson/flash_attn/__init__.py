"""Stub for flash_attn — no aarch64 wheel exists, and the Eagle remote code imports it at
module level before any config override can select a different attention implementation.

The benchmark runs with --attn-impl sdpa, which is the same attention computed by a
different kernel. Every symbol here is a callable that raises when INVOKED (not when
merely imported — see flash_attn_interface.py for why that distinction matters). If the
flash path is genuinely taken, the run dies loudly rather than producing an
unattributable number.
"""
__version__ = "0.0.0-stub"

class _LoudCallable:
    def __init__(self, name): self._n = name
    def __call__(self, *a, **k):
        raise RuntimeError(f"flash_attn.{self._n} was INVOKED — this is a STUB. See "
                           f"flash_attn/__init__.py. Do not trust this process's output.")
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"): raise AttributeError(k)
        return _LoudCallable(f"{self._n}.{k}")
    def __repr__(self): return f"<flash_attn stub {self._n}>"

def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _LoudCallable(name)
