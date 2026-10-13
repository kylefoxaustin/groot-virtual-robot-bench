"""Stub submodule — see flash_attn/__init__.py.

⚠️ DESIGN NOTE, learned the hard way: an attribute access is NOT a use. The Eagle remote
code does `from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func`
at module level — a BINDING, not a CALL. A stub that raises on __getattr__ cannot tell the
two apart and kills the import. So every name resolves to a callable that raises only when
INVOKED. Import succeeds; use dies loudly.
"""
class _LoudCallable:
    def __init__(self, name): self._n = name
    def __call__(self, *a, **k):
        raise RuntimeError(
            f"flash_attn.flash_attn_interface.{self._n} was INVOKED — this is a STUB. The run "
            f"was configured for sdpa attention; if this fires, the flash kernel path IS being "
            f"taken and the stub's premise is false. Install real flash-attn or fix the attn "
            f"implementation. Do not trust this process's output.")
    def __repr__(self): return f"<flash_attn stub {self._n}>"

def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _LoudCallable(name)
