"""Stub for pytorch3d.

Isaac-GR00T n1.5-release imports `pytorch3d.transforms as pt` at module level in
gr00t/data/transform/state_action.py and then NEVER CALLS IT — verified: zero `pt.`
references anywhere in the repo. The real package will not build against torch 2.9 here.

This stub exists only to satisfy the dead import. If any attribute is ever actually
touched it raises loudly rather than returning something plausible, because a silent
wrong rotation would be exactly the kind of defect that survives a benchmark.
"""
class _Loud:
    def __getattr__(self, name):
        raise RuntimeError(
            f"pytorch3d.transforms.{name} was CALLED, but this is a stub installed on the "
            f"assumption that it never is. That assumption is now false — install the real "
            f"pytorch3d and re-run. Do not trust any result from this process.")
transforms = _Loud()
