def __getattr__(name):
    raise RuntimeError(
        f"pytorch3d.transforms.{name} was CALLED — the stub's assumption is violated. "
        f"Install the real pytorch3d and re-run; do not trust this process's output.")
