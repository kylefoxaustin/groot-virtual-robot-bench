"""Stub for decord.

The Eagle2.5-VL processor module declares `import decord` at file scope for its VIDEO
path. transformers' dynamic-module loader runs check_imports() over that file and refuses
to load unless every declared import resolves — whether or not it is reached. We feed
IMAGES, not video, and the video path is never taken.

decord has no aarch64 wheels, so on Orin and Thor the choice is: build it from source to
satisfy an import that is never executed, or stub it.

Every attribute access raises. If the video path is ever actually taken, the run dies
loudly instead of silently decoding nothing — the same discipline as the pytorch3d stub.
"""
class _Loud:
    def __init__(self, name): self._n = name
    def __getattr__(self, k):
        raise RuntimeError(
            f"decord.{self._n}.{k} was CALLED. This is a stub installed because the Eagle "
            f"processor DECLARES decord for its video path, which we do not use. If this "
            f"fires, the video path IS being taken and the stub's premise is false — "
            f"install real decord and re-run. Do not trust this process's output.")
    def __call__(self, *a, **k): self.__getattr__("__call__")

def __getattr__(name):
    return _Loud(name)

class VideoReader(_Loud):
    def __init__(self, *a, **k): super().__init__("VideoReader"); self.__getattr__("__init__")
cpu = _Loud("cpu")
gpu = _Loud("gpu")
bridge = _Loud("bridge")
