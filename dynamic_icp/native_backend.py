"""Load the local C++ extension and reject a stale solver build."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def load_backend():
    try:
        from . import _native
    except ImportError as error:
        raise RuntimeError("C++ backend is unavailable. Run bash scripts/build_native.sh; "
                           "a compatible Open3D C++ library and pybind11 are required.") from error
    source = (Path(__file__).resolve().parent.parent / "Open3D/cpp/open3d/pipelines/registration"
              / "DopplerPt2PlaneICP.cpp")
    if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() != _native.solver_sha256:
        raise RuntimeError("The C++ solver changed after compilation. Run bash scripts/build_native.sh.")
    return _native
