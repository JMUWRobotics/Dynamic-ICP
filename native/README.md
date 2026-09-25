# C++ registration backend

The `_native` module compiles the pinned
`Open3D/cpp/open3d/pipelines/registration/DopplerPt2PlaneICP.cpp` and a small
NumPy binding. It links against an existing compatible Open3D C++ library for
point clouds, KD-tree correspondence search, and matrix utilities. The full
Open3D visualization/ML/Python package is not rebuilt or replaced.

## Build

Requirements: CMake ≥ 3.20, a C++14 compiler with OpenMP, Python development
headers, pybind11, and a **Doppler-capable Open3D 0.15.2 C++ installation**.
Its `PointCloud` and `RegistrationResult` layouts must match the pinned fork.
A stock Open3D pip wheel alone does not supply this C++ development installation.

The library, headers, and compiler ABI must be consistent. In particular, the
libstdc++ ABI setting must match the pinned fork. Do not link this extension
against an arbitrary Open3D release with a different class layout.

```bash
python -m pip install -e '.[native,test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/build_native.sh
```

For a different compatible installation:

```bash
bash scripts/build_native.sh -DOpen3D_DIR=/path/to/lib/cmake/Open3D
```

`PYTHON_EXECUTABLE` selects the interpreter; `BUILD_JOBS` defaults to 2.
The script initializes the pinned Open3D submodule if needed, verifies that its
Dynamic-ICP solver is present, configures Release mode, builds into
`build/native`, and runs the compiled-backend tests. The binary is placed in
`dynamic_icp/` and is ignored by Git.

## Use

Select `--backend cpp`, or a configuration with `"backend": "cpp"`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m dynamic_icp \
  --scans dataset/hercules_dynamic/scans --layout npz \
  --filename-time-unit ns --backend cpp \
  --gt dataset/hercules_dynamic/ground_truth.tum \
  --output outputs/hercules_demo_cpp
```

The NumPy backend remains available with `--backend numpy`. Both share scan
loading, voxelization, ego velocity, dynamic prediction, and target normals.
The C++ solver uses the same rotation/additive-translation update convention.
The optional displacement prior is implemented in both backends.

The module records the compiled solver SHA-256 in `run.json`. Loading it checks
the current solver source and raises if recompilation is needed. It does not
silently fall back to the previously installed Python Open3D binary.

## Validation

`tests/test_native.py` checks compiled updates against the NumPy equations at
nonidentity rotation and translation, using several Doppler weights and the
additional displacement prior. It also checks known-transform registration,
iteration limits, degeneracy, missing correspondences, and low-speed prior
suppression. Full pipeline tests are run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

The C++ backend currently supports fixed correspondence distance only. Use
`correspondence_range_scale=0`; other values raise a clear error. Rank-deficient
normal equations fail explicitly rather than returning an identity update.
