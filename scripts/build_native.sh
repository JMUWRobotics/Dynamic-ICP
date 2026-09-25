#!/usr/bin/env bash
# Rebuild only the Doppler registration code and NumPy bridge. The Open3D
# PointCloud/KD-tree support library must already be available to CMake.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_EXECUTABLE:-python3}"
build_jobs="${BUILD_JOBS:-2}"
solver_path="cpp/open3d/pipelines/registration/DopplerPt2PlaneICP.cpp"

if [[ ! -f "$repo_dir/Open3D/$solver_path" ]]; then
  git -C "$repo_dir" submodule update --init Open3D
fi
if [[ ! -f "$repo_dir/Open3D/$solver_path" ]]; then
  echo 'The pinned Open3D submodule does not contain the Dynamic-ICP solver.' >&2
  echo 'Run: git submodule update --init Open3D' >&2
  exit 1
fi

cmake -S "$repo_dir/native" -B "$repo_dir/build/native" \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v "$python_bin")" "$@"
cmake --build "$repo_dir/build/native" -j "$build_jobs"
cd "$repo_dir"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$python_bin" -m pytest -q tests/test_native.py
