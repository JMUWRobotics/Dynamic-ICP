FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libeigen3-dev \
        libgomp1 \
        libglu1-mesa-dev \
        libomp-dev \
        libtbb-dev \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
        xorg-dev \
    && rm -rf /var/lib/apt/lists/*

ARG BUILD_JOBS=2

WORKDIR /opt/dynamic-icp

# The Open3D submodule must be initialized before `docker build`.
COPY Open3D ./Open3D

RUN test -f Open3D/cpp/open3d/pipelines/registration/DopplerPt2PlaneICP.cpp \
    && cmake -S Open3D -B /tmp/open3d-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/open3d \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_PYTHON_MODULE=OFF \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_ISPC_MODULE=OFF \
        -DBUILD_GUI=OFF \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=OFF \
        -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_TBB=OFF \
    && cmake --build /tmp/open3d-build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/open3d-build \
    && rm -rf /tmp/open3d-build

COPY dynamic_icp ./dynamic_icp
COPY native ./native
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY dataset ./dataset
COPY pyproject.toml LICENSE README.md ./

ENV LD_LIBRARY_PATH=/opt/open3d/lib:/opt/open3d/lib64
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -e '.[native,test,viewer]' \
    && cmake -S native -B build/native -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpen3D_DIR=/opt/open3d/lib/cmake/Open3D \
        -DPython3_EXECUTABLE=/usr/bin/python3 \
    && cmake --build build/native --parallel "${BUILD_JOBS}" \
    && python3 -m pytest -q

ENTRYPOINT ["dynamic-icp"]
CMD ["--help"]
