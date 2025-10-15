#!/usr/bin/env bash

# © 2025 Hiroyuki Sakai

build_dir="${1:-build-asm+mcl}"
flags="${2:-}"

# Building OpenCV
mkdir -p "$build_dir"
cd "$build_dir/"
mkdir opencv
cd opencv/

cuda_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2 p' | tr -d -c 0-9)

cmake \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_STANDARD=17 \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -march=native ${flags}" \
-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native ${flags}" \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch \
-DWITH_CUDA=ON \
-DWITH_CUBLAS=ON \
-DOPENCV_EXTRA_MODULES_PATH=../../src/ext/opencv_contrib/modules \
-DBUILD_LIST=cudaarithm,cudev,cudaimgproc,highgui,ximgproc \
../../src/ext/opencv

make -j 16

cd ../

# Building pbrt-v3
mkdir pbrt-v3
cd pbrt-v3/

cmake \
-DOpenCV_BUILD_DIR_PREFIX="${build_dir}" \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
-DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -march=native ${flags}" \
-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native ${flags}" \
../../

make -j 16

cd ../../
