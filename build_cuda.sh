#!/bin/bash

cd llama.cpp || { echo "check if submodule llama.cpp is present"; exit 1; }
mkdir -p build || { echo "failed to create build dir"; exit 1; }
cd build 

cmake .. \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DLLAMA_STATIC=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_FLASH_ATTN=ON

cmake --build . --config Release -j "$(nproc)" || { echo "failed to compile"; exit 1; }