#!/usr/bin/env bash

# © 2025 Hiroyuki Sakai

# If adaptive sampling is not used, CUDA runtime can be reduced by adding -DDISABLE_DENOISE_VARS=1
./scripts/_build.sh build-asm+m "-DCASCADE_MODE=0 -DUSE_LEVC=0"
