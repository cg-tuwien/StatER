#!/usr/bin/env bash

# © 2025 Hiroyuki Sakai

mkdir -p out

if [[ -z "$1" ]]; then
    echo "Error: Missing scene."
    echo "Usage: $0 <scene-name> <scene-variant> <spp> <build-variant>"
    exit 1
fi

scene="$1"

if [[ -n "$2" ]]; then
    scene_variant="-$2"
else
    scene_variant=""
fi

message=0
if [[ -n "$3" ]]; then
    echo "To reproduce the results from the publication, render until $3 (total) SPP."
    message=1
fi

if [[ -n "$4" ]]; then
    build_variant="-$4"
else
    build_variant="-asm+mcl"
fi


if [ "$2" = "as" ]; then
    echo "With adaptive sampling enabled, your results may differ slightly from those"
    echo "reported in the publication. Small differences in numerical precision, which can"
    echo "easily happen, can cause the random number streams to diverge, leading to minor"
    echo "variations across the image."
    message=1
fi

if [ "$message" -eq 1 ]; then
    echo "================================================================================"
fi

# Activate rendering and denoising configuration
cp scenes/render-denoise"${scene_variant}".pbrt scenes/_active.pbrt

# Build path
build_dir="build${build_variant}"

# Check if the build path exists
if [[ ! -d "$build_dir" ]]; then
  echo "Error: Directory '$build_dir' does not exist. You need to compile non-default build configurations, for which you can use the scripts provided in scripts/alternate-builds/." >&2
  exit 1
fi

cd out/
../build"${build_variant}"/pbrt-v3/pbrt --writeimages ../scenes/"${scene}"/scene-stat.pbrt
cd ../
