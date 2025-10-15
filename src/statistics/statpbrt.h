// © 2025 Hiroyuki Sakai

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_STATISTICS_STATPBRT_H
#define PBRT_STATISTICS_STATPBRT_H

#ifndef DEBUG
#define DEBUG 0
#endif
// To change Box–Cox transformation from .25 to .5 for StatMC
#ifndef STATMC
#define STATMC 0
#endif

#ifndef DISABLE_DENOISE_VARS
#define DISABLE_DENOISE_VARS 0
#endif

#ifndef FULL_MULTI_TRANSFORM
#define FULL_MULTI_TRANSFORM 0 // OVERRIDES STATMC!
#endif

#ifndef FILM_MULTI_TRANSFORM_MODE
#define FILM_MULTI_TRANSFORM_MODE 2
#endif

// Helper macros to stringify values
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Macro to print macro name + value
#define SHOW_MACRO(m) std::cout << #m << " = " << TOSTRING(m) << std::endl;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "geometry.h"

namespace pbrt {

using cv::Vec3f;
using cv::Mat;
using cv::Mat_;
using cv::Mat1f;
using cv::Mat1i;
using cv::Mat3f;
using cv::cuda::GpuMat;

using Vec3  = cv::Vec<Float, 3>;
using Vec4  = cv::Vec<Float, 4>;
using Mat1  = Mat_<Float>;
using Mat3  = Mat_<Vec3>;

class Estimator;

}  // namespace pbrt

#endif  // PBRT_STATISTICS_STATPBRT_H
