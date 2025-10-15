// © 2025 Hiroyuki Sakai

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_STATISTICS_STATPATHINTEGRATOR_H
#define PBRT_STATISTICS_STATPATHINTEGRATOR_H

#include "pbrt.h"
#include "camera.h"
#include "integrator.h"
#include "lightdistrib.h"
#include "statistics/statpbrt.h"
#include "statistics/estimator.h"

namespace pbrt {

// We hardcode the vector indices here so that no expensive lookups must be performed during rendering.
enum BufferIndex {
    MaterialID = 0, // Float buffer vector indices
    Depth      = 1,
    Normal     = 0, // RGB buffer vector indices
    Albedo     = 1
};

enum StatTypeIndex {
    Radiance        = 0,
    MISBSDFWinRate  = 1,
    MISLightWinRate = 2,
    StatMaterialID  = 3,
    StatDepth       = 4,
    StatNormal      = 5,
    StatAlbedo      = 6,
    ItRadiance      = 7
};
constexpr unsigned char nStatTypeIndices = 8;

struct MISTally {
    unsigned char bsdf  = 0;
    unsigned char light = 0;
};

struct MISWinRate {
    Float bsdf  = 0;
    Float light = 0;
};

class StatPathIntegrator : public SamplerIntegrator {
    protected:
        struct Features {
            std::vector<Float> floats;
            std::vector<Spectrum> spectrums;
        };

    public:
        StatPathIntegrator(
            const unsigned int maxDepth,
            std::shared_ptr<const Camera> camera,
            std::shared_ptr<Sampler> sampler,
            const Bounds2i &pixelBounds,
            const uint64_t nIterations,
            const bool expIterations,
            const bool outputExpIterations,
            const bool enableMultiChannelStats,
            const bool enableAS,
            const bool denoiseFilm,
            const bool enableACRR,
            const bool enableSMIS,
            const bool calcProDenStats,
            const bool calcItStats,
            const unsigned char filterRadius,
            const float filterSD,
            const float meanZValue,
            const float varZValue,
            GBufferConfigs floatGBufConfigs,
            GBufferConfigs rgbGBufConfigs,
            StatTypeConfigs statTypeConfigs,
#if FULL_MULTI_TRANSFORM == 1
            std::vector<Float> lambdas,
#endif
            const Float rrThreshold,
            const std::string &lightSampleStrategy,
            const std::string &outputRegex
        ) : SamplerIntegrator(camera, sampler, pixelBounds),
            floatGBufConfigs(floatGBufConfigs),
            rgbGBufConfigs(rgbGBufConfigs),
            statTypeConfigs(statTypeConfigs),
            bufReg(camera->film->buffer),
            estimator(
                camera->film->buffer,
                this->statTypeConfigs,
                filterRadius,
                filterSD,
                meanZValue,
                varZValue,
                enableAS,
                denoiseFilm,
#if FULL_MULTI_TRANSFORM == 1
                lambdas.size(),
#endif
                bufReg
            ),
            maxDepth(maxDepth),
            nIterations(nIterations),
            expIterations(expIterations),
            outputExpIterations(outputExpIterations),
            enableMultiChannelStats(enableMultiChannelStats),
            enableAS(enableAS),
            enableACRR(enableACRR),
            enableSMIS(enableSMIS),
            calcProDenStats(calcProDenStats),
            calcItStats(calcItStats),
#if FULL_MULTI_TRANSFORM == 1
            lambdas(lambdas),
#endif
            rrThreshold(rrThreshold),
            lightSampleStrategy(lightSampleStrategy),
            outputRegex(outputRegex)
        {
            std::cout << "==== pbrt-v3 Macro Configuration Start ====" << std::endl;
            SHOW_MACRO(DEBUG);
            SHOW_MACRO(STATMC);
            SHOW_MACRO(DISABLE_DENOISE_VARS);
            SHOW_MACRO(FULL_MULTI_TRANSFORM);
            std::cout << "==== pbrt-v3 Macro Configuration End ====" << std::endl;
            estimator.PrintMacroConfiguration();
        };
        void Preprocess(const Scene &scene, Sampler &sampler);
        // Templated function pointer typedef
        template <typename T>
#if FULL_MULTI_TRANSFORM == 1
        using AddSampleFn = void (StatTile<T>::*)(const Point2i p, const Point2f inPxP, const T sample, const Float lambda);
#else
        using AddSampleFn = void (StatTile<T>::*)(const Point2i p, const Point2f inPxP, const T sample);
#endif
        template <typename T>
        inline AddSampleFn<T> GetAddSampleFn(const StatTypeConfig &cfg);
        template <typename T>
        void Render(const Scene &scene);
        void Render(const Scene &scene);
        inline void ReadFile(const std::string &filename, Buffer &buf);
        template <typename T>
        void Denoise(const Scene &scene);
        void Denoise(const Scene &scene);
        virtual Spectrum Li(
            const RayDifferential &ray,
            const Scene &scene,
            Sampler &sampler,
            MemoryArena &arena,
            Features &features,
            std::vector<Float> &avgLs,
            std::vector<MISWinRate> &misWinRates,
            std::vector<Spectrum> &L,
            std::vector<MISTally> &misTallies,
            unsigned int it
        ) const;

    protected:
        const GBufferConfigs floatGBufConfigs;
        const GBufferConfigs rgbGBufConfigs;
        const StatTypeConfigs statTypeConfigs;

        BufferRegistry bufReg; // Must be initialized before estimator; therefore, must be defined here, above it.
        Estimator estimator;

    private:
        inline Float GetY(const Float val) const {return val;}
        inline Float GetY(const Vec3 rgb) const {
            Spectrum spectrum = Spectrum::FromRGB(&rgb[0]);
            return spectrum.y();
        }
        template <typename T>
        inline T GetStatSample(const Spectrum &spectrum) const;

        const int maxDepth;
        const Float rrThreshold;
        const std::string lightSampleStrategy;
        std::unique_ptr<LightDistribution> lightDistribution;

        const uint64_t nIterations;
        const bool expIterations;
        const bool outputExpIterations;
        const bool enableMultiChannelStats;
        const bool enableAS;
        const bool enableACRR;
        const bool enableSMIS;
        const bool calcProDenStats;
        const bool calcItStats;

#if FULL_MULTI_TRANSFORM == 1
        std::vector<Float> lambdas;
#endif

        const std::string outputRegex;
};

template <>
inline Float StatPathIntegrator::GetStatSample(const Spectrum &spectrum) const {return spectrum.y();}
template <>
inline Vec3 StatPathIntegrator::GetStatSample(const Spectrum &spectrum) const {
    Float rgb[3];
    spectrum.ToRGB(rgb);
    return Vec3(rgb);
}

StatPathIntegrator *CreateStatPathIntegrator(
    const ParamSet &params,
    const ParamSet &extraParams,
    std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera
);

}  // namespace pbrt

#endif  // PBRT_STATISTICS_STATPATHINTEGRATOR_H
