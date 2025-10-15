// © 2025 Hiroyuki Sakai

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_STATISTICS_ESTIMATOR_H
#define PBRT_STATISTICS_ESTIMATOR_H

#include <cmath>
#include <unordered_set>
#include <cuda_runtime.h>

#include "pbrt.h"
#include "core/film.h"
#include "statistics/statpbrt.h"
#include "statistics/buffer.h"

namespace pbrt {

// Needed for moment calculation below (definition in .cpp)
inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

inline Vec3 operator/(const Vec3 &v1, const uint64_t &i) {
    return Vec3(v1[0] / i, v1[1] / i, v1[2] / i);
}

inline Float sqrt(const Float &val) {
    return std::sqrt(val);
}
inline Vec3 sqrt(const Vec3 &vec) {
    return Vec3(
        std::sqrt(vec[0]),
        std::sqrt(vec[1]),
        std::sqrt(vec[2])
    );
}


#if FULL_MULTI_TRANSFORM == 1
inline Float boxCox(const Float &val, const Float &l) {
    if (l == 1.f) {
        return val;
    }
#else
inline Float boxCox(const Float &val) {
  #if STATMC == 1
    const float l = .5f;
  #else
    const float l = .25f;
  #endif
#endif
    return (std::pow(val, l) - 1.f) / l;
}
#if FULL_MULTI_TRANSFORM == 1
inline Vec3 boxCox(const Vec3 &vec, const Float &l) {
    if (l == 1.f) {
        return vec;
    }
#else
inline Vec3 boxCox(const Vec3 &vec) {
  #if STATMC == 1
    const float l = .5f;
  #else
    const float l = .25f;
  #endif
#endif
    return Vec3(
        (std::pow(vec[0], l) - 1.f) / l,
        (std::pow(vec[1], l) - 1.f) / l,
        (std::pow(vec[2], l) - 1.f) / l
    );
}


enum CUDAGroupIndex {
    DenoiseMeanGrp = 0,
    CalcTargetSPPGrp = 1,
    CalcMeanVarGrp = 2
};
static constexpr unsigned char nCUDAGroupIndices = 3;

struct GBufferConfig {
    GBufferConfig(const std::string name = "") : name(name) {};
    std::string name;
    unsigned char index;
    bool enable = false;
};

struct GBufferConfigs {
    GBufferConfigs(
        std::vector<GBufferConfig> cfgs
    ) : configs(std::move(cfgs))
    {}
    GBufferConfig &operator[](size_t i) {
        return configs[i];
    };
    const GBufferConfig &operator[](size_t i) const {
        return configs[i];
    };

    unsigned char nEnabled = 0;
    std::vector<GBufferConfig> configs;
};


struct StatTypeConfig {
    bool enable = false;
    unsigned char type;
    unsigned char index;
    unsigned char nBounces = 0;
    unsigned char bounceStart = 0;
    unsigned char bounceEnd = 0;
    unsigned char nChannels = 1;
    bool transform = false;
    unsigned char maxMoment = 1;

    bool gBuffer = false;
    bool enableForVarFilter = false;
    bool enableForMeanFilter = false;
    Float varFilterSD;
    Float meanFilterSD;

    std::vector<unsigned char> cudaGroups = {};
};

struct StatTypeConfigs {
    StatTypeConfig &operator[](size_t i) {
        return configs[i];
    };
    const StatTypeConfig &operator[](size_t i) const {
        return configs[i];
    };

    unsigned char nEnabled = 0;
    std::vector<StatTypeConfig> configs;
};


template <typename T>
class Tile {
    public:
        Tile(const Bounds2i &pixelBounds)
          : pixelBounds(pixelBounds)
        {
            pixels = std::vector<T>(std::max(0, pixelBounds.Area()));
        }
        T &GetPixel(const Point2i &p) {
            // CHECK(InsideExclusive(p, pixelBounds)); // Getting rid of these checks significantly increases performance (note that CHECK is executed in release mode)
            return pixels[(p.y - pixelBounds.pMin.y) * (pixelBounds.pMax.x - pixelBounds.pMin.x) +
                          (p.x - pixelBounds.pMin.x)];
        }
        const T &GetPixel(const Point2i &p) const {
            // CHECK(InsideExclusive(p, pixelBounds)); // Getting rid of these checks significantly increases performance (note that CHECK is executed in release mode)
            return pixels[(p.y - pixelBounds.pMin.y) * (pixelBounds.pMax.x - pixelBounds.pMin.x) +
                          (p.x - pixelBounds.pMin.x)];
        }
        Bounds2i GetPixelBounds() const { return pixelBounds; }

    protected:
        const Bounds2i pixelBounds;
        std::vector<T> pixels;
};


template <typename T>
struct alignas(64) StatTilePixelStats {
    T mean{}, m2{}, m3{}, m4{};
    T c2X{}, c2Y{}, m1{};
};

template <typename T>
struct alignas(64) StatTilePixel {
    uint64_t n{};
    StatTilePixelStats<T> t0{};
    StatTilePixelStats<T> t1{};
};

// Safety checks
static_assert(sizeof(StatTilePixel<Float>) % 64 == 0, "StatTilePixel<Float> should be multiple of 64 bytes");
static_assert(sizeof(StatTilePixel<Vec3>)  % 64 == 0, "StatTilePixel<Vec3> should be multiple of 64 bytes");

template <typename T>
class StatTile : public Tile<StatTilePixel<T>> {
    public:
        StatTile(const Bounds2i &pixelBounds) : Tile<StatTilePixel<T>>(pixelBounds), filterTable(nullptr), filterTableSize(0)
        {}
        StatTile(
            const Bounds2i &pixelBounds, const Vector2f &filterRadius,
            const Float *filterTable, int filterTableSize
        ) : Tile<StatTilePixel<T>>(pixelBounds),
            pixelBounds(pixelBounds),
            filterRadius(filterRadius),
            invFilterRadius(1 / filterRadius.x, 1 / filterRadius.y),
            filterTable(filterTable),
            filterTableSize(filterTableSize)
        {}
#if FULL_MULTI_TRANSFORM == 1
        void AddSampleM1          (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddSample         (p, inPxP, sample,         &StatTile<T>::AddStatSampleM1);}
        void AddTransformSampleM1 (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddTransformSample(p, inPxP, sample, lambda, &StatTile<T>::AddStatSampleM1);}
        void AddSampleM2          (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddSample         (p, inPxP, sample,         &StatTile<T>::AddStatSampleM2);}
        void AddTransformSampleM2 (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddTransformSample(p, inPxP, sample, lambda, &StatTile<T>::AddStatSampleM2);}
        void AddSampleM3          (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddSample         (p, inPxP, sample,         &StatTile<T>::AddStatSampleM3);}
        void AddTransformSampleM3 (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddTransformSample(p, inPxP, sample, lambda, &StatTile<T>::AddStatSampleM3);}
        void AddSampleM4          (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddSample         (p, inPxP, sample,         &StatTile<T>::AddStatSampleM4);}
        void AddTransformSampleM4 (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddTransformSample(p, inPxP, sample, lambda, &StatTile<T>::AddStatSampleM4);}
        void AddSampleM4C         (const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddSample         (p, inPxP, sample,         &StatTile<T>::AddStatSampleM4C);}
        void AddTransformSampleM4C(const Point2i p, const Point2f inPxP, const T sample, const Float lambda) {AddTransformSample(p, inPxP, sample, lambda, &StatTile<T>::AddStatSampleM4C);}
#else
        void AddSampleM1          (const Point2i p, const Point2f inPxP, const T sample) {AddSample         (p, inPxP, sample, &StatTile<T>::AddStatSampleM1);}
        void AddTransformSampleM1 (const Point2i p, const Point2f inPxP, const T sample) {AddTransformSample(p, inPxP, sample, &StatTile<T>::AddStatSampleM1);}
        void AddSampleM2          (const Point2i p, const Point2f inPxP, const T sample) {AddSample         (p, inPxP, sample, &StatTile<T>::AddStatSampleM2);}
        void AddTransformSampleM2 (const Point2i p, const Point2f inPxP, const T sample) {AddTransformSample(p, inPxP, sample, &StatTile<T>::AddStatSampleM2);}
        void AddSampleM3          (const Point2i p, const Point2f inPxP, const T sample) {AddSample         (p, inPxP, sample, &StatTile<T>::AddStatSampleM3);}
        void AddTransformSampleM3 (const Point2i p, const Point2f inPxP, const T sample) {AddTransformSample(p, inPxP, sample, &StatTile<T>::AddStatSampleM3);}
        void AddSampleM4          (const Point2i p, const Point2f inPxP, const T sample) {AddSample         (p, inPxP, sample, &StatTile<T>::AddStatSampleM4);}
        void AddTransformSampleM4 (const Point2i p, const Point2f inPxP, const T sample) {AddTransformSample(p, inPxP, sample, &StatTile<T>::AddStatSampleM4);}
        void AddSampleM4C         (const Point2i p, const Point2f inPxP, const T sample) {AddSample         (p, inPxP, sample, &StatTile<T>::AddStatSampleM4C);}
        void AddTransformSampleM4C(const Point2i p, const Point2f inPxP, const T sample) {AddTransformSample(p, inPxP, sample, &StatTile<T>::AddStatSampleM4C);}
#endif

    private:
        inline void AddStatSampleM1(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample) {
            T &mean = stats.mean;

            // Use Meng's algorithm (https://arxiv.org/abs/1510.04923)
            const T d  = sample - mean;
            const T dN = d / n;

            mean += dN;
        }
        inline void AddStatSampleM2(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample) {
            T &mean = stats.mean;
            T &m2   = stats.m2;

            // Use Meng's algorithm (https://arxiv.org/abs/1510.04923)
            const T d   = sample - mean;
            const T dN  = d / n;

            mean += dN;
            m2   += d * (d - dN);
        }
        inline void AddStatSampleM3(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample) {
            T &mean = stats.mean;
            T &m2   = stats.m2;
            T &m3   = stats.m3;

            // Use Meng's algorithm (https://arxiv.org/abs/1510.04923)
            const T d   = sample - mean;
            const T d2  = d * d;
            const T dN  = d / n;
            const T dN2 = dN * dN;

            mean += dN;
            m2   +=                    d * (d      - dN      );
            m3   += - 3.f * dN  * m2 + d * (d2     - dN2     );
        }
        inline void AddStatSampleM4(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample) {
            T &mean = stats.mean;
            T &m2   = stats.m2;
            T &m3   = stats.m3;
            T &m4   = stats.m4;

            // Use Meng's algorithm (https://arxiv.org/abs/1510.04923)
            const T d   = sample - mean;
            const T d2  = d * d;
            const T dN  = d / n;
            const T dN2 = dN * dN;

            mean += dN;
            m2   +=                                    d * (d      - dN      );
            m3   +=                 - 3.f * dN  * m2 + d * (d2     - dN2     );
            m4   += - 4.f * dN * m3 - 6.f * dN2 * m2 + d * (d * d2 - dN * dN2);
        }
        inline void AddStatSampleM4C(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample) {
            T &mean = stats.mean;
            T &m2   = stats.m2;
            T &m3   = stats.m3;
            T &m4   = stats.m4;
            T &c2X  = stats.c2X;
            T &c2Y  = stats.c2Y;
            T &m1   = stats.m1;

            // Use Meng's algorithm (https://arxiv.org/abs/1510.04923)
            const T d   = sample - mean;
            const T d2  = d * d;
            const T dN  = d / n;
            const T dN2 = dN * dN;

            mean += dN;
            m2   +=                                    d * (d      - dN      );
            m3   +=                 - 3.f * dN  * m2 + d * (d2     - dN2     );
            m4   += - 4.f * dN * m3 - 6.f * dN2 * m2 + d * (d * d2 - dN * dN2);


            const float xD = (inPxP.x - 0.5f);
            const float yD = (inPxP.y - 0.5f);
            c2X += (sample - mean) * xD; // sample - mean instead of d since we use population mean for X.
            c2Y += (sample - mean) * yD; // sample - mean instead of d since we use population mean for Y.

            m1 += sqrt(d * (d - dN));
        }
        inline void AddSample(const Point2i p, const Point2f inPxP, const T sample, void (StatTile::*fn)(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample)) {
            StatTilePixel<T> &pixel = this->GetPixel(p);
            pixel.n++;
            (this->*fn)(pixel.n, pixel.t0, inPxP, sample);
        }
#if FULL_MULTI_TRANSFORM == 1
        inline void AddTransformSample(const Point2i p, const Point2f inPxP, const T sample, const Float lambda, void (StatTile::*fn)(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample)) {
#else
        inline void AddTransformSample(const Point2i p, const Point2f inPxP, const T sample, void (StatTile::*fn)(const uint64_t n, StatTilePixelStats<T> &stats, const Point2f inPxP, const T sample)) {
#endif
            StatTilePixel<T> &pixel = this->GetPixel(p);
            pixel.n++;
            (this->*fn)(pixel.n, pixel.t0, inPxP,        sample);
#if FULL_MULTI_TRANSFORM == 1
            (this->*fn)(pixel.n, pixel.t1, inPxP, boxCox(sample, lambda));
#else
            (this->*fn)(pixel.n, pixel.t1, inPxP, boxCox(sample));
#endif
        }

        const Bounds2i pixelBounds;
        const Vector2f filterRadius, invFilterRadius;
        const Float *filterTable;
        const int filterTableSize;
};


class Estimator {
    public:
        Estimator(
            const Buffer &filmBuffer,
            const StatTypeConfigs &statTypeConfigs,
            const unsigned char filterRadius,
            const float filterSD,
            const float meanZValue,
            const float varZValue,
            const bool enableAS,
            const bool denoiseFilm,
#if FULL_MULTI_TRANSFORM == 1
            const unsigned char nLambdas,
#endif
            BufferRegistry &reg
        ) : filmBuffer(filmBuffer),
            filmFilteredBuffer("filmD", Mat3(filmBuffer.mat.rows, filmBuffer.mat.cols)),
            width(filmBuffer.mat.cols),
            height(filmBuffer.mat.rows),
            filterRadius(filterRadius),
            filterDSFactor(-.5f / (filterSD * filterSD)),
            meanZValue(meanZValue),
            varZValue(varZValue),
            enableAS(enableAS),
            denoiseFilm(denoiseFilm),
#if FULL_MULTI_TRANSFORM == 1
            nLambdas(nLambdas),
#endif
            targetSPPBuffer("targetspp", Mat_<int>(filmBuffer.mat.rows, filmBuffer.mat.cols)),
            reg(reg)
        {
            floatBufferCounts = std::vector<unsigned char>(nCUDAGroupIndices, 0);
            rgbBufferCounts = std::vector<unsigned char>(nCUDAGroupIndices, 0);

            std::copy_if(statTypeConfigs.configs.begin(), statTypeConfigs.configs.end(), std::back_inserter(this->statTypeConfigs.configs), [](StatTypeConfig cfg){ return cfg.enable; }); // This statTypeConfigs only holds statTypeConfigs for enabled buffers!
            this->statTypeConfigs.nEnabled = this->statTypeConfigs.configs.size();

            reg.Register(filmFilteredBuffer);
            reg.Register(targetSPPBuffer);

            if (enableAS) {
                downloadBuffers.insert(&targetSPPBuffer);
            }

            if (denoiseFilm) {
                uploadBuffers.insert(&this->filmBuffer);
                downloadBuffers.insert(&this->filmFilteredBuffer);
            }

            cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED)); // pinned memory
            cv::cuda::stat_denoiser::setup();

            AllocateBuffers();
        }
        void RegisterVarGBuffer(Buffer &b, const Float filterSD);
        void RegisterMeanGBuffer(Buffer &b, const Float filterSD);
        void AllocateBuffers();

        template <typename T>
        std::vector<StatTile<T>> GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd) const;
        template <typename T>
        std::vector<std::vector<StatTile<T>>> GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd, const unsigned char n) const;
        template <typename T>
        inline void MergeTile(const StatTile<T> &tile, const unsigned char statTypeIndex, const unsigned char bounceIndex) const;
        template <typename T>
        void MergeTiles(const std::vector<StatTile<T>> &tiles, const StatTypeConfig &cfg) const;
        template <typename T>
        void MergeTiles(const std::vector<std::vector<StatTile<T>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const;
        template <typename T>
        inline void MergeTransformTile(const StatTile<T> &tile, const unsigned char statTypeIndex, const unsigned char bounceIndex) const;
        template <typename T>
        void MergeTransformTiles(const std::vector<StatTile<T>> &tiles, const StatTypeConfig &cfg) const;
        template <typename T>
        void MergeTransformTiles(const std::vector<std::vector<StatTile<T>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const;

        void Upload();
        void Download();
        void DenoiseVars();
        void DenoiseMeans();
        void CalcMeanVars();
        void CalcTargetSPPs(const uint64_t spp);
        void Synchronize();
        void PrintMacroConfiguration();

        BufferRegistry &reg;

        const unsigned short width;
        const unsigned short height;

        const bool enableAS;
        const bool denoiseFilm;

        const unsigned char filterRadius;
        const float filterDSFactor;
        const float meanZValue;
        const float varZValue;

        bool runCUDA = false;

        StatTypeConfigs statTypeConfigs;

        std::vector<unsigned char> floatBufferCounts;
        std::vector<unsigned char> rgbBufferCounts;

        Buffer filmBuffer;
        Buffer filmFilteredBuffer;

        std::vector<Buffer> varGBuffers;
        std::vector<Float>  varGBufferDRFactors;
        GpuMat varGBufferGPUPtrs;
        GpuMat varGBufferChannelCountsGPUMat;
        GpuMat varGBufferDRFactorsGPUMat;

        std::vector<Buffer> meanGBuffers;
        std::vector<Float>  meanGBufferDRFactors;
        GpuMat meanGBufferGPUPtrs;
        GpuMat meanGBufferChannelCountsGPUMat;
        GpuMat meanGBufferDRFactorsGPUMat;

        Buffer targetSPPBuffer;

        std::unordered_set<Buffer *> uploadBuffers;
        std::unordered_set<Buffer *> downloadBuffers;

        cv::cuda::Stream stream;

#define BUF_LIST(X) \
X(mean,  Float, Vec3) \
X(m2,    Float, Vec3) \
X(m3,    Float, Vec3) \
X(m4,    Float, Vec3) \
X(c2X,   Float, Vec3)  /* covariance with x */ \
X(c2Y,   Float, Vec3)  /* covariance with y */ \
X(m1,    Float, Vec3)  /* mean-absolute deviation */ \
X(jb,    Float, Float) /* Jarque–Bera score */ \
X(lnS2,  Float, Vec3)  /* Bonett ln(s2) */ \
X(se2,   Float, Vec3)  /* Bonett se^2 */ \
X(m2C,   Float, Vec3)  /* LEV-corrected m2 */ \
X(varD,  Float, Vec3)  /* denoised variance */ \
X(varCD, Float, Vec3)  /* denoised LEV-corrected variance */ \
X(discr, Float, Vec3)  /* Welch discriminator */ \

#define FILM_BUF_LIST(X) \
BUF_LIST(X) \
X(n,       int, int) \
X(meanVar, Float, Vec3) /* variance of mean */ \
X(meanD,   Float, Vec3) /* denoised mean */ \

#define BUF_DECL(name, type1, type3) std::vector<std::vector<Buffer>> name;
#define BUF_REF(name, type1, type3) &name,
        struct Buffers {
            BUF_LIST(BUF_DECL)
            std::vector<std::vector<std::vector<Buffer>> *> all() {
                return std::vector<std::vector<std::vector<Buffer>> *>{BUF_LIST(BUF_REF)};
            }
        };
        struct FilmBuffers {
            FILM_BUF_LIST(BUF_DECL)
            std::vector<std::vector<std::vector<Buffer>> *> all() {
                return std::vector<std::vector<std::vector<Buffer>> *>{
                    FILM_BUF_LIST(BUF_REF)
                };
            }
        };
#undef BUF_DECL
#undef BUF_REF

#define PTR_DECL(name, type1, type3) std::vector<GpuMat> name;
        struct GPUPtrs {
            BUF_LIST(PTR_DECL)
        };
        struct FilmGPUPtrs : GPUPtrs {
            FILM_BUF_LIST(PTR_DECL)
        };
#undef PTR_DECL

#if FULL_MULTI_TRANSFORM == 1
        const unsigned char nLambdas;
#endif

        FilmBuffers t0Bufs;
        Buffers     t1Bufs;

        FilmGPUPtrs fT0Ptrs;
        GPUPtrs     fT1Ptrs;
        FilmGPUPtrs rgbT0Ptrs;
        GPUPtrs     rgbT1Ptrs;
};

}  // namespace pbrt

#endif  // PBRT_STATISTICS_ESTIMATOR_H
