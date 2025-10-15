// © 2025 Hiroyuki Sakai

#include "statistics/estimator.h"
#include <opencv2/cudaimgproc.hpp>
#include "spectrum.h"
#include "statistics/statpath.h"

namespace pbrt {

void Estimator::RegisterVarGBuffer(Buffer &b, const Float filterSD) {
    varGBuffers.emplace_back(b);
    varGBufferDRFactors.emplace_back(-.5f / (filterSD * filterSD));
}

void Estimator::RegisterMeanGBuffer(Buffer &b, const Float filterSD) {
    meanGBuffers.emplace_back(b);
    meanGBufferDRFactors.emplace_back(-.5f / (filterSD * filterSD));
}


#define ALLOC_BUF(buffers, suffix, mat) { \
    Buffer &b = buffers[cfg.index].emplace_back("t" + std::to_string(cfg.type) + "-b" + jStr + suffix, mat); \
    reg.Register(b); \
} \

#define PREPARE_STAT_PTRS(buffers, fPtrs, rgbPtrs) { \
    std::vector<Mat> fPtrsCPUs(nCUDAGroupIndices); \
    std::vector<PtrStepSzb *> fPtrsCPUPtrs(nCUDAGroupIndices); \
    std::vector<Mat> rgbPtrsCPUs(nCUDAGroupIndices); \
    std::vector<PtrStepSzb *> rgbPtrsCPUPtrs(nCUDAGroupIndices); \
    for (unsigned char i = 0; i < nCUDAGroupIndices; i++) { \
        fPtrsCPUs[i] = Mat(1, floatBufferCounts[i], CV_8UC(sizeof(PtrStepSzb))); \
        fPtrsCPUPtrs[i] = fPtrsCPUs[i].ptr<PtrStepSzb>(); \
        rgbPtrsCPUs[i] = Mat(1, rgbBufferCounts[i], CV_8UC(sizeof(PtrStepSzb))); \
        rgbPtrsCPUPtrs[i] = rgbPtrsCPUs[i].ptr<PtrStepSzb>(); \
    } \
    for (unsigned char i = 0; i < cfgs.nEnabled; i++) { \
        if (cfgs[i].nChannels == 3) { \
            for (unsigned char k : cfgs[i].cudaGroups) { \
                if (k != CalcMeanVarGrp) { \
                    for (unsigned char j = 0; j < cfgs[i].nBounces; j++, rgbPtrsCPUPtrs[k]++) { \
                        *rgbPtrsCPUPtrs[k] = buffers[i][j].gpuMat; \
                    } \
                } else { \
                    *rgbPtrsCPUPtrs[k] = buffers[i][0].gpuMat; \
                    rgbPtrsCPUPtrs[k]++; \
                } \
            } \
        } else { \
            for (unsigned char k : cfgs[i].cudaGroups) { \
                if (k != CalcMeanVarGrp) { \
                    for (unsigned char j = 0; j < cfgs[i].nBounces; j++, fPtrsCPUPtrs[k]++) { \
                        *fPtrsCPUPtrs[k] = buffers[i][j].gpuMat; \
                    } \
                } else { \
                    *fPtrsCPUPtrs[k] = buffers[i][0].gpuMat; \
                    fPtrsCPUPtrs[k]++; \
                } \
            } \
        } \
    } \
    for (unsigned char i = 0; i < nCUDAGroupIndices; i++) { \
        fPtrs[i].upload(fPtrsCPUs[i], stream); \
        rgbPtrs[i].upload(rgbPtrsCPUs[i], stream); \
    } \
}

// *ptrsCPUPtr = mats[i] is an implicit conversion of GPUMat to PtrStepSzb, which can be addressed in the kernel.
#define PREPARE_G_PTRS(buffers, ptrs, channelCounts) { \
    unsigned char size = buffers.size(); \
    Mat ptrsCPU(1, size, CV_8UC(sizeof(PtrStepSzb))); \
    Mat channelCountsCPU(1, size, CV_8UC1); \
    PtrStepSzb *ptrsCPUPtr = ptrsCPU.ptr<PtrStepSzb>(); \
    unsigned char *channelCountsCPUPtr = channelCountsCPU.ptr<unsigned char>(); \
    for (unsigned char i = 0; i < size; i++, ptrsCPUPtr++, channelCountsCPUPtr++) { \
        *ptrsCPUPtr = buffers[i].gpuMat; \
        *channelCountsCPUPtr = buffers[i].gpuMat.channels(); \
    } \
    ptrs.upload(ptrsCPU, stream); \
    channelCounts.upload(channelCountsCPU, stream); \
} \

void Estimator::AllocateBuffers() {
    auto &cfgs = statTypeConfigs;

    for (auto bufs : t0Bufs.all()) {
        bufs->reserve(cfgs.nEnabled);
    }
    for (auto bufs : t1Bufs.all()) {
        bufs->reserve(cfgs.nEnabled);
    }

    for (unsigned char i = 0; i < cfgs.nEnabled; i++) {
        auto &cfg = cfgs[i]; // Note that configs have been filtered in the constructor to contain only the enabled ones.

        for (auto bufs : t0Bufs.all()) {
            bufs->emplace_back().reserve(cfg.nBounces);
        }
        for (auto bufs : t1Bufs.all()) {
            bufs->emplace_back().reserve(cfg.nBounces);
        }

        for (unsigned char j = cfg.bounceStart; j < cfg.bounceEnd; j++) {
            std::string jStr = std::to_string(j);

            if (cfg.nChannels == 3) {
                if (cfg.transform) {
                    // Allocate all buffers in BUF_LIST
                    #define BUF_ALLOC(name, type1, type3) ALLOC_BUF(t1Bufs.name, "-" #name, Mat_<type3>(height, width))
                    BUF_LIST(BUF_ALLOC)
                    #undef BUF_ALLOC   
                }
                // Allocate all buffers in FILM_BUF_LIST
                #define BUF_ALLOC(name, type1, type3) ALLOC_BUF(t0Bufs.name, "-film-" #name, Mat_<type3>(height, width))
                FILM_BUF_LIST(BUF_ALLOC)
                #undef BUF_ALLOC

                for (unsigned char k : cfg.cudaGroups) {
                    if (k != CalcMeanVarGrp) {
                        rgbBufferCounts[k]++;
                        runCUDA = true;
                    } else if (j == 0) { // Estimator variance estimate is only interesting for zeroth bounce and is calculated on the CPU instead of the GPU.
                        rgbBufferCounts[k]++;
                    }
                }
            } else if (cfg.nChannels == 1) {
                if (cfg.transform) {
                    // Allocate all buffers in BUF_LIST
                    #define BUF_ALLOC(name, type1, type3) ALLOC_BUF(t1Bufs.name, "-" #name, Mat_<type1>(height, width))
                    BUF_LIST(BUF_ALLOC)
                    #undef BUF_ALLOC   
                }
                // Allocate all buffers in FILM_BUF_LIST
                #define BUF_ALLOC(name, type1, type3) ALLOC_BUF(t0Bufs.name, "-film-" #name, Mat_<type1>(height, width))
                FILM_BUF_LIST(BUF_ALLOC)
                #undef BUF_ALLOC

                for (unsigned char k : cfg.cudaGroups) {
                    if (k != CalcMeanVarGrp) {
                        floatBufferCounts[k]++;
                        runCUDA = true;
                    } else if (j == 0) { // Estimator variance estimate is only interesting for zeroth bounce and is calculated on the CPU now instead of the GPU.
                        floatBufferCounts[k]++;
                    }
                }
            }

            if (cfg.gBuffer) {
                if (cfg.enableForMeanFilter) {
                    RegisterMeanGBuffer(t0Bufs.mean[i][j], cfg.meanFilterSD);
                    uploadBuffers.insert(&t0Bufs.mean[i][j]);
                }
                if (cfg.enableForVarFilter) {
                    RegisterVarGBuffer(t0Bufs.mean[i][j], cfg.varFilterSD);
                    uploadBuffers.insert(&t0Bufs.mean[i][j]);
                }
            }

            if (std::find(cfg.cudaGroups.begin(), cfg.cudaGroups.end(), DenoiseMeanGrp) != cfg.cudaGroups.end()) {
                if (cfg.transform) {
                    uploadBuffers.insert(&t1Bufs.mean[i][j]);
                    uploadBuffers.insert(&t1Bufs.m2[i][j]);
                    uploadBuffers.insert(&t1Bufs.m3[i][j]);
                    uploadBuffers.insert(&t1Bufs.m4[i][j]);
                    uploadBuffers.insert(&t1Bufs.c2X[i][j]);
                    uploadBuffers.insert(&t1Bufs.c2Y[i][j]);
                    uploadBuffers.insert(&t1Bufs.m1[i][j]);
                }

                uploadBuffers.insert(&t0Bufs.n[i][j]);
                uploadBuffers.insert(&t0Bufs.m2[i][j]);
                uploadBuffers.insert(&t0Bufs.m3[i][j]);
                uploadBuffers.insert(&t0Bufs.m4[i][j]);
                uploadBuffers.insert(&t0Bufs.c2X[i][j]);
                uploadBuffers.insert(&t0Bufs.c2Y[i][j]);
                uploadBuffers.insert(&t0Bufs.m1[i][j]);

#if DEBUG
                downloadBuffers.insert(&t0Bufs.varD [i][j]);
                downloadBuffers.insert(&t0Bufs.varCD[i][j]);
                downloadBuffers.insert(&t1Bufs.varD [i][j]);
                downloadBuffers.insert(&t1Bufs.varCD[i][j]);
#endif

                if (!(denoiseFilm && cfg.type == Radiance && j == 0)) { // Skip up/downloading radiance buffer at zeroth bounce because that's already covered by the film buffer.
                    uploadBuffers.insert(&t0Bufs.mean[i][j]);
                    downloadBuffers.insert(&t0Bufs.meanD[i][j]);
                }
            }

            if (std::find(cfg.cudaGroups.begin(), cfg.cudaGroups.end(), CalcTargetSPPGrp) != cfg.cudaGroups.end()) {
                if (cfg.transform) {
                    uploadBuffers.insert(&t1Bufs.m2[i][j]);
                    uploadBuffers.insert(&t1Bufs.m3[i][j]);
                    uploadBuffers.insert(&t1Bufs.m4[i][j]);
                }

                uploadBuffers.insert(&t0Bufs.n[i][j]);
                uploadBuffers.insert(&t0Bufs.m2[i][j]);
                uploadBuffers.insert(&t0Bufs.m3[i][j]);
                uploadBuffers.insert(&t0Bufs.m4[i][j]);

#if DEBUG
                downloadBuffers.insert(&t0Bufs.varD[i][j]);
                downloadBuffers.insert(&t1Bufs.varD[i][j]);
#endif
            }
        }
    }

    {
        using cv::cuda::PtrStepSzb;

        // Resize GPU pointer vectors
        #define GPU_PTR_RESIZE(name, type1, type3) fT0Ptrs.name.resize(nCUDAGroupIndices);
        FILM_BUF_LIST(GPU_PTR_RESIZE)
        #undef GPU_PTR_RESIZE
        #define GPU_PTR_RESIZE(name, type1, type3) fT1Ptrs.name.resize(nCUDAGroupIndices);
        BUF_LIST(GPU_PTR_RESIZE)
        #undef GPU_PTR_RESIZE
        #define GPU_PTR_RESIZE(name, type1, type3) rgbT0Ptrs.name.resize(nCUDAGroupIndices);
        FILM_BUF_LIST(GPU_PTR_RESIZE)
        #undef GPU_PTR_RESIZE
        #define GPU_PTR_RESIZE(name, type1, type3) rgbT1Ptrs.name.resize(nCUDAGroupIndices);
        BUF_LIST(GPU_PTR_RESIZE)
        #undef GPU_PTR_RESIZE

        // // Prepare GPU pointers pointing to the already allocated GPU buffers for the configured operations (DenoiseMeanGrp, CalcMeanVarGrp).
        #define PTR_PREPARE(name, type1, type3) PREPARE_STAT_PTRS(t0Bufs.name, fT0Ptrs.name, rgbT0Ptrs.name)
        FILM_BUF_LIST(PTR_PREPARE)
        #undef PTR_PREPARE
        #define GPU_PTR_PREPARE(name, type1, type3) PREPARE_STAT_PTRS(t1Bufs.name, fT1Ptrs.name, rgbT1Ptrs.name)
        BUF_LIST(GPU_PTR_PREPARE)
        #undef GPU_PTR_PREPARE


        PREPARE_G_PTRS(varGBuffers, varGBufferGPUPtrs, varGBufferChannelCountsGPUMat)
        PREPARE_G_PTRS(meanGBuffers, meanGBufferGPUPtrs, meanGBufferChannelCountsGPUMat)

        Mat varGBufferDRFactorsMat(varGBufferDRFactors);
        varGBufferDRFactorsGPUMat.upload(varGBufferDRFactorsMat, stream);
        Mat meanGBufferDRFactorsMat(meanGBufferDRFactors);
        meanGBufferDRFactorsGPUMat.upload(meanGBufferDRFactorsMat, stream);
    }
}

#undef ALLOC_BUF
#undef PREPARE_STAT_PTRS
#undef PREPARE_G_PTRS


template <typename T>
std::vector<StatTile<T>> Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd) const {
    return std::vector<StatTile<T>>(bounceEnd, StatTile<T>(tilePixelBounds));
}
template std::vector<StatTile<Float>> Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd) const;
template std::vector<StatTile<Vec3>>  Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd) const;

template <typename T>
std::vector<std::vector<StatTile<T>>> Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd, const unsigned char n) const {
    return std::vector<std::vector<StatTile<T>>>(bounceEnd, std::vector<StatTile<T>>(n, StatTile<T>(tilePixelBounds)));
}
template std::vector<std::vector<StatTile<Float>>> Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd, const unsigned char n) const;
template std::vector<std::vector<StatTile<Vec3>>>  Estimator::GetTiles(const Bounds2i &tilePixelBounds, const unsigned char bounceEnd, const unsigned char n) const;


template <typename T>
inline void Estimator::MergeTile(const StatTile<T> &tile, const unsigned char statTypeIndex, const unsigned char bounceIndex) const {
    for (const Point2i p : tile.GetPixelBounds()) {
        const unsigned int offset = p.y * width + p.x;
        const StatTilePixel<T> &tilePixel = tile.GetPixel(p);

        ((int *)t0Bufs.n   [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.n;
        ((T   *)t0Bufs.mean[statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.mean;
        ((T   *)t0Bufs.m2  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m2;
        ((T   *)t0Bufs.m3  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m3;
        ((T   *)t0Bufs.m4  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m4;
    }
}

template <typename T>
void Estimator::MergeTiles(const std::vector<StatTile<T>> &tiles, const StatTypeConfig &cfg) const {
    for (unsigned char j = 0; j < cfg.nBounces; j++)
        MergeTile(tiles[j+cfg.bounceStart], cfg.index, j);
}
template void Estimator::MergeTiles(const std::vector<StatTile<Float>> &tiles, const StatTypeConfig &cfg) const;
template void Estimator::MergeTiles(const std::vector<StatTile<Vec3>>  &tiles, const StatTypeConfig &cfg) const;

template <typename T>
void Estimator::MergeTiles(const std::vector<std::vector<StatTile<T>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const {
    for (unsigned char i = 0; i < cfgs.size(); i++) {
        auto &cfg = cfgs[i];
        for (unsigned char j = 0; j < cfg.nBounces; j++)
            MergeTile(tiles[j+cfg.bounceStart][i], cfg.index, j);
    }
}
template void Estimator::MergeTiles(const std::vector<std::vector<StatTile<Float>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const;
template void Estimator::MergeTiles(const std::vector<std::vector<StatTile<Vec3>>>  &tiles, const std::vector<StatTypeConfig> &cfgs) const;


template <typename T>
inline void Estimator::MergeTransformTile(const StatTile<T> &tile, const unsigned char statTypeIndex, const unsigned char bounceIndex) const {
    for (const Point2i p : tile.GetPixelBounds()) {
        const unsigned int offset = p.y * width + p.x;
        const StatTilePixel<T> &tilePixel = tile.GetPixel(p);

        ((int *)t0Bufs.n   [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.n;
        ((T   *)t0Bufs.mean[statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.mean;
        ((T   *)t0Bufs.m2  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m2;
        ((T   *)t0Bufs.m3  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m3;
        ((T   *)t0Bufs.m4  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m4;
        ((T   *)t0Bufs.c2X [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.c2X;
        ((T   *)t0Bufs.c2Y [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.c2Y;
        ((T   *)t0Bufs.m1  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t0.m1;

        ((T   *)t1Bufs.mean[statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.mean;
        ((T   *)t1Bufs.m2  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.m2;
        ((T   *)t1Bufs.m3  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.m3;
        ((T   *)t1Bufs.m4  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.m4;
        ((T   *)t1Bufs.c2X [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.c2X;
        ((T   *)t1Bufs.c2Y [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.c2Y;
        ((T   *)t1Bufs.m1  [statTypeIndex][bounceIndex].matPtr)[offset] = tilePixel.t1.m1;
    }
}

template <typename T>
void Estimator::MergeTransformTiles(const std::vector<StatTile<T>> &tiles, const StatTypeConfig &cfg) const {
    for (unsigned char j = 0; j < cfg.nBounces; j++)
        MergeTransformTile(tiles[j+cfg.bounceStart], cfg.index, j);
}
template void Estimator::MergeTransformTiles(const std::vector<StatTile<Float>> &tiles, const StatTypeConfig &cfg) const;
template void Estimator::MergeTransformTiles(const std::vector<StatTile<Vec3>>  &tiles, const StatTypeConfig &cfg) const;

template <typename T>
void Estimator::MergeTransformTiles(const std::vector<std::vector<StatTile<T>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const {
    for (unsigned char i = 0; i < cfgs.size(); i++) {
        auto &cfg = cfgs[i];
        for (unsigned char j = 0; j < cfg.nBounces; j++)
            MergeTransformTile(tiles[j+cfg.bounceStart][i], cfg.index, j);
    }
}
template void Estimator::MergeTransformTiles(const std::vector<std::vector<StatTile<Float>>> &tiles, const std::vector<StatTypeConfig> &cfgs) const;
template void Estimator::MergeTransformTiles(const std::vector<std::vector<StatTile<Vec3>>>  &tiles, const std::vector<StatTypeConfig> &cfgs) const;


void Estimator::Upload() {
    for (Buffer *b : uploadBuffers) {
#if DEBUG
        std::cout << "Uploading " << b->name << std::endl;
#endif
        b->upload(stream);
    }
}

void Estimator::Download() {
    for (Buffer *b : downloadBuffers) {
#if DEBUG
        std::cout << "Downloading " << b->name << std::endl;
#endif
        b->download(stream);
    }
}


void Estimator::DenoiseVars() {
    char grp = -1;
    if (rgbBufferCounts[DenoiseMeanGrp] > 0) {
        grp = DenoiseMeanGrp;
    } else if (rgbBufferCounts[CalcTargetSPPGrp] > 0) {
        grp = CalcTargetSPPGrp;
    }

    if (grp > -1) {
        cv::cuda::stat_denoiser::denoiseVars<float3>(
            rgbBufferCounts[grp],
            width,
            height,
            filterRadius,
            filterDSFactor,
            varZValue,
#if FULL_MULTI_TRANSFORM == 1
            nLambdas,
#endif
            rgbT0Ptrs.n[grp],
            rgbT0Ptrs.m2[grp],
            rgbT0Ptrs.m3[grp],
            rgbT0Ptrs.m4[grp],
            rgbT0Ptrs.c2X[grp],
            rgbT0Ptrs.c2Y[grp],
            rgbT0Ptrs.m1[grp],
            rgbT1Ptrs.m2[grp],
            rgbT1Ptrs.m3[grp],
            rgbT1Ptrs.m4[grp],
            rgbT1Ptrs.c2X[grp],
            rgbT1Ptrs.c2Y[grp],
            rgbT1Ptrs.m1[grp],
            varGBuffers.size(),
            varGBufferGPUPtrs,
            varGBufferChannelCountsGPUMat,
            varGBufferDRFactorsGPUMat,
            rgbT0Ptrs.jb[grp],
            rgbT0Ptrs.lnS2[grp],
            rgbT0Ptrs.se2[grp],
            rgbT0Ptrs.m2C[grp],
            rgbT0Ptrs.varD[grp],
            rgbT0Ptrs.varCD[grp],
            rgbT1Ptrs.jb[grp],
            rgbT1Ptrs.lnS2[grp],
            rgbT1Ptrs.se2[grp],
            rgbT1Ptrs.m2C[grp],
            rgbT1Ptrs.varD[grp],
            rgbT1Ptrs.varCD[grp],
            stream
        );
    }
}


void Estimator::DenoiseMeans() {
    if (rgbBufferCounts[DenoiseMeanGrp] > 0) {
        cv::cuda::stat_denoiser::denoiseFilm(
            rgbBufferCounts[DenoiseMeanGrp],
            width,
            height,
            filterRadius,
            filterDSFactor,
            meanZValue,
#if FULL_MULTI_TRANSFORM == 1
            nLambdas,
#endif
            rgbT0Ptrs.n[DenoiseMeanGrp],
            filmBuffer.gpuMat,
            rgbT0Ptrs.m2[DenoiseMeanGrp],
            rgbT0Ptrs.c2X[DenoiseMeanGrp],
            rgbT0Ptrs.c2Y[DenoiseMeanGrp],
            rgbT0Ptrs.m1[DenoiseMeanGrp],
            rgbT0Ptrs.varD[DenoiseMeanGrp],
            rgbT0Ptrs.varCD[DenoiseMeanGrp],
            rgbT1Ptrs.mean[DenoiseMeanGrp],
            rgbT1Ptrs.m2[DenoiseMeanGrp],
            rgbT1Ptrs.c2X[DenoiseMeanGrp],
            rgbT1Ptrs.c2Y[DenoiseMeanGrp],
            rgbT1Ptrs.m1[DenoiseMeanGrp],
            rgbT1Ptrs.varD[DenoiseMeanGrp],
            rgbT1Ptrs.varCD[DenoiseMeanGrp],
#if FULL_MULTI_TRANSFORM == 1 && FILM_MULTI_TRANSFORM_MODE == 3
            rgbT1Ptrs.jb[DenoiseMeanGrp],
#endif
            meanGBuffers.size(),
            meanGBufferGPUPtrs,
            meanGBufferChannelCountsGPUMat,
            meanGBufferDRFactorsGPUMat,
            rgbT0Ptrs.discr[DenoiseMeanGrp],
            rgbT1Ptrs.discr[DenoiseMeanGrp],
            filmFilteredBuffer.gpuMat,
            stream
        );
    }
}


template <typename T>
inline void calcMatMeanVar(const Mat &n, const Mat &m2, Mat &var) {
    for (int row = 0; row < m2.rows; ++row) {
        const int *nP = n.ptr<int>(row);
        const T *m2P = m2.ptr<T>(row);
        T *varP = var.ptr<T>(row);

        for (int col = 0; col < m2.cols; ++col) {
            const float nPF = static_cast<float>(*nP++);
            *varP++ = *m2P++ / ((nPF - 1.f) * nPF);
        }
    }
}

void Estimator::CalcMeanVars() {
    auto &cfgs = statTypeConfigs;

    for (unsigned char i = 0; i < cfgs.nEnabled; i++) {
        auto &cfg = cfgs[i];

        for (unsigned char j = cfg.bounceStart; j < cfg.bounceEnd; j++) {
            if (std::find(cfg.cudaGroups.begin(), cfg.cudaGroups.end(), CalcMeanVarGrp) != cfg.cudaGroups.end() && j == 0) {
                auto n = t0Bufs.n[i][j].mat;
                auto m2 = t0Bufs.m2[i][j].mat;
                auto var = t0Bufs.meanVar[i][j].mat;

                if (cfg.nChannels == 3) {
                    calcMatMeanVar<Vec3>(n, m2, var);
                } else {
                    calcMatMeanVar<Float>(n, m2, var);
                }
            }
        }
    }
}

void Estimator::CalcTargetSPPs(const uint64_t spp) {
    // The canonical way would be to find the StatTypeConfig index belonging to CalcTargetSPPGrp in statTypeConfigs,
    // but we already know that we use the radiance buffers for adaptive sampling.
    if (rgbBufferCounts[CalcTargetSPPGrp] > 0) {
        cv::cuda::stat_denoiser::calcTargetSPP<float3>(
            width,
            height,
            spp,
            t0Bufs.n[Radiance][0].gpuMat,
            t0Bufs.m2[Radiance][0].gpuMat,
            t0Bufs.varD[Radiance][0].gpuMat,
            targetSPPBuffer.gpuMat,
            stream
        );
    }
}

void Estimator::Synchronize() {
    cv::cuda::stat_denoiser::synchronize(stream);
}

void Estimator::PrintMacroConfiguration() {
    cv::cuda::stat_denoiser::printMacroConfiguration();
}

}  // namespace pbrt
