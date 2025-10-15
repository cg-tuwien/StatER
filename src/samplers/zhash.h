/*
 * Define a sampler that uses scrambled Z-ordering to distribute Sobol indexes.
 * 2020-05-20: Created by Abdalla Ahmed from an earlier version.
 * 2020-07-09: Adapted by Abdalla Ahmed to use a hash function.
 * 2020-08-17: Revised by Abdalla Ahmed for final submission with the paper.
 */


#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_ZHASH_H
#define PBRT_SAMPLERS_ZHASH_H

// samplers/zhash.h*
#include "sampler.h"
#include "samplers/zcommon.h"

namespace pbrt {
;

// ZHashSampler Declarations
class ZHashSampler : public PixelSampler {
  public:
    // ZHashSampler Public Methods
    ZHashSampler(
        int64_t samplesPerPixel,
        int nSampledDimensions,
        const Bounds2i &sampleBounds
    );
    void StartPixel(const Point2i &);
    std::unique_ptr<Sampler> Clone(int seed);
    int RoundCount(int count) const { return RoundUpPow2(count); }
  protected:
    // ZHashSampler Private Data and functions
    int resolution, log2Resolution;
    static const int DIMENSIONS = 1024;
    inline void Z1D(
        Point2i p,                                                              // Pixel coordinates
        int nSamplesPerPixelSample,
        int nPixelSamples,
        int depth,
        Float *samples,
        int dim
    );
    inline void Z2D(
        Point2i p,                                                              // Pixel coordinates
        int nSamplesPerPixelSample,
        int nPixelSamples,
        int depth,
        Point2f *samples,
        int dim
    );
    inline TileInfo getChildInfo(
        uint32_t X, uint32_t Y,                                                 // Absolute coordinates of child tile.
        TileInfo rootInfo,                                                      // ID and seqNo of root tile
        int digits,                                                             // Subdivision depth of parent tile; only these digits are considered!
        int dim                                                                 // Array of bytes indicating the ranks of children for each ID
    );
    inline uint32_t seqNo2position(
        uint32_t seqNo,
        uint32_t tileID,                                                        // ID of root node
        int bits,                                                               // depth in bits rather than base-4 digits, to handle odd powers of two
        int dim                                                                 // Table of permutations (per node ID)
    );
};

ZHashSampler *CreateZHashSampler(
    const ParamSet &params,
    const Bounds2i &sampleBounds
);


}  // namespace pbrt

#endif  // PBRT_SAMPLERS_ZHASH_H
