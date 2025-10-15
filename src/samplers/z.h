/*
 * Define a sampler that uses scrambled Z-ordering to distribute Sobol indexes.
 * 2020-05-20: Created by Abdalla Ahmed from an earlier version.
 * 2020-08-17: Revised by Abdalla Ahmed for inclusion with paper.
 */


#ifndef PBRT_SAMPLERS_Z_H
#define PBRT_SAMPLERS_Z_H

// samplers/z.h*
#include "sampler.h"
#include "samplers/zcommon.h"

extern unsigned ART_2x2_PRODUCTION[4096 * 4];

namespace pbrt {
;                                                                               // This is to fix the alignment for the KATE editor I am using.

// ZSampler Declarations
class ZSampler : public PixelSampler {
  public:
    // ZSampler Public Methods
    ZSampler(
        int64_t samplesPerPixel,
        int nSampledDimensions,
        const Bounds2i &sampleBounds,
        int alphabetSize,                                                       // Set alphabetSize to 0 to simulate Morton ordering.
        unsigned *preComputedTable = NULL
    );
    void StartPixel(const Point2i &);
    std::unique_ptr<Sampler> Clone(int seed);
    int RoundCount(int count) const { return RoundUpPow2(count); }
  protected:
    // ZSampler Private Data and functions
    int resolution, log2Resolution;
    static const int DIMENSIONS = 1024;                                         // This is taken equal to PBRT's global Sobol sampler
    int tileCount;
    unsigned *production;
    unsigned char *allRanks;
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

ZSampler *CreateZSampler(
    const ParamSet &params,
    const Bounds2i &sampleBounds
);

ZSampler *CreateZ_ARTSampler(
    const ParamSet &params,
    const Bounds2i &sampleBounds
);

ZSampler *CreateMortonSampler(                                                  // Please note that an actual Morton sampler should be faster.
    const ParamSet &params,
    const Bounds2i &sampleBounds
);

}                                                                               // namespace pbrt

#endif                                                                          // PBRT_SAMPLERS_Z_H
