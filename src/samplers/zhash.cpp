/*
 * Implement a sampler that uses scrambled Z-ordering
 * to distribute Sobol indexes.
 * 2020-05-20: Created by Abdalla Ahmed from an earlier version.
 * 2020-08-17: Revised by Abdalla Ahmed for inclusion with paper.
 */


// samplers/z.cpp*
#include "samplers/zhash.h"
#include "paramset.h"
#include "stats.h"

inline uint32_t hash(uint32_t seqNo, uint32_t dim) {
    const int BITS = 24;
    const uint32_t MASK = (1 << BITS) - 1;
    const uint32_t Z = 0x9E377A;                                                // Z / (1 << BITS) approximates 1 - golden ratio
    seqNo ^= dim * 0x555555;                                                    // This should embed the bits of dim with all the bits of seqNo
    uint32_t x = (seqNo * Z) & MASK;                                            // Fractional part
    return (x * 24) >> BITS;                                                    // Map to the desired range
}

namespace pbrt {

;

inline Float fixedPt2Float(uint32_t v) {
    return std::min(v * Float(2.3283064365386963e-10), OneMinusEpsilon);
}

// ZHashSampler Method Definitions

ZHashSampler::ZHashSampler(
    int64_t samplesPerPixel,
    int nSampledDimensions,
    const Bounds2i &sampleBounds
) : PixelSampler(RoundUpPow2(samplesPerPixel), nSampledDimensions) {
    if (!IsPowerOf2(samplesPerPixel)) {
        Warning(
            "Pixel samples being rounded up to power of 2 "
            "(from %" PRId64 " to %" PRId64 ").",
                samplesPerPixel, RoundUpPow2(samplesPerPixel));
    }
    resolution = RoundUpPow2(
        std::max(sampleBounds.Diagonal().x, sampleBounds.Diagonal().y)
    );
    log2Resolution = Log2Int(resolution);
}

inline void ZHashSampler::Z1D(
    Point2i p,                                                                  // Pixel coordinates
    int nSamplesPerPixelSample,
    int nPixelSamples,
    int depth,
    Float *samples,
    int dim
) {
    TileInfo pixelTile = getChildInfo(                                          // Retrieve ID and seqNo of pixel tile
        p.x, p.y,                                                               // Pixel coordinates
        {0, 1},                                                                 // A root tile representing the whole film
        depth,                                                                  // Subdivide down to pixel level
        dim
    );
    uint32_t total = nPixelSamples * nSamplesPerPixelSample;
    int bits = CountTrailingZeros(total);
    uint32_t base = pixelTile.seqNo << bits;
    uint32_t fixedPt = Sobol32(base, ZMatrix1stD);
    for (uint32_t i = 0; i < total; i++) {
        uint32_t sampleNo = i ^ (i >> 1);                                       // Generate a Gray code
        uint32_t seqNo = base + sampleNo;
        uint32_t position = seqNo2position(sampleNo, pixelTile.id, bits, dim);
        samples[position] = fixedPt2Float(fixedPt);
        fixedPt ^= ZMatrix1stD[CountTrailingZeros(i + 1)];
    }
}


inline void ZHashSampler::Z2D(
    Point2i p,                                                                  // Pixel coordinates
    int nSamplesPerPixelSample,
    int nPixelSamples,
    int depth,
    Point2f *samples,
    int dim
) {
    TileInfo pixelTile = getChildInfo(                                          // Retrieve ID and seqNo of pixel tile
        p.x, p.y,                                                               // Pixel coordinates
        {0, 1},                                                                 // A root tile representing the whole film
        depth,                                                                  // Subdivide down to pixel level
        dim
    );
    uint32_t total = nPixelSamples * nSamplesPerPixelSample;
    int bits = CountTrailingZeros(total);
    uint32_t base = pixelTile.seqNo << bits;
    uint32_t fixedPt1 = Sobol32(base, ZMatrix1stD);
    uint32_t fixedPt2 = Sobol32(base, ZMatrix2ndD);
    for (uint32_t i = 0; i < total; i++) {
        uint32_t sampleNo = i ^ (i >> 1);                                       // Generate a Gray code
        uint32_t seqNo = base + sampleNo;
        uint32_t position = seqNo2position(sampleNo, pixelTile.id, bits, dim);
        samples[position].x = fixedPt2Float(fixedPt1);
        samples[position].y = fixedPt2Float(fixedPt2);
        fixedPt1 ^= ZMatrix1stD[CountTrailingZeros(i + 1)];
        fixedPt2 ^= ZMatrix2ndD[CountTrailingZeros(i + 1)];
    }
}

inline TileInfo ZHashSampler::getChildInfo(
uint32_t X, uint32_t Y,                                                     // Absolute coordinates of child tile.
TileInfo rootInfo,                                                          // ID and seqNo of root tile
int digits,                                                                 // Subdivision depth of parent tile; only these digits are considered!
int dim                                                        // Array of bytes indicating the ranks of children for each ID
) {
    uint32_t tileID = rootInfo.id;
    uint32_t seqNo = rootInfo.seqNo;
    int shift = (digits - 1) * shift1D;
    for (int i = 0; i < digits; i++) {
        seqNo <<= shift2D;
        unsigned y = (Y >> shift) & mask1D;
        unsigned x = (X >> shift) & mask1D;
        unsigned childNo = (y << shift1D) | x;
        uint32_t rank = childNo2rank[ hash(tileID, dim) ][childNo];
        seqNo |= rank;
        tileID = (tileID << shift2D) | childNo;
        shift -= shift1D;
    }
    return {tileID, seqNo};
}

inline uint32_t ZHashSampler::seqNo2position(
    uint32_t seqNo,
    uint32_t tileID,                                                            // ID of root node
    int bits,                                                                   // depth in bits rather than base-4 digits, to handle odd powers of two
    int dim                                                                     // Sampled dimension
) {
    unsigned position = 0;
    if (bits & 1) {                                                             // Special treatment for an odd power of two, hence not a power of 4
        bits--;
        unsigned rank = (seqNo >> bits) & mask1D;                               // Theoretically, masking should not be needed
        unsigned childNo = rank ^ (rank2childNo[ hash(tileID, dim) ][0] & 1);       // We choose only child 0 or 1, and we use the LSB of rank1 as a permutation
        position |= childNo << bits;
    }
    while (bits > 0) {
        bits -= 2;
        unsigned rank = (seqNo >> bits) & mask2D;
        unsigned childNo = rank2childNo[ hash(tileID, dim) ][rank];
        position |= childNo << bits;
        tileID = (tileID << shift2D) | childNo;
    }
    return position;
}

void ZHashSampler::StartPixel(const Point2i &p) {
    ProfilePhase _(Prof::StartPixel);
    int dim = 0;
    // Generate 1D and 2D pixel sample components using Z-Sobol
    for (size_t i = 0; i < samples1D.size(); ++i) {
        Z1D(
            p, 1, samplesPerPixel, log2Resolution,
            &samples1D[i][0], dim
        );
        dim++;
    }
    for (size_t i = 0; i < samples2D.size(); ++i) {
        Z2D(
            p, 1, samplesPerPixel, log2Resolution,
            &samples2D[i][0], dim
        );
        dim++;
    }
    // Generate 1D and 2D array samples using Z-Sobol
    for (size_t i = 0; i < samples1DArraySizes.size(); ++i) {
        Z1D(
            p, samples1DArraySizes[i], samplesPerPixel, log2Resolution,
            &sampleArray1D[i][0], dim
        );
        dim++;
    }
    for (size_t i = 0; i < samples2DArraySizes.size(); ++i) {
        Z2D(
            p, samples2DArraySizes[i], samplesPerPixel, log2Resolution,
            &sampleArray2D[i][0], dim
        );
        dim++;
    }
    PixelSampler::StartPixel(p);
}

std::unique_ptr<Sampler> ZHashSampler::Clone(int seed) {
    ZHashSampler *lds = new ZHashSampler(*this);
    lds->rng.SetSequence(seed);
    return std::unique_ptr<Sampler>(lds);
}

ZHashSampler *CreateZHashSampler(
    const ParamSet &params, const Bounds2i &sampleBounds
) {
    int nsamp = params.FindOneInt("pixelsamples", 16);
    int sd = params.FindOneInt("dimensions", 4);
    if (PbrtOptions.quickRender) nsamp = 1;
    return new ZHashSampler(nsamp, sd, sampleBounds);
}

}                                                                               // namespace pbrt
