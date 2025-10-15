/*
 * Implement a sampler that uses scrambled Z-ordering
 * to distribute Sobol indexes.
 * 2020-05-20: Created by Abdalla Ahmed from an earlier version.
 * 2020-08-17: Revised by Abdalla Ahmed for final submission with the paper.
 */


// samplers/z.cpp*
#include "samplers/z.h"
#include "paramset.h"
#include "stats.h"

namespace pbrt {

;

// ZSampler Method Definitions

ZSampler::ZSampler(
    int64_t samplesPerPixel,
    int nSampledDimensions,
    const Bounds2i &sampleBounds,
    int alphabetSize,
    unsigned *preComputedTable
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
    if (alphabetSize <= 0) {                                                    // We use this as a shorthand for Morton ordering.
        Warning("Using vanilla Morton ordering\n");
        tileCount = 1;
        production = new unsigned[4];
        for (int i = 0; i < 4; i++) production[i] = 0;                          // Same tile
        allRanks = new unsigned char[DIMENSIONS];
        for (int i = 0; i < DIMENSIONS; i++) allRanks[i] = 0;                   // Morton ordering.
    }
    else {
        Warning("Using alphabet size %d for permutation\n", alphabetSize);
        tileCount = alphabetSize;
        if (preComputedTable != NULL) {
            production = preComputedTable;                                      // It's user's responsibility to make sure that the table matches the given size
        }
        else {
            production = new unsigned[tileCount * 4];
            for (int i = 0; i < tileCount * 4; i++) {                               // Construct the production rule.
                production[i] = rand() % tileCount;
            }
        }
        allRanks = new unsigned char[DIMENSIONS * tileCount];
        for (int i = 0; i < DIMENSIONS * tileCount; i++) {
            allRanks[i] = rand() % 24;
        }
    }
}

inline void ZSampler::Z1D(
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


inline void ZSampler::Z2D(
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

inline TileInfo ZSampler::getChildInfo(
    uint32_t X, uint32_t Y,                                                     // Absolute coordinates of child tile.
    TileInfo rootInfo,                                                          // ID and seqNo of root tile
    int digits,                                                                 // Subdivision depth of parent tile; only these digits are considered!
    int dim                                                        // Array of bytes indicating the ranks of children for each ID
) {
    unsigned char *ranks = &allRanks[dim * tileCount];
    uint32_t tileID = rootInfo.id;
    uint32_t seqNo = rootInfo.seqNo;
    int shift = (digits - 1) * shift1D;
    for (int i = 0; i < digits; i++) {
        seqNo <<= shift2D;
        unsigned y = (Y >> shift) & mask1D;
        unsigned x = (X >> shift) & mask1D;
        unsigned childNo = (y << shift1D) | x;
        uint32_t rank = childNo2rank[ ranks[tileID] ][childNo];
        seqNo |= rank;
        tileID = production[tileID * 4 + childNo];
        shift -= shift1D;
    }
    return {tileID, seqNo};
}

inline uint32_t ZSampler::seqNo2position(
    uint32_t seqNo,
    uint32_t tileID,                                                            // ID of root node
    int bits,                                                                   // depth in bits rather than base-4 digits, to handle odd powers of two
    int dim                                                                     // Sampled dimension
) {
    unsigned char *ranks = &allRanks[dim * tileCount];
    unsigned position = 0;
    if (bits & 1) {                                                             // Special treatment for an odd power of two, hence not a power of 4
        bits--;
        unsigned rank = (seqNo >> bits) & mask1D;                               // Theoretically, masking should not be needed
        unsigned childNo = rank ^ (rank2childNo[ ranks[tileID] ][0] & 1);       // We choose only child 0 or 1, and we use the LSB of rank1 as a permutation
        position |= childNo << bits;
    }
    while (bits > 0) {
        bits -= 2;
        unsigned rank = (seqNo >> bits) & mask2D;
        unsigned childNo = rank2childNo[ ranks[tileID] ][rank];
        position |= childNo << bits;
        tileID = production[tileID * 4 + childNo];
    }
    return position;
}

void ZSampler::StartPixel(const Point2i &p) {
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

std::unique_ptr<Sampler> ZSampler::Clone(int seed) {
    ZSampler *lds = new ZSampler(*this);
    lds->rng.SetSequence(seed);
    return std::unique_ptr<Sampler>(lds);
}

ZSampler *CreateZSampler(
    const ParamSet &params, const Bounds2i &sampleBounds
) {
    int nsamp = params.FindOneInt("pixelsamples", 16);
    int sd = params.FindOneInt("dimensions", 4);
    int alphabetSize = params.FindOneInt("alphabetSize", 4096);                 // Taken from recommended table size in ART paper. We experimented with 256 and it worked fine.
    if (PbrtOptions.quickRender) nsamp = 1;
    return new ZSampler(nsamp, sd, sampleBounds, alphabetSize);
}

ZSampler *CreateZ_ARTSampler(
    const ParamSet &params, const Bounds2i &sampleBounds
) {
    int nsamp = params.FindOneInt("pixelsamples", 16);
    int sd = params.FindOneInt("dimensions", 4);
    if (PbrtOptions.quickRender) nsamp = 1;
    return new ZSampler(nsamp, sd, sampleBounds, 4096, ART_2x2_PRODUCTION);
}

ZSampler *CreateMortonSampler(
    const ParamSet &params, const Bounds2i &sampleBounds
) {
    int nsamp = params.FindOneInt("pixelsamples", 16);
    int sd = params.FindOneInt("dimensions", 4);
    if (PbrtOptions.quickRender) nsamp = 1;
    return new ZSampler(nsamp, sd, sampleBounds, 0);
}


}                                                                               // namespace pbrt
