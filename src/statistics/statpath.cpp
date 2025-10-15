// © 2025 Hiroyuki Sakai

#include "statistics/statpath.h"
#include "progressreporter.h"
#include "scene.h"

#include <filesystem>

#include "materials/disney.h"
#include "materials/fourier.h"
#include "materials/glass.h"
#include "materials/hair.h"
#include "materials/kdsubsurface.h"
#include "materials/matte.h"
#include "materials/metal.h"
#include "materials/mirror.h"
#include "materials/mixmat.h"
#include "materials/plastic.h"
#include "materials/substrate.h"
#include "materials/subsurface.h"
#include "materials/translucent.h"
#include "materials/uber.h"
#include "samplers/random.h"
#include "textures/constant.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

void StatPathIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution = CreateLightSampleDistribution(lightSampleStrategy, scene);
}

template <typename T>
inline StatPathIntegrator::AddSampleFn<T> StatPathIntegrator::GetAddSampleFn(const StatTypeConfig &cfg) {
    if (cfg.transform) {
        if (cfg.maxMoment == 5) {
            return &StatTile<T>::AddTransformSampleM4C;
        } else if (cfg.maxMoment == 4) {
            return &StatTile<T>::AddTransformSampleM4;
        } else if (cfg.maxMoment == 3) {
            return &StatTile<T>::AddTransformSampleM3;
        } else if (cfg.maxMoment == 2) {
            return &StatTile<T>::AddTransformSampleM2;
        } else if (cfg.maxMoment == 1) {
            return &StatTile<T>::AddTransformSampleM1;
        }
    } else {
        if (cfg.maxMoment == 5) {
            return &StatTile<T>::AddSampleM4C;
        } else if (cfg.maxMoment == 4) {
            return &StatTile<T>::AddSampleM4;
        } else if (cfg.maxMoment == 3) {
            return &StatTile<T>::AddSampleM3;
        } else if (cfg.maxMoment == 2) {
            return &StatTile<T>::AddSampleM2;
        } else if (cfg.maxMoment == 1) {
            return &StatTile<T>::AddSampleM1;
        }
    }

    return nullptr;
}

template <typename T>
void StatPathIntegrator::Render(const Scene &scene) {
    namespace chrono = std::chrono;

    using std::cout;
    using std::endl;
    using std::shared_ptr;
    using std::unique_ptr;
    using std::vector;

    Preprocess(scene, *sampler);
    uint64_t spp = sampler->samplesPerPixel;

    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    const Bounds2i sampleBounds = camera->film->GetSampleBounds();
    const Vector2i sampleExtent = sampleBounds.Diagonal();
    const unsigned char tileSize = 16;
    const Point2i nTiles(
        (sampleExtent.x + tileSize - 1) / tileSize,
        (sampleExtent.y + tileSize - 1) / tileSize
    );
    const unsigned short nTilesTotal = nTiles.x * nTiles.y;

    const unsigned char nFloatGBufs = floatGBufConfigs.nEnabled;
    const unsigned char nRGBGBufs = rgbGBufConfigs.nEnabled;

    const StatTypeConfigs &sCfgs = statTypeConfigs;

    const unsigned char nLs = std::max((int)sCfgs[Radiance].bounceEnd, 1); // We need at least one item for the film

    // Declare per-tile samplers
    vector<unique_ptr<Sampler>> tileSamplers(nTilesTotal);

    // Declare tiles
    vector<shared_ptr<FilmTile>> filmTiles(nTilesTotal);
    vector<vector<StatTile<T>>> lTiles(nTilesTotal);
    vector<vector<StatTile<Vec3>>> itLTiles(nTilesTotal);
    vector<vector<vector<StatTile<Float>>>> misTallyTiles(nTilesTotal);
    vector<vector<vector<StatTile<Float>>>> floatGBufTiles(nTilesTotal);
    vector<vector<vector<StatTile<Vec3>>>> rgbGBufTiles(nTilesTotal);

    OutputBufferSelection outBufSel(bufReg, std::regex(outputRegex), camera->film->filename);

    const vector<StatTypeConfig> featureCfgs = {sCfgs[StatMaterialID], sCfgs[StatDepth], sCfgs[StatNormal], sCfgs[StatAlbedo]};
    vector<StatTypeConfig> enabledFloatGBufCfgs;
    vector<StatTypeConfig> enabledRGBGBufCfgs;
    std::copy_if(featureCfgs.begin(), featureCfgs.end(), std::back_inserter(enabledFloatGBufCfgs), [](auto &item) {return item.enable && item.nChannels == 1;});
    std::copy_if(featureCfgs.begin(), featureCfgs.end(), std::back_inserter(enabledRGBGBufCfgs),   [](auto &item) {return item.enable && item.nChannels == 3;});

    // Prepare functions pointers for adding samples
#if FULL_MULTI_TRANSFORM == 1
    void (StatTile<T>    ::*AddLSampleFn)           (const Point2i p, const Point2f inPxP, const T     sample, const Float lambda) = GetAddSampleFn<T>    (sCfgs[Radiance]);
    void (StatTile<Float>::*AddMISWinRateSampleFn)  (const Point2i p, const Point2f inPxP, const Float sample, const Float lambda) = GetAddSampleFn<Float>(sCfgs[MISBSDFWinRate]);
    void (StatTile<Float>::*AddFloatGBufferSampleFn)(const Point2i p, const Point2f inPxP, const Float sample, const Float lambda) = GetAddSampleFn<Float>(sCfgs[StatMaterialID]);
    void (StatTile<Vec3> ::*AddRGBGBufferSampleFn)  (const Point2i p, const Point2f inPxP, const Vec3  sample, const Float lambda) = GetAddSampleFn<Vec3> (sCfgs[StatNormal]);
    void (StatTile<Vec3> ::*AddItLSampleFn)         (const Point2i p, const Point2f inPxP, const Vec3  sample, const Float lambda) = GetAddSampleFn<Vec3> (sCfgs[ItRadiance]);
#else
    void (StatTile<T>    ::*AddLSampleFn)           (const Point2i p, const Point2f inPxP, const T     sample) = GetAddSampleFn<T>    (sCfgs[Radiance]);
    void (StatTile<Float>::*AddMISWinRateSampleFn)  (const Point2i p, const Point2f inPxP, const Float sample) = GetAddSampleFn<Float>(sCfgs[MISBSDFWinRate]);
    void (StatTile<Float>::*AddFloatGBufferSampleFn)(const Point2i p, const Point2f inPxP, const Float sample) = GetAddSampleFn<Float>(sCfgs[StatMaterialID]);
    void (StatTile<Vec3> ::*AddRGBGBufferSampleFn)  (const Point2i p, const Point2f inPxP, const Vec3  sample) = GetAddSampleFn<Vec3> (sCfgs[StatNormal]);
    void (StatTile<Vec3> ::*AddItLSampleFn)         (const Point2i p, const Point2f inPxP, const Vec3  sample) = GetAddSampleFn<Vec3> (sCfgs[ItRadiance]);
#endif

    auto RenderLoop = [&](const int nIterations) {
        ParallelFor2D([&](Point2i tile) {
            // Compute sample bounds for tile
            const unsigned short x0 = sampleBounds.pMin.x + tile.x * tileSize;
            const unsigned short x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
            const unsigned short y0 = sampleBounds.pMin.y + tile.y * tileSize;
            const unsigned short y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
            const Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

            const unsigned int tileIndex = tile.y * nTiles.x + tile.x;

            filmTiles     [tileIndex] = camera->film->GetFilmTile(tileBounds);
            tileSamplers  [tileIndex] = sampler->Clone(tileIndex);
            lTiles        [tileIndex] = estimator.GetTiles<T>    (camera->film->GetActualTileBounds(tileBounds), sCfgs[Radiance].bounceEnd);
            misTallyTiles [tileIndex] = estimator.GetTiles<Float>(camera->film->GetActualTileBounds(tileBounds), sCfgs[MISBSDFWinRate].bounceEnd, 2);
            floatGBufTiles[tileIndex] = estimator.GetTiles<Float>(camera->film->GetActualTileBounds(tileBounds), 1, nFloatGBufs);
            rgbGBufTiles  [tileIndex] = estimator.GetTiles<Vec3> (camera->film->GetActualTileBounds(tileBounds), 1, nRGBGBufs);
        }, nTiles);

        for (unsigned int i = 1, j = 1; i <= nIterations; i++) {
            if (calcItStats) {
                // Iteration tiles are resetted every iteration; don't worry about performance, this is only run if for non-performance critical runs
                ParallelFor2D([&](Point2i tile) {
                    // Compute sample bounds for tile
                    const unsigned short x0 = sampleBounds.pMin.x + tile.x * tileSize;
                    const unsigned short x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                    const unsigned short y0 = sampleBounds.pMin.y + tile.y * tileSize;
                    const unsigned short y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                    const Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

                    const unsigned int tileIndex = tile.y * nTiles.x + tile.x;

                    itLTiles[tileIndex] = estimator.GetTiles<Vec3> (camera->film->GetActualTileBounds(tileBounds), sCfgs[ItRadiance].bounceEnd);
                }, nTiles);
            }


            chrono::steady_clock::time_point begin = chrono::steady_clock::now();

            ProgressReporter reporter(nTilesTotal, "Rendering");
            {
                ProfilePhase _(Prof::StatPathRender);

                camera->film->Clear();

                ParallelFor2D([&](Point2i tile) {
                    // Render section of image corresponding to _tile_

                    // Allocate _MemoryArena_ for tile
                    MemoryArena arena;

                    // Compute sample bounds for tile
                    const unsigned short x0 = sampleBounds.pMin.x + tile.x * tileSize;
                    const unsigned short x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                    const unsigned short y0 = sampleBounds.pMin.y + tile.y * tileSize;
                    const unsigned short y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                    const Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                    const Bounds2i actualTileBounds = camera->film->GetActualTileBounds(tileBounds);
                    LOG(INFO) << "Starting image tile " << tileBounds;

                    const unsigned int tileIndex = tile.y * nTiles.x + tile.x;

                    const unique_ptr<Sampler>       &tileSampler   = tileSamplers  [tileIndex];
                    const shared_ptr<FilmTile>      &filmTile      = filmTiles     [tileIndex];
                    vector<StatTile<T>>             &lTile         = lTiles        [tileIndex];
                    vector<StatTile<Vec3>>          &itLTile       = itLTiles      [tileIndex];
                    vector<vector<StatTile<Float>>> &misTallyTile  = misTallyTiles [tileIndex];
                    vector<vector<StatTile<Float>>> &floatGBufTile = floatGBufTiles[tileIndex];
                    vector<vector<StatTile<Vec3>>>  &rgbGBufTile   = rgbGBufTiles  [tileIndex];

                    Features features;
                    features.floats    = vector<Float>   (nFloatGBufs);
                    features.spectrums = vector<Spectrum>(nRGBGBufs);

                    vector<Float> avgLs(sCfgs[Radiance].bounceEnd);
                    vector<MISWinRate> misWinRates(sCfgs[MISBSDFWinRate].bounceEnd);

                    vector<Spectrum> Ls(nLs);
                    Spectrum &L = Ls[0];
                    vector<MISTally> misTallies(sCfgs[MISBSDFWinRate].bounceEnd);

                    // Loop over pixels in tile to render them
                    for (const Point2i pixel : tileBounds) {
                        {
                            ProfilePhase pp(Prof::StartPixel);
                            tileSampler->StartPixel(pixel);
                        }

                        // Do this check after the StartPixel() call; this keeps the usage of RNG values from
                        // (most) Samplers that use RNGs consistent, which improves reproducability debugging.
                        if (!InsideExclusive(pixel, pixelBounds)) {
                            continue;
                        }

                        const Point2i actualPixel(pixel - camera->film->croppedPixelBounds.pMin);
                        const unsigned int offset = actualPixel.y * camera->film->width + actualPixel.x;

                        unsigned int n = 0;
                        unsigned int targetSPP = spp;
                        if (i > 1) {
                            if (enableAS) {
                                n = estimator.t0Bufs.n[Radiance][0].mat.ptr<int>()[offset];
                                targetSPP = estimator.targetSPPBuffer.mat.ptr<int>()[offset];
                            } else {
                                if (expIterations) {
                                    n = spp << (i - 2);
                                    targetSPP = n;
                                } else {
                                    n = (i - 1) * spp;
                                }
                            }

                            // Activate this part to stop rendering after a set amount of SPP.
#if 0
                            const unsigned int equalSPP = 123; // Set SPP here.
                            unsigned int totalSPP = n + targetSPP;
                            if (totalSPP > equalSPP)
                                totalSPP = equalSPP;
                            tileSampler->SetSPP(totalSPP);
#else
                            tileSampler->SetSPP(n + targetSPP);
#endif

                            if (!tileSampler->SetSampleNumber(n)) {
                                continue;
                            }
                        }


                        do {
                            // Initialize _CameraSample_ for current sample
                            const CameraSample cameraSample = tileSampler->GetCameraSample(pixel);

                            // Generate camera ray for current sample
                            RayDifferential ray;
                            // const Float rayWeight = camera->GenerateRay(cameraSample, &ray);
                            const Float rayWeight = camera->GenerateRayDifferential(cameraSample, &ray);

                            if (enableAS) {
                                const unsigned int nIterationsLeft = nIterations - i + 1;
                                ray.ScaleDifferentials(1.f / std::sqrt((Float)n + nIterationsLeft * targetSPP));
                            } else {
                                ray.ScaleDifferentials(1.f / std::sqrt((Float)(expIterations ? spp << (nIterations-1) : nIterations * targetSPP))); // Note: multiplication with nIterations produces a difference vs. vanilla path tracing!
                            }
                            ++nCameraRays;

                            if (i > 1) {
                                if (enableACRR) {
                                    for (unsigned char j = sCfgs[Radiance].bounceStart; j < sCfgs[Radiance].bounceEnd; j++) {
                                        avgLs[j] = GetY(estimator.t0Bufs.meanD[sCfgs[Radiance].index][j - sCfgs[Radiance].bounceStart].mat.ptr<T>()[offset]);
                                    }
                                }
                                if (enableSMIS) {
                                    for (unsigned char j = sCfgs[MISBSDFWinRate].bounceStart; j < sCfgs[MISBSDFWinRate].bounceEnd; j++) {
                                        misWinRates[j].bsdf  = estimator.t0Bufs.meanD[sCfgs[MISBSDFWinRate ].index][j].mat.ptr<Float>()[offset];
                                        misWinRates[j].light = estimator.t0Bufs.meanD[sCfgs[MISLightWinRate].index][j].mat.ptr<Float>()[offset];
                                    }
                                }
                            }

                            for (auto &myFloat : features.floats) {
                                myFloat = 0.f;
                            }
                            for (auto &mySpectrum : features.spectrums) {
                                mySpectrum = 0.f;
                            }
                            for (auto &myL : Ls) {
                                myL = 0.f;
                            }
                            for (auto &myMISTally : misTallies) {
                                myMISTally.bsdf  = 0;
                                myMISTally.light = 0;
                            }

                            if (rayWeight > 0)
                                Li(
                                    ray, scene, *tileSampler, arena, features,
                                    avgLs, misWinRates, Ls, misTallies, i
                                );

                            // Issue warning if unexpected radiance value returned
                            if (L.HasNaNs()) {
                                LOG(ERROR) << StringPrintf(
                                    "Not-a-number radiance value returned for pixel (%d, %d), sample %d. Setting to black.",
                                    pixel.x, pixel.y, (int)tileSampler->CurrentSampleNumber()
                                );
                                L = Spectrum(0.f);
                            } else if (L.y() < -1e-5) {
                                LOG(ERROR) << StringPrintf(
                                    "Negative luminance value, %f, returned for pixel (%d, %d), sample %d. Setting to black.",
                                    L.y(), pixel.x, pixel.y, (int)tileSampler->CurrentSampleNumber()
                                );
                                L = Spectrum(0.f);
                            } else if (std::isinf(L.y())) {
                                LOG(ERROR) << StringPrintf(
                                    "Infinite luminance value returned for pixel (%d, %d), sample %d. Setting to black.",
                                    pixel.x, pixel.y, (int)tileSampler->CurrentSampleNumber()
                                );
                                L = Spectrum(0.f);
                            }
                            VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " << ray << " -> L = " << L;

                            // Add camera ray's contribution to tiles
                            filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

                            // // Within/in-pixel position for covariance calculation
                            Point2f inPxP = Point2f(cameraSample.pFilm - Point2f(actualPixel));

#if FULL_MULTI_TRANSFORM == 1
                            for (unsigned char j = 0; j < lambdas.size(); j++) {
                                (lTile[j].*AddLSampleFn)(actualPixel, inPxP, GetStatSample<T>(Ls[0]), lambdas[j]);
                            }
#else
                            for (unsigned char j = sCfgs[Radiance].bounceStart; j < sCfgs[Radiance].bounceEnd; j++) {
                                (lTile[j].*AddLSampleFn)(actualPixel, inPxP, GetStatSample<T>(Ls[j]));
                            }
#endif
                            for (unsigned char j = sCfgs[ItRadiance].bounceStart; j < sCfgs[ItRadiance].bounceEnd; j++) {
#if FULL_MULTI_TRANSFORM == 1
                                (itLTile[j].*AddItLSampleFn)(actualPixel, inPxP, GetStatSample<Vec3>(Ls[j]), 1.f);
#else
                                (itLTile[j].*AddItLSampleFn)(actualPixel, inPxP, GetStatSample<Vec3>(Ls[j]));
#endif
                            }
                            for (unsigned char j = sCfgs[MISBSDFWinRate].bounceStart; j < sCfgs[MISBSDFWinRate].bounceEnd; j++) {
#if FULL_MULTI_TRANSFORM == 1
                                (misTallyTile[j][0].*AddMISWinRateSampleFn)(actualPixel, inPxP, misTallies[j].bsdf,  1.f);
                                (misTallyTile[j][1].*AddMISWinRateSampleFn)(actualPixel, inPxP, misTallies[j].light, 1.f);
#else
                                (misTallyTile[j][0].*AddMISWinRateSampleFn)(actualPixel, inPxP, misTallies[j].bsdf);
                                (misTallyTile[j][1].*AddMISWinRateSampleFn)(actualPixel, inPxP, misTallies[j].light);
#endif
                            }
                            for (unsigned char j = 0; j < nFloatGBufs; j++) {
#if FULL_MULTI_TRANSFORM == 1
                                (floatGBufTile[0][j].*AddFloatGBufferSampleFn)(actualPixel, inPxP, features.floats[j], 1.f);
#else
                                (floatGBufTile[0][j].*AddFloatGBufferSampleFn)(actualPixel, inPxP, features.floats[j]);
#endif
                            }
                            for (unsigned char j = 0; j < nRGBGBufs; j++) {
#if FULL_MULTI_TRANSFORM == 1
                                (rgbGBufTile[0][j].*AddRGBGBufferSampleFn)(actualPixel, inPxP, GetStatSample<Vec3>(features.spectrums[j]), 1.f);
#else
                                (rgbGBufTile[0][j].*AddRGBGBufferSampleFn)(actualPixel, inPxP, GetStatSample<Vec3>(features.spectrums[j]));
#endif
                            }

                            // Free _MemoryArena_ memory from computing image sample value
                            arena.Reset();
                        } while (tileSampler->StartNextSample());
                    }
                    LOG(INFO) << "Finished image tile " << tileBounds;

                    // Merge tiles into buffers
                    camera->film->MergeFilmTile(filmTile);
                    if (sCfgs[Radiance].enable) {
                        if (sCfgs[Radiance].transform) {
                            estimator.MergeTransformTiles(lTile, sCfgs[Radiance]);
                        } else {
                            estimator.MergeTiles(lTile, sCfgs[Radiance]);
                        }
                    }
                    if (sCfgs[ItRadiance].enable) {
                        estimator.MergeTiles(itLTile, sCfgs[ItRadiance]);
                    }
                    if (sCfgs[MISBSDFWinRate].enable && sCfgs[MISLightWinRate].enable) {
                        estimator.MergeTiles(misTallyTile, {sCfgs[MISBSDFWinRate], sCfgs[MISLightWinRate]});
                    }
                    estimator.MergeTiles(floatGBufTile, enabledFloatGBufCfgs);
                    estimator.MergeTiles(rgbGBufTile, enabledRGBGBufCfgs);

                    reporter.Update();
                }, nTiles);

                reporter.Done();
            }
            LOG(INFO) << "Rendering finished";

            camera->film->UpdateImage();

            if (calcProDenStats) {
                estimator.CalcMeanVars();
            }

            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            auto renderTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            cout << "Iteration: " << i << endl;
            cout << "SPP: " << (expIterations ? spp << std::max((int)i - 2, 0) : spp) << endl;
            cout << "Rendering time [ns]: " << renderTime << endl;

            if (!estimator.runCUDA) {
                cout << "CUDA time [ns]: " << 0 << endl;
            } else {
                begin = chrono::steady_clock::now();
                estimator.Upload();
#if DISABLE_DENOISE_VARS == 0
                estimator.DenoiseVars();
#endif
                estimator.DenoiseMeans();
                if (enableAS)
                    estimator.CalcTargetSPPs(expIterations ? spp << (i - 1) : spp);
                estimator.Download();
                estimator.Synchronize();

                end = chrono::steady_clock::now();
                auto cudaTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
                cout << "CUDA time [ns]: " << cudaTime << endl;
            }

            begin = chrono::steady_clock::now();
            if (PbrtOptions.writeImages || PbrtOptions.displayImages) {
                if (expIterations || !outputExpIterations || spp << (j - 1) == i * spp) {
                    outBufSel.PrepareOutput();
                    const unsigned int outputSPP = expIterations ? spp << (i - 1) : i * spp;
                    if (PbrtOptions.writeImages) {
                        outBufSel.Write(std::to_string(outputSPP));
                    }
                    if (PbrtOptions.displayImages) {
                        outBufSel.Display(std::to_string(outputSPP));
                    }
                    j++;
                }
            }
            end = chrono::steady_clock::now();
            cout << "Output time [ns]: " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;
        }
    };

    if (PbrtOptions.warmUp) {
        cout << "==== Warm-Up Start ====" << endl;
        RenderLoop(1);
        cout << "==== Warm-Up End ====" << endl;
    }

    RenderLoop(nIterations);
}

void StatPathIntegrator::Render(const Scene &scene) {
    if (enableMultiChannelStats) {
        Render<Vec3>(scene);
    } else {
        Render<Float>(scene);
    }
}

inline void StatPathIntegrator::ReadFile(const std::string &filename, Buffer &buf) {
    cv::imread(filename, cv::IMREAD_UNCHANGED).convertTo(buf.mat, buf.mat.type());
    if (buf.mat.channels() == 3)
        cv::cvtColor(buf.mat, buf.mat, cv::COLOR_BGR2RGB);
    buf.matPtr = buf.mat.ptr();
}

template <typename T>
void StatPathIntegrator::Denoise(const Scene &scene) {
    namespace chrono = std::chrono;

    using std::shared_ptr;
    using std::unique_ptr;
    using std::string;
    using std::vector;

    uint64_t spp = sampler->samplesPerPixel;

    // Render image tiles in parallel

    const StatTypeConfigs &sCfgs = statTypeConfigs;
    OutputBufferSelection outBufSel(bufReg, std::regex(outputRegex), camera->film->filename);

    auto DenoiseLoop = [&](const int nIterations) {
        for (unsigned int i = 1; i <= nIterations; i++) {
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();

            const uint64_t currentSPP = expIterations ? spp << (i - 1) : i * spp;
            const string prefix = outBufSel.GetFilenameStem() + "-" + std::to_string(currentSPP) + "-";

            if (std::filesystem::exists(prefix + "film.pfm")) {
                ReadFile(prefix + "film.pfm", camera->film->buffer);
            }

            vector<string> filenames; // std::string in opencv2.4, but cv::String in 3.0
            {
                const string path = prefix + "*.pfm";
                cv::glob(path, filenames, false);
            }

            for (auto fn : filenames) {
                const string id = fn.substr(fn.find(prefix) + prefix.length(), fn.find(".pfm") - (fn.find(prefix) + prefix.length()));

                // Extract indices and suffix from filename
                int t = -1;
                int b = -1;
                string suffix;
                {
                    std::smatch match;
                    if (std::regex_search(id, match, std::regex("^t([0-9]+)-")))         t = std::stoi(match[1]);
                    if (std::regex_search(id, match, std::regex("^t[0-9]+-b([0-9]+)-"))) b = std::stoi(match[1]);
                    if (std::regex_search(id, match, std::regex("^t[0-9]+-b[0-9]+-(.*)"))) suffix = match[1];
                }

                int tI = -1;
                for (const auto& cfg : sCfgs.configs) {
                    if (cfg.type == t) {
                        tI = cfg.index;
                    }
                }

                if (false) {}
                #define BUF_ELIF(name, type1, type3) else if (suffix == "film-" #name) ReadFile(fn, estimator.t0Bufs.name[tI][b]);
                FILM_BUF_LIST(BUF_ELIF)
                #undef BUF_ELIF
                #define BUF_ELIF(name, type1, type3) else if (suffix == #name) ReadFile(fn, estimator.t1Bufs.name[tI][b]);
                BUF_LIST(BUF_ELIF)
                #undef BUF_ELIF
            }

            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            auto renderTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            std::cout << "Iteration: " << i << std::endl;
            std::cout << "I/O time [ns]: " << renderTime << std::endl;

            {
                begin = chrono::steady_clock::now();
                estimator.Upload();
#if DISABLE_DENOISE_VARS == 0
                estimator.DenoiseVars();
#endif
                estimator.DenoiseMeans();
                if (enableAS)
                    estimator.CalcTargetSPPs(expIterations ? spp << (i - 1) : spp);
                estimator.Download();
                estimator.Synchronize();

                end = chrono::steady_clock::now();
                auto cudaTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
                std::cout << "CUDA time [ns]: " << cudaTime << std::endl;
            }

            begin = chrono::steady_clock::now();
            if (PbrtOptions.writeImages || PbrtOptions.displayImages) {
                outBufSel.PrepareOutput();
                if (PbrtOptions.writeImages) {
                    outBufSel.Write(std::to_string(currentSPP));
                }
                if (PbrtOptions.displayImages) {
                    outBufSel.Display(std::to_string(currentSPP));
                }
            }
            end = chrono::steady_clock::now();
            std::cout << "Output time [ns]: " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << std::endl;
        }
    };

    if (PbrtOptions.warmUp) {
        std::cout << "==== Warm-Up Start ====" << std::endl;
        DenoiseLoop(1);
        std::cout << "==== Warm-Up End ====" << std::endl;
    }

    DenoiseLoop(nIterations);
}

void StatPathIntegrator::Denoise(const Scene &scene) {
    if (enableMultiChannelStats)
        Denoise<Vec3>(scene);
    else
        Denoise<Float>(scene);
}

Spectrum EstimateDirectSMIS(const Interaction &it, const Point2f &uScattering,
                            const Light &light, const Point2f &uLight,
                            const Scene &scene, Sampler &sampler,
                            MemoryArena &arena,
                            MISWinRate &misWinRate,
                            MISTally &misTally,
                            bool handleMedia, bool specular) {
    bool disableLight = misWinRate.light < 0.001f && misTally.light == 0 && (misWinRate.bsdf  >= 0.001f || misTally.bsdf  > 0); // Light strategy never won and BSDF strategy won at least once over light strategy
    bool disableBSDF  = misWinRate.bsdf  < 0.001f && misTally.bsdf  == 0 && (misWinRate.light >= 0.001f || misTally.light > 0); // Vice versa

    BxDFType bsdfFlags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;

    bool gotoLightAndDie = false;

    // Under the condition that this sampling takes place, we would like
    // to know which of the sampling strategies light vs. BSDF is better.
    // Here, the goal is to get the relative performance between the two
    // sampling strategies and not the overall contribution to the image.
    //
    // Therefore, every sampling attempt should count as an attempt, which
    // is later used to normalize the success rate (and not only the times
    // where a weight is actually calculated).
    //
    // The MIS weight is actually just an indicator for the relative
    // usefulness of a sampling strategy. The MIS weights are useful for
    // This purpose because they exhibit a direct linear relationship to
    // the contributions of the samples. If we were to use raw PDF values
    // instead, spikes would overpower the estimate because the average
    // would be preferably constituted by high PDF samples, whereas
    // near-constant PDFs would have much lower averages. In terms of MIS
    // weights, however, near-constant PDFs could also play a large role
    // if the other sampling strategy is near zero in these regions.
    //
    // Another possibility is to calculate the average contributions, but
    // that might be too noisy.

    light:

    if (!disableLight || IsDeltaLight(light.flags)) {
        VisibilityTester visibility;
        Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
        VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
                << wi << ", pdf: " << lightPdf;
        if (lightPdf > 0 && !Li.IsBlack()) {
            // Compute BSDF or phase function's value for light sample
            Spectrum f;
            if (it.IsSurfaceInteraction()) {
                // Evaluate BSDF for light sampling strategy
                const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
                f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                    AbsDot(wi, isect.shading.n);
                scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
                VLOG(2) << "  surf f*dot :" << f << ", scatteringPdf: " << scatteringPdf;
            } else {
                // Evaluate phase function for light sampling strategy
                const MediumInteraction &mi = (const MediumInteraction &)it;
                Float p = mi.phase->p(mi.wo, wi);
                f = Spectrum(p);
                scatteringPdf = p;
                VLOG(2) << "  medium p: " << p;
            }
            if (!f.IsBlack()) {
                // Compute effect of visibility for light source sample
                if (handleMedia) {
                    Li *= visibility.Tr(scene, sampler);
                    VLOG(2) << "  after Tr, Li: " << Li;
                } else {
                  if (!visibility.Unoccluded(scene)) {
                    VLOG(2) << "  shadow ray blocked";
                    Li = Spectrum(0.f);
                  } else
                    VLOG(2) << "  shadow ray unoccluded";
                }

                // Add light's contribution to reflected radiance
                if (!Li.IsBlack()) {
                    if (IsDeltaLight(light.flags))
                        Ld += f * Li / lightPdf;
                    else {
                        Float weight = PowerHeuristic(1, lightPdf, 1, scatteringPdf);

                        Spectrum contr = f * Li / lightPdf;

                        if (weight > .5f)
                            misTally.light++;
                        else {
                            misTally.bsdf++;
                            disableBSDF = false;
                        }

                        if (disableBSDF)
                            Ld += contr;
                        else
                            Ld += contr * weight;
                    }
                }
            }
        }
    }

    if (gotoLightAndDie)
        return Ld;

    // Sample BSDF with multiple importance sampling
    if (!disableBSDF && !IsDeltaLight(light.flags)) {
        Spectrum f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        } else {
            // Sample scattered direction for medium interactions
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
            scatteringPdf;
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) {return Ld;}
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }

            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            bool foundSurfaceInteraction =
                handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                            : scene.Intersect(ray, &lightIsect);

            // Add light contribution from material sampling
            Spectrum Li(0.f);
            if (foundSurfaceInteraction) {
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);

            if (!Li.IsBlack()) {
                Spectrum contr = f * Li * Tr / scatteringPdf;
                
                if (weight > .5f)
                    misTally.bsdf++;
                else {
                    misTally.light++;
                    if (disableLight) {
                        disableLight = false;
                        gotoLightAndDie = true;
                    }
                }

                if (disableLight)
                    Ld += contr;
                else
                    Ld += contr * weight;
            }
        }
    }

    if (gotoLightAndDie)
        goto light;

    return Ld;
}

Spectrum UniformSampleOneLightSMIS(const Interaction &it, const Scene &scene,
                                   MemoryArena &arena, Sampler &sampler,
                                   MISWinRate &misWinRate,
                                   MISTally &misTally,
                                   bool handleMedia, const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    int nLights = int(scene.lights.size());
    if (nLights == 0) return Spectrum(0.f);
    int lightNum;
    Float lightPdf;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sampler.Get1D(), &lightPdf);
        if (lightPdf == 0) return Spectrum(0.f);
    } else {
        lightNum = std::min((int)(sampler.Get1D() * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }
    const std::shared_ptr<Light> &light = scene.lights[lightNum];
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    return EstimateDirectSMIS(it, uScattering, *light, uLight,
                              scene, sampler, arena,
                              misWinRate,
                              misTally,
                              handleMedia,
                              false) / lightPdf;
}

Spectrum StatPathIntegrator::Li(
    const RayDifferential &r,
    const Scene &scene, Sampler &sampler,
    MemoryArena &arena,
    Features &features,
    std::vector<Float> &avgLs,
    std::vector<MISWinRate> &misWinRates,
    std::vector<Spectrum> &Ls,
    std::vector<MISTally> &misTallies,
    unsigned int it
) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    const unsigned char nLs = Ls.size();
    std::vector<Spectrum> betas(nLs, 1.f);
    Spectrum &beta = betas[0];
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;

    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << Ls[0]
                << ", beta = " << beta;

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        const Distribution1D *distrib;

        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                const Spectrum Le = isect.Le(-ray.d);
                for (unsigned char i = 0; i < nLs; i++) // HSTODO: possible optimization: write contribution into only one value and add up afterward (would avoid the iteration every time here)
                    Ls[i] += betas[i] * Le;
                VLOG(2) << "Added Le -> L = " << Ls[0];
            } else {
                for (const auto &light : scene.infiniteLights) {
                    const Spectrum Le = light->Le(ray);
                    for (unsigned char i = 0; i < nLs; i++)
                        Ls[i] += betas[i] * Le;
                }
                VLOG(2) << "Added infinite area lights -> L = " << Ls[0];
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        } else if (bounces == 0) {
            const std::vector<GBufferConfig> &fCfgs = floatGBufConfigs.configs;
            const std::vector<GBufferConfig> &rgbCfgs = rgbGBufConfigs.configs;
            if (fCfgs  [MaterialID].enable) features.floats   [fCfgs  [MaterialID].index] = isect.primitive->GetMaterial()->GetId();
            if (fCfgs  [Depth     ].enable) features.floats   [fCfgs  [Depth     ].index] = ray.tMax;
            if (rgbCfgs[Normal    ].enable) features.spectrums[rgbCfgs[Normal    ].index] = Spectrum::FromRGB(&isect.shading.n.x);
            if (rgbCfgs[Albedo    ].enable) features.spectrums[rgbCfgs[Albedo    ].index] = isect.primitive->GetMaterial()->GetAlbedo(&isect);
        }

        distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0) {
            ++totalPaths;
            Spectrum Ld;
            if (enableSMIS && bounces < statTypeConfigs[MISBSDFWinRate].bounceEnd) {
                Ld = UniformSampleOneLightSMIS(isect, scene, arena,
                                               sampler,
                                               misWinRates[bounces],
                                               misTallies[bounces],
                                               false, distrib);
            }
            else
                Ld = UniformSampleOneLight(isect, scene, arena,
                                           sampler, false, distrib);
            for (unsigned char i = 0; i < nLs; i++) {
                const Spectrum betaLd = betas[i] * Ld;
                if (i == 0) {
                    VLOG(2) << "Sampled direct lighting Ld = " << betaLd;
                    if (betaLd.IsBlack()) ++zeroRadiancePaths;
                }
                CHECK_GE(betaLd.y(), 0.f);
                Ls[i] += betaLd;
            }
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                          BSDF_ALL, &flags);
        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        {
            const Spectrum bsdfBeta = f * AbsDot(wi, isect.shading.n) / pdf;
            for (unsigned char i = 0; i <= bounces && i < nLs; i++)
                betas[i] *= bsdfBeta;
        }
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable (not implemented for albedo)
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            for (unsigned char i = 0; i <= bounces && i < nLs; i++)
                betas[i] *= S / pdf; // HSTODO: test caching

            // Account for the direct subsurface scattering component
            Spectrum Ld;
            if (enableSMIS && bounces < statTypeConfigs[MISBSDFWinRate].bounceEnd) {
                Ld = UniformSampleOneLightSMIS(pi, scene, arena, sampler,
                                               misWinRates[bounces],
                                               misTallies[bounces],
                                               false, lightDistribution->Lookup(pi.p));
            }
            else
                Ld = UniformSampleOneLight(pi, scene, arena, sampler,
                                           false, lightDistribution->Lookup(pi.p));
            for (unsigned char i = 0; i < nLs; i++)
                Ls[i] += betas[i] * Ld;

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                           BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            for (unsigned char i = 0; i <= bounces && i < nLs; i++)
                betas[i] *= f * AbsDot(wi, pi.shading.n) / pdf; // HSTODO: test caching
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        if (bounces > 3) {
            Float avgL = 1.f;

            if (enableACRR && it > 1) {
                unsigned short index = bounces+1;
                if (index >= statTypeConfigs[Radiance].bounceEnd)
                    index = statTypeConfigs[Radiance].bounceEnd - 1;
                avgL = avgLs[index] / avgLs[0];
            }

            const Spectrum rrBeta = beta * etaScale;
            Float survivalRate = rrBeta.MaxComponentValue() * avgL;
            if (survivalRate < rrThreshold) {
                Float q = std::max((Float).05f, 1 - survivalRate);
                if (sampler.Get1D() < q) break;
                for (unsigned char i = 0; i < nLs; i++)
                    betas[i] /= 1 - q;
                DCHECK(!std::isinf(beta.y()));
            }
        }
    }
    ReportValue(pathLength, bounces);

    return Ls[0];
}

StatPathIntegrator *CreateStatPathIntegrator(
    const ParamSet &params,
    const ParamSet &extraParams,
    std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera
) {
    using std::find;
    using std::string;
    using std::vector;

    const int maxDepth = extraParams.FindOneInt("integratormaxdepth", params.FindOneInt("maxdepth", 5));
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    {
        int np;
        const int *pb = params.FindInt("pixelbounds", &np);        
        if (pb) {
            if (np != 4) {
                Error("Expected four values for \"pixelbounds\" parameter. Got %d.", np);
            } else {
                pixelBounds = Intersect(pixelBounds, Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
                if (pixelBounds.Area() == 0)
                    Error("Degenerate \"pixelbounds\" specified.");
            }
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    string lightStrategy = params.FindOneString("lightsamplestrategy", "spatial");

    const unsigned int  nIterations = params.FindOneInt("iterations", 16);
    const bool expIterations = params.FindOneBool("expiterations", true);
    const bool outputExpIterations = params.FindOneBool("outputexpiterations", expIterations);
    const unsigned char nTrackedBounces = extraParams.FindOneInt("integratortrackedbounces", params.FindOneInt("trackedbounces", maxDepth));
    const bool enableMultiChannelStats = params.FindOneBool("multichannelstats", true);
    if (!enableMultiChannelStats) {
        Error("Disabling \"multichannelstats\" is not supported.");
        exit(1);
    }

    const bool enableAS = params.FindOneBool("adaptivesampling", false);
    const bool denoiseFilm = params.FindOneBool("denoisefilm", false);
    const bool enableACRR = params.FindOneBool("acrr", false);
    if (enableACRR) {
        Error("ACRR is untested and might be broken.");
    }
    const bool enableSMIS = params.FindOneBool("smis", false);
    if (enableSMIS) {
        Error("SMIS is not supported.");
        exit(1);
    }
    const bool calcStats = params.FindOneBool("calcstats", false);
    const bool transform = params.FindOneBool("transform", true);
    const unsigned char maxMoment = params.FindOneInt("maxmoment", 5);
    const bool calcStatMCStats = params.FindOneBool("calcstatmcstats", false);
    const bool calcProDenStats = params.FindOneBool("calcprodenstats", false);
    const bool calcMoonStats = params.FindOneBool("calcmoonstats", false);
    const bool calcGBufs = params.FindOneBool("calcgbuffers", false);
    const bool calcItStats = params.FindOneBool("calcitstats", false);
    if (calcItStats) {
        Error("Iteration statistics untested and might be broken.");
    }

    const float meanZValue = params.FindOneFloat("meanzvalue", 1.95996f); // Unused
    if (fabsf(meanZValue - 1.95996f) > Epsilon) {
        Error("The option \"meanzvalue\" has currently no effect.");
    }
    const float varZValue = params.FindOneFloat("varzvalue", 2.80703f);
    const float filterSD = params.FindOneFloat("filtersd", 10.f);
    const unsigned char filterRadius = params.FindOneInt("filterradius", 20);

#if FULL_MULTI_TRANSFORM == 1
    int nLambdas = 0;
    const Float *lambdasPtr = params.FindFloat("lambdas", &nLambdas);
    std::vector<float> lambdas;
    if (lambdasPtr != nullptr && nLambdas > 0) {
        lambdas.assign(lambdasPtr, lambdasPtr + nLambdas);
    } else {
        lambdas = {1.f};
    }
#endif

    // The sequence of the configs must correspond to the indices given by BufferIndex in statpath.h
    GBufferConfigs floatGBufCfgs({
        GBufferConfig("materialid"),
        GBufferConfig("depth")
    });
    GBufferConfigs rgbGBufCfgs({
        GBufferConfig("normal"),
        GBufferConfig("albedo")
    });

    StatTypeConfigs statTypeCfgs;
    statTypeCfgs.configs = vector<StatTypeConfig>(nStatTypeIndices);

    // Set stat configs
    {
        // Radiance
        if (enableAS || denoiseFilm || enableACRR || calcStats || calcStatMCStats || calcProDenStats || calcMoonStats) {
            auto &cfg = statTypeCfgs[Radiance];
            cfg.enable = true;
            cfg.type = Radiance;
            cfg.index = statTypeCfgs.nEnabled++;
            cfg.bounceStart = 0;
#if FULL_MULTI_TRANSFORM == 1
            cfg.bounceEnd = nLambdas ? nLambdas : 1;
#else
            cfg.bounceEnd = enableACRR ? nTrackedBounces : 1;
#endif
            cfg.nBounces = cfg.bounceEnd - cfg.bounceStart;
            if (enableMultiChannelStats) {
                cfg.nChannels = 3;
            }
            if (calcProDenStats || calcMoonStats) {
                // Noisy variance required; calculate up to second moment
                cfg.maxMoment = 2;
            }
            if (calcStatMCStats) {
                // StatMC requires up to the third moment
                cfg.transform = true;
                cfg.maxMoment = 3;
            }
            if (enableAS || denoiseFilm || enableACRR || calcStats) {
                // Denoised variance required; transform samples and calculate up to fourth moment
                cfg.transform = transform; // Hacky override to make timings more accurate!
                cfg.maxMoment = maxMoment; // Hacky override to make timings more accurate!
            }

            // CUDA execution groups
            if (enableAS) {
                cfg.cudaGroups.push_back(CalcTargetSPPGrp);
            }
            if (denoiseFilm || enableACRR) {
                cfg.cudaGroups.push_back(DenoiseMeanGrp);
            }
            if (calcProDenStats) {
                cfg.cudaGroups.push_back(CalcMeanVarGrp);
            }
        }

        // SMIS (TODO: BROKEN! Channel number of one WILL make problems, as the denoiser only handles three-channel buffers.)
        if (enableSMIS) {
            auto &cfgBSDF = statTypeCfgs[MISBSDFWinRate];
            cfgBSDF.enable = true;
            cfgBSDF.type = MISBSDFWinRate;
            cfgBSDF.index = statTypeCfgs.nEnabled++;
            cfgBSDF.bounceStart = 0;
            cfgBSDF.bounceEnd = nTrackedBounces;
            cfgBSDF.nBounces = nTrackedBounces;
            cfgBSDF.nChannels = 1;
            cfgBSDF.transform = true;
            cfgBSDF.maxMoment = 5; // We don't track the fifth moment; 5 is code for covariances.
            cfgBSDF.cudaGroups.push_back(DenoiseMeanGrp);

            auto &cfgLight = statTypeCfgs[MISLightWinRate];
            cfgLight.enable = true;
            cfgLight.type = MISLightWinRate;
            cfgLight.index = statTypeCfgs.nEnabled++;
            cfgLight.bounceStart = 0;
            cfgLight.bounceEnd = nTrackedBounces;
            cfgLight.nBounces = nTrackedBounces; 
            cfgLight.nChannels = 1;
            cfgLight.transform = true;
            cfgLight.maxMoment = 5; // We don't track the fifth moment; 5 is code for covariances.
            cfgLight.cudaGroups.push_back(DenoiseMeanGrp);
        }

        // Iteration statistics (UNTESTED, might be BROKEN!)
        if (calcItStats) {
            auto &cfg  = statTypeCfgs[ItRadiance];
            cfg.enable = true;
            cfg.type = ItRadiance;
            cfg.index = statTypeCfgs.nEnabled++;
            cfg.bounceStart = 0;
            cfg.bounceEnd = 1;
            cfg.nBounces = 1;
            cfg.nChannels = 3;
            cfg.transform = false;
            cfg.maxMoment = 2;
        }
    }

    // Set G-buffer configs
    {
        int nMeanNames = 0;
        int nMeanSDs = 0;
        const string *meanNamesPtr = params.FindString("meanfilterbuffers", &nMeanNames);
        const string *meanNamesPtrEnd = meanNamesPtr + nMeanNames;
        const Float *meanSDsPtr = params.FindFloat ("meanfilterbuffersds", &nMeanSDs);
        if (nMeanNames != nMeanSDs) {
            Error("Size of meanfilterbuffers and meanfilterbuffersds must match.");
            exit(1);
        }
        
        int nVarNames = 0;
        int nVarSDs = 0;
        const string *varNamesPtr = params.FindString("varfilterbuffers", &nVarNames);
        const string *varNamesPtrEnd = varNamesPtr + nVarNames;
        const Float *varSDsPtr = params.FindFloat ("varfilterbuffersds", &nVarSDs);
        if (nVarNames != nVarSDs) {
            Error("Size of varfilterbuffers and varfilterbuffersds must match.");
            exit(1);
        }

        if (enableAS || denoiseFilm || enableACRR  || enableSMIS || calcStats || calcStatMCStats || calcProDenStats || calcMoonStats || calcGBufs) {
            for (auto &i : {vector<unsigned char>{MaterialID, StatMaterialID},
                                                      {Depth,      StatDepth}}) {
                auto &gCfg = floatGBufCfgs[i[0]];
                auto &cfg = statTypeCfgs[i[1]];

                if (const string *namePtr = find(meanNamesPtr, meanNamesPtrEnd, gCfg.name); namePtr != meanNamesPtrEnd) {
                    gCfg.enable = true;
                    cfg.enable = true;
                    if (enableAS || denoiseFilm || enableACRR || enableSMIS) {
                        cfg.enableForMeanFilter = true;
                        cfg.meanFilterSD = meanSDsPtr[namePtr - meanNamesPtr];
                    }
                }
                if (const string *namePtr = find(varNamesPtr, varNamesPtrEnd, gCfg.name); namePtr != varNamesPtrEnd) {
                    gCfg.enable = true;
                    cfg.enable = true;
                    if (enableAS || denoiseFilm || enableACRR || enableSMIS) {
                        cfg.enableForVarFilter = true;
                        cfg.varFilterSD = varSDsPtr[namePtr - varNamesPtr];
                    }
                }

                if (gCfg.enable) {
                    gCfg.index = floatGBufCfgs.nEnabled++;
                }
                if (cfg.enable) {
                    cfg.type = i[1];
                    cfg.index = statTypeCfgs.nEnabled++;
                    cfg.bounceStart = 0;
                    cfg.bounceEnd = 1;
                    cfg.nBounces = 1;
                    cfg.nChannels = 1;
                    cfg.transform = false;
                    cfg.maxMoment = 1;
                    cfg.gBuffer = true;
                    if (calcProDenStats) {
                        cfg.maxMoment = 2;
                        cfg.cudaGroups.push_back(CalcMeanVarGrp);
                    }
                }
            }

            for (auto &i : {vector<unsigned char>{Normal, StatNormal}, {Albedo, StatAlbedo}}) {
                auto &gCfg = rgbGBufCfgs[i[0]];
                auto &cfg = statTypeCfgs[i[1]];

                if (const string *namePtr = find(meanNamesPtr, meanNamesPtrEnd, gCfg.name); namePtr != meanNamesPtrEnd) {
                    gCfg.enable = true;
                    cfg.enable = true;
                    if (enableAS || denoiseFilm || enableACRR || enableSMIS) {
                        cfg.enableForMeanFilter = true;
                        cfg.meanFilterSD = meanSDsPtr[namePtr - meanNamesPtr];
                    }
                }
                if (const string *namePtr = find(varNamesPtr, varNamesPtrEnd, gCfg.name); namePtr != varNamesPtrEnd) {
                    gCfg.enable = true;
                    cfg.enable = true;
                    if (enableAS || denoiseFilm || enableACRR || enableSMIS) {
                        cfg.enableForVarFilter = true;
                        cfg.varFilterSD = varSDsPtr[namePtr - varNamesPtr];
                    }
                }

                if (gCfg.enable) {
                    gCfg.index = rgbGBufCfgs.nEnabled++;
                }
                if (cfg.enable) {
                    cfg.type = i[1];
                    cfg.index = statTypeCfgs.nEnabled++;
                    cfg.bounceStart = 0;
                    cfg.bounceEnd = 1;
                    cfg.nBounces = 1;
                    cfg.nChannels = 3;
                    cfg.transform = false;
                    cfg.maxMoment = 1;
                    cfg.gBuffer = true;
                    if (calcProDenStats) {
                        cfg.maxMoment = 2;
                        cfg.cudaGroups.push_back(CalcMeanVarGrp);
                    }
                }
            }
        }
    }

    string outputRegex = params.FindOneString("outputregex", "film.*");

    return new StatPathIntegrator(
        maxDepth, camera, sampler, pixelBounds,
        nIterations,
        expIterations,
        outputExpIterations,
        enableMultiChannelStats,
        enableAS,
        denoiseFilm,
        enableACRR,
        enableSMIS,
        calcProDenStats,
        calcItStats,
        filterRadius,
        filterSD,
        meanZValue,
        varZValue,
        floatGBufCfgs,
        rgbGBufCfgs,
        statTypeCfgs,
#if FULL_MULTI_TRANSFORM == 1
        lambdas,
#endif
        rrThreshold, lightStrategy,
        outputRegex
    );
}

}  // namespace pbrt
