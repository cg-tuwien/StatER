# Statistical Error Reduction for Monte Carlo Rendering

![Comparison of different denoising methods on the "Veach Egg" scene.](https://users.cg.tuwien.ac.at/~hiroyuki/StatER/static/images/veach-bidir-zoomplot.png)

This repository contains our implementation of our research paper ["Statistical Error Reduction for Monte Carlo Rendering" [Sakai et al. 2025]](https://www.cg.tuwien.ac.at/StatER).
Our implementation is based on [StatMC](https://github.com/cg-tuwien/StatMC.git), itself built on [pbrt-v3](https://github.com/mmp/pbrt-v3), and offers the following:

- a new unidirectional path-tracing integrator `StatPathIntegrator`, which tracks the required statistics for multi-transform denoising, variance denoising, adaptive sampling, and linear explained-variance correction, described in our paper,
- [OpenCV](https://opencv.org/) integration for buffer management and CUDA abstraction,
- support for our CUDA denoiser implemented on top of OpenCV (hosted on a [separate repository](https://github.com/cg-tuwien/StatER-opencv_contrib.git)),
- albedo lookup tables for faster and more accurate albedo queries (compared to pbrt-v3's own `rho()` function), and
- support for the [tev image viewer](https://github.com/Tom94/tev).

The extensions are mostly implemented in [`src/statistics/`](src/statistics) and [`src/display/`](src/display).

With the focus on research, this code is not intended for production.
We appreciate your feedback, questions, and reports of any issues you encounter; feel free to [contact us](https://www.cg.tuwien.ac.at/staff/HiroyukiSakai)!


## Build Instructions

### Prerequisites

We developed our denoiser using [CUDA 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive) and [OpenCV 4.8.1](https://github.com/opencv/opencv/releases/tag/4.8.1).
Note that [later CUDA versions (>= 12.4) are incompatible with OpenCV 4.8.1](https://github.com/opencv/opencv_contrib/issues/3690).

For reproducing the results presented in our paper, we recommend using Clang 16.0.6 on Ubuntu 22.04 LTS or Linux Mint 20 (as used for the paper).
While we have successfully tested GCC 11.4.0, it produces slightly different results (mostly due to differences in random number generation).

In the following, we describe two alternative ways to build our code: an automatic approach tested for Ubuntu 22.04 LTS and a manual approach, which we recommend if you want to retrace the steps of the build process or use another operating system.

### Automatic Building (for a fresh Ubuntu 22.04 LTS install)

1.  Clone this repository (OpenCV will be cloned automatically as a submodule):
    ```bash
    git clone --recursive https://github.com/cg-tuwien/StatER.git
    cd StatER/
    ```

2.  Install dependencies:
    ```bash
    sudo ./scripts/_install-dependencies.sh
    ```

3.  Build our code:
    ```bash
    ./scripts/_build.sh
    ```
    Our version of the pbrt binary should now be located in `build-asm+mcl/pbrt-v3/`.

### Docker Build Instructions

We have prepared files required to run our project within a Docker container [here](docker).

### Manual Building

Skip this if you have used the automatic approach [above](#automatic-building-for-a-fresh-ubuntu-2204-lts-install).

1.  Clone this repository (OpenCV will be cloned automatically as a submodule):
    ```bash
    git clone --recursive https://github.com/cg-tuwien/StatER.git
    cd StatER/
    ```

2.  Make sure that CUDA 12.3, as well as the packages `cmake`, `clang`, `libstdc++-12-dev`, and `zlib1g-dev`, are installed.
    The packages may vary depending on the operating system.

#### Building OpenCV

1.  In the root directory of the repository, create the directories for building OpenCV:
    ```bash
    mkdir build-asm+mcl
    cd build-asm+mcl/
    mkdir opencv
    cd opencv/
    ```

2.  Build OpenCV according to [these instructions](https://github.com/cg-tuwien/StatER-opencv_contrib#building-opencv) using the directories `../../src/ext/opencv` and `../../src/ext/opencv_contrib` for `<opencv_source_directory>` and `<opencv_contrib>`.
    We build OpenCV and pbrt separately to have better control over the individual builds.

3.  Change to the `build-asm+mcl/` directory for building pbrt in the next step:
    ```bash
    cd ../
    ```

#### Building pbrt-v3

1.  In the [previously created](#building-opencv) `build-asm+mcl/` directory, create the build directory for pbrt-v3:
    ```bash
    mkdir pbrt-v3
    cd pbrt-v3/
    ```

2.  Create the CMake buildsystem:
    ```bash
    cmake \
    -DOpenCV_BUILD_DIR_PREFIX="build-asm+mcl" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -march=native" \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native" \
    ../../
    ```

3.  Build pbrt-v3
    ```bash
    make -j 16
    ```


## Usage

The directory [`scenes/`](scenes) contains configurations and scene description files for reproducing the results in our paper.
We do not include the complete scenes; they can be downloaded by running the [`scripts/_download-scenes.sh`](scripts/_download-scenes.sh) shell script from the root directory of the repository:
```bash
./scripts/_download-scenes.sh
```
Note that the scenes are hosted on creator's or licensee's websites and are subject to being changed or taken down without prior notice: we are not responsible for (the availability of) the content hosted on other websites (except for [pbrt-v3's measure-one scene](https://www.pbrt.org/scenes-v3), which we rehost for your convenience).

Once the scenes are downloaded, you can reproduce the results from our paper by simply running the shell scripts for the corresponding figures in the [`scripts/`](scripts) directory. E.g.,
```bash
./scripts/1-veach-bidir.sh
```
Result images will be saved to the `out/` directory.
The prefixes (here, `1`) refer to the figure number in the [paper](https://www.cg.tuwien.ac.at/research/publications/2025/sakai-2025-stater/sakai-2025-stater-paper.pdf) and [supplementary document](https://www.cg.tuwien.ac.at/research/publications/2025/sakai-2025-stater/sakai-2025-stater-Supplementary%20Document.pdf).
The images generated by these scripts should generally match our paper results, which we have published in lossless format [here](https://doi.org/10.48436/kny07-dvd93) (including references).
Image filenames indicate the number of samples per pixel (SPP) used for rendering.
To reproduce a figure, pbrt must be run up to the corresponding SPP value.

By default, the generated images correspond to the "MCL" variant from the paper.
To enable adaptive sampling ("ASM+MCL"), pass the `as` option to the script:
```bash
./scripts/1-veach-bidir.sh as
```

You can also build alternative variants of our code:
```bash
./scripts/alternate-builds/build-asm+m.sh   # "[ASM+]M"    built into build-asm+m/
./scripts/alternate-builds/build-asm+mc.sh  # "[ASM+]MC"   built into build-asm+mc/
./scripts/alternate-builds/build-asn+mcl.sh # "[ASN+]MCL"  built into build-asn+mcl/
./scripts/alternate-builds/build-mt.sh      # "MT{1-5}"    built into build-mt/
./scripts/alternate-builds/build-mt-jb.sh   # "MT{1–5}-JB" built into build-mt-jb/
```
Refer to our paper and supplementary document for detailed explanations of these variants.

Once an alternate build has been created, you can specify it as an additional parameter.
Here are a few examples:
```bash
./scripts/12-furball.sh as     build-asm-mc # ASM-MC
./scripts/13-car.sh     ""     build-asm-m  # M
./scripts/S6-lamp.sh    mt1    build-mt     # MT1 (MCL)
./scripts/S8-house.sh   mt5-as build-mt-jb  # MT5-JB (ASM-MCL)
```

Feel free to experiment with different scenes and configurations.
For a starting point, refer to the [quick reference](#quick-reference) further below.


## Notes on Reproducing Our Results

We reproduced the results in our paper by following the instructions on this page using two machines:

1. a desktop PC equipped with an AMD Ryzen 9 5950X CPU, an NVIDIA RTX 3080 Ti GPU, and running Linux Mint 20, as well as
2. a virtual machine hosted on a computing cluster equipped with an AMD EPYC 7413 CPU, an NVIDIA A40 GPU, and running Ubuntu 22.04 LTS.

**Despite following our instructions for reproducibility, differences in hardware, operating systems, compilers, and other factors may still lead to minor variations in the generated images.**

### Comparisons

In this repository, we do not include implementations of the neural denoisers we compared against in our paper.
For those comparisons, we have used the commits linked here:

- [NVIDIA's OptiX](https://github.com/DeclanRussell/NvidiaAIDenoiser/tree/86e59a46aa12470b8e420c84a1dd5446109e5e3c)
- [Intel's OIDN](https://github.com/RenderKit/oidn/tree/49a0a43bdab45fbb6bec7b084d743387634da9c3)
- [ProDen [Firmino et al. 2022]](https://github.com/ArthurFirmino/progressive-denoising/tree/3500970acb321c07278057ff6413b57823f365a8)

For [Moon et al.'s confidence-interval approach [2013]](https://doi.org/10.1111/cgf.12004) and [StatMC [Sakai et al. 2024]](https://www.cg.tuwien.ac.at/StatMC), we used the implementation and instructions provided in the [StatMC repository](https://github.com/cg-tuwien/StatMC).

### Additional Steps for Reproducing Specific Figures

With the scripts provided [above](#usage), it is possible to generate all main results of our publication.
If you wish to reproduce the results of our ablation studies, we have documented the required compilation settings [here](https://github.com/cg-tuwien/StatER-opencv_contrib/blob/master/modules/cudaimgproc/src/cuda/stat_denoiser.cu#L83).
Please also note the comments for the individual compilation flags [here](https://github.com/cg-tuwien/StatER-opencv_contrib/blob/master/modules/cudaimgproc/src/cuda/stat_denoiser.cu#L22).
If you need assistance with these steps, we are happy to help—please [contact us](https://www.cg.tuwien.ac.at/staff/HiroyukiSakai)!


## Quick Reference

### Additional Command-Line Options

Our version of the pbrt executable extends the original with the following options:

| Option | Description |
| - | - |
| `--writeimages` | Write images to disk. |
| `--displayserver <socket>` | Write images to the specified network socket (format `<IP address>:<port number>`). |
| `--baseseed <num>` | Use the specified base seed for `RandomSampler`. |
| `--denoise` | Skip rendering and use prerendered images on disk instead (useful for performing multiple denoising passes without rerendering). |
| `--warmup` | Perform a warm-up iteration (useful for consistent performance measurements). |

### Extended Scene Description Format

Most of the configuration is done in the scene description files.
In the following, we provide an overview over our extensions to the original scene description format.

#### StatPathIntegrator Options

We have extended the original format with options for our `StatPathIntegrator`.
To illustrate, here is an example configuration, which utilizes all relevant options:

```
Integrator "statpath"
  "integer  maxdepth"            [65]
  "bool     expiterations"       ["true"]
  "bool     outputexpiterations" ["true"]
  "integer  iterations"          [13]

  "bool     adaptivesampling" ["false"]
  "bool     denoisefilm"      ["true"]
  "bool     calcstats"        ["false"]
  "bool     calcstatmcstats"  ["false"]
  "bool     calcprodenstats"  ["false"]
  "bool     calcmoonstats"    ["false"]
  "bool     calcgbuffers"     ["false"]

  "float    varcizvalue"  [2.80703]
  "float    filtersd"     [10]
  "integer  filterradius" [20]

  "string   varfilterbuffers"    ["albedo" "normal"]
  "float    varfilterbuffersds"  [0.02 0.1]
  "string   meanfilterbuffers"   ["albedo" "normal"]
  "float    meanfilterbuffersds" [0.02 0.1]

  "string   outputregex" ["film|filmD"]
```

The following table summarizes all available options for our `StatPathIntegrator`:

| Type | Name | Default Value | Description |
| - | - | - | - |
| integer | `maxdepth` | `5` | Same as in the [original](https://pbrt.org/fileformat-v3#integrators): "Maximum length of a light-carrying path sampled by the integrator." |
| integer[4] | `pixelbounds` | (Entire image) | Same as in the [original](https://pbrt.org/fileformat-v3#integrators): "Subset of image to sample during rendering; in order, values given specify the starting and ending x coordinates and then starting and ending y coordinates. (This functionality is primarily useful for narrowing down to a few pixels for debugging.)" |
| float | `rrthreshold` | `1` | Same as in the [original](https://pbrt.org/fileformat-v3#integrators): "Determines when Russian roulette is applied to paths: when the maximum spectral component of the path contribution falls beneath this value, Russian roulette starts to be used." |
| string | `lightsamplestrategy` | `"spatial"` | Same as in the [original](https://pbrt.org/fileformat-v3#integrators): "Technique used for sampling light sources. Options include 'uniform', which samples all light sources uniformly, 'power', which samples light sources according to their emitted power, and 'spatial', which computes light contributions in regions of the scene and samples from a related distribution." |
| bool | `expiterations` | `true` | Our integrator operates iteratively, with each iteration comprising a rendering and denoising pass. `true` enables exponential growth of the total number of samples per pixel for rendering (e.g., 4, 16, 64, etc.), while `false` enables linear growth (e.g., 4, 8, 12, etc.). The (initial) number of samples per pixel (4 in the examples) is specified via the `pixelsamples` option of the `Sampler`. |
| bool | `outputexpiterations` | `expiterations` | `true` only outputs images for exponential iterations (e.g., 4, 16, 64, etc.), even if `expiterations` is `false`. |
| integer | `iterations` | `16` | Total number of iterations |
| bool | `adaptivesampling` | `false` | `true` enables adaptive sampling. |
| bool | `denoisefilm` | `false` | `true` enables denoising of the rendered image. |
| bool | `calcstats` | `false` | `true` enables the calculation of G-buffers and statistics required by our denoiser. Use this option to precompute everything required for denoising without performing the denoising itself. |
| bool | `calcstatmcstats` | `false` | `true` enables the calculation of G-buffers and statistics required by [StatMC [Sakai et al. 2024]](https://www.cg.tuwien.ac.at/StatMC). |
| bool | `calcprodenstats` | `false` | `true` enables the calculation of G-buffers and statistics required by [ProDen [Firmino et al. 2022]](https://doi.org/10.1111/cgf.14454). |
| bool | `calcmoonstats` | `false` | `true` enables the calculation of G-buffers and statistics required by [Moon et al.'s confidence-interval approach [2013]](https://doi.org/10.1111/cgf.12004). |
| bool | `calcgbuffers` | `false` | `true` enables the calculation of G-buffers required by [NVIDIA's OptiX denoiser](https://developer.nvidia.com/optix-denoiser) and [Intel's OIDN](https://www.openimagedenoise.org/). |
| float | `filtersd` | `10.0` | Standard deviation of the denoising filter kernel |
| integer | `filterradius` | `20` | Radius of the denoising filter kernel (limiting the kernel to a finite number of pixels) |
| string[] | `varfilterbuffers` | `["albedo" "normal"]` | G-buffers for variance denoising; possible options are `materialid`, `depth`, `normal`, `albedo`. `materialid` refers to unique numbers that are assigned to different materials by the renderer. For fair comparisons, we used albedos and normals only. |
| float[] | `varfilterbuffersds` | `[0.02 0.1]` | Standard deviations associated with the G-buffers for variance denoising ($\sigma_r$ as described in [one of the original joint-bilateral-filter papers](https://hhoppe.com/flash.pdf)); lower values make the filter more discriminative. |
| string[] | `meanfilterbuffers` | `["albedo" "normal"]` | G-buffers for mean denoising; possible options are `materialid`, `depth`, `normal`, `albedo`. `materialid` refers to unique numbers that are assigned to different materials by the renderer. For fair comparisons, we used albedos and normals only. |
| float[] | `meanfilterbuffersds` | `[0.02 0.1]` | Standard deviations associated with the G-buffers for mean denoising ($\sigma_r$ as described in [one of the original joint-bilateral-filter papers](https://hhoppe.com/flash.pdf)); lower values make the filter more discriminative. |
| string | `outputregex` | `film.*` | Regular expression specifying the buffers to output (to disk or network socket as determined by the `--writeimages` and `--displayserver` [command-line options](#additional-command-line-options)); buffers whose unique names match the specified regular expression are output. This way of specification provides a high degree of flexibility, e.g., `film.*\|t0-.*` matches all buffers whose name begins with `film` or `t0-`. We provide a complete list of buffers [below](#buffer-system). |

#### Including Files

Similarly to [pbrt-v4](https://pbrt.org/fileformat-v4#include-import), our scene description format supports file includes:
```
Include "../_active.pbrt"
```
We have implemented this feature to quickly switch between rendering and denoising configurations without changing the scene description file itself.
We provide the following configurations in the [`scenes/`](scenes) directory:

| Configuration File | Description |
| - | - |
| [`render-denoise*.pbrt`](scenes/render-denoise.pbrt) | Render and denoise using our denoiser |

As can be seen in the scripts for reproducing the figures, a configuration file is activated by overwriting `scenes/_active.pbrt` with it.
Once a configuration is activated, pbrt can be run normally, supplying the desired scene description file as parameter, e.g.,:
```bash
./pbrt ../../scenes/bathroom/scene-stat.pbrt
```

### Buffer System

This section provides an overview of the buffer system in `StatPathIntegrator`, which enables working with various inputs for and outputs from our denoiser.
Note that a more detailed description goes beyond the scope of this overview; for more details, refer to the code itself or [contact us](https://www.cg.tuwien.ac.at/staff/HiroyukiSakai)!

There are five **buffer types**:

| Index | Name | Box-Cox Transformation | Description |
| - | - | - | - |
| 0 | `Radiance` | applied | Monte Carlo radiance estimate |
| 1 | `StatMaterialID` | not applied | Material ID G-buffer |
| 2 | `StatDepth` | not applied | Depth G-buffer |
| 3 | `StatNormal` | not applied | Normal G-buffer |
| 4 | `StatAlbedo` | not applied | Albedo G-buffer |

The Box-Cox transformation of radiance samples in our multi-transform setting makes our approach more robust to non-normality; details can be found in [our paper](https://www.cg.tuwien.ac.at/StatER).
These types are enabled as required by the [`StatPathIntegrator` configuration](#extended-scene-description-format).
In particular, `filterbuffers` determines the enabled G-buffer types.

Each **enabled** type is assigned a consecutively numbered ID (for performance reasons).
For instance, if denoising, normals and albedos are enabled, IDs would be assigned as follows:

| ID | Name |
| - | - |
| 0 | `Radiance` |
| 1 | `StatNormal` |
| 2 | `StatAlbedo` |

For each enabled type, a set of buffers is created.
Based on the configuration and these rules, the following buffers are potentially created:

| Type | Name | Description |
| - | - | - |
| RGB | `film` | Noisy rendered image |
| RGB | `filmD` | Denoised rendered image |
| RGB | `tX-b0-mean` | Sample mean of transformed samples for type `X` |
| RGB | `tX-b0-m2` | Sum of squared deviations of transformed samples for type `X` (division by the number of samples gives the second sample central moment) |
| RGB | `tX-b0-m3` | Sum of cubed deviations of transformed samples for type `X` (division by the number of samples gives the third sample central moment) |
| RGB | `tX-b0-m4` | Sum of fourth powers of deviations of transformed samples for type `X` (division by the number of samples gives the fourth sample central moment) |
| RGB | `tX-b0-c2X` | Cumulative covariance of transformed samples and pixel positions along x direction for type `X` (division by the number of samples minus one gives the Bessel-corrected covariance) |
| RGB | `tX-b0-c2Y` | Cumulative covariance of transformed samples and pixel positions along y direction for type `X` (division by the number of samples minus one gives the Bessel-corrected covariance) |
| RGB | `tX-b0-m1` | Total absolute deviation of transformed samples for type `X` (division by the number of samples gives the mean absolute deviation) |
| RGB | `tX-b0-jb` | Jarque–Bera score of transformed samples for type `X` |
| RGB | `tX-b0-lnS2` | Bonett's `ln(s2)` of transformed samples for type `X` |
| RGB | `tX-b0-se2` | Bonett's squared standard error `se^2` of transformed samples for type `X` |
| RGB | `tX-b0-m2C` | LEV-corrected sum of squared deviations of transformed samples for type `X` (division by the number of samples gives the second sample central moment) |
| RGB | `tX-b0-varD` | Denoised variance of transformed samples for type `X` |
| RGB | `tX-b0-varCD` | Denoised LEV-corrected variance of transformed samples for type `X` |
| RGB | `tX-b0-discr` | Welch discriminator of transformed samples for type `X` (used to cache per-pixel discriminators for mean denoising) |

All `tX-b0-*` buffers above are also available for untransformed samples with the prefix `tX-b0-film-*`.
In addition, the following buffers are additionally used for untransformed samples:

| Type | Name | Description |
| - | - | - |
| integer | `tX-b0-film-n` | Number of samples taken for type `X` | 
| RGB | `tX-b0-film-meanVar` | Sample variance of untransformed samples for type `X` |
| RGB | `tX-b0-film-meanD` | Denoised mean of untransformed samples for type `X` |

Depending on the configuration, some buffers may be disabled, and GPU-computed buffers must be [downloaded to the CPU](src/statistics/estimator.cpp#L355) before they can be output.
For `mt[-jb]` builds, only non-`film` buffers are used for storing the transformed statistics, with `b1` to `b8` specifying the index of the transformation.
Covering the effects of every possible configuration is beyond the scope of this README.
For more information, please check the code directly or [reach out to us](https://www.cg.tuwien.ac.at/staff/HiroyukiSakai).
The [`outputregex` option](#extended-scene-description-format) provides a convenient way to select output buffers.

### Limitations

`StatPathIntegrator` supports the `BoxFilter` only.


## Acknowledgments

We thank [Thomas Auzinger](https://auzinger.name/) for providing LaTeX plugins, [José Dias Curto](https://ciencia.iscte-iul.pt/authors/jose-joaquim-dias-curto/cv) for support with confidence intervals, and [Markus Schütz](https://www.cg.tuwien.ac.at/staff/MarkusSch%C3%BCtz) for assistance with the CUDA implementation. We also thank the creators of the scenes we used: [Benedikt Bitterli](https://benedikt-bitterli.me/) for ["Veach, Bidir Room"](https://benedikt-bitterli.me/resources/) (Figs. 1, S12), ["Cornell Box"](https://benedikt-bitterli.me/resources/) (Fig. 2), and ["Fur Ball"](https://benedikt-bitterli.me/resources/) (Fig. 12); [Jay-Artist](https://blendswap.com/profile/1574) for ["Country Kitchen"](https://blendswap.com/blend/5156) (Figs. 4, 5, 10, S2, S10, S16); [Mareck](https://www.blendswap.com/profile/53736) for ["Contemporary Bathroom"](https://blendswap.com/blend/13303) (Figs. 7, 14, S13); [thecali](https://blendswap.com/profile/215428) for ["4060.b Spaceship"](https://blendswap.com/blend/13489) (Fig. 9); [piopis](https://blendswap.com/profile/10550) for ["Old Vintage Car"](https://blendswap.com/blend/14205) (Fig. 13); [Cem Yuksel](http://www.cemyuksel.com/) for ["Straight Hair"](http://www.cemyuksel.com/research/hairmodels/) (Fig. S4) and ["Curly Hair"](http://www.cemyuksel.com/research/hairmodels/) (Fig. S5); [UP3D](https://blendswap.com/profile/4758) for ["Little Lamp"](https://blendswap.com/blend/6885) (Fig. S6); [axel](https://blendswap.com/profile/945886) for ["Glass of Water"](https://blendswap.com/blend/3915) (Fig. S7); [MrChimp2313](https://blendswap.com/profile/10069) for ["Victorian Style House"](https://blendswap.com/blend/12687) (Fig. S8); [NovaAshbell](https://blendswap.com/profile/135376) for ["Japanese Classroom"](https://blendswap.com/blend/13632) (Fig. S9); and [Beeple](https://www.beeple-crap.com/) for ["Zero-Day"](https://www.beeple-crap.com/resources) (Fig. S11). Statistical simulation studies were conducted using the [Austrian Scientific Computing (ASC)](https://asc.ac.at/) infrastructure. This work has been funded by the Vienna Science and Technology Fund (WWTF) [Grant ID: 1047379/ICT22028]. This research was funded in whole or in part by the Austrian Science Fund (FWF) [[10.55776/F77]](https://doi.org/10.55776/F77). For open-access purposes, the author has applied a CC BY public copyright license to any author-accepted manuscript version arising from this submission. The authors acknowledge TU Wien Bibliothek for financial support through its Open Access Funding Programme.

