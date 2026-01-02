# SERGHEI

Simulation Environment for Geomorphology, Hydrodynamics and Ecohydrology in Integrated form

[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.8159947.svg)](https://doi.org/10.5281/zenodo.8159947)
[![BSD3 License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![doi](https://img.shields.io/badge/rsd-serghei-00a3e3.svg)](https://helmholtz.software/software/serghei)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)


# Dependencies

SERGHEI has the following dependencies:

+ [Kokkos](https://github.com/kokkos/kokkos) handles the
parallelization
+ [Parallel NetCDF](https://github.com/Parallel-NetCDF/PnetCDF) writes
output files
+ We use [R](https://www.r-project.org/) scripts for postprocessing (optional)

Both Kokkos and PNetCDF are linked as git submodules in this project. If you are unfamiliar with submodules, you can simply clone this repo together with the submodules with the `git clone --recurse-submodules` command, e.g.,

```
git clone --recurse-submodules https://gitlab.com/serghei-model/serghei.git ./serghei
```

PnetCDF is often available in Linux distributions via package managers, and also as software modules in HPC systems. It is recommended to use these system wide installations, and only fall back on building the PNetCDF from source if there's no other option.

# Building SERGHEI with CMake
The CMake workflow is the recommended approach to build SERGHEI. This assumes you have cloned the Kokkos submodule (the recommended approach is to simply clone with the submodules).

1. Build Kokkos
```
bash buildKokkos [GPU_ARCHITECTURE]
``` 
The `GPU_ARCHITECTURE` argument is optional (it defaults to a shared memory CPU architecture). If providedi, it must be a valid [GPU architecture string as defined by Kokkos](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures), and matching your GPU architecture. Only the `COMPUTE_CAPABILITY` part of the string is required. That is, whereas the Kokkos CMake keyword string is `Kokkos_ARCH_AMPERE80`, only `AMPERE80` needs to be provided as `GPU_ARCHITECTURE` above.

This configures and builds Kokkos with the target backend and architecture.

2. Configure and build SERGHEI
In the root SERGHEI directory

```
cmake -S ./ -DUSE_SYSTEM_KOKKOS=ON [-DSERGHEI_OPTION -DSERGHEI_OPTION ...]
make
```

A recommended option for beginners is to include a minimal set of test cases is 
```
cmake -S ./ -DUSE_SYSTEM_KOKKOS=ON -DSERGHEI_ENABLE_TESTS=ON
```

Note that when builing SERGHEI, there is no backend/architecture selection (this was done in the Kokkos build step).

Build options for SERGHEI are passed to CMake as usual, e.g., `-DSERGHEI_WRITE_HZ=ON`. Read the [documentation on the available build options](https://gitlab.com/serghei-model/serghei/-/wikis/CMake-build-options).

An alternative option is to build Kokkos on the fly:
```
cmake -S ./ -B ./build -DKokkos_ENABLE_MYBACKEND=ON -DKokkos_ARCH_MYARCH=ON
```
where `MYBACKEND` should be replaced by one of the available Kokkos [backends](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#device-backends), with the corresponding [Kokkos architecture keyword](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures) replacing `MYARCH`. This also allows to pass futher [Kokkos compile options](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cmake-keywords) to the build.

**Note**: building SERGHEI only with `make` is deprecated and no longer supported. The documentation for this can be found [here](https://gitlab.com/serghei-model/serghei/-/wikis/User-Guide/Legacy-and-deprecated).

# Running SERGHEI

Once installed, SERGHEI can be invoked by:

```
$ mpirun -n N ./serghei inputDir/ outputDir/ M
```

where

+ `N`: number of MPI tasks (subdomains). This must be in accordance
with the partition chosen in `parameters.input`
+ `inputDir`: directory where the input files are located
+ `outputDir`: directory where the output files will be located
+ `M`: number of threads (OpenMP) or number of GPUs per resource set
(GPUs)

## Examples and test cases

SERGHEI ships with a few minimal tests to check that the build has been succesful. These can be bound in the `bin/tests` directory. These binaries are self documented. Try running them without arguments to get help on how to run them.

For more sophisticated cases, which also illustrate the input files, take a look at the [collection of test cases](https://gitlab.com/serghei-model/serghei-tests).

To run the test case located at `cases/paraboloid2`, execute SERGHEI with 

```
$ mpirun -n 2 /path/to/serghei/bin/serghei ../cases/paraboloid2/ output/ 4
```

Depending on the architecture, this command causes different things to
happen:

1. If the code has been compiled for CPU, this means that it would be
2 subdomains (MPI tasks) parallelized with 4 threads per subdomain
(OpenMP).
2. If the code has been compiled for GPU, this means that it would be
8 subdomains (MPI tasks). The code is run on 2 nodes, each of them
containing 4 GPUs.

Similarly, the use of `mpirun` is conditioned to the execution with
MPI and the corresponding architecture. For example, the code can be
run just using:

```
/path/to/serghei/bin/serghei ./cases/paraboloid2/ output/ 1
```

# Surface-Subsurface Coupling

SERGHEI supports coupled simulations between the shallow water equation (SWE) solver for surface flow and the Richards equation (RE) solver for subsurface flow. The coupling mechanism supports two important features: **multi-resolution coupling** (`dxRatio`) and **asynchronous time stepping** (`dt_ratio`). These features allow for efficient simulations where surface and subsurface domains can use different spatial and temporal resolutions.

## Multi-Resolution Coupling (dxRatio)

### Theory

Multi-resolution coupling allows the subsurface domain to use a coarser grid resolution than the surface domain. This is motivated by the observation that surface flow often requires fine spatial resolution to capture complex topography, small-scale features, and rapid flow dynamics, while subsurface flow may be adequately resolved at coarser scales due to smoother spatial variations in soil properties and hydraulic heads.

The grid resolution ratio `dxRatio` is defined as:
```
dxRatio = Δx_subsurface / Δx_surface ≥ 1
```

When `dxRatio = 1`, both domains use the same grid resolution. When `dxRatio > 1`, the subsurface grid is coarser by a factor of `dxRatio` in each horizontal direction, resulting in a total reduction in cell count by a factor of `dxRatio²`.

### Algorithm

The multi-resolution coupling algorithm involves three main operations:

1. **Grid Dimension Calculation**: 
   - Subsurface grid dimensions: `nx_sub = nx_surf / dxRatio`, `ny_sub = ny_surf / dxRatio`
   - Subsurface grid spacing: `Δx_sub = Δx_surf × dxRatio`
   - Surface domain dimensions must be divisible by `dxRatio` to ensure perfect nesting

2. **Data Aggregation (Surface → Subsurface)**:
   - **Surface elevation (z)**: Averaged over all surface cells within each subsurface cell footprint
   - **Surface water depth (h)**: Averaged over wet surface cells only within each subsurface cell footprint
   - **Rainfall rate**: Averaged over all valid surface cells within each subsurface cell footprint

3. **Data Distribution (Subsurface → Surface)**:
   - **Exchange flux (qss)**: Distributed from subsurface cells to surface cells
   - Flux conservation: Each subsurface cell flux `qss_gw` is distributed to `dxRatio²` surface cells
   - Distribution formula: `qss_surf = qss_gw / dxRatio²` (ensures flux per unit area is consistent)

### Implementation Details

The implementation in `src/GwInit.h` handles initialization and grid setup:

- **Domain dimensions** (lines 96-112): Subsurface grid dimensions are computed based on `dxRatio`
- **Elevation aggregation** (lines 150-189): Surface elevation is aggregated to subsurface cells during initialization
- **Index mapping functions** in `src/GwDomain.h` (lines 58-72): Helper functions map between surface and subsurface cell indices

During time integration in `src/serghei.h`:

- **Rainfall aggregation** (lines 246-279): Rainfall rates are aggregated from fine surface grid to coarse subsurface grid
- **Water depth aggregation** (lines 286-325): Surface water depth is aggregated before each subsurface solve
- **Exchange flux distribution** (lines 376-392): Exchange fluxes computed on subsurface grid are distributed to surface cells

The exchange flux is computed in `src/GwBC.h` for the top boundary condition (direction 6) and stored in `gw.qss_gw` per subsurface cell (line 435). This flux is then distributed to surface cells in `serghei.h`.

### Justification

Multi-resolution coupling provides significant computational savings:
- **Memory reduction**: Subsurface memory scales as `1/dxRatio²` for 2D horizontal grids
- **Computational cost reduction**: Subsurface solve cost scales approximately as `1/dxRatio²` for explicit/implicit time stepping
- **Maintains accuracy**: For many applications, subsurface flow can be adequately resolved at coarser scales, especially when soil properties vary smoothly

The approach maintains mass conservation through proper flux distribution: the total exchange flux from a subsurface cell equals the sum of distributed fluxes to its corresponding surface cells.

### Handling Wetting and Drying in Multi-Resolution Coupling

A critical challenge in multi-resolution coupling occurs when a subsurface cell spans multiple surface cells, some of which are wet and some are dry. This situation is common during wetting and drying cycles. The model handles this case through a carefully designed approach that ensures correct pressure head computation, exchange rate calculation, and mass conservation.

**1. Surface Water Depth Aggregation (Wet-Cell Averaging)**:
   - When aggregating surface water depth `h` to subsurface cells, only **wet surface cells** are considered (those with `h > hmin`)
   - The aggregated value `hs` is computed as the average over wet cells only:
     ```
     hs = (Σ h_wet) / n_wet    if n_wet > 0
     hs = 0.0                   if n_wet = 0 (all cells dry)
     ```
   - This approach ensures that the subsurface boundary condition reflects the actual wetted area, not an average that includes dry cells

**2. Pressure Head at Subsurface Boundary**:
   - The pressure head at the subsurface ghost cell (`gw.h(iGhost,1)`) is set directly from the aggregated `hs` value
   - This represents the effective boundary condition for the subsurface solver
   - When `hs = 0.0` (all surface cells are dry), the boundary condition represents a dry state, allowing for exfiltration if the subsurface pressure is high enough

**3. Exchange Flux Computation**:
   - The exchange flux `qss_gw` is computed per subsurface cell using the aggregated pressure head
   - The flux type (infiltration, exfiltration, or no-flow) is determined based on:
     - Whether the ghost cell has water (`hs > 0`) → infiltration or ponding
     - Whether subsurface pressure exceeds surface elevation → exfiltration
   - The computed flux represents the total exchange for the entire subsurface cell area

**4. Flux Distribution to Surface Cells**:
   - The subsurface exchange flux `qss_gw` is distributed uniformly to **all** surface cells within the subsurface cell footprint, regardless of their wet/dry state
   - Distribution formula: `qss_surf = qss_gw / (dxRatio²)`
   - This uniform distribution ensures mass conservation: the total flux from the subsurface cell equals the sum of distributed fluxes
   - **Important**: Dry surface cells can receive exchange flux, which will increase their water depth and may cause them to become wet

**5. Mass Conservation and Area Representation**:
   - **Pressure head**: Represents the wetted area (average over wet cells only)
   - **Exchange flux**: Represents the total flux for the entire subsurface cell area
   - **Flux distribution**: Ensures mass conservation by uniformly distributing the total flux
   - This approach correctly handles cases where:
     - Partially wetted subsurface cells (some surface cells wet, some dry)
     - Fully wetted subsurface cells (all surface cells wet)
     - Fully dry subsurface cells (all surface cells dry)

The implementation in `src/serghei.h` (lines 295-324) performs the wet-cell averaging during aggregation, and `src/GwBC.h` (lines 290-302) uses this aggregated value to set boundary conditions and compute exchange fluxes. The uniform distribution back to surface cells (lines 377-392 in `serghei.h`) ensures proper mass accounting even when individual surface cells transition between wet and dry states.

## Asynchronous Time Stepping (dt_ratio)

### Theory

Asynchronous time stepping allows the subsurface domain to use a larger time step than the surface domain. This is motivated by the different time scales of surface and subsurface flow processes: surface flow is typically faster and requires smaller time steps for stability, while subsurface flow evolves more slowly and can use larger time steps.

The time step ratio `dt_ratio` is defined as:
```
dt_ratio = Δt_subsurface / Δt_surface ≥ 1
```

When `dt_ratio = 1`, both domains use synchronous time stepping (same time step). When `dt_ratio > 1`, the subsurface domain uses a time step that is `dt_ratio` times larger than the surface time step.

### Algorithm

The asynchronous coupling algorithm operates as follows:

1. **Time Step Calculation**:
   - Surface time step `Δt_sw` is computed based on CFL condition and stability constraints
   - Subsurface time step `Δt_gw = dt_ratio × Δt_sw`, but constrained by `dt_max` upper bound
   - If `Δt_gw > dt_max`, then `Δt_gw = dt_max` and surface time step may be adjusted

2. **Time Integration Loop**:
   - Surface domain advances one step: `t_surf += Δt_sw`
   - Subsurface domain advances multiple steps until it catches up to or exceeds surface time:
     ```
     while (t_subsurf + Δt_gw ≤ t_surf) {
         solve_subsurface();
         t_subsurf += Δt_gw;
     }
     ```
   - Exchange fluxes are computed and applied at the end of each surface time step

3. **Data Exchange Timing**:
   - Surface water depth is aggregated and passed to subsurface before each subsurface solve
   - Exchange fluxes computed by subsurface are distributed to surface domain
   - Surface domain uses these fluxes to update water depth at the end of its time step

### Implementation Details

The implementation in `src/serghei.h` handles asynchronous time stepping:

- **Initialization** (lines 206-221): Initial subsurface time step is set as `gdom.dt = gdom.dt_ratio * dom.dt`
- **Time step update** (lines 494-523): After each surface step, subsurface time step is recalculated based on updated surface time step and `dt_ratio`
- **Synchronous mode** (lines 331-341): When `dt_ratio = 1.0`, subsurface solves once per surface step, with time synchronization
- **Asynchronous mode** (lines 342-358): When `dt_ratio > 1.0`, a while loop executes multiple subsurface steps until subsurface time catches up to surface time

The condition `gdom.etime + gdom.dt <= dom.etime` ensures that a full subsurface time step can be taken before exceeding the current surface time. This maintains temporal alignment between the two domains.

### Justification

Asynchronous time stepping provides significant computational savings:
- **Reduced number of subsurface solves**: With `dt_ratio = 2`, the subsurface solver runs approximately half as many times as the surface solver
- **Maintains stability**: Subsurface time step is constrained by `dt_max` to ensure numerical stability
- **Accurate coupling**: The exchange flux computation uses the most recent surface state, ensuring accurate representation of surface-subsurface interaction

The implementation ensures that:
- Subsurface time never exceeds surface time (avoiding extrapolation)
- Exchange fluxes are always computed using current surface conditions
- Mass conservation is maintained through proper flux accounting

## Combined Multi-Resolution and Asynchronous Coupling

When both `dxRatio > 1` and `dt_ratio > 1` are used together, the computational savings multiply:
- **Memory savings**: `1/dxRatio²`
- **Compute savings per time step**: `1/dxRatio²`
- **Time step savings**: `1/dt_ratio`
- **Total speedup potential**: Approximately `dxRatio² × dt_ratio` for the subsurface component

However, these parameters should be chosen carefully based on:
- **Physical requirements**: Sufficient resolution to capture important flow features
- **Numerical stability**: Time steps must satisfy CFL conditions
- **Accuracy considerations**: Coarser grids and larger time steps may introduce errors

Both `dxRatio` and `dt_ratio` are specified in the `subsurface.input` file and validated during initialization to ensure proper domain nesting and stability.

# Known issues

+ The `clang` compiler may fail to correctly load the OpenMP
library. Thus, if SERGHEI is compiled with `clang`, OpenMP may not be
available.
+ `gcc-10` has trouble compiling Parallel NetCDF and throws a type
  mismatch errors. The errors can be turned into warnings by passing
  ```
  FCFLAGS="-fallow-argument-mismatch" FFLAGS="-fallow-argument-mismatch"
  ```
  to `configure` and `make`. See [this github issue](https://github.com/Unidata/netcdf-fortran/issues/212).

# How to cite 
Please cite the software using the corresponding [SERGHEI Zenodo DOI](https://doi.org/10.5281/zenodo.8159947), and if necessary with the specific release DOI.

You can refer to the [SERGHEI-SWE paper](https://gmd.copernicus.org/articles/16/977/2023/) for the shallow water module SERGHEI-SWE.
```
@Article{Caviedes2023-serghei-swe,
AUTHOR = {Caviedes-Voulli\`eme, D. and Morales-Hern\'andez, M. and Norman, M. R. and \"Ozgen-Xian, I.},
TITLE = {SERGHEI (SERGHEI-SWE) v1.0: a performance portable high-performance parallel-computing shallow-water solver for hydrology and environmental hydraulics},
JOURNAL = {Geoscientific Model Development Discussions},
VOLUME = {16}
YEAR = {2023},
PAGES = {977--1008},
DOI = {10.5194/gmd-16-977-2023}
}
```

The Richards solver (RE module) is presented in the [SERGHEI-RE paper](https://gmd.copernicus.org/articles/18/547/2025/).
```
@Article{Li2025-serghei-re,
  author           = {Li, Zhi and Rickert, Gregor and Zheng, Na and Zhang, Zhibo and Özgen-Xian, Ilhan and Caviedes-Voullième, Daniel},
  journal          = {Geoscientific Model Development},
  title            = {SERGHEI v2.0: introducing a performance-portable, high-performance, three-dimensional variably saturated subsurface flow solver (SERGHEI-RE)},
  year             = {2025},
  month            = jan,
  number           = {2},
  pages            = {547--562},
  volume           = {18},
  doi              = {10.5194/gmd-18-547-2025},
}
```

The Lagrangian particle module is presented in the [SERGHEI-LPT paper](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-722/egusphere-2025-722.pdfhttps://gmd.copernicus.org/articles/18/7399/2025/).
```
@Article{Valles2025-serghei-lpt,
  author    = {Vallés, Pablo and Morales-Hernández, Mario and Roeber, Volker and García-Navarro, Pilar and Caviedes-Voullième, Daniel},
  journal   = {Geoscientific Model Development},
  title     = {SERGHEI v2.1: a Lagrangian model for passive particle transport using a two-dimensional shallow water model (SERGHEI-LPT)},
  year      = {2025},
  issn      = {1991-9603},
  month     = oct,
  number    = {20},
  pages     = {7399--7416},
  volume    = {18},
  doi       = {10.5194/gmd-18-7399-2025},
}
```