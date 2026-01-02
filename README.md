# SERGHEI (Experimental Fork)

**⚠️ NON-OFFICIAL EXPERIMENTAL VERSION ⚠️**

This is a non-official, experimental fork of SERGHEI (Simulation Environment for Geomorphology, Hydrodynamics and Ecohydrology in Integrated form) used for testing and developing experimental new algorithms and model capabilities. 

**This repository is for research and development purposes only. For the official, stable version of SERGHEI, please visit the [official GitLab repository](https://gitlab.com/serghei-model/serghei).**

## Experimental Features

This fork includes experimental implementations of:
- **Multi-resolution coupling** (`dxRatio`) between surface and subsurface domains
- **Asynchronous time stepping** (`dt_ratio`) for coupled SWE-RE simulations
- Enhanced handling of wetting and drying in multi-resolution coupling scenarios

These features are under active development and testing. Use at your own risk.


# Dependencies

SERGHEI has the following dependencies:

+ [Kokkos](https://github.com/kokkos/kokkos) handles the
parallelization
+ [Parallel NetCDF](https://github.com/Parallel-NetCDF/PnetCDF) writes
output files
+ We use [R](https://www.r-project.org/) scripts for postprocessing (optional)

Both Kokkos and PNetCDF are linked as git submodules in this project. If you are unfamiliar with submodules, you can clone this repo together with the submodules using:

```
git clone --recurse-submodules <repository-url> ./serghei-flex
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

Note that when building SERGHEI, there is no backend/architecture selection (this was done in the Kokkos build step).

Build options for SERGHEI are passed to CMake as usual, e.g., `-DSERGHEI_WRITE_HZ=ON`. For details on available build options, refer to the official SERGHEI documentation or the CMakeLists.txt file.

An alternative option is to build Kokkos on the fly:
```
cmake -S ./ -B ./build -DKokkos_ENABLE_MYBACKEND=ON -DKokkos_ARCH_MYARCH=ON
```
where `MYBACKEND` should be replaced by one of the available Kokkos backends, with the corresponding Kokkos architecture keyword replacing `MYARCH`.

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

This experimental fork may include test cases in the `bin/tests` directory. These binaries are self-documented. Try running them without arguments to get help on how to run them.

Example test cases may be available in the `cases/` directory. To run a test case, execute SERGHEI with:

```
$ mpirun -n N /path/to/serghei/bin/serghei <inputDir> <outputDir> M
```

where `N` is the number of MPI tasks, `M` is the number of threads (OpenMP) or GPUs per resource set, `inputDir` is the directory containing input files, and `outputDir` is where output files will be written.

For the official test cases and examples, please refer to the [official SERGHEI repository](https://gitlab.com/serghei-model/serghei-tests).

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

The implementation in `src/GwInit.h` handles initialization and grid setup. Subsurface grid dimensions are computed based on `dxRatio`:

```cpp
// Set subsurface domain dimensions based on dxRatio
if (gdom.dxRatio == 1) {
    // Same resolution: use existing code path
    gdom.nx = dom.nx;
    gdom.ny = dom.ny;
    gdom.nx_glob = dom.nx_glob;
    gdom.ny_glob = dom.ny_glob;
    gdom.dx = dom.dxConst;
    gdom.dy = dom.dxConst;
} else {
    // Multi-resolution: subsurface is coarser
    gdom.nx = dom.nx / gdom.dxRatio;
    gdom.ny = dom.ny / gdom.dxRatio;
    gdom.nx_glob = dom.nx_glob / gdom.dxRatio;
    gdom.ny_glob = dom.ny_glob / gdom.dxRatio;
    gdom.dx = dom.dxConst * gdom.dxRatio;
    gdom.dy = dom.dxConst * gdom.dxRatio;
}
```

Surface elevation is aggregated to subsurface cells during initialization:

```cpp
// Aggregate surface elevation for subsurface cell
real z_sum = 0.0;
int n_valid = 0;
if (gdom.dxRatio == 1) {
    // Same resolution: direct mapping
    iGlobSW = packIndicesUniformGrid(dom.ny + 2*hc, dom.nx + 2*hc, jj, ii);
    if (!state.isnodata(iGlobSW)) {
        z_sum = state.z(iGlobSW);
        n_valid = 1;
    }
} else {
    // Multi-resolution: aggregate over surface cells
    int i_sw_start = (par.i_beg + ii - hc) * gdom.dxRatio;
    int j_sw_start = (par.j_beg + jj - hc) * gdom.dxRatio;
    int i_sw_end = i_sw_start + gdom.dxRatio;
    int j_sw_end = j_sw_start + gdom.dxRatio;
    
    // Clamp to valid surface domain range and aggregate
    for (int j_sw = j_sw_start; j_sw < j_sw_end; j_sw++) {
        for (int i_sw = i_sw_start; i_sw < i_sw_end; i_sw++) {
            // ... get local surface cell index and accumulate z_sum ...
            if (!state.isnodata(iGlobSW_local)) {
                z_sum += state.z(iGlobSW_local);
                n_valid++;
            }
        }
    }
}
```

Helper functions in `src/GwDomain.h` map between surface and subsurface cell indices:

```cpp
// Get subsurface cell index from surface cell indices
KOKKOS_INLINE_FUNCTION int getGwIndexFromSw(int i_sw, int j_sw, int nx_sw) const {
    int i_gw = i_sw / dxRatio;
    int j_gw = j_sw / dxRatio;
    return j_gw * nx + i_gw;
}

// Get surface cell index range within a subsurface cell
KOKKOS_INLINE_FUNCTION void getSwIndicesInGw(int i_gw, int j_gw, 
                                              int &i_sw_start, int &j_sw_start,
                                              int &i_sw_end, int &j_sw_end) const {
    i_sw_start = i_gw * dxRatio;
    j_sw_start = j_gw * dxRatio;
    i_sw_end = (i_gw + 1) * dxRatio;
    j_sw_end = (j_gw + 1) * dxRatio;
}
```

During time integration in `src/serghei.h`, rainfall rates are aggregated from the fine surface grid to the coarse subsurface grid:

```cpp
if (gdom.dxRatio == 1) {
    Kokkos::deep_copy(gdom.rainRate, ss.swss.rainRate);
} else {
    // Aggregate rainfall from surface to subsurface
    Kokkos::parallel_for("aggregate_rainfall", gdom.nCell, KOKKOS_LAMBDA(int idom) {
        int ii, jj, kk;
        gdom.unpackIndices(idom, kk, jj, ii);
        if (kk == 0) {  // Only aggregate for top layer
            real rain_sum = 0.0;
            int n_valid = 0;
            int i_sw_start = ii * gdom.dxRatio;
            int j_sw_start = jj * gdom.dxRatio;
            int i_sw_end = (ii + 1) * gdom.dxRatio;
            int j_sw_end = (jj + 1) * gdom.dxRatio;
            
            for (int j_sw = j_sw_start; j_sw < j_sw_end && j_sw < dom.ny; j_sw++) {
                for (int i_sw = i_sw_start; i_sw < i_sw_end && i_sw < dom.nx; i_sw++) {
                    int iGlobSW = dom.getIndex(j_sw * dom.nx + i_sw);
                    if (!state.isnodata(iGlobSW)) {
                        rain_sum += ss.swss.rainRate(iGlobSW);
                        n_valid++;
                    }
                }
            }
            if (n_valid > 0) {
                int iGlobGW = gdom.getHaloExtension(ii, jj, kk);
                gdom.rainRate(iGlobGW) = rain_sum / n_valid;
            }
        }
    });
}
```

Exchange fluxes computed on the subsurface grid are distributed to surface cells:

```cpp
if (gdom.dxRatio == 1) {
    // Same resolution: direct copy from qss_gw to qss
    Kokkos::parallel_for("copy_qss_dx1", dom.nCell, KOKKOS_LAMBDA(int idom) {
        if (idom >= 0 && idom < gdom.nCell) {
            state.qss(idom) = gw.qss_gw(idom);
        }
    });
} else {
    // Distribute qss_gw to surface cells
    Kokkos::parallel_for("distribute_qss", dom.nCell, KOKKOS_LAMBDA(int idom) {
        int i_sw, j_sw;
        unpackIndicesUniformGrid(idom, dom.ny, dom.nx, j_sw, i_sw);
        int i_gw = i_sw / gdom.dxRatio;
        int j_gw = j_sw / gdom.dxRatio;
        int iGlobGW = j_gw * gdom.nx + i_gw;
        
        if (iGlobGW >= 0 && iGlobGW < gdom.nCell) {
            // Distribute flux: qss_sw = qss_gw / (dxRatio²)
            state.qss(idom) = gw.qss_gw(iGlobGW) / (gdom.dxRatio * gdom.dxRatio);
        } else {
            state.qss(idom) = 0.0;
        }
    });
}
```

The exchange flux is computed in `src/GwBC.h` for the top boundary condition (direction 6) and stored in `gw.qss_gw` per subsurface cell. This flux is then distributed to surface cells as shown above.

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

The implementation in `src/serghei.h` performs the wet-cell averaging during aggregation:

```cpp
// Aggregate surface depth: average over wet cells only
// First initialize all cells to zero
Kokkos::deep_copy(gw.hs, 0.0);
// Then aggregate for top layer physical cells
Kokkos::parallel_for("aggregate_hs", gdom.nCell, KOKKOS_LAMBDA(int idom) {
    int ii, jj, kk;
    gdom.unpackIndices(idom, kk, jj, ii);
    if (kk == 0) {  // Only aggregate for top layer
        real h_sum = 0.0;
        int n_wet = 0;
        int i_sw_start = ii * gdom.dxRatio;
        int j_sw_start = jj * gdom.dxRatio;
        int i_sw_end = (ii + 1) * gdom.dxRatio;
        int j_sw_end = (jj + 1) * gdom.dxRatio;
        
        for (int j_sw = j_sw_start; j_sw < j_sw_end && j_sw < dom.ny; j_sw++) {
            for (int i_sw = i_sw_start; i_sw < i_sw_end && i_sw < dom.nx; i_sw++) {
                int iGlobSW = dom.getIndex(j_sw * dom.nx + i_sw);
                real h_sw = state.h(iGlobSW);
                if (h_sw > state.hmin && !state.isnodata(iGlobSW)) {
                    h_sum += h_sw;
                    n_wet++;
                }
            }
        }
        // Store aggregated value using halo extension index
        int iGlobGW = gdom.getHaloExtension(ii, jj, kk);
        if (n_wet > 0) {
            gw.hs(iGlobGW) = h_sum / n_wet;
        } else {
            gw.hs(iGlobGW) = 0.0;
        }
    }
});
```

In `src/GwBC.h`, this aggregated value is used to set boundary conditions and compute exchange fluxes:

```cpp
// Get index for hs lookup
int iGlobGW;
if (gdom.dxRatio == 1) {
    iGlobGW = (jj) * gdom.nxhc + (ii);  // 2D indexing for surface field
} else {
    iGlobGW = iGlob;  // Multi-resolution: use halo index directly
}
gw.h(iGhost,1) = gw.hs(iGlobGW);
// get sw-gw exchange type
if (gw.h(iGhost,1) > 0.0) {
    real q_infilt = 2.0 * ks * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dz(iGlob) - ks;
    if (-q_infilt * gdom.dt <= gw.h(iGhost,1)) {
        swgw_type(ibc) = 1;
    } else {
        swgw_type(ibc) = 2;
    }
} else {
    // exfiltration
    if (gw.h(iGlob,1) > gw.h(iGhost,1) + 0.5*gdom.dz(iGlob)) {
        swgw_type(ibc) = 1;
    } else {
        swgw_type(ibc) = 0;  // no flow
    }
}
// Store exchange flux per subsurface cell
if (iGlobGW >= 0 && iGlobGW < gdom.nCell) {
    gw.qss_gw(iGlobGW) = gw.q(iGhost,2);
}
```

The uniform distribution back to surface cells in `src/serghei.h` ensures proper mass accounting even when individual surface cells transition between wet and dry states:

```cpp
// Distribute qss_gw to surface cells
Kokkos::parallel_for("distribute_qss", dom.nCell, KOKKOS_LAMBDA(int idom) {
    int i_sw, j_sw;
    unpackIndicesUniformGrid(idom, dom.ny, dom.nx, j_sw, i_sw);
    int i_gw = i_sw / gdom.dxRatio;
    int j_gw = j_sw / gdom.dxRatio;
    int iGlobGW = j_gw * gdom.nx + i_gw;
    
    if (iGlobGW >= 0 && iGlobGW < gdom.nCell) {
        // Distribute flux: qss_sw = qss_gw / (dxRatio²)
        state.qss(idom) = gw.qss_gw(iGlobGW) / (gdom.dxRatio * gdom.dxRatio);
    } else {
        state.qss(idom) = 0.0;
    }
});
```

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

The implementation in `src/serghei.h` handles asynchronous time stepping. During initialization, the initial subsurface time step is set based on `dt_ratio`:

```cpp
#if SERGHEI_SWE_RE
// Set initial gw.dt based on dt_ratio
// But first we need to compute initial sw.dt
tint.computeDt(state,dom,io);
// Use dt_ratio to set initial gw.dt
gdom.dt = gdom.dt_ratio * dom.dt;
if (gdom.dt > gdom.dt_max) {
    gdom.dt = gdom.dt_max;
}
// If gw.dt < sw.dt, synchronize sw.dt to gw.dt
if (gdom.dt < dom.dt) {
    dom.dt = gdom.dt;
}
#endif
```

After each surface step, the subsurface time step is recalculated based on the updated surface time step and `dt_ratio`:

```cpp
#if SERGHEI_SWE_RE
tint.computeDt(state,dom,io);
#if SERGHEI_RE_MODEL
// Use dt_ratio to determine coupling mode
// Calculate what gw.dt should be based on ratio
real gdom_dt_target = gdom.dt_ratio * dom.dt;
// Respect dt_max as upper bound
if (gdom_dt_target > gdom.dt_max) {
    gdom_dt_target = gdom.dt_max;
}
// When gdom_dt_target >= dom.dt: use dt_ratio * dom.dt (asynchronous when dt_ratio > 1)
// When gdom_dt_target < dom.dt: set dom.dt = gdom_dt_target (synchronize surface to subsurface)
if (gdom_dt_target >= dom.dt) {
    gdom.dt = gdom_dt_target;
} else {
    gdom.dt = gdom_dt_target;
    dom.dt = gdom.dt;
}
// For synchronous coupling (dt_ratio = 1), ensure both use the same dt
if (gdom.dt_ratio == 1.0) {
    if (dom.dt < gdom.dt) {
        gdom.dt = dom.dt;
    } else {
        dom.dt = gdom.dt;
    }
}
#endif
#endif
```

The time integration loop handles both synchronous and asynchronous modes:

```cpp
if (gdom.dt_ratio == 1.0) {
    // Synchronous: run exactly once, ensure time sync
    gdom.etime = dom.etime;
    // PC scheme or Modified Picard scheme
    if (gdom.gw_scheme == 1) {
        gwf.pca_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
    } else {
        gwf.picard_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
    }
} else {
    // Asynchronous: run while subsurface can take a full step before surface time
    // Subsurface advances by gdom.dt (which is dt_ratio * dom.dt) per step
    // Condition: gdom.etime + gdom.dt <= dom.etime ensures we can take a full step
    while (gdom.etime + gdom.dt <= dom.etime) {
        if (gdom.gw_scheme == 1) {
            gwf.pca_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
        } else {
            gwf.picard_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
        }
        // Advance subsurface time after solver completes
        gdom.etime += gdom.dt;
    }
}
```

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

**Note**: This is an experimental fork. When citing SERGHEI, please cite the official version using the corresponding [SERGHEI Zenodo DOI](https://doi.org/10.5281/zenodo.8159947), and if necessary with the specific release DOI.

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