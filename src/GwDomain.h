#ifndef _GW_DOMAIN_H_
#define _GW_DOMAIN_H_

#include "define.h"
#include "Domain.h"

class GwDomain : public Domain {

public:
    // Time stepping options
    real dt_init, dt_max, dtOld;
    real dt_ratio;  // Fixed ratio between gw.dt and sw.dt (dt_ratio >= 1.0, default 1.0 for synchronous coupling)
    // Subsurface domain dimensions
    real thickH, topZ, zll, dx, dy, dz_multiplier, dz_base;
    int nz, nz_glob, nhalo, nxhc, nyhc, nzhc;
    int nCellSw, nCellSwMem;
    // Multi-resolution coupling
    int dxRatio;  // Grid size ratio (subsurface/surface), default = 1
    // Domain properties
    int nSoilID, aev, hmin;
    int hasRoot, hasET;
    // Baseline VG parameters;
    real wcs, wcr, Ks, alpha, n;
    real wc_ic;
    // Numerical scheme
    int gw_scheme, cg_iter = 1000000;
	real cg_tol = 1e-8;
    // Kokkos views
    realArr x, y, z, dz, depth, sinx, cosx, siny, cosy, rainRate, evapRate, etpmRate;
    intArr isnodata, onboundary;

    // other variables
    int BCtype;
    realArr globalBuffer;
    geometry::point extent[2];

    KOKKOS_INLINE_FUNCTION void unpackIndicesGw(int const iGlob, int nz, int ny, int nx, int &k, int &j, int &i) const{
      unpackIndicesUniformGrid(iGlob,nz,ny,nx,k,j,i);
    };
    KOKKOS_INLINE_FUNCTION void unpackIndices(int const iGlob, int &k, int &j, int &i) const{
        unpackIndicesUniformGrid(iGlob, nz, ny, nx, k, j, i);
    };

    KOKKOS_INLINE_FUNCTION void unpackIndicesHalo(int const iGlob, int &k, int &j, int &i) const{
        unpackIndicesUniformGrid(iGlob, nzhc, nyhc, nxhc, k, j, i);
    };

    KOKKOS_INLINE_FUNCTION int getHaloExtension(const int i, const int j, const int k) const {
        return( (hc+k)*(nx+2*hc)*(ny+2*hc) + (hc+j)*(nx+2*hc) + hc+i ); //index for the extended domain (including halo cells)
    };

    KOKKOS_INLINE_FUNCTION int getSubdomainExtension(const Parallel &par, const int i, const int j, const int k) const{
        return( k*nx_glob*ny_glob + (par.j_beg+j)*nx_glob + par.i_beg+i ); //index for the subdomain (par.j_beg+j,par.i_beg+i)
    };
    
    // Multi-resolution index mapping functions
    // Get subsurface cell index from surface cell indices (i_sw, j_sw are surface indices)
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
    // Initialize surface domain
    // void initialise() {
    //     // physical cells onlys
    //     #if SERGHEI_MESH_UNIFORM
    //     nCell = nx*ny*nz;
    //     nCellMem = (ny+2*hc)*(nx+2*hc)*(nz+2*hc);
    //     nCellGlobal = nx_glob * ny_glob * nz_glob;
    //     #endif
    //     globalBuffer = realArr("globalBuffer", nCellGlobal);
    // };

    void fetchFieldFromGlobalBuffer(const Parallel &par, realArr &data){
        Kokkos::parallel_for("fetch_from_global_buffer", nCell , KOKKOS_CLASS_LAMBDA (int iGlob) {
            int i,j,k;
            unpackIndices(iGlob,k,j,i);
            int ii1 = getHaloExtension(i,j,k);
            int ii2 = getSubdomainExtension(par,i,j,k);
            data(ii1) = globalBuffer(ii2);
        });
    }

};


#endif
