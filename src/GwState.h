/* -*- mode: c++; c-default-style: "linux" -*- */

/**
 * Subsurface model container
 **/

#ifndef _GW_STATE_H_
#define _GW_STATE_H_


#if SERGHEI_RE_MODEL

#include "const.h"
#include "define.h"
#include "SArray.h"
#include "Indexing.h"
#include "GwDomain.h"
#include "GwMatrix.h"
#include "GwMPI.h"
#include "GwSolver.h"
#include "State.h"
#include <set>
#include <math.h>


// declare functions
real wc2h(real wc, real alpha, real n, real wcs, real wcr);
real h2wc(real h, real alpha, real n, real wcs, real wcr);
real h2kr(real h, real alpha, real n);
real h2dwcdh(real h, real alpha, real n, real wcs, real wcr);
real h2dKdh(real h, real alpha, real n, real Ks);

class GwState   {

public:
    // GW variables
    realArr2 h, q, wc, k;
    // matrix system
    realArr2 resi, coef;
    realArr qss, qss_gw, hs;  // qss: per surface cell, qss_gw: per subsurface cell (for multi-resolution)
    realArr hs_fine, qss_fine;  // Fine-resolution surface depth and exchange flux (for dxRatio > 1)
    // source/sink flow rate
    realArr ssflow;
    //
    // reasoning
    // ---------
    // soil variables are stored in a table, soilID maps the cell to
    // van Genuchten parameters in the table soilTable
    intArr soilID;
    realArr vgTable; // van Genuchten parameters: Ks, Phi, ThetaS, ThetaR, n, m, alpha
    int nVGparam; /* number of van Genuchten parameters */
    // std::string initialMode;
    int initialMode;
    real initialValue;

    // Allocate state variables for groundwater
    inline void allocate (GwDomain &gdom) {
        nVGparam = NVG;

        h = realArr2("h", gdom.nCellMem, 2);
        wc = realArr2("wc", gdom.nCellMem, 3);
        k = realArr2("k", gdom.nCellMem, 4);
        q = realArr2("q", gdom.nCellMem, 3);
        soilID = intArr ("soilID", gdom.nCellMem);
        vgTable = realArr ("vg", nVGparam * gdom.nSoilID);
        coef = realArr2("coef", gdom.nCell, 8);

		ssflow = realArr("ss", 1);
		ssflow(0) = 0.0;
        qss = realArr("qss", gdom.nCellSw);  // Exchange rate per surface cell
        qss_gw = realArr("qss_gw", gdom.nCell);  // Exchange rate per subsurface cell (for multi-resolution)
        // Initialize qss arrays to zero at simulation start
        // qss_gw retains values between GW solver executions for temporal continuity
        Kokkos::deep_copy(qss, 0.0);
        Kokkos::deep_copy(qss_gw, 0.0);
        // hs: surface water depth aggregated to subsurface cells
        // When dxRatio==1: uses surface cell memory (same as state.h)
        // When dxRatio>1: uses subsurface cell memory (aggregated values)
        hs = realArr("hs", (gdom.dxRatio == 1) ? gdom.nCellSwMem : gdom.nCellMem);
        // Fine-resolution views for multi-resolution coupling (dxRatio > 1)
        // hs_fine: individual surface cell depths (for fine-resolution flux computation)
        // qss_fine: fine-resolution exchange fluxes (computed in GwBC.h, copied to state.qss in serghei.h)
        hs_fine = realArr("hs_fine", gdom.nCellSw);  // Always fine resolution (surface cell count)
        qss_fine = realArr("qss_fine", gdom.nCellSw);  // Always fine resolution (surface cell count)
    }

};



/* --------------------------------------------------
    Soil constitutive functions
-------------------------------------------------- */
KOKKOS_INLINE_FUNCTION real wc2h(real wc, real alpha, real n, real wcs, real wcr) {
    real h, m, eps=1e-7;
    m = 1.0 - 1.0 / n;
    if (wc - wcr < eps) {wc = wcr + eps;}
    if (wc < wcs) {
        h = -(1.0/alpha) * pow(pow((wcs - wcr)/(wc - wcr),(1.0/m)) - 1.0,(1.0/n));
        return h;
    }
    else    {return 0.0;}
}

KOKKOS_INLINE_FUNCTION real h2wc(real h, real alpha, real n, real wcs, real wcr) {
    real wc, m, s;
    m = 1.0 - 1.0 / n;
    s = pow(1.0 + pow(fabs(alpha*h), n), -m);
    // calculate water content
    if (h >= 0.0)  {wc = wcs;}
    else    {wc = wcr + (wcs-wcr) * s;}
    if (wc > wcs)    {wc = wcs;}
    else if (wc < wcr)   {wc = wcr;}
    return wc;
}

KOKKOS_INLINE_FUNCTION real h2kr(real h, real alpha, real n) {
    real m, s, kr;
    m = 1.0 - 1.0 / n;
    s = pow(1.0 + pow(fabs(alpha*h), n), -m);
    kr = pow(s,0.5) * pow(1-pow(1-pow(s,1.0/m),m), 2.0);
    if (h > 0.0 || kr > 1.0) {kr = 1.0;}
    return kr;
}

KOKKOS_INLINE_FUNCTION real h2dwcdh(real h, real alpha, real n, real wcs, real wcr)  {
    real dwcdh, wc1, wc2, dh=1e-3;
    wc1 = h2wc(h, alpha, n, wcs, wcr);
    wc2 = h2wc(h+dh, alpha, n, wcs, wcr);
    if (h > 0.0)    {dwcdh = 0.0;}
    else {dwcdh = (wc2 - wc1) / dh;}
    return dwcdh;
}

KOKKOS_INLINE_FUNCTION real h2dKdh(real h, real alpha, real n, real Ks)  {
    real dkdh, k1, k2, dh=1e-3;
    k1 = h2kr(h, alpha, n);
    k2 = h2kr(h+dh, alpha, n);
    if (h > 0.0)    {dkdh = 0.0;}
    else {dkdh = 0.5 * Ks * (k2 - k1) / dh;}
    return dkdh;
}

#endif

#endif
