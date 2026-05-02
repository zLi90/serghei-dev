#pragma once

#include "State.h"
#include "netcdfIO.h"
#include "units.h"

#ifndef SERGHEI_NETCDF_FORCING
#define SERGHEI_NETCDF_FORCING 0
#endif

#if SERGHEI_NETCDF_FORCING

class AtmosphericForcing{
  private:
    int nextForcingTime;
    ncStreamStrings *ncS;
    // Set to true to partition the atmospheric forcing grid across MPI ranks
    // (currently only works correctly  when domAF is spatially aligned with dom).
    // Set to false to replicate the full rain grid on every rank,
    // which is safe regardless of the relative extents of the two grids.
    static constexpr bool allowDecomposition = false;

  public:
    ncStream nc;
    Domain domAF;
    realArr time;
    realArr rain;

  inline int initialise(const Parallel &par, Domain &dom, ncStreamStrings &ncs, realArr &rainRate){
    nc.initialise(ncs);
    if(par.masterproc) std::cout << BDASH << "Initialising NetCDF atmospheric forcing from " << CYAN << nc.s->fname << RESET << std::endl;
    nc.readNetCDFheader(par, domAF, 1);
    domAF.hc=0;
    domAF.get2Ddecomposition(par);
    if constexpr(!allowDecomposition){
      // Replicate the full rain grid on every rank.
      // Safe when domAF and dom have different physical extents: a decomposed
      // domAF would not spatially overlap its matching dom subdomain, causing
      // getCellForPoint to return -1 for valid cells and producing zero-rain bands.
      domAF.i_beg = 0; domAF.i_end = domAF.nx_glob - 1;
      domAF.j_beg = 0; domAF.j_end = domAF.ny_glob - 1;
      domAF.nx    = domAF.nx_glob;
      domAF.ny    = domAF.ny_glob;
    }
    domAF.initialiseSpatial();
    nc.readNetCDFcoordinates(par,domAF);
    domAF.getExtent();
    #if SERGHEI_DEBUG_INPUT_NETCDF
      domAF.print();
    #endif

    // initialise the NetCDF variables
    nc.varTime.initialise(nc.s->t, nc.s->fname, "time");
    nc.varRain.initialise(nc.s->rain, nc.s->fname,"rain");


    // allocate memory for reading the rain from netcdf.
    // This must be the size of the physical domain for the rain, WITHOUT HALOS
    rain = realArr("ncRain",domAF.nCell);

    // allocate the rainRate for the source terms.
    // This is of size of the computational domain, just like state variables
    rainRate = realArr ("rainRate", dom.nCellMem);
    dom.isRain=1;

    if(!nc.readTime(par,domAF,time)) return SERGHEI_ERROR;

    nextForcingTime=0;

    if(par.masterproc) std::cout << GOK << "Atmospheric forcing initialised" << std::endl;
    return SERGHEI_OK;
  }

  int readRain(const Parallel &par){
    nc.varRain.tStride = nextForcingTime;
    int err = nc.varRain.readField(par,domAF,rain);
    if(err != NC_NOERR){
      if(err == NC_ENOTVAR){
        std::cerr << RERROR << "'" << nc.varRain.s->name << "'  variable not found in " << nc.s->fname << std::endl;
      }
      else{
        ncwrap(err,__LINE__);
      }
      return SERGHEI_ERROR;
    }

    // unit transformation and nodata check
    nc.varRain.factor = Units::rainFactor(nc.varRain.s->units);
    #if SERGHEI_DEBUG_RAINFALL
    Kokkos::printf("%s[DEBUG] %s%s Rain units: %s\tRain factor: %e\n",GRAY,__PRETTY_FUNCTION__,RESET,nc.varRain.s->units.c_str(),nc.varRain.factor);
    #endif

    Kokkos::parallel_for("distributedRain",domAF.nCell, KOKKOS_CLASS_LAMBDA (int ii){
      if(rain(ii) <= nc.varRain.nodata || std::isnan(rain(ii))) rain(ii) = 0;
      rain(ii) *= nc.varRain.factor;
    });
    Kokkos::fence();
    nextForcingTime++;


    return SERGHEI_OK;
  }

  inline void computeRain(const Domain &dom, const State &state, realArr &rainRate){
    #if SERGHEI_DEBUG_RAINFALL
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "nextForcingTime: " << nextForcingTime << " time: " << dom.etime << " =? " << time(nextForcingTime) << std::endl;
    #endif

    if(dom.etime >= time(nextForcingTime)){
      if(nextForcingTime+1 > nc.nTime){
        // if the time is beyond what is available in the file, set rain to zero
        Kokkos::deep_copy(rain,0.);
      }else{
        if(!readRain(*dom.Par)){
          std::cerr << RERROR << "Error reading rain from << " CYAN << nc.s->fname << RESET << std::endl;
          abort();
        }
        domAF.reorderViewToRasterIndexing(rain);
      }
    }

    #if SERGHEI_DEBUG_INPUT_NETCDF
    std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << ":" << RESET << std::endl;
    #endif
    // interpolate from the rain grid
    Kokkos::parallel_for("distributedRain",dom.nCell, KOKKOS_CLASS_LAMBDA (int iGlob){
      int ii = dom.getIndex(iGlob);
      if(!state.isnodata(ii)){
        geometry::point p;
        p = dom.getCellCenter(iGlob);
        int jj = domAF.getCellForPoint(p);
        rainRate(ii) = rain(jj);  // piecewise constant closest neighbor
        //if(dom.etime < 1) Kokkos::printf("iGlob=%d\tii=%d\tjj=%d\tpx=%6e\tpy=%6e\n",iGlob,ii,jj,p(_X),p(_Y));
        //Kokkos::printf("%.6e %.6e\n", rainRate(ii), rain(jj));
      }
    });
    Kokkos::fence();

    #if SERGHEI_DEBUG_RAINFALL
      real maxvalAF,minvalAF;
      Kokkos::parallel_reduce("debug_nc_rain",domAF.nCell,KOKKOS_CLASS_LAMBDA(int ii, real &maxval, real &minval){
        maxval = max(rain(ii),maxval);
        minval = min(rain(ii),minval);
      }, Kokkos::Max<real>(maxvalAF),Kokkos::Min<real>(minvalAF));
      real maxval,minval;
      Kokkos::parallel_reduce("debug_rainRate",dom.nCell,KOKKOS_CLASS_LAMBDA(int ii, real &maxval, real &minval){
        maxval = max(rainRate(ii),maxval);
        minval = min(rainRate(ii),minval);
      }, Kokkos::Max<real>(maxval),Kokkos::Min<real>(minval));
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " Min/max rainfall intensity: " << minval << "/" << maxval << " and from atmforcing: " << minvalAF << "/" << maxvalAF <<  std::endl;
    #endif
  }



};
#endif
