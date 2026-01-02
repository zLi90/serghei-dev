
#ifndef _STATE_H_
#define _STATE_H_

#include "define.h"
#include "SArray.h"
#include "Indexing.h"
#include <set>

#ifndef SERGHEI_MAXFLOOD
#define SERGHEI_MAXFLOOD 0
#endif

typedef struct{
  real h=0;
  real hu=0;
  real hv=0;
  #if SERGHEI_VERTICAL_VELOCITY
    real w=0.0;
  #endif
  real z=0;
} swState;



class State {

public:

  // SW variables
  realArr h;
  realArr hu;
  realArr hv;
  #if SERGHEI_LPT
    realArr tparticles;
    intArr market;
  #endif
  #if SERGHEI_VERTICAL_VELOCITY
    realArr w;
    realArr dZ_X;
    realArr  dZ_Y;
    realArr dU_X;
    realArr  dU_Y;
  #endif

  //elevation
  realArr z;

  //roughness
  realArr roughness;
  real hmin;

  //deltaFluxes
  realArr dsw0; //3 variables (h,hu,hv). left and south contribs
  realArr dsw1; //3 variables (h,hu,hv). right and north contribs

  // surface-subsurface exchange flux
  realArr qss;

  boolArr isnodata; //contains 0 if is a regular cell, 1 if is nodata cell
  intArr isBound; //positive values for inlet boundaries, negative values for outlet bvoundaries, 0 for inner cells

  #if SERGHEI_MAXFLOOD
    realArr hMax;
    realArr momentumMax;
    realArr time_hMax;
  #endif


  inline void allocate(Domain &dom){
    h 				= realArr( "h" , dom.nCellMem );
    hu 			= realArr( "hu" , dom.nCellMem );
    hv 			= realArr( "hv" , dom.nCellMem );
    #if SERGHEI_LPT
      tparticles 				= realArr( "tparticles" , dom.nCellMem );
      market 					= intArr( "market" , dom.nCellMem );
      Kokkos::parallel_for("initialise_tparticles",dom.nCell,KOKKOS_CLASS_LAMBDA(int iGlob2) {
        int ii2 = dom.getIndex(iGlob2);
        tparticles(ii2)=0.0;
        market(ii2)=0;
      });
    #endif
    #if SERGHEI_VERTICAL_VELOCITY
      w  =  realArr("w", dom.nCellMem);
      dZ_X =  realArr("dZ_X", dom.nCellMem);
      dZ_Y =  realArr("dZ_Y", dom.nCellMem);
      dU_X =  realArr("dU_X", dom.nCellMem);
      dU_Y =  realArr("dU_Y", dom.nCellMem);
    #endif
    z 				= realArr( "z" , dom.nCellMem );
    roughness 	= realArr( "roughness" , dom.nCellMem );
    isnodata 	= boolArr( "isnodata" , dom.nCellMem );
	  isBound 	= intArr( "isBound" , dom.nCellMem );
    dsw0 			= realArr( "dsw0" , 3*dom.nCellMem );
    dsw1 			= realArr( "dsw1" , 3*dom.nCellMem );
    qss 				= realArr( "qss" , dom.nCell );

    #if SERGHEI_MAXFLOOD
      hMax = realArr("hMax",dom.nCellMem);
      momentumMax = realArr("momMax",dom.nCellMem);
      time_hMax = realArr("timehMax",dom.nCellMem);

      Kokkos::parallel_for("initialise_maxflood",dom.nCell,KOKKOS_CLASS_LAMBDA(int iGlob) {
        int ii = dom.getIndex(iGlob);
        hMax(ii) = 0;
        momentumMax(ii)=0;
        time_hMax(ii)=0;
      });
    #endif

    Kokkos::deep_copy(h, 0);
    Kokkos::deep_copy(hu, 0);
    Kokkos::deep_copy(hv, 0);
    Kokkos::deep_copy(z, 0);
    Kokkos::deep_copy(roughness, 0);
    Kokkos::deep_copy(isBound, 0);
    Kokkos::deep_copy(isnodata, false);
    Kokkos::deep_copy(dsw0, 0);
    Kokkos::deep_copy(dsw1, 0);
    #if SERGHEI_VERTICAL_VELOCITY
      Kokkos::deep_copy(w, 0);
      Kokkos::deep_copy(dZ_X, 0);
      Kokkos::deep_copy(dZ_Y, 0);
      Kokkos::deep_copy(dU_X, 0);
      Kokkos::deep_copy(dU_Y, 0);
    #endif
	if(dom.id==0) std::cout << GOK << "State allocated and initialised" << std::endl;
  }

  inline void filterDomain(const Domain &dom){
    Kokkos::parallel_for("filter_domain",dom.nCell,KOKKOS_CLASS_LAMBDA(int iGlob) {
      int ii = dom.getIndex(iGlob);
      if(isnodata(ii)){
        h(ii) = SERGHEI_NAN;
        hu(ii) = SERGHEI_NAN;
        hv(ii) = SERGHEI_NAN;
      }
    });
  }
};


class ShallowWater{
public:
  std::string initialMode;
  std::string frictionModel;
  std::string roughnessInput;
  std::set<std::string> initialModes = {"dry","h","h+z","file","netcdf"};
  std::set<std::string> frictionModels = {"none","manning","darcyweisbach","chezy"};
  real roughness = 0 ;
  real initialValue = 0;
  real hmin = -1;
};

#endif
