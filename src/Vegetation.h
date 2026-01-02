// Vegetation model container

#ifndef _VEGSTATE_H_
#define _VEGSTATE_H_

#include "define.h"
#include "SArray.h"
#include "Indexing.h"
#include "Domain.h"
#include <set>

#ifndef SERGHEI_VEGETATION_MODEL
#define SERGHEI_VEGETATION_MODEL 0
#endif


#if VEGETATION_MODEL
// Class containing vegetation data
class VegetationState{

public:
  realArr biomass;
  realArr uptake;

  inline void allocate(Domain &dom){
    biomass = realArr("biomass", dom.nCellMem);
    uptake = realArr("biomass", dom.nCellMem);
  }
};


// To parse vegetation model control
class VegetationParser{
  public:
  std::string model;
  std::string initalMode;
  std::set<std::string> initialModes = {"bare","biomass","file"};
};

};

#endif

#endif
