#pragma once

#include "define.h"

#ifndef SERGHEI_MAPCLASS
#define SERGHEI_MAPCLASS 0
#endif
#ifndef SERGHEI_DEBUG_MAPCLASS
#define SERGHEI_DEBUG_MAPCLASS 0
#endif

// Forward declaration
template<bool Enabled>
class MapClass_;

// Base class with common interface
template<bool Enabled>
class MapClassBase {
public:
  bool use = false;
  int nClass = 0;
  int maxid = 0;
  ushortArr mapClass;   // map of classes
  
  ~MapClassBase() = default;
  int allocate(const Domain &dom) = delete;
  int validate(const Domain &dom, const boolArr &mask) = delete;
};

// Full implementation 
template<>
class MapClass_<true> : public MapClassBase<true> {
  
public:
  realArr table;        // table to map the class id to values
  
  int inline allocate(const Domain &dom) {
    if(!use) return SERGHEI_OK;
    mapClass = ushortArr("mapClass", dom.nCell);
    return SERGHEI_OK;
  }

  int validate(const Domain &dom, const boolArr &mask) {
    std::cout << BDASH << "Validating map classes..." << std::endl;
    if(!use) return SERGHEI_OK;
    int errout = 0;
    int errwrong = 0;

    Kokkos::parallel_reduce("validateMapValue", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob, int &eout, int &ewrong){
      int ii = dom.getIndex(iGlob);
      if(!mask(ii)){
        if(mapClass(iGlob) > maxid) eout = 1;
        if(std::isnan(table(mapClass(iGlob)))){
          ewrong = 1;
        }
      }
    }, Kokkos::Sum<int>(errout), Kokkos::Sum<int>(errwrong));
    Kokkos::fence();
    
    if(errout > 0) std::cerr << RERROR << "Class ID is out of bounds of table data provided." << std::endl;
    if(errwrong > 0) std::cerr << RERROR << "There are " << errwrong << " class IDs in the map which have no equivalent in the class table." << std::endl;
    if(errout || errwrong) return SERGHEI_ERROR;

    return SERGHEI_OK;
  }

  template <typename T> int assign(const Domain &dom, T &mapValue) const {
    if(!use) return SERGHEI_OK;
    if constexpr(SERGHEI_DEBUG_MAPCLASS) std::cout << GGD << __PRETTY_FUNCTION__ << "::" << __LINE__ << std::endl;

    Kokkos::parallel_for("assignMapValue", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
      int ii = dom.getIndex(iGlob);
      mapValue(ii) = table(mapClass(iGlob));
    });
    Kokkos::fence();
    
    if constexpr(SERGHEI_DEBUG_MAPCLASS) std::cout << GGD << __PRETTY_FUNCTION__ << "::" << __LINE__ << std::endl;
    
    return SERGHEI_OK;
  }

  template <typename T> int assignMap(const Domain &dom, T &newmap) const {
    if(!use) return SERGHEI_OK;
    if constexpr(SERGHEI_DEBUG_MAPCLASS) std::cout << GGD << __PRETTY_FUNCTION__ << "::" << __LINE__ << std::endl;

    Kokkos::parallel_for("assignMap", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
      int ii = dom.getIndex(iGlob);
      newmap(ii) = mapClass(iGlob);
    });
    Kokkos::fence();
    return SERGHEI_OK;
  }
};

// Stub specialization
template<>
class MapClass_<false> : public MapClassBase<false> {
  
public:
  int inline allocate(const Domain &dom) {
    return SERGHEI_OK;
  }

  int validate(const Domain &dom, const boolArr &mask) {
    return SERGHEI_OK;
  }

  template <typename T> int assign(const Domain &dom, T &mapValue) const {
    return SERGHEI_OK;
  }

  template <typename T> int assignMap(const Domain &dom, T &newmap) const {
    return SERGHEI_OK;
  }
};

// Type alias: automatically selects correct specialization depending on the build options
using MapClass = MapClass_<SERGHEI_MAPCLASS>;