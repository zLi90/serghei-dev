#pragma once

#include "const.h"
#include "math.h"
#include "mpi.h"
#include <Kokkos_Core.hpp>

// Topology boundary conditions
#define TOPOLOGY_BC_PERIODIC 1
#define TOPOLOGY_BC_REFLECTIVE 2
#define TOPOLOGY_BC_TRANSMISSIVE 3

// Subsurface initial mode
#define IC_SAT 1
#define IC_H 2
#define IC_WC 3
#define IC_WT 4

#define SERGHEI_FLOAT 1
#define SERGHEI_DOUBLE 2

#ifndef SERGHEI_REAL
#define SERGHEI_REAL SERGHEI_DOUBLE
#endif

#ifndef SERGHEI_MPI_REAL
  #if SERGHEI_REAL == SERGHEI_DOUBLE
    #define SERGHEI_MPI_REAL MPI_DOUBLE
  #elif SERGHEI_REAL == SERGHEI_FLOAT
    #define SERGHEI_MPI_REAL MPI_FLOAT
  #endif
#endif

#if SERGHEI_REAL == SERGHEI_DOUBLE
#define SERGHEI_VTK_REAL "DOUBLE"
#elif SERGHEI_REAL == SERGHEI_FLOAT
#define SERGHEI_VTK_REAL "FLOAT"
#endif

#if SERGHEI_REAL == SERGHEI_DOUBLE
typedef double real;
#define TOL_MASS_ERROR TOL8
#define TOLDRY TOL12
#define TOL_ZERO_MOMENTUM TOL12
#define TOL_WETDRY TOL12
#define TOL_MACHINE_ACCURACY TOL12
#endif
#if SERGHEI_REAL == SERGHEI_FLOAT
typedef float real;
#define TOL_MASS_ERROR TOL5
#define TOLDRY TOL6
#define TOL_ZERO_MOMENTUM TOL12
#define TOL_WETDRY TOL12
#define TOL_MACHINE_ACCURACY TOL6
#endif

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(KOKKOS_ENABLE_SYCL)
#include <CL/sycl.hpp>
#endif

typedef Kokkos::View<real *, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> realArr;
typedef Kokkos::View<int *, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> intArr;
typedef Kokkos::View<bool *, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> boolArr;
typedef Kokkos::View<double *, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> doubleArr;
typedef Kokkos::View<real**, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> realArr2;
typedef Kokkos::View<int**, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> intArr2;
typedef Kokkos::View<ushort*, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> ushortArr;
typedef Kokkos::View<uchar*, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> ucharArr;
typedef Kokkos::View<float*, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> floatArr;

typedef ushort color;
typedef ushortArr colorArr;


KOKKOS_INLINE_FUNCTION real operator"" _fp(long double x)
{
    return static_cast<real>(x);
}

KOKKOS_INLINE_FUNCTION double mypow ( double const x , double const p ) { return pow (x,p); }
KOKKOS_INLINE_FUNCTION float  mypow ( float  const x , float  const p ) { return powf(x,p); }
KOKKOS_INLINE_FUNCTION double mysqrt( double const x ) { return sqrt (x); }
KOKKOS_INLINE_FUNCTION float  mysqrt( float  const x ) { return sqrtf(x); }
KOKKOS_INLINE_FUNCTION double myfabs( double const x ) { return fabs (x); }
KOKKOS_INLINE_FUNCTION float  myfabs( float  const x ) { return fabsf(x); }
KOKKOS_INLINE_FUNCTION int  myfabs( int  const x ) { return abs(x); }
KOKKOS_INLINE_FUNCTION double mylog( double const x ) { return log (x); }
KOKKOS_INLINE_FUNCTION float  mylog( float  const x ) { return logf(x); }



template <class T1, class T2> KOKKOS_INLINE_FUNCTION T1 min (T1 const v1, T2 const v2){
  if (v1 < v2) { return (T1) v1; }
  else         { return (T1) v2; }
}

template <class T1, class T2>
KOKKOS_INLINE_FUNCTION T1 max(T1 const v1, T2 const v2){
    if (v1 > v2){return (T1)v1;}
    else        {return (T1)v2; }
}

template <class T>
KOKKOS_INLINE_FUNCTION int sgn(T const val)
{
    return (T(0) < val) - (val < T(0));
}


#include "SArray.h"
// Define Kokkos::View of SArray
typedef Kokkos::View<SArray<real,2>*, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>> realS2Arr;


class ParserLine{
public:
  std::string line;
  std::string key ;
  std::stringstream value;

  void lowercase(){
    std::for_each(line.begin(), line.end(), [](char & c) {
      c = ::tolower(c);
    });
  }

  void parse(){
    // make sure key and value are clean (in case of reuse)
    key.clear();
    value.clear();
     // Remove spaces and tabs from the line
     // line.erase (std::remove(line.begin(), line.end(), ' '), line.end());
     // line.erase (std::remove(line.begin(), line.end(), '\t'), line.end());

     // If the line isn't empty and doesn't begin with a comment specifier, split it based on the colon
     if (!line.empty() && line.find("//",0) != 0) {

       // Find the colon
       uint splitloc = line.find(':',0);

       // Store the key and value strings
       key   = line.substr(0,splitloc);

       // Remove spaces and tabs from the key
       key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
       key.erase(std::remove(key.begin(), key.end(), '\t'), key.end());
       std::string val = line.substr(splitloc+1,line.length()-splitloc);

       // Check for comments after values
       size_t splitter = val.find("//",0);
       std::string strloc;
       if (splitter != std::string::npos){
         strloc = val.substr(0,splitter);
       } else {
         strloc = val;
       }

       // Transform the val into a string stream for convenience
       value.clear();
       value.str(strloc);
     }
   }

   void print(){
     std::cout << "line: " << line << "\tkey: " << key << "\tvalue: " << value.str() << std::endl;
   }
};


/*
  TimeSeries provides a construct/class to store time series.
 */
class TimeSeries{

public:

  int np;     // number of points in time
  int nc;       // number of grid cells with different time series values
  int nx = 1; // number of partitions in x direction
  int ny = 1; // number of partitions in y direction

  realArr time;
  realArr value;
  realArr2 values; // this is not a nice solution
  int timeIndex = 0;

  //Time serries applied to polygon
  int nver = 0;
  realArr xPoly;
  realArr yPoly;

/*
  // WARNING valid only for piece-wise constant time data
  inline real interpolate (real const &t, int spaceIndex){
    if(t >= time (np - 1)){
      timeIndex = np - 1;
    }
    else{
	     if (t >= time (timeIndex + 1)) timeIndex++;
    }
    return (value (np * spaceIndex + timeIndex));
  }
*/
  void initialise(int n){
    np = n;
    time = realArr("time",np);
    value = realArr("value",np);
  };

  void initPolygon(int n){
    nver = n;
    xPoly = realArr("xPoly",nver);
    yPoly = realArr("yPoly",nver);
  };

};

