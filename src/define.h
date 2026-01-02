
#ifndef _CONST_H_
#define _CONST_H_

#include "const.h"
#include "math.h"
#include "mpi.h"
#include <Kokkos_Core.hpp>

// SWE boundary conditions
#define BC_PERIODIC 1
#define BC_REFLECTIVE 2
#define BC_TRANSMISSIVE 3

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


template <class T1, class T2>
KOKKOS_INLINE_FUNCTION T1 min(T1 const v1, T2 const v2)
{
    if (v1 < v2)
    {
        return (T1)v1;
    }
    else
    {
        return (T1)v2;
    }
}

template <class T1, class T2>
KOKKOS_INLINE_FUNCTION T1 max(T1 const v1, T2 const v2)
{
    if (v1 > v2)
    {
        return (T1)v1;
    }
    else
    {
        return (T1)v2;
    }
}

template <class T>
KOKKOS_INLINE_FUNCTION int sgn(T const val)
{
    return (T(0) < val) - (val < T(0));
}

#endif
