#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "define.h"
#include "SArray.h"
#include "mpi.h"

#define SERGHEI_MASTERPROC 0

class Parallel {

public:

  int nranks;
  int myrank;
  int nproc_x;
  int nproc_y;
  int nproc_z;
  int px;
  int py;
  int nthreads; //if is 0, then take what comes from hwloc
  ulong i_beg;
  ulong j_beg;
  ulong i_end;
  ulong j_end;
  int masterproc;
  SArray<int,3,3> neigh;
};

inline void printKokkosInitArguments(Parallel const &par){
  std::cout << "MPI Rank " << par.myrank << "\tnum_threads " << Kokkos::num_threads() << std::endl;
  #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  std::cout << "MPI Rank " << par.myrank << "\tdevice_id " << Kokkos::device_id() << std::endl;
  #endif
  // Note: Kokkos::num_devices() was removed in newer Kokkos versions
  //std::cout << "MPI Rank " << par.myrank << "\tskip_device " << Kokkos::skip_device() << std::endl;
}

#ifdef KOKKOS_ENABLE_CUDA
inline int printKokkosCuda(Kokkos::InitializationSettings const &args, Parallel const &par){
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "MPI Rank " << par.myrank << "\tcudaGetDeviceCount = " << deviceCount << std::endl;

  int device = args.get_device_id();
  cudaDeviceProp deviceProp;
  cudaError_t result = cudaGetDeviceProperties(&deviceProp, device);
  if (cudaSuccess != result) {
    printf("Failed to query device properties: %s on MPI rank %d\n", cudaGetErrorString(result),par.myrank);
    return 0;
  }

  cudaUUID_t devUUID = deviceProp.uuid;
  char uuid_string[20];
  sprintf(uuid_string, "%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx", devUUID.bytes[0], devUUID.bytes[1], devUUID.bytes[2], devUUID.bytes[3], devUUID.bytes[4], devUUID.bytes[5], devUUID.bytes[6], devUUID.bytes[7], devUUID.bytes[8], devUUID.bytes[9], devUUID.bytes[10], devUUID.bytes[11], devUUID.bytes[12], devUUID.bytes[13], devUUID.bytes[14], devUUID.bytes[15]);

  std::cout << "MPI Rank " << par.myrank <<"\tfinding device " << device << "\t -> Device name: " << deviceProp.name << "\tUUID " << uuid_string << std::endl;
  return 1;
}
#endif


#endif
