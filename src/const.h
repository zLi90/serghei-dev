/* -*- mode: mode: c++; c-default-style: "linux" -*- */

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <iostream>

#define SERGHEI_OK 1
#define SERGHEI_ERROR 0

// Index for variables
#define idH 0
#define idHU 1
#define idHV 2
#define idZ 3
#define idR 4
// Index for variable for IO
#define ioH idH
#define ioH idH
#define ioHU idHU
#define ioHV idHV
#define ioZ idZ
#define ioR idR
#define ioHZ 1000
#define ioU 1001
#define ioV 1002

// Some physical constants
#define GRAV 9.81
#define SQRTGRAV 3.132091953
#define RHOA 1.225
#define PI 3.14159

#define RHOW 1000.0
#define RHOS 2650.0
#define VISCMU 0.0012349 //Water dynamic viscosity (Pa*s) at 12oC
#define VISCNU 0.0000012356 //Water kinematic viscosity (m2/s) at 12oC

// Some tolerances
#define TOL4 1e-4
#define TOL5 1e-5
#define TOL6 1e-6
#define TOL6NEG -1e-6
#define TOL8 1e-8
#define TOL8NEG -1e-8
#define TOL9 1e-9
#define TOL9NEG -1e-9
#define TOL12 1e-12
#define TOL12NEG -1e-12
#define TOL14 1e-14
#define TOL14NEG -1e-14
#define TOL15 1e-15
#define TOL15NEG -1e-15

#define ZERO TOL12

#define NVG 7

// model component defaults

#ifndef SERGHEI_SWE_RE
#define SERGHEI_SWE_RE 0
#endif

#ifndef SERGHEI_RE_MODEL
#define SERGHEI_RE_MODEL 0
#endif

#ifndef SERGHEI_SWE_MODEL
#define SERGHEI_SWE_MODEL 1
#endif

#ifndef SERGHEI_KOKKOSKERNELS_SOLVER
#define SERGHEI_KOKKOSKERNELS_SOLVER 1
#endif

// friction model (0-->upwind or 1-->pointwise-centered)
#ifndef SERGHEI_POINTWISE_FRICTION
#define SERGHEI_POINTWISE_FRICTION 0
#endif

#define SERGHEI_FRICTION_MANNING 1
#define SERGHEI_FRICTION_DARCYWEISBACH 2
#define SERGHEI_FRICTION_CHEZY 3
#ifndef SERGHEI_FRICTION_MODEL
#define SERGHEI_FRICTION_MODEL SERGHEI_FRICTION_MANNING
#endif

// min depth from which water is stopped
// #define hmin 0.001

// No Data ThresHold
#define NDTH 9999.0

#define SERGHEI_NAN NAN

// PNETCDF parameters
#define PNETCDF_N_INPUT_VARIABLES 9 // number of variables in an initial input file


// program options
#ifndef SERGHEI_DEBUG_PARALLEL_DECOMPOSITION
#define SERGHEI_DEBUG_PARALLEL_DECOMPOSITION 0 // debug the subdomains ranks and neighbours
#endif
#ifndef SERGHEI_DEBUG_SUBSURFACE
#define SERGHEI_DEBUG_SUBSURFACE 0
#endif
#ifndef SERGHEI_DEBUG_WORKFLOW
#define SERGHEI_DEBUG_WORKFLOW 0
#endif
#ifndef SERGHEI_DEBUG_BOUNDARY
#define SERGHEI_DEBUG_BOUNDARY 0
#endif
#ifndef SERGHEI_SERGHEI_DEBUG_RAINFALL
#define SERGHEI_SERGHEI_DEBUG_RAINFALL 0
#endif
#ifndef SERGHEI_DEBUG_INFILTRATION
#define SERGHEI_DEBUG_INFILTRATION 0
#endif
#ifndef SERGHEI_DEBUG_KOKKOS_SETUP
#define SERGHEI_DEBUG_KOKKOS_SETUP 0
#endif
#ifndef SERGHEI_DEBUG_DT
#define SERGHEI_DEBUG_DT 0
#endif
#ifndef SERGHEI_DEBUG_MASS_CONS
#define SERGHEI_DEBUG_MASS_CONS 0
#endif
#ifndef SERGHEI_DEBUG_SCALAR_TRANSPORT
#define SERGHEI_DEBUG_SCALAR_TRANSPORT 0
#endif
#ifndef SERGHEI_VEGETATION_MODEL
#define SERGHEI_VEGETATION_MODEL 0
#endif
#ifndef SERGHEI_EROSIVE_SHEAR
#define SERGHEI_EROSIVE_SHEAR 0 //erosive shear forcing calculation
#define SERGHEI_EROSIVE_SHEAR_FORMULATION 1 //mode for erosive shear forcing: 1->shear strees
#endif
#ifndef SERGHEI_SCALAR_TRANSPORT
#define SERGHEI_SCALAR_TRANSPORT 0
#endif
#ifndef SERGHEI_SEDIMENT_TRANSPORT
#define SERGHEI_SEDIMENT_TRANSPORT 0
#endif
#ifndef SERGHEI_UPWIND_BED
#define SERGHEI_UPWIND_BED 0
#endif
#ifndef SERGHEI_SUSPENDED_SEDIMENT
#define SERGHEI_SUSPENDED_SEDIMENT 0
#endif
#ifndef SERGHEI_BEDLOAD_SEDIMENT
#define SERGHEI_BEDLOAD_SEDIMENT 0
#endif
#ifndef SERGHEI_SCALAR_DIFFUSION 
#define SERGHEI_SCALAR_DIFFUSION 0
#endif
#ifndef SERGHEI_ANALYTICAL_DIFFUSION 
#define SERGHEI_ANALYTICAL_DIFFUSION 0
#endif
#ifndef SERGHEI_DEBUG_MPI
#define SERGHEI_DEBUG_MPI 0
#endif
#ifndef SERGHEI_DEBUG_OUTPUT
#define SERGHEI_DEBUG_OUTPUT 0
#endif
#ifndef SERGHEI_DEBUG_INPUT_NETCDF
#define SERGHEI_DEBUG_INPUT_NETCDF 0
#endif
#ifndef SERGHEI_DEBUG_RAINFALL
#define SERGHEI_DEBUG_RAINFALL 0
#endif
#ifndef SERGHEI_LPT
#define SERGHEI_LPT 0
#endif
#ifndef SERGHEI_LPT_DIFFUSIVE
#define SERGHEI_LPT_DIFFUSIVE 0
#endif
#ifndef SERGHEI_LPT_MICROPLASTICS
#define SERGHEI_LPT_MICROPLASTICS 0
#endif
#ifndef SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION
#define SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION 0
#endif
#ifndef SERGHEI_PARTICLE_NO_OUTPUT
#define SERGHEI_PARTICLE_NO_OUTPUT 0
#endif
#ifndef SERGHEI_VERTICAL_VELOCITY
#define SERGHEI_VERTICAL_VELOCITY 0
#endif
#ifndef SERGHEI_VERTICAL_VELOCITY_Z
#define SERGHEI_VERTICAL_VELOCITY_Z 0
#endif
#ifndef SERGHEI_DEBUG_SEDIMENT
#define SERGHEI_DEBUG_SEDIMENT 0
#endif
#ifndef SERGHEI_SWE_DRY_RUNOFF_START_DT
#define SERGHEI_SWE_DRY_RUNOFF_START_DT 0
#endif

//colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define GRAY   "\033[90m"      /* Gray */
#define BOLD "\033[1m"  /* Bold */
//mesagges
#define GSTAR GREEN << "[**] " << RESET
#define GOK GREEN << "[OK] " << RESET
#define GIO MAGENTA << "[IO] " << RESET
#define BEXC BLUE << "[!] " << RESET
#define BDASH BLUE << "[-] " << RESET
#define YEXC YELLOW << "[!] " << RESET
#define REXC RED << "[!] " << RESET
#define RERROR RED << "[ERROR] " << RESET
#define GGD GRAY << "[DEBUG] " << RESET
#define NCERROR RED << "[NETCDF ERROR] " << RESET

#define NO_DATA -999;

int const OUT_NETCDF = 1;
int const OUT_VTK = 2;
int const OUT_BIN = 3;

int const OUTPUT_PRECISION = 6;

#ifndef SERGHEI_MESH_UNIFORM
#define SERGHEI_MESH_UNIFORM 1
#endif

int getIOvarID(std::string vname){
    int varid=-1;
    if(!vname.compare("h")) varid=ioH;
    if(!vname.compare("hu")) varid=ioHV;
    if(!vname.compare("hv")) varid=ioHU;
    if(!vname.compare("z")) varid=ioZ;
    if(!vname.compare("h+z")) varid=ioHZ;
    if(!vname.compare("roughness")) varid=ioR;
    if(!vname.compare("u")) varid=ioU;
    if(!vname.compare("v")) varid=ioV;
    if(varid<0){
        std::cerr << RERROR << "IO variable " << vname << " not handled by " << __FUNCTION__ << std::endl;

    return -1;
  }
  return varid;
}
#endif
