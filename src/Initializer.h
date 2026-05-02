/* -*- mode: c++; c-default-style: "linux" -*- */

#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include "define.h"
#include "Exchange.h"
#include "TimeIntegrator.h"
#include "Indexing.h"
#include "Parser.h"
#include "SourceSink.h"

#include "GwState.h"
#include "GwDomain.h"


class Initializer{

public:

	bool read = 1;

  void initializeMPI( int *argc , char ***argv , Parallel &par ) {
  	#if SERGHEI_DEBUG_WORKFLOW
	    std::cerr << GGD "Initialising MPI " << std::endl;
	  #endif
    int ierr = MPI_Init( argc , argv );
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&par.nranks);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&par.myrank);

    //Determine if I'm the master process
    if (par.myrank == SERGHEI_MASTERPROC) {
      par.masterproc = 1;
    } else {
      par.masterproc = 0;
    }
  }

  int initialize(State &state, SourceSinkData &ss, ExternalBoundaries &ebc, Domain &dom, Parallel &par, TimeIntegrator &tint, surfaceIntegrator &sint, boundaryIntegrator &bint, Parser &parser, Exchange &exch, FileIO &io){

    if(read) if(!parser.readDimensions(dom, state, par, io)) return 0;

    dom.buildDomainDecomposition(par);

    dom.initialise();
    state.allocate(dom);

		if(read) if(!parser.readInputFiles(dom, state, ss, ebc, par, io)) return 0;

		#if SERGHEI_MAPCLASS
		//if(read) if(!parser.readLandUseClasses(par,io,state)) return SERGHEI_ERROR;
			if(read) if(!parser.readMapClasses(par,io,state.landuse,"landuse.input")) return SERGHEI_ERROR;
			state.landuse.allocate(dom);
    	if(read) if(!parser.readLandUse(par, io, dom, state)) return SERGHEI_ERROR;
    	state.landuse.assign(dom,state.roughness);
   	 	if(par.masterproc) std::cout << GOK << "Roughness assigned from land use" << std::endl << std::endl;

			//if(read) if(!parser.readMapClasses(par,io,ss.inf.soilmap,"soilmap.input")) return SERGHEI_ERROR;
			ss.inf.soilmap.allocate(dom);
			if(read) if(!parser.readSoilMap(par,io,dom,ss,state)) return SERGHEI_ERROR;
			ss.inf.soilmap.assignMap(dom,ss.inf.infLabel);
			ss.inf.constCap = ss.inf.soilmap.table;
		#endif

    dom.getStatistics();
		ss.initializeSW(dom,state,par);
    #if SERGHEI_INPUT_NETCDF
      if(read) if(!parser.readInfiltrationRateNetCDF(io,dom,ss.inf,par)) return 0;
    #endif

    sint.initialize(state,dom,ss);
    bint.initialize(ebc.extbc);

    #if SERGHEI_SCALAR_TRANSPORT
      state.ade.initialise(dom,par,state.h,io.conf.st);
    #endif

    #if SERGHEI_SEDIMENT_TRANSPORT
      state.sediment.initialise(dom,par,io.conf.st,io.conf.sedt);
    #endif

    if(par.masterproc){
      if(!parser.createOutputDir(io.outFolder)){
	return 0;
      }
    }

    #if SERGHEI_SCALAR_TRANSPORT
      exch.addMaxPack(state.ade.nScalar);
    #endif
    #if SERGHEI_SEDIMENT_TRANSPORT
      exch.addMaxPack(state.sediment.movableBedLevel);
    #endif
    exch.iniMPI(state,dom,par);

    //initialize boundaries
    boundaryIni(state,dom,par,ebc.extbc);

		// initialise source/sink at start time
		ss.ComputeSWSourceSink(state,dom);

    // Output the initial model state
		#if SERGHEI_SWE_MODEL
    if(io.allowIni) io.outputIni(state, dom, ss, par);
		#endif

    return 1;

  }

	inline void boundaryIni(State &state, const Domain &dom, const Parallel &par, const std::vector<ExtBC> &extbc){

		//impose boundary conditions in the outer (full) domain in the case of reflective boundary conditions. Periodic and transmissive are default in exchangeIniMPI

		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
			//boundary conditions (halo) for outer domain (periodic/transmissive by default with the exchange in exchangeIniMPI)
			if(dom.iE){
				boundaryReflectiveE(state,dom);
			}
 			if(dom.iN){
				boundaryReflectiveN(state,dom);
			}
			if(dom.iW){
				boundaryReflectiveW(state,dom);
			}
			if(dom.iS){
				boundaryReflectiveS(state,dom);
			}
		}

		//numerical boundaries. Remove the high walls
		for (int k = 0; k < extbc.size(); k ++) {
			removeElevationNumericalBoundaries(state, extbc[k], dom,par);
		}

	}

	inline void removeElevationNumericalBoundaries(State &state, const ExtBC &extbc, const Domain &dom, const Parallel &par){

		Kokkos::parallel_for("remove_elevation_numerical_boundaries", extbc.ncellsBC, KOKKOS_LAMBDA (int iGlob) {
			int ii = extbc.bcells[iGlob];
			int i, j;
			unpackIndicesUniformGrid(ii,dom.ny+2*dom.hc,dom.nx+2*dom.hc,j,i);

			if(i==dom.nx+dom.hc-1 && dom.iE){
				state.z(ii+dom.di)=state.z(ii);
			}
			if(i==dom.hc && dom.iW) {
				state.z(ii-dom.di)=state.z(ii);
			}
			if(j==dom.ny+dom.hc-1 && dom.iS){
				state.z(ii+dom.dj)=state.z(ii);
			}
			if(j==dom.hc && dom.iN){
				state.z(ii-dom.dj)=state.z(ii);
			}

		});

	}


	inline void boundaryReflectiveE(State &state,const Domain &dom){

		Kokkos::parallel_for("boundaryReflectiveE", dom.ny*dom.hc , KOKKOS_LAMBDA (int iGlob) {
			int rx, ry;
			unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
			int ii=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx-1;
			//east boundary
			state.z(ii+rx+1) 	= 1e4;
			state.h(ii+rx+1)	=0.0;
			state.hu(ii+rx+1)	=0.0;
			state.hv(ii+rx+1)	=0.0;

      if(state.isBound(ii)==0){ //no open boundary
        state.isBound(ii+rx+1)=SERGHEI_BC_OUTER_REFLECTIVE_CELL;
      }

		 });
	}

	inline void boundaryReflectiveN(State &state, const Domain &dom){

		Kokkos::parallel_for("boundaryReflectiveN", dom.hc*dom.nx , KOKKOS_LAMBDA (int iGlob) {
      	int rx, ry;
      	unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
			int ii = dom.hc*(dom.nx+2*dom.hc) + dom.hc+rx;
			//north boundary
			state.z(ii-(ry+1)*(dom.nx+2*dom.hc))	=1e4;
			state.h(ii-(ry+1)*(dom.nx+2*dom.hc))	=0.0;
			state.hu(ii-(ry+1)*(dom.nx+2*dom.hc))=0.0;
			state.hv(ii-(ry+1)*(dom.nx+2*dom.hc))=0.0;

      if(state.isBound(ii)==0){ //no open boundary
        state.isBound(ii-(ry+1)*(dom.nx+2*dom.hc))=SERGHEI_BC_OUTER_REFLECTIVE_CELL;
      }

		 });
	}

	inline void boundaryReflectiveW(State &state, const Domain &dom){

		Kokkos::parallel_for("boundaryReflectiveW", dom.ny*dom.hc , KOKKOS_LAMBDA (int iGlob) {
			int rx, ry;
			unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
			int ii=(dom.hc+ry)*(dom.nx+2*dom.hc) + dom.hc;
			//west boundary
			state.z(ii-rx-1)	=1e4;
			state.h(ii-rx-1)	=0.0;
			state.hu(ii-rx-1)	=0.0;
			state.hv(ii-rx-1)	=0.0;

      if(state.isBound(ii)==0){ //no open boundary
        state.isBound(ii-rx-1)=SERGHEI_BC_OUTER_REFLECTIVE_CELL;
      }

		 });
	}

	inline void boundaryReflectiveS(State &state, const Domain &dom){

		Kokkos::parallel_for("boundaryReflectiveS", dom.hc*dom.nx , KOKKOS_LAMBDA (int iGlob) {
      	int rx, ry;
      	unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
			int ii=(dom.hc+dom.ny-1)*(dom.nx+2*dom.hc)+dom.hc+rx;
			//south boundary
			state.z(ii+(ry+1)*(dom.nx+2*dom.hc))	=1e4;
			state.h(ii+(ry+1)*(dom.nx+2*dom.hc))	=0.0;
			state.hu(ii+(ry+1)*(dom.nx+2*dom.hc))=0.0;
			state.hv(ii+(ry+1)*(dom.nx+2*dom.hc))=0.0;

      if(state.isBound(ii)==0){ //no open boundary
        state.isBound(ii+(ry+1)*(dom.nx+2*dom.hc))=SERGHEI_BC_OUTER_REFLECTIVE_CELL;
      }

		 });
	}

};

#endif
