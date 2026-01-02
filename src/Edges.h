
#ifndef _EDGES_H_
#define _EDGES_H_

#include "define.h"
#include "Parallel.h"
#include "SArray.h"
#include "Domain.h"
#include "Exchange.h"
#include "Indexing.h"
#include "Solvers.h"


class Edges {

Kokkos::Timer timer,timerdt;

public :


	inline void computeDeltaStateSW(State &state, Domain &dom, Exchange &exch, Parallel &par){
    	timer.reset();
	 	solve(state, dom, exch, par);
		computeTimeStepReduction(dom, state);
    	dom.timers.swe.flux.total += timer.seconds();
	}



	inline void solve(State &state, Domain &dom, Exchange &exch, Parallel &par){

    // Riemann solver X
    computeDeltaFluxXRoe(state, dom, par);
	// Riemann solver Y
	 computeDeltaFluxYRoe(state, dom, par);

  }


  inline void computeDeltaFluxXRoe(State &state, Domain const &dom,  Parallel &par) {
    
    Kokkos::parallel_for("computeDeltaFluxXRoe", dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
      int i, j, ncells;
		int id1,id2;
      unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
		if(i>hc-2 && i<dom.nx+hc && j>hc-1 && j<dom.ny+hc){ //note the hc-2 (first valid halo-inner wall) 
			SArray<real,3> upwM, upwP;
			SArray<real,5> s1,s2; //3 sw variables plus z and roughness
			
			ncells=dom.nCellMem;
			id1=iGlob; //j*(dom.nx+2*hc)+i
			id2=j*(dom.nx+2*hc)+i+1;

			s1(idH)=state.h(id1);
			s2(idH)=state.h(id2);
			
			bool nodata = state.isnodata(id1) || state.isnodata(id2);

			if((s1(idH)>0. || s2(idH)>0.) && !nodata && !(dom.iW&&i==hc-1) && !(dom.iE&&i==dom.nx+hc-1)){ //avoid dry-pair, nodata and boundary cells
				s1(idHU)=state.hu(id1);
				s2(idHU)=state.hu(id2);
				s1(idHV)=state.hv(id1);
				s2(idHV)=state.hv(id2);
				s1(idZ)=state.z(id1);
				s2(idZ)=state.z(id2);
				s1(idR)=state.roughness(id1);
				s2(idR)=state.roughness(id2);

				roeSolver(s1, s2, upwM, upwP, dom.dt ,dom.dx(), 1,0);

				state.dsw0(id1) = upwM(0);
				state.dsw0(id1+ncells) = upwM(1);
				state.dsw0(id1+2*ncells) = upwM(2);
				
				state.dsw1(id2) = upwP(0);
				state.dsw1(id2+ncells) = upwP(1);
				state.dsw1(id2+2*ncells) = upwP(2);

			}
		}
		
	});
  }


  inline void computeDeltaFluxYRoe(State &state, Domain const &dom,  Parallel &par) {
    
    Kokkos::parallel_for( "computeDeltaFluxXRoe",dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
      int i, j, ncells;
		int id1,id2;
      unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
		if(i>hc-1 && i<dom.nx+hc && j>hc-2  && j<dom.ny+hc){ //note the hc-2 (first valid halo-inner wall)
			SArray<real,3> upwM, upwP;
			SArray<real,5> s1,s2; //3 sw variables plus z and roughness
			
			ncells=dom.nCellMem;
			id1=iGlob; //j*(dom.nx+2*hc)+i
			id2=(j+1)*(dom.nx+2*hc)+i;

			s1(idH)=state.h(id1);
			s2(idH)=state.h(id2);

			bool nodata = state.isnodata(id1) || state.isnodata(id2);

			if((s1(idH)>0. || s2(idH)>0.) && !nodata && !(dom.iN&&j==hc-1) && !(dom.iS&&j==dom.ny+hc-1)){ //avoid dry-pair, nodata and boundary cells
				s1(idHU)=state.hu(id1);
				s2(idHU)=state.hu(id2);
				s1(idHV)=state.hv(id1);
				s2(idHV)=state.hv(id2);
				s1(idZ)=state.z(id1);
				s2(idZ)=state.z(id2);
				s1(idR)=state.roughness(id1);
				s2(idR)=state.roughness(id2);

				roeSolver(s1, s2, upwM, upwP, dom.dt ,dom.dx(), 0,-1);

				 //note that we have sum to not overwrite the x-contributions
				state.dsw0(id1) += upwM(0);
				state.dsw0(id1+ncells) += upwM(1);
				state.dsw0(id1+2*ncells) += upwM(2);
				
				state.dsw1(id2) += upwP(0);
				state.dsw1(id2+ncells) += upwP(1);
				state.dsw1(id2+2*ncells) += upwP(2);

			}
		}
		
	});
  }


	inline void computeTimeStepReduction(Domain &dom, State &state) {
		real dtloc=dom.dt;
		timerdt.reset();
		Kokkos::parallel_reduce("computeTimeStepReduction", dom.nCell , KOKKOS_LAMBDA (int iGlob, real &dt) {
    		int ii = dom.getIndex(iGlob);

			dt=min(dt,dom.dt);
			real h=state.h(ii);
			real dh=state.dsw0(ii)+state.dsw1(ii);
			if(dh>TOLDRY){ //only positive dh can make negative water depths
				real dt_candidate = (h+TOLDRY)*dom.dx()/dh;
				// Prevent dt from becoming too small (minimum 1e-10 seconds)
				dt_candidate = max(dt_candidate, 1e-10);
				dt=min(dt, dt_candidate); 
			}

		} , Kokkos::Min<real>(dtloc) );

	  Kokkos::fence();

	  dom.timers.swe.flux.dt += timerdt.seconds();
	  dom.dt = dtloc;
	  
	  // Ensure dt doesn't become zero or negative
	  if(dom.dt < 1e-10) {
		  dom.dt = 1e-10;
	  }
	  
	  if(dom.nsubdom>1){
	  	timerdt.reset();
	  	int ierr = MPI_Allreduce(&dtloc, &dom.dt, 1, SERGHEI_MPI_REAL , MPI_MIN, MPI_COMM_WORLD);
	  	dom.timers.swe.flux.mpi += timerdt.seconds();
	  	// Re-check after MPI reduction
	  	if(dom.dt < 1e-10) {
	  		dom.dt = 1e-10;
	  	}
	  }
	  #if SERGHEI_DEBUG_DT
	  std::cout << "time = " << dom.etime << "\tdt_neg = " << dom.dt << std::endl;
	  #endif
	}


};

#endif
