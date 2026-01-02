#ifndef _GW_INTEGRATOR_H_
#define _GW_INTEGRATOR_H_

#if SERGHEI_RE_MODEL

#include "GwState.h"
#include "SourceSink.h"
#include "GwDomain.h"
#include "GwBC.h"
#include "Indexing.h"

class GwIntegrator	{

	Kokkos::Timer timer;

	public:
		real Vtot, Vtot_glob;
		real Vexch, Vexch_glob;

		SourceSink *ss;
		GwState *gw;
		GwDomain *gdom;
		std::vector<GwBC>* gwbc;
		std::vector<GwSS>* gwss;

		int ncellsBC, ncellsBC_glob, ncellsIT, ncellsIT_glob;
		real QinBC, QoutBC, QinSS, QoutSS;
		real QinBC_glob, QoutBC_glob, QinSS_glob, QoutSS_glob;


		void initialize(GwState &gw_, GwDomain &gdom_, std::vector<GwBC> &gwbc_, std::vector<GwSS> &gwss_){
			gw = &gw_;
			gdom = &gdom_;
			gwss = &gwss_;
			gwbc = &gwbc_;
		}

		void integrate(GwState const &gw, GwDomain const &gdom, std::vector<GwBC> &gwbc, std::vector<GwSS> &gwss)	{
	 		int ierr=0;
	 		// get total volume
	 		Vtot = 0;
	  		Kokkos::parallel_reduce(gdom.nCell , KOKKOS_LAMBDA (int idx, real &tmp) {
	  			int ii, jj, kk, iGlob;
            	gdom.unpackIndices(idx, kk, jj, ii);
            	iGlob = (hc+kk)*gdom.nxhc*gdom.nyhc + (hc+jj)*gdom.nxhc + ii + hc;
            	tmp += gw.wc(iGlob,1) * gdom.dx * gdom.dx * gdom.dz(iGlob);
    		} , Kokkos::Sum<real>(Vtot) );
			Kokkos::fence();
			Vtot_glob = 0.0;
			ierr=MPI_Allreduce(&Vtot, &Vtot_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			// get surface-subsurface exchange rate [m3/s]
			Vexch = 0;
			#if SERGHEI_SWE_MODEL
			// Always use qss_gw, which is populated in GwBC.h for both dxRatio=1 and dxRatio>1
			Kokkos::parallel_reduce(gdom.ny*gdom.nx, KOKKOS_LAMBDA (int idx, real &tmp) {
			        int ii, jj, iGlob;
			        unpackIndicesUniformGrid(idx, gdom.ny, gdom.nx, jj, ii);
			        iGlob = jj*gdom.nx + ii;
			        tmp += gw.qss_gw(iGlob) * gdom.dx * gdom.dy;
			} , Kokkos::Sum<real>(Vexch) );
			Kokkos::fence();
			#endif
			Vexch_glob = 0.0;
			ierr=MPI_Allreduce(&Vexch, &Vexch_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			// boundary flow
			QinBC = 0.0;
			QoutBC = 0.0;
			for (int k = 0; k < gwbc.size(); k++) {
				int _ncellsBC;
				MPI_Allreduce(&(gwbc[k].ncellsBC), &_ncellsBC, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				ncellsBC += _ncellsBC;
				QinBC += gwbc[k].Qinflow;
				QoutBC += gwbc[k].Qoutflow;
			}
			QinBC_glob=0.0;
			QoutBC_glob=0.0;
        	ncellsBC_glob = 0;
			ierr=MPI_Allreduce(&QinBC, &QinBC_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			ierr=MPI_Allreduce(&QoutBC, &QoutBC_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		    ierr=MPI_Allreduce(&ncellsBC, &ncellsBC_glob, 1, MPI_INT , MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			// source/sink terms
			QinSS = 0.0;
			QoutSS = 0.0;
			for (int k = 0; k < gwss.size(); k++) {
				int _ncellsIT;
				MPI_Allreduce(&(gwss[k].ncellsIT), &_ncellsIT, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				ncellsIT += _ncellsIT;
				QinSS += gwss[k].Qinflow;
				QoutSS += gwss[k].Qoutflow;
			}
			QinSS_glob=0.0;
			QoutSS_glob=0.0;
        	ncellsIT_glob = 0;
			ierr=MPI_Allreduce(&QinSS, &QinSS_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			ierr=MPI_Allreduce(&QoutSS, &QoutSS_glob, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		    ierr=MPI_Allreduce(&ncellsIT, &ncellsIT_glob, 1, MPI_INT , MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

		}


};

#endif

#endif
