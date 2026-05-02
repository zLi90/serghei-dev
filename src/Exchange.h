#ifndef _EXCHANGE_H_
#define _EXCHANGE_H_

#include "define.h"
#include "mpi.h"
#include "Indexing.h"


class Exchange {

protected:

  int  maxPack = SERGHEI_N_VARS_SWE;

  MPI_Request sReq[2];
  MPI_Request rReq[2];

  MPI_Status  sStat[2];
  MPI_Status  rStat[2];

  int nPack;
  int nUnpack;

  realArr haloSendBufS;
  realArr haloSendBufN;
  realArr haloSendBufW;
  realArr haloSendBufE;
  realArr haloRecvBufS;
  realArr haloRecvBufN;
  realArr haloRecvBufW;
  realArr haloRecvBufE;

  Kokkos::Timer timerMPI;   // should be used only to wrap around MPI calls
  Kokkos::Timer timerHalo;  // general timer for the entire halo exchange process
  Kokkos::Timer timerST;    // timer for bed elevation halo exchange
  Kokkos::Timer timerADE;	// timer for scalar transport halo exchange
	//Kokkos::View<real*,Kokkos::LayoutStride> bufdata;
	realArr bufdata;

public:

  inline void allocate(Domain &dom) {
	// haloSendBuf* and haloRecvBuf* arrays are 1D containers (arrays) which should be able to store all state variables, for the exchanging band
    haloSendBufS = realArr("haloSendBufS",maxPack*dom.hc*dom.nx);
    haloSendBufN = realArr("haloSendBufN",maxPack*dom.hc*dom.nx);
    haloSendBufW = realArr("haloSendBufW",maxPack*dom.ny*dom.hc);
    haloSendBufE = realArr("haloSendBufE",maxPack*dom.ny*dom.hc);
    haloRecvBufS = realArr("haloRecvBufS",maxPack*dom.hc*dom.nx);
    haloRecvBufN = realArr("haloRecvBufN",maxPack*dom.hc*dom.nx);
    haloRecvBufW = realArr("haloRecvBufW",maxPack*dom.ny*dom.hc);
    haloRecvBufE = realArr("haloRecvBufE",maxPack*dom.ny*dom.hc);
    // initialisation is not necessary
    /*
    Kokkos::deep_copy(haloSendBufS,0.);
    Kokkos::deep_copy(haloSendBufN,0.);
    Kokkos::deep_copy(haloSendBufW,0.);
    Kokkos::deep_copy(haloSendBufE,0.);
    Kokkos::deep_copy(haloRecvBufS,0.);
    Kokkos::deep_copy(haloRecvBufN,0.);
    Kokkos::deep_copy(haloRecvBufW,0.);
    Kokkos::deep_copy(haloRecvBufE,0.);
    */

		#if SERGHEI_SCALAR_TRANSPORT
			bufdata = realArr("bufdata",dom.nCellMem);
		#endif
  }

	inline void addMaxPack(const int &n){
		maxPack += n;
	}

  inline void haloInit() {
    nPack   = 0;
    nUnpack = 0;
  }


	// direct transfer x-halo columns
	inline void haloTransfer_x_bool(Domain &dom, Parallel &par, boolArr &a){
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif

		int nHalo = dom.ny * dom.hc;

		boolArr haloSendBoolW;
		boolArr haloSendBoolE;
		haloSendBoolW = boolArr("haloSendBoolW",nHalo);
    haloSendBoolE = boolArr("haloSendBoolE",nHalo);

		boolArr haloRecvBoolW;
		boolArr haloRecvBoolE;
		haloRecvBoolW = boolArr("haloRecvBoolW",nHalo);
    haloRecvBoolE = boolArr("haloRecvBoolE",nHalo);

    Kokkos::parallel_for(nHalo , KOKKOS_LAMBDA (int iGlob) {
			int rx,ry;
			unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
			int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc) + dom.hc + rx;		// west
			int ii2=(dom.hc+ry)*(dom.nx+2*dom.hc) + dom.hc + dom.nx-1+rx;	// east
      haloSendBoolW(iGlob) = a(ii1);
      haloSendBoolE(iGlob) = a(ii2);
    });

		if (par.nproc_x > 1) {
			Kokkos::fence();	// ensure no kernels are running, we need everything available in host memory
			//Pre-post the receives
			MPI_Irecv( haloRecvBoolW.data() , nHalo , MPI_CXX_BOOL , par.neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
			MPI_Irecv( haloRecvBoolE.data() , nHalo , MPI_CXX_BOOL , par.neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );

			//Send the data
			MPI_Isend( haloSendBoolW.data() , nHalo , MPI_CXX_BOOL , par.neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
			MPI_Isend( haloSendBoolE.data() , nHalo , MPI_CXX_BOOL , par.neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );

			//Wait for the sends and receives to finish
			MPI_Waitall(2, sReq, sStat);
			MPI_Waitall(2, rReq, rStat);
		}

    Kokkos::parallel_for(nHalo , KOKKOS_LAMBDA (int iGlob) {
			int rx,ry;
			unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
			int ii1h=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc-rx-1;
			int ii2h=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx+rx;
      a(ii1h) = haloRecvBoolW(iGlob);
      a(ii2h) = haloRecvBoolE(iGlob);
    });

  }

	// direct transfer y-halo rows
	inline void haloTransfer_y_bool(Domain &dom, Parallel &par, boolArr &a){
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif

		int nHalo = dom.hc * dom.nx;

		boolArr haloSendBoolN;
		boolArr haloSendBoolS;
		haloSendBoolN = boolArr("haloSendBoolN",nHalo);
    haloSendBoolS = boolArr("haloSendBoolS",nHalo);

		boolArr haloRecvBoolN;
		boolArr haloRecvBoolS;
		haloRecvBoolN = boolArr("haloRecvBoolN",nHalo);
    haloRecvBoolS = boolArr("haloRecvBoolS",nHalo);

    Kokkos::parallel_for(nHalo , KOKKOS_LAMBDA (int iGlob) {
			int rx,ry;
			unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
			int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc) + dom.hc + rx;			// north
			int ii2=(dom.hc+dom.ny-1+ry)*(dom.nx + 2*dom.hc) + dom.hc + rx;	// south
      haloSendBoolN(iGlob) = a(ii1);
      haloSendBoolS(iGlob) = a(ii2);
    });

		if (par.nproc_y > 1) {
			Kokkos::fence();	// ensure no kernels are running, we need everything available in host memory
			//Pre-post the receives
			MPI_Irecv( haloRecvBoolN.data() , nHalo , MPI_CXX_BOOL , par.neigh(0,1) , 0 , MPI_COMM_WORLD , &rReq[0] );
			MPI_Irecv( haloRecvBoolS.data() , nHalo , MPI_CXX_BOOL , par.neigh(2,1) , 1 , MPI_COMM_WORLD , &rReq[1] );

			//Send the data
			MPI_Isend( haloSendBoolN.data() , nHalo , MPI_CXX_BOOL , par.neigh(0,1) , 1 , MPI_COMM_WORLD , &sReq[0] );
			MPI_Isend( haloSendBoolS.data() , nHalo , MPI_CXX_BOOL , par.neigh(2,1) , 0 , MPI_COMM_WORLD , &sReq[1] );

			//Wait for the sends and receives to finish
			MPI_Waitall(2, sReq, sStat);
			MPI_Waitall(2, rReq, rStat);
		}

    Kokkos::parallel_for(nHalo , KOKKOS_LAMBDA (int iGlob) {
			int rx,ry;
			unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
			int ii1h=(dom.hc-ry-1)*(dom.nx+2*dom.hc) + dom.hc + rx;
			int ii2h=(dom.hc+dom.ny+ry)*(dom.nx+2*dom.hc) + dom.hc + rx;
      a(ii1h) = haloRecvBoolN(iGlob);
      a(ii2h) = haloRecvBoolS(iGlob);
    });

  }




	// packs state variable a into haloSendBuf arrays
	#if SERGHEI_SCALAR_TRANSPORT
		inline void pullSingleScalarHaloX(const int &iphi, ADEsolver &ade, Domain &dom, realArr &a){
			Kokkos::parallel_for("pull_single_scalar_haloX", dom.ny*dom.hc, KOKKOS_LAMBDA(int iGlob){
				int rx,ry;
				unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
				int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;		// west
				int ii2=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx-1+rx;	// east
					a(ii1) = ade.hphi(ii1,iphi);
					a(ii2) = ade.hphi(ii2,iphi);
			});
		}
	#endif

	inline void haloPack_x(Domain &dom, const realArr &a){
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
    haloPack_x_ext(dom, a, haloSendBufW, haloSendBufE, nPack);
    nPack = nPack + 1;		// to keep track of how many variable are getting packed
  }


  inline void haloPack_x_ext(Domain &dom, const realArr &a, realArr &haloSendBufW, realArr &haloSendBufE, int const nPack) {
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
		// span the x-halo columns
    Kokkos::parallel_for("haloPack_x_span", dom.ny*dom.hc , KOKKOS_LAMBDA (int iGlob) {
		int rx,ry;
		int nGlob = dom.ny*dom.hc;
	 	unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
	   int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;		// west
	   int ii2=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx-1+rx;	// east
      haloSendBufW(nPack*nGlob+iGlob) = a(ii1);
      haloSendBufE(nPack*nGlob+iGlob) = a(ii2);
    });
  }


	#if SERGHEI_SCALAR_TRANSPORT
		inline void pullSingleScalarHaloY(const int &iphi, ADEsolver &ade, Domain &dom, realArr &a){
			Kokkos::parallel_for("pull_single_scalar_haloY", dom.hc*dom.nx, KOKKOS_LAMBDA(int iGlob){
				int rx,ry;
				unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
				int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;			// north
				int ii2=(dom.hc+dom.ny-1+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;	// south
					a(ii1) = ade.hphi(ii1,iphi);
					a(ii2) = ade.hphi(ii2,iphi);
			});
		}
	#endif

  inline void haloPack_y(Domain &dom, const realArr &a) {
    haloPack_y_ext(dom, a, haloSendBufS, haloSendBufN, nPack);
    nPack = nPack + 1;       // to keep track of how many variable are getting packed
  }

  inline void haloPack_y_ext(Domain &dom, const realArr &a, realArr &haloSendBufS, realArr &haloSendBufN, int const nPack) {
	  	// span the y-halo rows
    Kokkos::parallel_for("haloPack_y_span", dom.hc*dom.nx , KOKKOS_LAMBDA (int iGlob) {
	 	int rx,ry;
		int nGlob = dom.hc*dom.nx;
		unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
	 	int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;			// north
	 	int ii2=(dom.hc+dom.ny-1+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;	// south
		haloSendBufN(nPack*nGlob+iGlob) = a(ii1);
		haloSendBufS(nPack*nGlob+iGlob) = a(ii2);

    });
  }

	// unpacks haloRecvBuf arrays into state variable a
  inline void haloUnpack_x(Domain &dom, realArr &a) {
    haloUnpack_x_ext(dom, a, haloRecvBufW, haloRecvBufE, nUnpack);
    nUnpack = nUnpack + 1;
  }

  inline void haloUnpack_x_ext(Domain &dom, realArr &a, realArr &haloRecvBufW, realArr &haloRecvBufE, int const nUnpack) {
    Kokkos::parallel_for("haloUnpack_x_span", dom.ny*dom.hc , KOKKOS_LAMBDA (int iGlob) {
	 	  int rx,ry;
	 	  int nGlob = dom.ny*dom.hc;
		  unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
	    int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc-rx-1;
	    int ii2=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx+rx;
		  a(ii1) = haloRecvBufW(nUnpack*nGlob+iGlob);
		  a(ii2) = haloRecvBufE(nUnpack*nGlob+iGlob);
    });
  }

	#if SERGHEI_SCALAR_TRANSPORT
		inline void pushSingleScalarHaloX(const int &iphi, realArr &a, Domain &dom, ADEsolver &ade){
			Kokkos::parallel_for("push_single_scalar_haloX", dom.ny*dom.hc, KOKKOS_LAMBDA(int iGlob){
				int rx,ry;
				unpackIndicesUniformGrid(iGlob,dom.ny,dom.hc,ry,rx);
				int ii1=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc-rx-1;
				int ii2=(dom.hc+ry)*(dom.nx+2*dom.hc)+dom.hc+dom.nx+rx;
				ade.hphi(ii1,iphi) = a(ii1);
				ade.hphi(ii2,iphi) = a(ii2);
			});
		}
	#endif
	// just a wrapper to deal with buffered data in the class
  /*inline void haloUnpack_bufdata_x(Domain &dom) {
		haloUnpack_x(dom,bufdata);
  }	*/


  inline void haloUnpack_y(Domain &dom, realArr &a) {
    haloUnpack_y_ext(dom, a, haloRecvBufS, haloRecvBufN, nUnpack);
    nUnpack = nUnpack + 1;
  }

  inline void haloUnpack_y_ext(Domain &dom, realArr &a, realArr &haloRecvBufS, realArr &haloRecvBufN, int const nUnpack) {
    Kokkos::parallel_for( "haloUnpack_y_span",dom.hc*dom.nx , KOKKOS_LAMBDA (int iGlob) {
		  int rx,ry;
		  int nGlob = dom.hc*dom.nx;
		  unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
	 	  int ii1=(dom.hc-ry-1)*(dom.nx+2*dom.hc)+dom.hc+rx;
	 	  int ii2=(dom.hc+dom.ny+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;
      a(ii1) = haloRecvBufN(nUnpack*nGlob+iGlob);
      a(ii2) = haloRecvBufS(nUnpack*nGlob+iGlob);
    });
  }

	#if SERGHEI_SCALAR_TRANSPORT
		inline void pushSingleScalarHaloY(const int &iphi, realArr &a, Domain &dom, ADEsolver &ade){
			Kokkos::parallel_for("push_single_scalar_haloY", dom.hc*dom.nx , KOKKOS_LAMBDA(int iGlob){
				int rx,ry;
				unpackIndicesUniformGrid(iGlob,dom.hc,dom.nx,ry,rx);
				int ii1=(dom.hc-ry-1)*(dom.nx+2*dom.hc)+dom.hc+rx;
				int ii2=(dom.hc+dom.ny+ry)*(dom.nx+2*dom.hc)+dom.hc+rx;
				ade.hphi(ii1,iphi) = a(ii1);
				ade.hphi(ii2,iphi) = a(ii2);
			});
		}
	#endif
	// just a wrapper to deal with buffered data in the class
	/*inline void haloUnpack_bufdata_y(Domain &dom){
		haloUnpack_y(dom,bufdata);
	}	*/



	// MPI wrapper to exchange east/west halo regions
	inline int haloExchange_x(Domain &dom, Parallel &par) {
	#if SERGHEI_DEBUG_MPI
	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
	#endif
    	int ierr;
		ierr=1;

    	if (par.nproc_x > 1) {
      		Kokkos::fence("haloExchangeMPI");	// ensure no kernels are running, we need everything available in host memory
			timerMPI.reset();

			//Pre-post the receives
      		ierr = MPI_Irecv( haloRecvBufW.data() , nPack*dom.ny*dom.hc , SERGHEI_MPI_REAL , par.neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
      		ierr = MPI_Irecv( haloRecvBufE.data() , nPack*dom.ny*dom.hc , SERGHEI_MPI_REAL , par.neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );

     		//Send the data
      		ierr = MPI_Isend( haloSendBufW.data() , nPack*dom.ny*dom.hc , SERGHEI_MPI_REAL , par.neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
      		ierr = MPI_Isend( haloSendBufE.data() , nPack*dom.ny*dom.hc , SERGHEI_MPI_REAL , par.neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );

      		//Wait for the sends and receives to finish
      		ierr = MPI_Waitall(2, sReq, sStat);
      		ierr = MPI_Waitall(2, rReq, rStat);
			dom.timers.swe.halo.mpi += timerMPI.seconds();
    	}

		return ierr;

	}


	inline int haloExchange_y(Domain &dom, Parallel &par) {
	#if SERGHEI_DEBUG_MPI
	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
	#endif
   	int ierr;
		ierr=1;

    	if (par.nproc_y > 1) {
    		Kokkos::fence("haloExchangeMPI");
			timerMPI.reset();

			//Pre-post the receives
			ierr = MPI_Irecv( haloRecvBufN.data() , nPack*dom.hc*dom.nx , SERGHEI_MPI_REAL , par.neigh(0,1) , 0 , MPI_COMM_WORLD , &rReq[0] );
			ierr = MPI_Irecv( haloRecvBufS.data() , nPack*dom.hc*dom.nx , SERGHEI_MPI_REAL , par.neigh(2,1) , 1 , MPI_COMM_WORLD , &rReq[1] );

			//Send the data
			ierr = MPI_Isend( haloSendBufN.data() , nPack*dom.hc*dom.nx , SERGHEI_MPI_REAL , par.neigh(0,1) , 1 , MPI_COMM_WORLD , &sReq[0] );
			ierr = MPI_Isend( haloSendBufS.data() , nPack*dom.hc*dom.nx , SERGHEI_MPI_REAL , par.neigh(2,1) , 0 , MPI_COMM_WORLD , &sReq[1] );

			//Wait for the sends and receives to finish
			ierr = MPI_Waitall(2, sReq, sStat);
			ierr = MPI_Waitall(2, rReq, rStat);

			dom.timers.swe.halo.mpi += timerMPI.seconds();
    	}

		return ierr;
	}



	inline int imposeTransmissiveOuterHalo_x(Domain &dom, Parallel &par) {
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
		if(dom.iW){
			haloTransmissive(nPack*dom.ny*dom.hc, haloSendBufW, haloRecvBufW);
		}
		if(dom.iE){
			haloTransmissive(nPack*dom.ny*dom.hc, haloSendBufE, haloRecvBufE);
		}
		return 1;
	}
	inline int imposeReflectiveOuterHalo_x(Domain &dom, Parallel &par) {
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
		if(dom.iW){
			haloReflective(nPack*dom.ny*dom.hc, haloRecvBufW);
		}
		if(dom.iE){
			haloReflective(nPack*dom.ny*dom.hc, haloRecvBufE);
		}
		return 1;
	}

	inline int imposeTransmissiveOuterHalo_y(Domain &dom, Parallel &par) {
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
		if(dom.iN){
			haloTransmissive(nPack*dom.hc*dom.nx, haloSendBufN, haloRecvBufN);
		}
		if(dom.iS){
			haloTransmissive(nPack*dom.hc*dom.nx, haloSendBufS, haloRecvBufS);
		}
		return 1;
	}
	inline int imposeReflectiveOuterHalo_y(Domain &dom, Parallel &par) {
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nPack = " << nPack << std::endl;
		#endif
		if(dom.iN){
			haloReflective(nPack*dom.hc*dom.nx, haloRecvBufN);
		}
		if(dom.iS){
			haloReflective(nPack*dom.hc*dom.nx, haloRecvBufS);
		}
		return 1;
	}

  /*inline void haloPeriodic(const int ncells, realArr const &haloSendBuf1, realArr const &haloSendBuf2, realArr &haloRecvBuf1, realArr &haloRecvBuf2) {
    Kokkos::parallel_for("haloPeriodic", ncells , KOKKOS_LAMBDA (int iGlob) {
		haloRecvBuf1(iGlob) = haloSendBuf2(iGlob);
      haloRecvBuf2(iGlob) = haloSendBuf1(iGlob);
    });
  }*/

  inline void haloReflective(const int ncells, realArr &haloRecv) {
    Kokkos::parallel_for("haloReflective", ncells , KOKKOS_LAMBDA (int iGlob) {
      haloRecv(iGlob) = 0.0;
    });
  }

  inline void haloTransmissive(const int ncells, realArr const &haloSend, realArr &haloRecv) {
    Kokkos::parallel_for("haloTransmissive", ncells , KOKKOS_LAMBDA (int iGlob) {
		haloRecv(iGlob) = haloSend(iGlob);
    });
  }




	// Generic MPI exchange for a single realArr field (e.g., h or z)
	inline void exchangeMPIfield(Domain &dom, Parallel &par, realArr &field){
		if constexpr(SERGHEI_DEBUG_MPI) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		// Exchange in x-direction
		haloInit();
		haloPack_x(dom, field);
		haloExchange_x(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_x(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE) imposeReflectiveOuterHalo_x(dom, par);
		haloUnpack_x(dom, field);

		// Exchange in y-direction
		haloInit();
		haloPack_y(dom, field);
		haloExchange_y(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_y(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE) imposeReflectiveOuterHalo_y(dom, par);
		haloUnpack_y(dom, field);
	}

  inline void exchangeMPIz(State &state, Domain &dom,  Parallel &par){
    exchangeMPIfield(dom, par, state.z);
  }
  inline void exchangeMPIh(State &state, Domain &dom,  Parallel &par){
    exchangeMPIfield(dom, par, state.h);
  }

/*
	// high level wrapper for bed level
	inline void exchangeMPIz(State &state, Domain &dom,  Parallel &par){
	if constexpr(SERGHEI_DEBUG_MPI)	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

		 // Exchange depth in x-direction
		 haloInit      ();
		 haloPack_x   (dom, state.z);		// re-orders the entries in state.z array which are on the east/west halo regions into data packs which will be sent east and west
		 //dom.timers.sedPack += timer.seconds();
		 //timer.reset();
		 haloExchange_x(dom, par);			// MPI send/receives the data packs, and handles boundaries
		 //dom.timers.sedExchange += timer.seconds();
		 //timer.reset();
     if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
		  imposeTransmissiveOuterHalo_x(dom, par);
     }
     if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
      imposeReflectiveOuterHalo_x(dom, par);
     }
		 haloUnpack_x (dom, state.z);		// re-order the updated data packs back into the halo regions in state.z array


		 // Exchange depth in y-direction
		 haloInit      ();
		 haloPack_y   (dom, state.z);		// re-orders the entries in state.z array which are on the north/south halo regions into data packs which will be sent north and south
		 //dom.timers.sedPack += timer.seconds();
		 //timer.reset();
		 haloExchange_y(dom, par);			// MPI send/receives the data packs, and handles boundaries
		 //dom.timers.sedExchange += timer.seconds();
		 //timer.reset();
		 if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
      imposeTransmissiveOuterHalo_y(dom, par);
     }
     if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
      imposeReflectiveOuterHalo_y(dom, par);
     }
		 haloUnpack_y (dom, state.z);		// re-order the updated data packs back into the state.z array
		 //dom.timers.sedPack += timer.seconds();
		 #if SERGHEI_SEDIMENT_TRANSPORT
      dom.timers.st.halo += timerST.seconds();
    #endif
	}

	// high level wrapper for water depth
	inline void exchangeMPIh(State &state, Domain &dom,  Parallel &par){
	#if SERGHEI_DEBUG_MPI
	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	#endif
		Kokkos::fence("exchangeMPIh-start");
		timerHalo.reset();	// only to keep track of time

		 // Exchange depth in x-direction
		 haloInit      ();
		 haloPack_x   (dom, state.h);		// re-orders the entries in state.h array which are on the east/west halo regions into data packs which will be sent east and west
		 haloExchange_x(dom, par);			// MPI send/receives the data packs, and handles boundaries
			// the outer boundaries are assumed to be periodic by default in the previous exchange
			// we now need to manage boundary conditions (halo) for outer domain
			if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
				imposeTransmissiveOuterHalo_x(dom, par);
			}
			if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
				imposeReflectiveOuterHalo_x(dom, par);
			}
		 haloUnpack_x (dom, state.h);		// re-order the updated data packs back into the halo regions in state.h array

		 // Exchange depth in y-direction
		 Kokkos::fence("exchangeMPIh-x-done");	// ensure X-halo unpack is visible before packing Y boundary rows
		 haloInit      ();
		 haloPack_y   (dom, state.h);		// re-orders the entries in state.h array which are on the north/south halo regions into data packs which will be sent north and south
		 haloExchange_y(dom, par);			// MPI send/receives the data packs, and handles boundaries
			// the outer boundaries are assumed to be periodic by default in the previous exchange
			// we now need to manage boundary conditions (halo) for outer domain
			if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
				imposeTransmissiveOuterHalo_y(dom, par);
			}
			if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
				imposeReflectiveOuterHalo_y(dom, par);
			}
		 haloUnpack_y (dom, state.h);		// re-order the updated data packs back into the state.h array
		 Kokkos::fence("exchangeMPIh-end");
    	 dom.timers.swe.halo.total += timerHalo.seconds();
	}
*/

	// high level wrapper for momentum
	inline void exchangeMPIhuhv(State &state, Domain &dom,  Parallel &par){
	#if SERGHEI_DEBUG_MPI
	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	#endif

		Kokkos::fence("exchangeMPIhuhv-start");
		timerHalo.reset();

		 // Exchange momentum in x-direction
		 haloInit      ();
		 haloPack_x   (dom, state.hu);
		 haloPack_x   (dom, state.hv);
		 haloExchange_x(dom, par);
			// the outer boundaries are assumed to be periodic by default in the previous exchange
			// we now need to manage boundary conditions (halo) for outer domain
			if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_x(dom, par);
			if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE) imposeReflectiveOuterHalo_x(dom, par);
		 haloUnpack_x (dom, state.hu);
		 haloUnpack_x (dom, state.hv);

		 // Exchange momentum in y-direction
		 Kokkos::fence("exchangeMPIhuhv-x-done");	// ensure X-halo unpack is visible before packing Y boundary rows
		 haloInit      ();
		 haloPack_y   (dom, state.hu);
		 haloPack_y   (dom, state.hv);
		 haloExchange_y(dom, par);
		 // the outer boundaries are assumed to be periodic by default in the previous exchange
		 // we now need to manage boundary conditions (halo) for outer domain
		 if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_y(dom, par);
		 if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE) imposeReflectiveOuterHalo_y(dom, par);
		 haloUnpack_y (dom, state.hu);
		 haloUnpack_y (dom, state.hv);

		 Kokkos::fence("exchangeMPIhuhv-end");
		 dom.timers.swe.halo.total += timerHalo.seconds();
	}

#if SERGHEI_SCALAR_TRANSPORT
	inline void exchangeMPIscalars(State &state, Domain &dom,  Parallel &par){
		#if SERGHEI_DEBUG_MPI
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		timerADE.reset();

		 haloInit      ();
		 for(int iphi=0; iphi<state.ade.nScalar; iphi++){
			//bufdata = Kokkos::subview(state.ade.hphi,Kokkos::ALL,iphi);
			pullSingleScalarHaloX(iphi, state.ade, dom, bufdata);
			haloPack_x(dom, bufdata);
		 }
		 haloExchange_x(dom, par);
		// the outer boundaries are assumed to be periodic by default in the previous exchange
		// we now need to manage boundary conditions (halo) for outer domain
		if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
			imposeTransmissiveOuterHalo_x(dom, par);
		}
		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
			imposeReflectiveOuterHalo_x(dom, par);
		}
		 for(int iphi=0; iphi<state.ade.nScalar; iphi++){
			//bufdata = Kokkos::subview(state.ade.hphi,Kokkos::ALL,iphi);
		 	//haloUnpack_bufdata_x(dom);
			haloUnpack_x(dom, bufdata);
			pushSingleScalarHaloX(iphi, bufdata, dom, state.ade);
		 }

		 haloInit      ();
		 for(int iphi=0; iphi<state.ade.nScalar; iphi++){
			//bufdata = Kokkos::subview(state.ade.hphi,Kokkos::ALL,iphi);
			pullSingleScalarHaloY(iphi, state.ade, dom, bufdata);
		 	haloPack_y   (dom, bufdata);
		 }
		 haloExchange_y(dom, par);
			// the outer boundaries are assumed to be periodic by default in the previous exchange
			// we now need to manage boundary conditions (halo) for outer domain
			if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE){
				imposeTransmissiveOuterHalo_y(dom, par);
			}
			if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE){
				imposeReflectiveOuterHalo_y(dom, par);
			}
		 for(int iphi=0; iphi<state.ade.nScalar; iphi++){
			//bufdata = Kokkos::subview(state.ade.hphi,Kokkos::ALL,iphi);
		 	//haloUnpack_bufdata_y(dom);
			haloUnpack_y(dom,bufdata);
			pushSingleScalarHaloY(iphi, bufdata, dom, state.ade);
		 }
		 dom.timers.ade.halo += timerADE.seconds();
	}
#endif

	// high level wrapper to do the initial exchange of surface parameters
	inline void iniMPI(State &state, Domain &dom, Parallel &par){
	  if constexpr(SERGHEI_DEBUG_MPI) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

		// Allocate the MPI exchange buffers
		allocate(dom);

		// Transfer isnodata index to internal halos
		haloTransfer_x_bool(dom, par, state.isnodata);
		haloTransfer_y_bool(dom, par, state.isnodata);

    /*
		haloInit      ();
		haloPack_x   (dom, state.roughness);
		haloExchange_x(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_x(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE)   imposeReflectiveOuterHalo_x  (dom, par);
		haloUnpack_x (dom, state.roughness);

		// Exchange bed elevation and roughness in y-direction
		haloInit      ();
		haloPack_y   (dom, state.roughness);
		haloExchange_y(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_TRANSMISSIVE) imposeTransmissiveOuterHalo_y(dom, par);
		if(dom.BCtype==TOPOLOGY_BC_REFLECTIVE)   imposeReflectiveOuterHalo_y  (dom, par);
		haloUnpack_y (dom, state.roughness);
    */

		exchangeMPIz(state,dom,par);
		exchangeMPIh(state,dom,par);
		exchangeMPIfield(dom, par, state.roughness);
    exchangeMPIhuhv(state,dom,par);
		#if SERGHEI_SCALAR_TRANSPORT
			exchangeMPIscalars(state,dom,par);
		#endif

	}

};

#endif
