#ifndef _GWMPI_H_
#define _GWMPI_H_

#include "define.h"
#include "Exchange.h"
#include "mpi.h"
#include "Indexing.h"
#include "Parallel.h"

class GwMPI : public Exchange {

protected:

    int const maxPack = 3; //the number of sw variables

    MPI_Request sReq[2];
    MPI_Request rReq[2];

    MPI_Status  sStat[2];
    MPI_Status  rStat[2];

    int nBufS, nBufN, nBufW, nBufE;
    intArr idxBufS, idxBufN, idxBufW, idxBufE;
    realArr haloSendBufS, haloSendBufN, haloSendBufW, haloSendBufE;
    realArr haloRecvBufS, haloRecvBufN, haloRecvBufW, haloRecvBufE;

public:
    real MPItime=0;
    real exchangeTime=0;

    inline void allocate(GwDomain &gdom) {
        int ii, jj, kk, nxhalo, nyhalo, idxS, idxN, idxW, idxE;
        if (gdom.hc != 1)    {
            std::cerr << RERROR "ERROR: hc must be 1 for subsurface halo exchange!" << std::endl;
        }
        nBufS = gdom.nx*gdom.nz;
        nBufN = gdom.nx*gdom.nz;
        nBufW = gdom.ny*gdom.nz;
        nBufE = gdom.ny*gdom.nz;
        // haloSendBuf* and haloRecvBuf* arrays are 1D containers (arrays) which should be able to
        // store all state variables, for the exchanging band
        haloSendBufS = realArr("haloSendBufS",nBufS);
        haloSendBufN = realArr("haloSendBufN",nBufN);
        haloSendBufW = realArr("haloSendBufW",nBufW);
        haloSendBufE = realArr("haloSendBufE",nBufE);
        haloRecvBufS = realArr("haloRecvBufS",nBufS);
        haloRecvBufN = realArr("haloRecvBufN",nBufN);
        haloRecvBufW = realArr("haloRecvBufW",nBufW);
        haloRecvBufE = realArr("haloRecvBufE",nBufE);
        // store the cell id of the buffers
        idxBufS = intArr("idxBufS",nBufS);
        idxBufN = intArr("idxBufN",nBufN);
        idxBufW = intArr("idxBufW",nBufW);
        idxBufE = intArr("idxBufE",nBufE);
        // calculate buffer index
        nxhalo = gdom.nx+2*gdom.hc;    nyhalo = gdom.ny+2*gdom.hc;
        idxW = 0;   idxE = 0;   idxS = 0;   idxN = 0;
        for (int iGlob = 0; iGlob < gdom.nCellMem; iGlob++)  {
            gdom.unpackIndicesGw(iGlob, gdom.nz+2*gdom.hc, gdom.ny+2*gdom.hc, gdom.nx+2*gdom.hc, kk, jj, ii);
            if (ii == 0)    {
                if (jj > 0 & jj < gdom.ny+1 & kk > 0 & kk < gdom.nz+1)    {idxBufW(idxW) = iGlob;  idxW += 1;}
            }
            else if (ii == gdom.nx+1)   {
                if (jj > 0 & jj < gdom.ny+1 & kk > 0 & kk < gdom.nz+1)    {idxBufE(idxE) = iGlob;  idxE += 1;}
            }
            if (jj == 0)    {
                if (ii > 0 & ii < gdom.nx+1 & kk > 0 & kk < gdom.nz+1)    {idxBufN(idxN) = iGlob;  idxN += 1;}
            }
            else if (jj == gdom.ny+1)   {
                if (ii > 0 & ii < gdom.nx+1 & kk > 0 & kk < gdom.nz+1)    {idxBufS(idxS) = iGlob;  idxS += 1;}
            }
        }
    }


    // Top level MPI exchange functions for realArr2
    inline void mpi_sendrecv(realArr2 var, GwDomain &gdom, Parallel &par){
        int ierr, nx = gdom.nx+2*gdom.hc;
        Kokkos::Timer timer;	// only to keep track of time
        // x direction
        if (par.nproc_x > 1) {
            for (int ii = 0; ii < var.extent(1); ii++) {
                haloPack_x(haloSendBufE, haloSendBufW, var, idxBufE, idxBufW, ii);
                ierr = haloSendRecv_x(gdom, par);
                haloUnPack_x(haloRecvBufE, haloRecvBufW, var, idxBufE, idxBufW, ii);
            }
        }
        // y direction
        if (par.nproc_y > 1) {
            for (int ii = 0; ii < var.extent(1); ii++) {
                haloPack_y(haloSendBufS, haloSendBufN, var, idxBufS, idxBufN, ii, nx);
                ierr = haloSendRecv_y(gdom, par);
                haloUnPack_y(haloRecvBufS, haloRecvBufN, var, idxBufS, idxBufN, ii);
            }
        }
        if (par.masterproc) {exchangeTime += timer.seconds();}
    }

    inline void mpi_sendrecv1(realArr var, GwDomain &gdom, Parallel &par){
        int ierr, nx = gdom.nx+2*gdom.hc;
        Kokkos::Timer timer;	// only to keep track of time
        // x direction
        if (par.nproc_x > 1) {
            haloPack_x1(haloSendBufE, haloSendBufW, var, idxBufE, idxBufW);
            ierr = haloSendRecv_x(gdom, par);
            haloUnPack_x1(haloRecvBufE, haloRecvBufW, var, idxBufE, idxBufW);
        }
        // y direction
        if (par.nproc_y > 1) {
            haloPack_y1(haloSendBufS, haloSendBufN, var, idxBufS, idxBufN, nx);
            ierr = haloSendRecv_y(gdom, par);
            haloUnPack_y1(haloRecvBufS, haloRecvBufN, var, idxBufS, idxBufN);
        }
        if (par.masterproc) {exchangeTime += timer.seconds();}
    }

    //  ---------------------------------------------
    //  Exchange in the E-W direction
    //  ---------------------------------------------
    inline void haloPack_x(realArr bufp, realArr bufm, realArr2 a, intArr idxp, intArr idxm, int col)    {
        Kokkos::parallel_for(nBufE , KOKKOS_LAMBDA (int idx) {
            bufm(idx) = a(idxm(idx)+1,col);
            bufp(idx) = a(idxp(idx)-1,col);
        });
    }

    // MPI Exchange
    inline int haloSendRecv_x(GwDomain &gdom, Parallel &par)    {
        int ierr=1, nPack=1;
    	if (par.nproc_x > 1) {
          	Kokkos::fence();	// ensure no kernels are running, we need everything available in host memory

            //Pre-post the receives
            ierr = MPI_Irecv( haloRecvBufW.data() , nBufW , SERGHEI_MPI_REAL , par.neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
            ierr = MPI_Irecv( haloRecvBufE.data() , nBufE , SERGHEI_MPI_REAL , par.neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
            //Send the data
            ierr = MPI_Isend( haloSendBufW.data() , nBufW , SERGHEI_MPI_REAL , par.neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
            ierr = MPI_Isend( haloSendBufE.data() , nBufE , SERGHEI_MPI_REAL , par.neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );

          	//Wait for the sends and receives to finish
          	ierr = MPI_Waitall(2, sReq, sStat);
          	ierr = MPI_Waitall(2, rReq, rStat);
    	}
        return ierr;
    }

    // Receive buffers
    inline void haloUnPack_x(realArr bufp, realArr bufm, realArr2 a, intArr idxp, intArr idxm, int col)    {
        Kokkos::parallel_for(nBufE , KOKKOS_LAMBDA (int idx) {
             a(idxm(idx),col) = bufm(idx);
             a(idxp(idx),col) = bufp(idx);
        });
    }

    // For realArr
    inline void haloPack_x1(realArr bufp, realArr bufm, realArr a, intArr idxp, intArr idxm)    {
        Kokkos::parallel_for(nBufE , KOKKOS_LAMBDA (int idx) {
            bufm(idx) = a(idxm(idx)+1);
            bufp(idx) = a(idxp(idx)-1);
        });
    }
    inline void haloUnPack_x1(realArr bufp, realArr bufm, realArr a, intArr idxp, intArr idxm)    {
        Kokkos::parallel_for(nBufE , KOKKOS_LAMBDA (int idx) {
             a(idxm(idx)) = bufm(idx);
             a(idxp(idx)) = bufp(idx);
        });
    }
    //  ---------------------------------------------
    //  End exchange in the E-W direction
    //  ---------------------------------------------









    //  ---------------------------------------------
    //  Exchange in the N-S direction
    //  ---------------------------------------------

    inline void haloPack_y(realArr bufp, realArr bufm, realArr2 a, intArr idxp, intArr idxm, int col, int nx)    {
        Kokkos::parallel_for(nBufS , KOKKOS_LAMBDA (int idx) {
            bufm(idx) = a(idxm(idx)+nx,col);
            bufp(idx) = a(idxp(idx)-nx,col);
        });
    }

    // MPI Exchange
    inline int haloSendRecv_y(GwDomain &gdom, Parallel &par)    {
        int ierr=1, nPack=1;
    	if (par.nproc_y > 1) {
          	Kokkos::fence();	// ensure no kernels are running, we need everything available in host memory

            //Pre-post the receives
            ierr = MPI_Irecv( haloRecvBufN.data() , nBufN , SERGHEI_MPI_REAL , par.neigh(0,1) , 0 , MPI_COMM_WORLD , &rReq[0] );
            ierr = MPI_Irecv( haloRecvBufS.data() , nBufS , SERGHEI_MPI_REAL , par.neigh(2,1) , 1 , MPI_COMM_WORLD , &rReq[1] );
            //Send the data
            ierr = MPI_Isend( haloSendBufN.data() , nBufN , SERGHEI_MPI_REAL , par.neigh(0,1) , 1 , MPI_COMM_WORLD , &sReq[0] );
            ierr = MPI_Isend( haloSendBufS.data() , nBufS , SERGHEI_MPI_REAL , par.neigh(2,1) , 0 , MPI_COMM_WORLD , &sReq[1] );

          	//Wait for the sends and receives to finish
          	ierr = MPI_Waitall(2, sReq, sStat);
          	ierr = MPI_Waitall(2, rReq, rStat);
    	}
        return ierr;
    }

    // Receive buffers
    inline void haloUnPack_y(realArr bufp, realArr bufm, realArr2 a, intArr idxp, intArr idxm, int col)    {
        Kokkos::parallel_for(nBufS , KOKKOS_LAMBDA (int idx) {
             a(idxm(idx),col) = bufm(idx);
             a(idxp(idx),col) = bufp(idx);
        });
    }

    // For realArr
    inline void haloPack_y1(realArr bufp, realArr bufm, realArr a, intArr idxp, intArr idxm, int nx)    {
        Kokkos::parallel_for(nBufS , KOKKOS_LAMBDA (int idx) {
            bufm(idx) = a(idxm(idx)+nx);
            bufp(idx) = a(idxp(idx)-nx);
        });
    }
    inline void haloUnPack_y1(realArr bufp, realArr bufm, realArr a, intArr idxp, intArr idxm)    {
        Kokkos::parallel_for(nBufS , KOKKOS_LAMBDA (int idx) {
             a(idxm(idx)) = bufm(idx);
             a(idxp(idx)) = bufp(idx);
        });
    }
    //  ---------------------------------------------
    //  End exchange in the N-S direction
    //  ---------------------------------------------




};

#endif
