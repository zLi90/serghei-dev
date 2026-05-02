/* -*- mode: c++ -*- */

#ifndef _GWBC_H_
#define _GWBC_H_

#if SERGHEI_RE_MODEL

#include "define.h"
#include "Indexing.h"
#include "GwDomain.h"
#include "GwState.h"
#include "Parallel.h"

// subsurface bc types
#define SUB_BC_NOFLOW 1
#define SUB_BC_H_CONST 2
#define SUB_BC_Q_CONST 3
#define SUB_BC_WT_CONST 4
#define SUB_BC_H_T 5
#define SUB_BC_Q_T 6
#define SUB_BC_WT_T 7
#define SUB_BC_SWE 8
#define SUB_BC_FD 9

// subsurface bc directions
#define XPLUS 1
#define XMINUS 2
#define YPLUS 3
#define YMINUS 4
#define ZPLUS 5
#define ZMINUS 6


class GwBC {
// this class is safe to invoke in a parallel region
public:
	int ncellsBC = 0; //number of bcells
	int ncellsIT = 0;	// number of internal source/sink cells
	intArr bcells, gcells, icells; //array of indexes of boundary cells
    intArr swgw_type; // type of surface-subsurface exchange: 0: No ponding, no sw-gw exchange, 1: Ponding with large h, 2: Ponding with small h
	int location, bctype, isInDomain, direction;
    realArr bcvals, bcdata;
	TimeSeries ts;
	real Qtot, Qinflow, Qoutflow;

	MPI_Comm comm;	// communicator for ranks associated to the BC

	inline int find_bcells(GwState &gw, std::string &id, GwDomain &gdom, Parallel &par, int nPoly, realArr &xPoly, realArr &yPoly){
		int foundInSubdom; // to keep track of which subdomains are associated to this boundary
		std::vector<int> tmpbcells; //array of indexes of boundary cells
		std::vector<int> tmpgcells; //array of indexes of ghost cells
		std::vector<int> subdomains;	// keeps track of which subdomains are associated to the BC
		
		// Loop over the entire domain to find bc cells
		int onBoundary;

		for (int kk = 0; kk < gdom.nz; kk++) {
			for (int jj = 0; jj < gdom.ny; jj++) {
				for (int ii = 0; ii < gdom.nx; ii++) {
					int iGlob = (gdom.hc+kk)*gdom.nxhc*gdom.nyhc + (gdom.hc+jj)*gdom.nxhc + ii + gdom.hc;
					foundInSubdom = -1;
		            real xCoord = gdom.xll + ( par.i_beg + ii + 0.5) * gdom.dx;
		            real yCoord = gdom.yll + gdom.ny_glob*gdom.dx - ( par.j_beg + jj + 0.5) * gdom.dx;
		            
		            // Check if the cell is on the boundary 
		            onBoundary = 0;
		            if (direction == ZMINUS || direction == ZPLUS)	{
						onBoundary = 1;
		            }
		            else {
		            	if (gdom.isnodata(iGlob) == 0)	{
		            		if (ii == 0 || ii == gdom.nx-1 || jj == 0 || jj == gdom.ny-1)	{onBoundary = 1;}
		            		else {
		            			if (gdom.isnodata(iGlob+1) == 1 && direction == XPLUS) {onBoundary = 1;}
		            			else if (gdom.isnodata(iGlob-1) == 1 && direction == XMINUS) {onBoundary = 1;}
		            			else if (gdom.isnodata(iGlob+gdom.nxhc) == 1 && direction == YPLUS) {onBoundary = 1;}
		            			else if (gdom.isnodata(iGlob-gdom.nxhc) == 1 && direction == YMINUS) {onBoundary = 1;}
		            		}
		            	}
		            }
		       		// only keep cells on the boundary
		            if (geometry::isInsidePoly(nPoly, xPoly, yPoly, xCoord, yCoord) && onBoundary == 1){
		                // If on top/bottom boundary, only the top/bottom layer counts
		                if (direction == ZPLUS) {
							if (kk == gdom.nz-1) {
								tmpbcells.push_back(iGlob);
								tmpgcells.push_back(iGlob+gdom.nxhc*gdom.nyhc);
							}
						}
		                else if (direction == ZMINUS)    {
							if (kk == 0)  {
								tmpbcells.push_back(iGlob);
								tmpgcells.push_back(iGlob-gdom.nxhc*gdom.nyhc);
							}
						}
		                // Otherwise (lateral boundary), all cells in the vertical direction are included
		                // NOTE: In the future, this should be customized to allow only certain vertical layers to be included
		                else {
							tmpbcells.push_back(iGlob);
							if (direction == XPLUS)	{tmpgcells.push_back(iGlob+1);}
							else if (direction == XMINUS)	{tmpgcells.push_back(iGlob-1);}
							else if (direction == YPLUS)	{tmpgcells.push_back(iGlob+gdom.nxhc);}
							else if (direction == YMINUS)	{tmpgcells.push_back(iGlob-gdom.nxhc);}
							
							// save on boundary info
		            		gdom.onboundary(iGlob) = 1;
							
						}
		            }		            
				}
			}
		}

		ncellsBC=int(tmpbcells.size());
		if(ncellsBC > 0) foundInSubdom = par.myrank; // if at least one cell in this subdomain (rank) is in the BC, tag as found
		

		int ncells_all;
        MPI_Allreduce(&ncellsBC, &ncells_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		int *subdoms;
		subdoms = (int*) malloc(par.nranks * sizeof(int));
		MPI_Allgather(&foundInSubdom,1,MPI_INT,subdoms,1,MPI_INT,MPI_COMM_WORLD);
		#if SERGHEI_DEBUG_BOUNDARY
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "GW ncellsBC " << ncells_all << std::endl;
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "GW BC subdomains: " ;
			for (int i=0; i<par.nranks; i++){
				std::cout << GGD << " ";
				if(subdoms[i]==par.myrank) std::cout << RED;
				std::cout << subdoms[i] << "\t"<< RESET ;
			}
			std::cout << std::endl;
		#endif

		for(int i=0; i<par.nranks; i++){
			if(subdoms[i] >= 0){
				subdomains.push_back(subdoms[i]);
			}
		}
		#if SERGHEI_DEBUG_BOUNDARY
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "GW Consolidated BC subdomains = ";
			for(int i=0; i<subdomains.size(); i++){
				std::cout << GGD << " " ;
				std::cout << subdomains[i] << "\t";
			}
 			std::cout << std::endl;
		#endif
		MPI_Group group, subgroup;
		MPI_Comm_group(MPI_COMM_WORLD,&group);
		MPI_Group_incl(group,subdomains.size(),subdomains.data(),&subgroup);
		MPI_Comm_create(MPI_COMM_WORLD,subgroup,&comm);

		//we need the total boundary cells detected by all subdomain to launch an error otherwise
		if(ncells_all>0){
			bcells=intArr("bcells", ncellsBC);
			gcells=intArr("gcells", ncellsBC);
            if (bctype == SUB_BC_SWE)   {swgw_type = intArr("swgw_type", ncellsBC);}
			#ifdef KOKKOS_ENABLE_CUDA
				cudaMemcpyAsync( bcells.data() , tmpbcells.data() , ncellsBC*sizeof(int) , cudaMemcpyHostToDevice );
				cudaMemcpyAsync( gcells.data() , tmpgcells.data() , ncellsBC*sizeof(int) , cudaMemcpyHostToDevice );
				cudaDeviceSynchronize();
			#else
				std::memcpy(bcells.data(), tmpbcells.data(), ncellsBC*sizeof(int));
				std::memcpy(gcells.data(), tmpgcells.data(), ncellsBC*sizeof(int));
			#endif
		}
		else{
			if(par.masterproc){
				std::cerr << RERROR << "No boundary cells found for subsurface boundary with id '" << id << "'" << std::endl;
			}
			return 0;
		}
		return 1;
	}


    
    



    // Apply subsurface boundary conditions
    inline void applyHBC(GwState &gw, GwDomain &gdom, Parallel &par) {
		// Check if on global boundaries
		bool onBoundary = 0;
		if (direction == 2 && par.px == 0)	{onBoundary = 1;}
		else if (direction == 1 && par.px == par.nproc_x-1)	{onBoundary = 1;}
		else if (direction == 4 && par.py == 0)	{onBoundary = 1;}
		else if (direction == 3 && par.py == par.nproc_y-1)	{onBoundary = 1;}
		else if (direction == 5 || direction == 6)	{onBoundary = 1;}

        // Kokkos::Timer timer;
        if (ncellsBC > 0 && onBoundary == 1) {
            real hbc;
			// interpolate if time-series boundary value is read
            if (bctype == SUB_BC_H_T || bctype == SUB_BC_WT_T) {
				if (ts.nc == 1)	{hbc = interpolateLinear(ts, gdom.etime);}
			}
            // zero gradient if Q BC is specified
            if (bctype == SUB_BC_Q_CONST || bctype == SUB_BC_Q_T || bctype == SUB_BC_FD)   {
                Kokkos::parallel_for("gw_bc_h", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                    int iGlob = bcells[ibc], iGhost = gcells[ibc];
					gw.h(iGhost,1) = gw.h(iGlob,1);
                });
            }
			// H CONST 
			else if (bctype == SUB_BC_H_CONST)	{
				Kokkos::parallel_for("gw_bc_h", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                    int ivg, iGlob = bcells[ibc], iGhost = gcells[ibc];
                    real wcs, wcr, alpha, n;
                    ivg = gw.soilID(iGlob) * gw.nVGparam;
                    wcs = gw.vgTable(ivg+2);    wcr = gw.vgTable(ivg+3);
                    n = gw.vgTable(ivg+4);  alpha = gw.vgTable(ivg+6);
					gw.h(iGhost,1) = bcvals(ibc);
					gw.wc(iGhost,1) = h2wc(gw.h(iGhost,1), alpha, n, wcs, wcr);
                });
			}
            // WT CONST
            else if (bctype == SUB_BC_WT_CONST) {
				Kokkos::parallel_for("gw_bc_wt", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                    int ivg, iGlob = bcells[ibc], iGhost = gcells[ibc];
                    real wcs, wcr, alpha, n;
                    ivg = gw.soilID(iGlob) * gw.nVGparam;
                    wcs = gw.vgTable(ivg+2);    wcr = gw.vgTable(ivg+3);
                    n = gw.vgTable(ivg+4);  alpha = gw.vgTable(ivg+6);
					gw.h(iGhost,1) = bcvals(ibc) - gdom.z(iGlob);
					gw.wc(iGhost,1) = h2wc(gw.h(iGhost,1), alpha, n, wcs, wcr);
                });
            }
            // H Time series
            else if (bctype == SUB_BC_H_T)  {
                Kokkos::parallel_for("gw_bc_wt", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                    int ivg, iGlob = bcells[ibc], iGhost = gcells[ibc];
                    real wcs, wcr, alpha, n;
                    ivg = gw.soilID(iGlob) * gw.nVGparam;
                    wcs = gw.vgTable(ivg+2);    wcr = gw.vgTable(ivg+3);
                    n = gw.vgTable(ivg+4);  alpha = gw.vgTable(ivg+6);
					gw.h(iGhost,1) = hbc;
					gw.wc(iGhost,1) = h2wc(gw.h(iGhost,1), alpha, n, wcs, wcr);
                });
            }
            // Prescribed water table BC
            else if (bctype == SUB_BC_WT_T)  {

            	findTimeBlock(ts, gdom.etime);
  				int t_idx = ts.timeIndex;
  				int t_next = t_idx + 1;
  				if (t_idx == ts.np - 1) {t_next = t_idx;}

  				//std::cout << t_idx << ", " << ts.np << ", " << ts.nc << "\n";
                Kokkos::parallel_for("gw_bc_wt", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                    int ivg, iGlob = bcells[ibc], iGhost = gcells[ibc];
                    real wcs, wcr, alpha, n, wtbc;
                    ivg = gw.soilID(iGlob) * gw.nVGparam;
                    wcs = gw.vgTable(ivg+2);    wcr = gw.vgTable(ivg+3);
                    n = gw.vgTable(ivg+4);  alpha = gw.vgTable(ivg+6);
					// interpolate cell-by-cell water table
					if (ts.nc > 1)	{
  						wtbc = ts.values(t_idx,ibc) + 
  							(ts.values(t_next,ibc) - ts.values(t_idx,ibc))/(ts.time(t_next)-ts.time(t_idx))*(gdom.etime-ts.time(t_idx));
                        gw.h(iGhost,1) = wtbc - gdom.z(iGlob);
					}
					else {gw.h(iGhost,1) = hbc - gdom.z(iGlob);}
					gw.wc(iGhost,1) = h2wc(gw.h(iGhost,1), alpha, n, wcs, wcr);
                });
            }
            // Surface-subsurface exchange
            else if (bctype == SUB_BC_SWE)  {
				#if SERGHEI_SWE_MODEL
				if (direction == 6)	{
					Kokkos::parallel_for("gw_bc_swe", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
	                    int iGlob = bcells[ibc], iGhost = gcells[ibc], ivg = gw.soilID(iGlob) * NVG;
	                    real ks = gw.vgTable(ivg);
						int ii, jj, kk, iGlobSW;
						gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
						iGlobSW = jj*gdom.nxhc + ii;
						gw.h(iGhost,1) = gw.hs(iGlobSW);
						// get sw-gw exchange type
						if (gw.h(iGhost,1) > 0.0)    {
							real q_infilt = 2.0 * ks * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dz(iGlob) - ks;
							if (-q_infilt * gdom.dt <= gw.h(iGhost,1))   {swgw_type(ibc) = 1;}
							else {swgw_type(ibc) = 2;}
						}
						else {
							// exfiltration
							if (gw.h(iGlob,1) > gw.h(iGhost,1) + 0.5*gdom.dz(iGlob)) {swgw_type(ibc) = 1;}
							// no flow
							else {swgw_type(ibc) = 0;}
						}
	                });
				}
				else {
					if (par.masterproc)	{std::cerr << RERROR "BC direction must be 6 for SW-GW exchange boundary! " << "\n";}
				}
				#endif
            }
        }
        // gdom.timers.gw += timer.seconds();
    }


    // Apply boundary for K
    inline void applyKBC(GwState &gw, GwDomain &gdom, Parallel &par) {
		// Check if on global boundaries
		bool onBoundary = 0;
		if (direction == 2 && par.px == 0)	{onBoundary = 1;}
		else if (direction == 1 && par.px == par.nproc_x-1)	{onBoundary = 1;}
		else if (direction == 4 && par.py == 0)	{onBoundary = 1;}
		else if (direction == 3 && par.py == par.nproc_y-1)	{onBoundary = 1;}
		else if (direction == 5 || direction == 6)	{onBoundary = 1;}

        if (ncellsBC > 0 && onBoundary == 1) {
            Kokkos::parallel_for("gw_bc", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                int iGlob = bcells[ibc], iGhost = gcells[ibc], ivg = gw.soilID(iGlob) * NVG;
                real ks = gw.vgTable(ivg);
				if (direction == 1)	{
					if (gw.h(iGhost,1) >= 0.0 || gw.h(iGlob,1) >= 0.0)   {gw.k(iGlob,0) = ks;}
					else {gw.k(iGlob,0) = ks * gw.k(iGlob,3);}
				}
				else if (direction == 2)	{
					if (gw.h(iGhost,1) >= 0.0 || gw.h(iGlob,1) >= 0.0)   {gw.k(iGhost,0) = ks;}
					else {gw.k(iGhost,0) = ks * gw.k(iGlob,3);}
				}
				else if (direction == 3)	{
					if (gw.h(iGhost,1) >= 0.0 || gw.h(iGlob,1) >= 0.0)   {gw.k(iGlob,1) = ks;}
					else {gw.k(iGlob,1) = ks * gw.k(iGlob,3);}
				}
				else if (direction == 4)	{
					if (gw.h(iGhost,1) >= 0.0 || gw.h(iGlob,1) >= 0.0)   {gw.k(iGhost,1) = ks;}
					else {gw.k(iGhost,1) = ks * gw.k(iGlob,3);}
				}
				else if (direction == 5)	{
					if (bctype == SUB_BC_FD)	{gw.k(iGlob,2) = ks * gw.k(iGlob,3);}
					else {
						if (gw.h(iGhost,1) >= 0.0 || gw.h(iGlob,1) >= 0.0)   {gw.k(iGlob,2) = ks;}
						else {gw.k(iGlob,2) = 0.5 * ks * (gw.k(iGlob,3) + gw.k(iGhost,3));}
					}
				}
				else if (direction == 6)	{
					if (gw.h(iGhost,1) >= 0.0)   {gw.k(iGhost,2) = ks;}
					//else {gw.k(iGhost,2) = 0.0;}
					else {gw.k(iGhost,2) = 0.5 * ks * (gw.k(iGlob,3) + gw.k(iGhost,3));}
				}
            });
        }
    }

    // Apply boundary conditions for Q
    inline void applyQBC(GwState &gw, GwDomain &gdom, Parallel &par) {
		// Check if on global boundaries
		bool onBoundary = 0;
		if (direction == 2 && par.px == 0)	{onBoundary = 1;}
		else if (direction == 1 && par.px == par.nproc_x-1)	{onBoundary = 1;}
		else if (direction == 4 && par.py == 0)	{onBoundary = 1;}
		else if (direction == 3 && par.py == par.nproc_y-1)	{onBoundary = 1;}
		else if (direction == 5 || direction == 6)	{onBoundary = 1;}

        if (ncellsBC > 0 && onBoundary == 1) {
            real qbc;
			// interpolate if boundary flux is a time series
            if (bctype == SUB_BC_Q_T) {qbc = interpolateLinear(ts, gdom.etime);}
	  	    switch (bctype) {
              	default:
					// Note that the default settings do not need to be applied for all GwBC functions
					// because all functions in GwBC.h read the same input settings
                    std::cerr << RERROR "Boundary type: " << bctype << " not recognized for flux boundary." << std::endl;
        	        std::cerr << RERROR "No boundary condition applied." << std::endl;
        	        exit(EXIT_FAILURE);
                    break;
                case SUB_BC_H_CONST:
                case SUB_BC_WT_CONST:
                case SUB_BC_H_T:
                case SUB_BC_WT_T:
                    Kokkos::parallel_for("gw_bc_h_const", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                        int iGlob = bcells[ibc], iGhost = gcells[ibc];
						if (direction == 1)	{
							gw.q(iGlob,0) = 2.0 * gw.k(iGlob,0) * (gw.h(iGlob+1,1) - gw.h(iGlob,1)) / gdom.dx;
						}
						else if (direction == 2)	{
							gw.q(iGhost,0) = 2.0 * gw.k(iGhost,0) * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dx;
						}
						else if (direction == 3)	{
							gw.q(iGlob,1) = 2.0 * gw.k(iGlob,1) * (gw.h(iGlob+gdom.nxhc,1) - gw.h(iGlob,1)) / gdom.dy;
						}
						else if (direction == 4)	{
							gw.q(iGhost,1) = 2.0 * gw.k(iGhost,1) * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dy;
						}
						else if (direction == 5)	{
							gw.q(iGlob,2) = 2.0 * gw.k(iGlob,2) * (gw.h(iGlob+gdom.nxhc*gdom.nyhc,1) - gw.h(iGlob,1)) / gdom.dz(iGlob) - gw.k(iGhost,2);
						}
						else if (direction == 6)	{
							gw.q(iGhost,2) = 2.0 * gw.k(iGhost,2) * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dz(iGlob) - gw.k(iGhost,2);
						}
                    });
                    break;
                case SUB_BC_SWE:
					#if SERGHEI_SWE_MODEL
					if (direction == 6)	{
						Kokkos::parallel_for("gw_swe_fd", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
	                        int ii, jj, kk, iGlobSW, iGlob = bcells[ibc], iGhost = gcells[ibc], ivg = gw.soilID(iGlob) * NVG;
	                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
	                        iGlobSW = (jj-1)*gdom.nx + ii - 1;
	                        real wcs = gw.vgTable(ivg+2);
                            if (gdom.isnodata(iGlob) == 0)  {
    							if (swgw_type(ibc) == 0)    {
    								gw.q(iGhost,2) = 0.0;
    							}
    							else if (swgw_type(ibc) == 2)   {
    								gw.q(iGhost,2) = -gw.h(iGhost,1) / gdom.dt;
    							}
    							else {
    								gw.q(iGhost,2) = 2.0 * gw.k(iGhost,2) * (gw.h(iGlob,1) - gw.h(iGhost,1)) / gdom.dz(iGlob) - gw.k(iGhost,2);
    							}
                            }
                            else {gw.q(iGhost,2) = 0.0;}
							// Get exchange flux
							gw.qss(iGlobSW) = gw.q(iGhost,2);
	                    });
					}
					else {
						if (par.masterproc)	{std::cerr << RERROR "BC direction must be 6 for SW-GW exchange boundary! " << "\n";}
					}
					#endif
                case SUB_BC_Q_CONST:
					Kokkos::parallel_for("gw_bc_q_const", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
						int ii, jj, kk, iGlobSW, iGlob = bcells[ibc], iGhost = gcells[ibc];
						if (direction == 1)	{
							gw.q(iGlob,0) = bcvals(ibc);
						}
						else if (direction == 2)	{
							gw.q(iGhost,0) = bcvals(ibc);
						}
						else if (direction == 3)	{
							gw.q(iGlob,1) = bcvals(ibc);
						}
						else if (direction == 4)	{
							gw.q(iGhost,1) = bcvals(ibc);
						}
						else if (direction == 6)	{
							// rainfall
							gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
							iGlobSW = jj*gdom.nxhc + ii;
							#if SERGHEI_SWE_MODEL
                            // soil evaporation when surface domain is dry
                            // NOTE: should add a limiter to reduce evaporation when soil is dry, ZhiLi20250111
                            if (gw.h(iGhost,1) <= 0.0)  {gw.q(iGhost,2) = bcvals(ibc);}
                            #else
							gw.q(iGhost,2) = bcvals(ibc);
							if (gdom.isRain)    {gw.q(iGhost,2) -= gdom.rainRate(iGlobSW);}
							#endif
						}
					});
					break;
                case SUB_BC_Q_T:
                    Kokkos::parallel_for("gw_bc_q_const", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                        int ii, jj, kk, iGlobSW, iGlob = bcells[ibc], iGhost = gcells[ibc];
						if (direction == 1)	{
							gw.q(iGlob,0) = qbc;
						}
						else if (direction == 2)	{
							gw.q(iGhost,0) = qbc;
						}
						else if (direction == 3)	{
							gw.q(iGlob,1) = qbc;
						}
						else if (direction == 4)	{
							gw.q(iGhost,1) = qbc;
						}
						else if (direction == 6)	{
							// rainfall
							gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
							iGlobSW = jj*gdom.nxhc + ii;
							#if SERGHEI_SWE_MODEL
                            // soil evaporation when surface domain is dry
                            // NOTE: should add a limiter to reduce evaporation when soil is dry, ZhiLi20250111
                            if (gw.h(iGhost,1) <= 0.0)  {gw.q(iGhost,2) = qbc;}
                            #else
							gw.q(iGhost,2) = qbc;
							if (gdom.isRain)    {gw.q(iGhost,2) -= gdom.rainRate(iGlobSW);}
							#endif
						}
                    });
                    break;
                case SUB_BC_FD:
					if (direction == 5)	{
						Kokkos::parallel_for("gw_bc_fd", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
	                        int iGlob = bcells[ibc];
							gw.q(iGlob,2) = -gw.k(iGlob,2);
	                    });
					}
					else {
						if (par.masterproc)	{std::cerr << RERROR "BC direction must be 5 for free-drainage boundary! " << "\n";}
					}
            }
            // get the total flow rate across the boundary
            if (direction == 1)	{
            	Kokkos::parallel_reduce("reducex", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
					int iGlob = bcells[ibc];
					tmp += gw.q(iGlob,0) * gdom.dz(iGlob) * gdom.dy;
				}, Kokkos::Sum<real>(Qtot));
				if (Qtot > 0)	{Qinflow = Qtot;}
				else {Qoutflow = -Qtot;}
            }
            else if (direction == 2)	{
            	Kokkos::parallel_reduce("reducex", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
					int iGlob = bcells[ibc], iGhost = gcells[ibc];
					tmp += gw.q(iGhost,0) * gdom.dz(iGlob) * gdom.dy;
				}, Kokkos::Sum<real>(Qtot));
				if (Qtot < 0)	{Qinflow = -Qtot;}
				else {Qoutflow = Qtot;}
            }
            else if (direction == 3)	{
            	Kokkos::parallel_reduce("reducey", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
					int iGlob = bcells[ibc];
					tmp += gw.q(iGlob,1) * gdom.dz(iGlob) * gdom.dx;
				}, Kokkos::Sum<real>(Qtot));
				if (Qtot > 0)	{Qinflow = Qtot;}
				else {Qoutflow = -Qtot;}
            }
            else if (direction == 4)	{
            	Kokkos::parallel_reduce("reducey", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
					int iGlob = bcells[ibc], iGhost = gcells[ibc];
					tmp += gw.q(iGhost,1) * gdom.dz(iGlob) * gdom.dx;
				}, Kokkos::Sum<real>(Qtot));
				if (Qtot < 0)	{Qinflow = -Qtot;}
				else {Qoutflow = Qtot;}
            }
            else if (direction == 5)	{
            	Kokkos::parallel_reduce("reducez", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
					int iGlob = bcells[ibc];
					tmp += gw.q(iGlob,2) * gdom.dx * gdom.dy;
				}, Kokkos::Sum<real>(Qtot));
				if (Qtot > 0)	{Qinflow = Qtot;}
				else {Qoutflow = -Qtot;}
            }
            else if (direction == 6)    {
                Kokkos::parallel_reduce("reducez", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc, real &tmp){
                    int iGlob = bcells[ibc], iGhost = gcells[ibc];
                    tmp += gw.q(iGhost,2) * gdom.dx * gdom.dy;
                }, Kokkos::Sum<real>(Qtot));
                if (Qtot < 0)    {Qinflow = -Qtot;}
                else {Qoutflow = Qtot;}
            }
        }
    }


    // Apply boundary conditions to the matrix coefficients
    inline void applyMatBC(GwState &gw, GwDomain &gdom, Parallel &par) {
		// Check if on global boundaries
		bool onBoundary = 0;
		if (direction == 2 && par.px == 0)	{onBoundary = 1;}
		else if (direction == 1 && par.px == par.nproc_x-1)	{onBoundary = 1;}
		else if (direction == 4 && par.py == 0)	{onBoundary = 1;}
		else if (direction == 3 && par.py == par.nproc_y-1)	{onBoundary = 1;}
		else if (direction == 5 || direction == 6)	{onBoundary = 1;}

        if (ncellsBC > 0 && onBoundary == 1) {
            real qbc;
            if (bctype == SUB_BC_Q_T) {qbc = interpolateLinear(ts, gdom.etime);}
	  	    switch (bctype) {
                case SUB_BC_H_CONST:    case SUB_BC_WT_CONST:   case SUB_BC_H_T:    case SUB_BC_WT_T:
                    Kokkos::parallel_for("gw_bc_h_const", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
                        int ii, jj, kk, idom, iGlobSW, iGlob = bcells[ibc], iGhost = gcells[ibc];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
						iGlobSW = jj*gdom.nxhc + ii;
						if (direction == 1)	{
							gw.coef(idom,1) = gw.coef(idom,1) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,1) * gw.h(iGlob+1,1);
						}
						else if (direction == 2)	{
							gw.coef(idom,2) = gw.coef(idom,2) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,2) * gw.h(iGlob-1,1);
						}
						else if (direction == 3)	{
							gw.coef(idom,3) = gw.coef(idom,3) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,3) * gw.h(iGlob-gdom.nxhc,1);
						}
						else if (direction == 4)	{
							gw.coef(idom,4) = gw.coef(idom,4) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,4) * gw.h(iGlob-gdom.nxhc,1);
						}
						else if (direction == 5)	{
							gw.coef(idom,5) = gw.coef(idom,5) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,5) * gw.h(iGlob+gdom.nxhc*gdom.nyhc,1);
						}
						else if (direction == 6)	{
							gw.coef(idom,6) = gw.coef(idom,6) * 2.0;
							gw.coef(idom,7) -= gw.coef(idom,6) * gw.h(iGlob-gdom.nxhc*gdom.nyhc,1);
							// if (gdom.isEvap)	{
							// 	gw.coef(idom,7) -= gdom.dt * gdom.evapRate(iGlobSW) / gdom.dz(iGlob);
							// }
						}
                    });
                    break;
                case SUB_BC_SWE:
                	#if SERGHEI_SWE_MODEL
					if (direction == 6)	{
						Kokkos::parallel_for("gw_bc_swe_const", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
	                        int ii, jj, kk, idom, iGlobSW, iGlob = bcells[ibc];
	                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
	                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
							iGlobSW = jj*gdom.nxhc + ii;
							if (swgw_type(ibc) == 2)    {
								real q_infilt = gw.h(iGlob-gdom.nxhc*gdom.nyhc,1) / gdom.dt;
								gw.coef(idom,7) -= gdom.dt * gw.k(iGlob-gdom.nxhc*gdom.nyhc,2) / gdom.dz(iGlob);
								gw.coef(idom,7) += gdom.dt * q_infilt / gdom.dz(iGlob);
								gw.coef(idom,6) = 0.0;
							}
							else if (swgw_type(ibc) == 0)	{
								gw.coef(idom,7) -= gdom.dt * gw.k(iGlob-gdom.nxhc*gdom.nyhc,2) / gdom.dz(iGlob);
								gw.coef(idom,6) = 0.0;
							}
							else {
								gw.coef(idom,6) = gw.coef(idom,6) * 2.0;
								gw.coef(idom,7) -= gw.coef(idom,6) * gw.h(iGlob-gdom.nxhc*gdom.nyhc,1);
							}
							// evaporation
							if (gdom.isEvap)	{
								gw.coef(idom,7) -= gdom.dt * gdom.evapRate(iGlobSW) / gdom.dz(iGlob);
							}
	                    });
					}
                    else {
						if (par.masterproc)	{std::cerr << RERROR "BC direction must be 6 for sw-gw boundary! " << "\n";}
					}
                    #endif
                    break;
				case SUB_BC_Q_CONST:
					Kokkos::parallel_for("gw_bc_q", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
						int ii, jj, kk, idom, iGlobSW, iGlob = bcells[ibc];
						gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
						idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
						iGlobSW = jj*gdom.nxhc + ii;
						if (direction == 1)	{
							gw.coef(idom,7) += gdom.dt * bcvals(ibc) / gdom.dx;
							gw.coef(idom,1) = 0.0;
						}
						else if (direction == 2)	{
							gw.coef(idom,7) -= gdom.dt * bcvals(ibc) / gdom.dx;
							gw.coef(idom,2) = 0.0;
						}
						else if (direction == 3)	{
							gw.coef(idom,7) += gdom.dt * bcvals(ibc) / gdom.dx;
							gw.coef(idom,3) = 0.0;
						}
						else if (direction == 4)	{
							gw.coef(idom,7) -= gdom.dt * bcvals(ibc) / gdom.dx;
							gw.coef(idom,4) = 0.0;
						}
						else if (direction == 5)	{
							gw.coef(idom,5) = 0.0;
						}
						else if (direction == 6)	{
							#if SERGHEI_SWE_MODEL
                            if (gw.h(iGlob-gdom.nxhc*gdom.nyhc,1) <= 0.0)   {
                                gw.coef(idom,7) -= gdom.dt * bcvals(ibc) / gdom.dz(iGlob);
                            }
                            #else
							gw.coef(idom,7) -= gdom.dt * gw.k(iGlob-gdom.nxhc*gdom.nyhc,2) / gdom.dz(iGlob);
							gw.coef(idom,7) -= gdom.dt * bcvals(ibc) / gdom.dz(iGlob);
							// if (gdom.isRain)    {gw.coef(idom,7) += gdom.dt * gdom.rainRate(iGlobSW) / gdom.dz(iGlob);}
							// if (gdom.isEvap)	{gw.coef(idom,7) -= gdom.dt * gdom.evapRate(iGlobSW) / gdom.dz(iGlob);}
							gw.coef(idom,6) = 0.0;
							#endif
						}
					});
					break;
                case SUB_BC_Q_T:    case SUB_BC_FD:
                    Kokkos::parallel_for("gw_bc_q_fd", ncellsBC, KOKKOS_CLASS_LAMBDA (int ibc){
					// for (int ibc = 0; ibc < ncellsBC; ibc++)	{
                        int ii, jj, kk, idom, iGlobSW, iGlob = bcells[ibc];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
						iGlobSW = jj*gdom.nxhc + ii;
						if (direction == 1)	{
							gw.coef(idom,7) += gdom.dt * qbc / gdom.dx;
							gw.coef(idom,1) = 0.0;
						}
						else if (direction == 2)	{
							gw.coef(idom,7) -= gdom.dt * qbc / gdom.dx;
							gw.coef(idom,2) = 0.0;
						}
						else if (direction == 3)	{
							gw.coef(idom,7) += gdom.dt * qbc / gdom.dx;
							gw.coef(idom,3) = 0.0;
						}
						else if (direction == 4)	{
							gw.coef(idom,7) -= gdom.dt * qbc / gdom.dx;
							gw.coef(idom,4) = 0.0;
						}
						else if (direction == 5)	{
							gw.coef(idom,5) = 0.0;
						}
						else if (direction == 6)	{
							#if SERGHEI_SWE_MODEL
                            if (gw.h(iGlob-gdom.nxhc*gdom.nyhc,1) <= 0.0)   {
                                gw.coef(idom,7) -= gdom.dt * qbc / gdom.dz(iGlob);
                            }
                            #else

							gw.coef(idom,7) -= gdom.dt * gw.k(iGlob-gdom.nxhc*gdom.nyhc,2) / gdom.dz(iGlob);
							gw.coef(idom,7) -= gdom.dt * qbc / gdom.dz(iGlob);
							// if (gdom.isRain)    {gw.coef(idom,7) += gdom.dt * gdom.rainRate(iGlobSW) / gdom.dz(iGlob);}
							// if (gdom.isEvap)	{gw.coef(idom,7) -= gdom.dt * gdom.evapRate(iGlobSW) / gdom.dz(iGlob);}
							gw.coef(idom,6) = 0.0;
							#endif
						}
                    });
					// }
                    break;
            }
        }
    }


};

class SubsurfaceBoundaries{
// This class should not be invoked form a parallel region as it contains strings
public:
  	std::string BoundaryTypes[9] = {"NOFLOW","CONST_H","CONST_Q","CONST_WT","H_TIMESERIES","Q_TIMESERIES","WT_TIMESERIES","SWEXCHANGE","FREE_DRAINAGE"};
	std::vector<std::string> id;
	std::vector<GwBC> gwbc;
};

#endif

#endif
