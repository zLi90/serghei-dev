/* -*- mode: c++ -*- */

#ifndef _BC_H_
#define _BC_H_

#include "define.h"
#include "Indexing.h"

// outer boundary direction definitions
#define SERGHEI_BC_OUTER_REFLECTIVE_CELL 9999

// these definitions are meant for hydraulics
#define SWE_BC_PERIODIC 1
#define SWE_BC_REFLECTIVE 2
#define SWE_BC_TRANSMISSIVE 3
#define SWE_BC_CRITICAL 5
#define SWE_BC_H_CONST 6
#define SWE_BC_Q_CONST 7
#define SWE_BC_WSE_CONST 8
#define SWE_BC_FREE_OUTFLOW 9
#define SWE_BC_HZ_T_INLET 10
#define SWE_BC_HZ_T_OUTLET 11
#define SWE_BC_Q_T 12


KOKKOS_INLINE_FUNCTION real criticalDepth(real hu, real hv, real Fr){
  return(cbrt((hu*hu+hv*hv)/(GRAV*Fr*Fr)));
}

class ExtBC {
// this class is safe to invoke in a parallel region
public:
	int ncellsBC = 0; //number of bcells
	intArr bcells; //array of indexes of boundary cells
	real normalx, normaly; //direction set by user for inflow/outflow
	int location; //1->west, 2->north, 3->east, 4-> south
	int bctype;
	int isInDomain;
    realArr bcvals;
	real outflowDischarge;
	real outflowAccumulated = 0;
	real inflowDischarge;
	real inflowAccumulated = 0;
	real adjustedVolume = 0;

	#if SERGHEI_SUSPENDED_SEDIMENT
		real outflowSolidDischarge;
		real outflowSolidAccumulated = 0;
		real inflowSolidDischarge;
		real inflowSolidAccumulated = 0;
		real adjustedSolidVolume = 0;
	#endif

	TimeSeries hydrograph;
	real netQ,netVol;

  real hzMin=1E6; // lowest water surface in boundary cross section
  real zMin=1E6; // lowest bed elevation in boundary cross section
	real zMax=-1E6;
  int nzMin=0;

	MPI_Comm comm;	// communicator for ranks associated to the BC


public:

	inline int find_bcells(State &state, std::string &id, const Domain &dom, Parallel &par, int nPoly, realArr &xPoly, realArr &yPoly){
		#if SERGHEI_DEBUG_BOUNDARY
			std::cout << GGD << "find_bcells called for boundary id =" << RED << id << RESET << std::endl;
		#endif
		Kokkos::Timer timermpi;
		int foundInSubdom; // to keep track of which subdomains are associated to this boundary
		std::vector<int> tmpbcells; //array of indexes of boundary cells
		std::vector<int> subdomains;	// keeps track of which subdomains are associated to the BC

		for(int iGlob=0; iGlob<dom.nCell; iGlob++){
			int i,j;
			dom.unpackIndices(iGlob,j,i);
			int ii = dom.getHaloExtension(i,j);
			foundInSubdom = -1;
			if(!state.isnodata(ii)){
				if((j==0 && dom.iN) || (j==dom.ny-1 && dom.iS) || (i==0 && dom.iW) || (i==dom.nx-1 && dom.iE) ||
				state.isnodata(ii+1) || state.isnodata(ii-1) ||
				state.isnodata(ii-(dom.nx+2*dom.hc)) || state.isnodata(ii+(dom.nx+2*dom.hc))){
				//boundary domain || nodata neighbours
					real xCoord = dom.xll + ( par.i_beg + i + 0.5) * dom.dxConst;
					real yCoord = dom.yll + dom.ny_glob*dom.dxConst - ( par.j_beg + j + 0.5) * dom.dxConst;
					if(geometry::isInsidePoly(nPoly,xPoly, yPoly, xCoord, yCoord)){
						tmpbcells.push_back(ii);
				  }
			  }
		  }
    }


		ncellsBC=int(tmpbcells.size());
		if(ncellsBC > 0) foundInSubdom = par.myrank; // if at least one cell in this subdomain (rank) is in the BC, tag as found

		int ncells_all;

		timermpi.reset();
   	MPI_Allreduce(&ncellsBC, &ncells_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if(par.nranks > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();

		int *subdoms;
		subdoms = (int*) malloc(par.nranks * sizeof(int));
		timermpi.reset();
		MPI_Allgather(&foundInSubdom,1,MPI_INT,subdoms,1,MPI_INT,MPI_COMM_WORLD);
		if(par.nranks > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();

		#if SERGHEI_DEBUG_BOUNDARY
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " rank " << RED << par.myrank << RESET << std::endl;
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " ncellsBC " << RED << ncells_all << RESET << std::endl;
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " BC subdomains: " ;
			for (int i=0; i<par.nranks; i++){
				std::cout << " ";
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
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "Consolidated " << RED << subdomains.size() << RESET << " BC subdomains = ";
			for(int i=0; i<subdomains.size(); i++){
				std::cout << " " ;
				std::cout << subdomains[i] << "\t";
			}
 			std::cout << std::endl;
		#endif
		timermpi.reset();
		MPI_Group group, subgroup;
		MPI_Comm_group(MPI_COMM_WORLD,&group);
		MPI_Group_incl(group,subdomains.size(),subdomains.data(),&subgroup);
		MPI_Comm_create(MPI_COMM_WORLD,subgroup,&comm);
		if(par.nranks > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();

    //int err;
    // we need the total boundary cells detected by all subdomain to launch an error otherwise
    if (ncells_all > 0) {
      bcells = intArr("bcells", ncellsBC);
#if defined(KOKKOS_ENABLE_CUDA)
      int err = cudaMemcpyAsync(bcells.data(), tmpbcells.data(), ncellsBC * sizeof(int),
                      cudaMemcpyHostToDevice);
      err = cudaDeviceSynchronize();
#elif defined(KOKKOS_ENABLE_HIP)
      int err = hipMemcpyAsync(bcells.data(), tmpbcells.data(), ncellsBC * sizeof(int),
                      hipMemcpyHostToDevice);
      err = hipDeviceSynchronize();
#elif defined(KOKKOS_ENABLE_SYCL)
      sycl::queue q{sycl::gpu_selector_v};
      q.memcpy(bcells.data(), tmpbcells.data(), ncellsBC * sizeof(int));
      q.wait();
#else
      std::memcpy(bcells.data(), tmpbcells.data(), ncellsBC * sizeof(int));
#endif
    }else{
			if(par.masterproc) std::cerr << RERROR << "No boundary cells found for external boundary with id '" << id << "'" << std::endl;
			return 0;
		}
	return 1;
}

	void inline getMinBedElevation(Domain const &dom, State &state){
		Kokkos::Timer timermpi;
			Kokkos::parallel_reduce("swe_bc_z_min", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &zMin){
			int ii = bcells[iGlob];
				real z = state.z(ii);
				zMin = min(zMin,z);
			}, Kokkos::Min<real>(zMin) );
		real zMin_all;
		timermpi.reset();
		MPI_Allreduce(&zMin, &zMin_all, 1, SERGHEI_MPI_REAL, MPI_MIN, comm);
		if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
		zMin = zMin_all;

		real zMax=-1E6;

		Kokkos::parallel_reduce("swe_bc_z_max", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &zMax){
		  int ii = bcells[iGlob];
			real z = state.z(ii);
			zMax = max(zMax,z);
		}, Kokkos::Max<real>(zMax) );
    real zMax_all;
	timermpi.reset();
    MPI_Allreduce(&zMax, &zMax_all, 1, SERGHEI_MPI_REAL, MPI_MAX, comm);
	if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
		zMax = zMax_all;

    nzMin=0;
    Kokkos::parallel_reduce("swe_bc_z_min_count", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, int &nzMin){
		  int ii = bcells[iGlob];
			if(state.z(ii) == zMin) nzMin++;
		}, Kokkos::Sum<int>(nzMin) );
    int nzMin_all;
	timermpi.reset();
    MPI_Allreduce(&nzMin, &nzMin_all, 1, MPI_INT, MPI_SUM, comm);
	if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
		nzMin = nzMin_all;

    #if SERGHEI_DEBUG_BOUNDARY
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "zMin = " << zMin << "\tnzMin = " << nzMin << std::endl;
    #endif
  }

	void inline flattenWaterSurface(State &state, Domain &dom){
		#if SERGHEI_DEBUG_BOUNDARY
		  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		Kokkos::Timer timermpi;
    	hzMin=zMin=1E6;
		real extraVol=0;
		real extraArea=0;

		// find the lowest water surface in the cross section
		Kokkos::parallel_reduce("swe_bc_flatten_hz", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &hzMin){
		  int ii = bcells[iGlob];
			real h = state.h(ii);
			real z = state.z(ii);
			if(h > state.hmin) hzMin = min(hzMin,h+z);
		}, Kokkos::Min<real>(hzMin) );

		real hzMin_all;
		timermpi.reset();
    	MPI_Allreduce(&hzMin, &hzMin_all, 1, SERGHEI_MPI_REAL, MPI_MIN, comm);
		if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
		hzMin = hzMin_all;

		// find the volume above the lowest water surface
		real area = dom.cellArea();	// WARNING assumes uniform mesh
		Kokkos::parallel_reduce("swe_bc_vol_minhz", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &volume){
			int ii = bcells[iGlob];
			real h = state.h(ii);
			real z = state.z(ii);
			if(h > state.hmin){
				if(z <= hzMin){
					volume += area * (h + z - hzMin);
				}else{
					volume += area * h;
				}
			}
		}, Kokkos::Sum<real>(extraVol) );
		real extraVol_all;

		timermpi.reset();
   	 	MPI_Allreduce(&extraVol, &extraVol_all, 1, SERGHEI_MPI_REAL, MPI_SUM, comm);
		if(dom.nsubdom > 1) dom.timers.swe.bc.mpi += timermpi.seconds();

		extraVol = extraVol_all;

		if(extraVol > ZERO){
			// compute the surface area with water level higher than the cross-sectional minimum
			Kokkos::parallel_reduce("swe_bc_area_minhz", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &sumarea){
				int ii = bcells[iGlob];
				real h = state.h(ii);
				real z = state.z(ii);
				if(h > state.hmin && z < hzMin && hzMin-z >0){
					sumarea += area;
				}
			}, Kokkos::Sum<real>(extraArea) );
			real extraArea_all;
			timermpi.reset();
    		MPI_Allreduce(&extraArea, &extraArea_all, 1, SERGHEI_MPI_REAL, MPI_SUM, comm);
			if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
			extraArea = extraArea_all;

			// compute and assign homogenised water surface
			real hz = hzMin + extraVol / extraArea;
			Kokkos::parallel_for("swe_bc_h_minhz", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob){
				int ii = bcells[iGlob];
				real h = state.h(ii);
				real z = state.z(ii);
				if(h > state.hmin){
					state.h(ii) = 0.;
					if(z <= hzMin) state.h(ii) = max(hz - z, (real) 0.);
				}
			});
		}
	}


	void inline distributeDischarge(State &state, Domain const &dom, real Q){
		#if SERGHEI_DEBUG_BOUNDARY
		  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		Kokkos::Timer timermpi;
		real hsum;
		Kokkos::parallel_reduce("swe_bc_h_weight", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &hsum){
			int ii = bcells[iGlob];
			hsum += state.h(ii);
		}, Kokkos::Sum<real>(hsum) );
		real hsum_all;
		timermpi.reset();
    	MPI_Allreduce(&hsum, &hsum_all, 1, SERGHEI_MPI_REAL, MPI_SUM, comm);
		if(dom.nsubdom > 1)	dom.timers.swe.bc.mpi += timermpi.seconds();
		hsum = hsum_all;

    int dryxs = 0;
		real dz, hz;
    if(hsum <= ZERO){ // dry cross section
      getMinBedElevation(dom,state);
      dryxs=1;
			dz = zMax-zMin;
			hz = zMin + 0.10*dz;	// initialise with 10% of the elevation difference in the cross section
			#if SERGHEI_DEBUG_BOUNDARY
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "zMin = " << zMin << "\tzMax = " << zMax << "\tdz = " << dz << "\thz = " << hz << std::endl;
			#endif
    }
		Kokkos::parallel_for("swe_bc_h_minhz", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob){
			int ii = bcells[iGlob];
			real h = state.h(ii);
      real z = state.z(ii);
      real weight = 0;
			if(!dryxs){ // wet cross section
        weight = h/hsum;
      }else{
				h = max(hz,z) - z;
				state.h(ii) = h;
        if(z == zMin) weight = 1./nzMin; // dry cross section
      }
			real ds = dom.dx();
			state.hu(ii) = Q * weight / ds * normalx;
			state.hv(ii) = Q * weight / ds * normaly;

			#if SERGHEI_DEBUG_BOUNDARY > 1
				std::cout << GGD  << GRAY << __PRETTY_FUNCTION__ << RESET << "Q = " << Q << "\tii = " << ii << "\tz = " << z << "\t h = " << h << "\tweight = " << weight << "\tds = " << ds  << "\t(hu,hv) = " << state.hu(ii) << " " << state.hv(ii) << std::endl;
			#endif
		});

	}


	#if SERGHEI_SEDIMENT_TRANSPORT
	void inline distributeBedChange(State &state, Domain const &dom){
	#if SERGHEI_DEBUG_BOUNDARY
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	#endif
		real DZsum;
		real Asum;
		Kokkos::parallel_reduce("sed_bc_sumVol", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob, real &var1, real &var2){
			int ii = bcells[iGlob];
			real h = state.h(ii);

			state.z(ii) += state.sediment.bedExchangeVol(ii);

			var1 += state.sediment.bedExchangeVol(ii);
			if(h>TOL12){
				var2 += dom.cellArea();
			}
		}, Kokkos::Sum<real>(DZsum), Kokkos::Sum<real>(Asum));

		real DZsum_all, Asum_all;
    MPI_Allreduce(&DZsum, &DZsum_all, 1, SERGHEI_MPI_REAL, MPI_SUM, comm);
		MPI_Allreduce(&Asum, &Asum_all, 1, SERGHEI_MPI_REAL, MPI_SUM, comm);
		DZsum = DZsum_all;
		Asum = Asum_all;

		if(Asum>TOL12){
			real excVol=DZsum/Asum;

			Kokkos::parallel_for("swe_bc_shareVol", ncellsBC, KOKKOS_CLASS_LAMBDA(int iGlob){
				int ii = bcells[iGlob];
				real h = state.h(ii);
				if(h>TOL12){
					state.z(ii) -= excVol*dom.cellArea();
				}
				#if SERGHEI_DEBUG_BOUNDARY > 1
					std::cout << GGD  << GRAY << __PRETTY_FUNCTION__ << RESET << "Q = " << Q << "\tii = " << ii << "\tz = " << z << "\t h = " << h << "\tweight = " << weight << "\tds = " << ds  << "\t(hu,hv) = " << state.hu(ii) << " " << state.hv(ii) << std::endl;
				#endif
			});
		}

	}
	#endif


  inline void apply(State &state, Domain &dom) {
    #if SERGHEI_DEBUG_WORKFLOW
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
	Kokkos::fence("bc.apply-end");
    Kokkos::Timer timer;

    real extraMass=0.0;


		#if SERGHEI_SEDIMENT_TRANSPORT
			if(ncellsBC > 0){
				switch (bctype){
					//outlet boundary conditions
					case SWE_BC_CRITICAL:
					case SWE_BC_H_CONST:
					case SWE_BC_WSE_CONST:
					case SWE_BC_FREE_OUTFLOW:
					case SWE_BC_HZ_T_OUTLET:

						distributeBedChange(state, dom);

						//do nothing already done in outletScalarFlux in ScalarTransport.h
					break;

					case SWE_BC_Q_CONST:
					case SWE_BC_HZ_T_INLET:
					case SWE_BC_Q_T:
						//to be defined for inlet boundaries with solutes (need an input file)
					break;


				}
			}
		#endif

    if(ncellsBC > 0){
	  	switch (bctype){
      	default:
	      	std::cerr << RERROR "Boundary type: " << bctype << " not recognised." << std::endl;
	        std::cerr << RERROR "No boundary condition applied." << std::endl;
	        exit(EXIT_FAILURE);
          break;

        case SWE_BC_CRITICAL: // critical flow boundary condition
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_CRITICAL" << std::endl;
				#endif
	      	Kokkos::parallel_for("swe_bc_critical",ncellsBC , KOKKOS_CLASS_LAMBDA (int iGlob){
		      	int ii=bcells[iGlob];
		        real h=state.h(ii);
		        if( h>=state.hmin) {
            	real hu=state.hu(ii);
		          real hv=state.hv(ii);
							//----------------------------
							real modQ=sqrt(hu*hu+hv*hv);
							real vel=modQ/h;
							if(vel/sqrt(GRAV*h)<1.){
								vel=1.0*sqrt(GRAV*h); //Froude 1.0 (critical)
							}
		          hu=vel*h*normalx;
		          hv=vel*h*normaly;
		          state.hu(ii)=hu;
		          state.hv(ii)=hv;
		        }
				  });
        	break;

        case SWE_BC_WSE_CONST: // constant free surface elevation
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_WSE_CONST" << std::endl;
				#endif
          Kokkos::parallel_reduce("swe_bc_swe_const",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumM){
            int ii = bcells[iGlob];
            real h = state.h(ii);
						if(h > TOL12){
							real hu= state.hu(ii);
							real hv= state.hv(ii);
							real z = state.z(ii);
							state.h(ii) = max(bcvals(0) - z, (real) 0.0); // enforce water depth positivity
							sumM += (state.h(ii)-h)*dom.cellArea();
							//orientation wrt to the outflow normal direction
							real modQ=sqrt(hu*hu+hv*hv);
							state.hu(ii)=normalx*modQ;
							state.hv(ii)=normaly*modQ;
						}
          }, Kokkos::Sum<real>(extraMass));
	      break;

	      case SWE_BC_H_CONST: // constant depth boundary condition
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_H_CONST" << std::endl;
				#endif
	        Kokkos::parallel_reduce("swe_bc_h_const",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumM){
	          int ii = bcells[iGlob];
			      real h = state.h(ii);
						if(h>TOL12){
							real hu= state.hu(ii);
							real hv= state.hv(ii);
							state.h(ii) = bcvals(0);
							sumM += (state.h(ii)-h)*dom.cellArea();
							//orientation wrt to the outflow normal direction
							real modQ=sqrt(hu*hu+hv*hv);
							state.hu(ii)=normalx*modQ;
							state.hv(ii)=normaly*modQ;
						}
          }, Kokkos::Sum<real>(extraMass) );
        break;

        case SWE_BC_Q_CONST: // constant inflow discharge boundary condition
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_Q_CONST" << std::endl;
				#endif
          Kokkos::parallel_reduce("swe_bc_q_const",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumM){
          int ii = bcells[iGlob];
          real h = state.h(ii);
          real qbc_x = bcvals(1);
 			    real qbc_y = bcvals(2);

			    //check if water depth is subcritical (Froude
			    //number less than 0.99). Otherwise impose
			    //boundary condition water depth
			    real inletFr=0.99;
			    //real hcr=cbrt((qbc_x*qbc_x+qbc_y*qbc_y)/(GRAV*inletFr*inletFr));
          real hcr = criticalDepth(qbc_x,qbc_y,inletFr);

			    //TODO: add this mass in case h is less than hcr
			    //state.h(ii) = max(h, hcr);
			    if (hcr > h) state.h(ii) = bcvals(0);

          sumM += (state.h(ii)-h)*dom.cellArea();

			    state.hu(ii) = qbc_x;
			    state.hv(ii) = qbc_y;
        }, Kokkos::Sum<real>(extraMass));
        break;

        case SWE_BC_FREE_OUTFLOW: // zero gradient or free boundary
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_FREE_OUTFLOW" << std::endl;
				#endif
          Kokkos::parallel_for("swe_bc_free",ncellsBC , KOKKOS_CLASS_LAMBDA (int iGlob){
            int ii=bcells[iGlob];
				    //orientation wrt to the outflow direction
						real hu= state.hu(ii);
						real hv= state.hv(ii);
						real modQ=sqrt(hu*hu+hv*hv);
						state.hu(ii)=normalx*modQ;
						state.hv(ii)=normaly*modQ;
          });
        break;

        case SWE_BC_HZ_T_INLET: // stage hydrograph inlet
        {
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_HZ_T_INLET" << std::endl;
				#endif
          real hzBC = interpolateLinear(hydrograph,dom.etime);
          #if SERGHEI_DEBUG_BOUNDARY
            std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "time = " << dom.etime << "\th+z = " << hzBC << std::endl;
          #endif
	        Kokkos::parallel_reduce("swe_bc_hz_t_inlet",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumM){
            int ii = bcells[iGlob];
            real h = state.h(ii);
				 		real hu= state.hu(ii);
				 		real hv= state.hv(ii);

			    	state.h(ii) = max(hzBC-state.z(ii), 0.0);
            sumM += (state.h(ii)-h)*dom.cellArea();

            /*
			    	//orientation wrt to the outflow normal direction
				 		real modQ=sqrt(hu*hu+hv*hv);
		       	state.hu(ii)=normalx*modQ;
		       	state.hv(ii)=normaly*modQ;
            */

					}, Kokkos::Sum<real>(extraMass) );
        	break;
        }

        case SWE_BC_HZ_T_OUTLET: // stage hydrograph outlet
      	{
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_HZ_T_OUTLET" << std::endl;
				#endif
          real hzBC = interpolateLinear(hydrograph,dom.etime);
          #if SERGHEI_DEBUG_BOUNDARY
            std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "time = " << dom.etime << "\th+z = " << hzBC << std::endl;
          #endif
	        Kokkos::parallel_reduce("swe_bc_hz_t_outlet",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumM){
            int ii = bcells[iGlob];
            real h = state.h(ii);
			    	state.h(ii) = max(hzBC-state.z(ii),(real) 0.0);
            sumM += (state.h(ii)-h)*dom.cellArea();

			    	//no orientation wrt to the outflow normal direction to allow tidal wave coming into the domain
					}, Kokkos::Sum<real>(extraMass) );
        	break;
        }

				case SWE_BC_Q_T:	// hydrograph
				{
				#if SERGHEI_DEBUG_BOUNDARY
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "SWE_BC_Q_T" << std::endl;
				#endif
					real Q = interpolateLinear(hydrograph,dom.etime);
					#if SERGHEI_DEBUG_BOUNDARY
            std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "HYDROGRAPH BC: time = " << dom.etime << "\tQ = " << Q << std::endl;
					#endif
					flattenWaterSurface(state,dom);
					distributeDischarge(state,dom,Q);
					//extraMass = Q;
					break;
				}
	  	} // end switch
  	} // endif ncellsBC

		#if SERGHEI_SCALAR_TRANSPORT
			if(ncellsBC > 0){
				switch (bctype){
					//outlet boundary conditions
					case SWE_BC_CRITICAL:
					case SWE_BC_H_CONST:
					case SWE_BC_WSE_CONST:
					case SWE_BC_FREE_OUTFLOW:
					case SWE_BC_HZ_T_OUTLET:

						//do nothing already done in outletScalarFlux in ScalarTransport.h
					break;

					case SWE_BC_Q_CONST:
					case SWE_BC_HZ_T_INLET:
					case SWE_BC_Q_T:
						//to be defined for inlet boundaries with solutes (need an input file)
					break;


				}
			}
		#endif

	adjustedVolume=extraMass; //extraMass per bc
	Kokkos::fence("bc.apply-end");
  	dom.timers.swe.bc.total += timer.seconds();
  }


  inline void integrate(State &state, Domain &dom) {
    #if SERGHEI_DEBUG_WORKFLOW
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
	Kokkos::fence("bc.integrate-begin");
    Kokkos::Timer timer;
    real outDischarge=0.0;
    real inDischarge=0.0;
    real totalDischarge=0.0;

	#if SERGHEI_SUSPENDED_SEDIMENT
    real outSolidDischarge=0.0;
    real inSolidDischarge=0.0;
	#endif
    real totalSolidDischarge=0.0;


    if(ncellsBC > 0){
			//discharge integration
			Kokkos::parallel_reduce("reduceDischargeBC",ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob, real &sumD, real &sumSD){
				int ii = bcells[iGlob];
				real qbound;
		    if( state.h(ii)>=state.hmin) {
					//the integration is done over all boundary walls according to the outflow direction
					qbound= (state.hu(ii)*sgn(normalx) + state.hv(ii)*sgn(normaly));
					sumD +=  qbound * dom.dx();
					#if SERGHEI_SUSPENDED_SEDIMENT
						for(int iphi=state.sediment.iphised; iphi<(state.sediment.iphised+state.sediment.nSed); iphi++){
							sumSD += qbound * state.ade.hphi(ii,iphi)/state.h(ii) * dom.dx();
						}
					#endif
				}
			}, Kokkos::Sum<real>(totalDischarge), Kokkos::Sum<real>(totalSolidDischarge));
			Kokkos::fence();

	    switch (bctype){
				case SWE_BC_CRITICAL:
				case SWE_BC_H_CONST:
				case SWE_BC_WSE_CONST:
				case SWE_BC_FREE_OUTFLOW:
      	case SWE_BC_HZ_T_OUTLET:
					outDischarge=totalDischarge;
					#if SERGHEI_SUSPENDED_SEDIMENT
						outSolidDischarge=totalSolidDischarge;
					#endif
					break;
				case SWE_BC_Q_CONST:
				case SWE_BC_HZ_T_INLET:
				case SWE_BC_Q_T:
					inDischarge=totalDischarge;
					#if SERGHEI_SUSPENDED_SEDIMENT
						inSolidDischarge=totalSolidDischarge;
					#endif
					break;
			  default:
					std::cerr << RERROR " Boundary type: " << bctype << " not recognized." << std::endl;
			 		std::cerr << RERROR " No boundary condition applied." << std::endl;
			 		exit(EXIT_FAILURE);
		  }
			#if SERGHEI_DEBUG_BOUNDARY
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "totalDischarge = " << totalDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "inDischarge = " << inDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "outDischarge = " << outDischarge << std::endl;
				#if SERGHEI_SUSPENDED_SEDIMENT
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "totalSolidDischarge = " << totalSolidDischarge << std::endl;
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "inSolidDischarge = " << inSolidDischarge << std::endl;
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "outSolidDischarge = " << outSolidDischarge << std::endl;
				#endif
			#endif
		}
    inflowDischarge = inDischarge; //inflowdischarge local per bc
    outflowDischarge = outDischarge; //outflowdischarge local per bc
    inflowAccumulated += inDischarge*dom.dt; //inflowAccumulated local per bc
	outflowAccumulated += outDischarge * dom.dt; //outflowaccumulated local per bc
		#if SERGHEI_SUSPENDED_SEDIMENT
			inflowSolidDischarge = inSolidDischarge; //inflowdischarge local per bc
			outflowSolidDischarge = outSolidDischarge; //outflowdischarge local per bc
			inflowSolidAccumulated += inSolidDischarge*dom.dt; //inflowAccumulated local per bc
			outflowSolidAccumulated += outSolidDischarge * dom.dt; //outflowaccumulated local per bc
		#endif

	Kokkos::fence("bc.integrate-end");
    dom.timers.swe.bc.integrate += timer.seconds();
	}

	inline void reduce(Parallel const &par){
		// only used to write out to file
		real Qin,Qout;

		MPI_Reduce(&inflowDischarge, &Qin, 1, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC,  MPI_COMM_WORLD);
		MPI_Reduce(&outflowDischarge, &Qout, 1, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC,  MPI_COMM_WORLD);
		netQ = Qin-Qout;

		MPI_Reduce(&inflowAccumulated, &Qin, 1, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC,  MPI_COMM_WORLD);
		MPI_Reduce(&outflowAccumulated, &Qout, 1, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC,  MPI_COMM_WORLD);
		netVol = Qin-Qout;
	}

	void setIsBound(State &state, int value) const{
		Kokkos::parallel_for("init_isBound", ncellsBC, KOKKOS_CLASS_LAMBDA (int iGlob){
	 		int ii = bcells[iGlob]; //extended domain index
			state.isBound(ii)=value;
		});
	}
};


class ExternalBoundaries{
// This class should not be invoked form a parallel region as it contains strings
public:
  	std::string BoundaryTypes[13] = {"NONE","PERIODIC","REFLECTIVE","TRANSMISSIVE","NONE","CRITICAL","CONSTANT DEPTH","CONSTANT INFLOW","CONSTANT WSELEVATION","FREE OUTFLOW","STAGE HYDROGRAPH INLET","STAGE HYDROGRAPH OUTLET", "HYDROGRAPH"};
	std::vector<std::string> id;
	std::vector<ExtBC> extbc;
};
#endif
