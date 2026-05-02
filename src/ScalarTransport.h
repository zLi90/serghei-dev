#ifndef _SCALAR_TRANSP_H_
#define _SCALAR_TRANSP_H_

#include "define.h"
#include "Domain.h"
#include "Parallel.h"
#include "rasterTools.h"
#include "Solvers.h"



#if SERGHEI_SCALAR_TRANSPORT

class ScalarTransport{
public:
  std::string dir;
  std::string fname="scalar.input";
  int nScalar=0;
  std::vector<std::string> scalarNames;  
  std::set<std::string> initialModes = {"none","constant","file","netcdf"};
  std::vector<real> initialValue;
  std::vector< SArray<real,2> > diffc;
  int mode = 0;	

  std::string initialMode="none";	

	int read(const Parallel &par){
		#if SERGHEI_DEBUG_SCALAR_TRANSPORT
			std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
		#endif
		std::string fNameIn = dir + fname;
		std::ifstream fInStream(fNameIn);
		std::string line;
		ParserLine pline;
		std::string tmpStr;

		if (fInStream.is_open()){
			while (std::getline(fInStream, line)) {
				pline.line = line;
				pline.lowercase();
				pline.parse();

				if(!pline.key.empty()){
					if(!strcmp("nscalar",pline.key.c_str())){
						pline.value >> nScalar;
						std::cout << BDASH "Number of scalars to transport: " << nScalar << std::endl;
					}
					if(nScalar>0){
						if(!strcmp("initialmode",pline.key.c_str())){
							pline.value >> initialMode;
						}
						if(!strcmp("scalarname",pline.key.c_str())){ 
							pline.value >> tmpStr;
							scalarNames.push_back(tmpStr);
						}
						if(!strcmp("diffc",pline.key.c_str())){
							SArray<real,2> tmp;
							pline.value >> tmp(0) >> tmp(1);
							diffc.push_back(tmp);
						}
						if(!strcmp("initialvalue",pline.key.c_str())){
							real temp;
							pline.value >> temp; 
							initialValue.push_back(temp); 
						}
					}
				}
			}
		}else{
			if (par.masterproc){
				std::cerr << RERROR "File " << fNameIn << " not found" << std::endl;
				return 0;
			}
		}

		//if(nScalar > 0) mode = 1;
		mode = 1; //always activated

		if(scalarNames.size() < nScalar || diffc.size() < nScalar){
			std::cout << RERROR "Inconsistent nScalar with data blocks provided in " << fNameIn << std::endl;
			std::cout << "nScalar = " << nScalar << std::endl;
			std::cout << "n diffc = " << diffc.size() << std::endl;
			std::cout << "n scalarNames = " << scalarNames.size() << std::endl;
			return(0);
		}
		std::cout << GOK "Scalar transport input read" << std::endl;
		return 1;
	};

};



class ADEsolver{

public:
  int nScalar=0;
	int nSed=0; //!=0 when sediment transport exists
  realArr2 hphi; // transported scalars
  realArr2 dhphi0; // equivalent to dsw0 for the transport
  realArr2 dhphi1; // equivalent to dsw1 for the transport

	//Kokkos::Timer timer;

  #if SERGHEI_SCALAR_DIFFUSION
  	realS2Arr diffc;
	realArr2 phiM; // auxiliary variable for the diffusion sub-step iterative method
   	//we need four realArr2 (one per neighbour interface) UCM: To be changed for adapative mesh
  	realArr2 ddiff0;
  	realArr2 ddiff1;
  	realArr2 ddiff2;
  	realArr2 ddiff3;

	// Sub-stepping state variables (for MPI-aware diffusion)
	int nS;           // number of sub-steps
	int currentSubStep; // current sub-step index
	real diffDt;       // diffusion time step (computed from stability)
  #endif

  realArr2 hphiSource; // source term for conservative scalar variable
	

	/////////////////
	void addScalars(const int n){
		nScalar += n;
	}

  void allocate(const Domain &dom){
		// nScalar needs to be set prior to calling
    #if SERGHEI_DEBUG_SCALAR_TRANSPORT
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    hphi = realArr2("hphi",dom.nCellMem	,nScalar);
    dhphi0 = realArr2("dhphi0",dom.nCellMem,nScalar);
    dhphi1 = realArr2("dhphi1",dom.nCellMem,nScalar);

    #if SERGHEI_SCALAR_DIFFUSION
    	phiM = realArr2("phiM",dom.nCellMem,nScalar);
    	ddiff0 = realArr2("ddiff0",dom.nCellMem,nScalar);
    	ddiff1 = realArr2("ddiff1",dom.nCellMem,nScalar);
    	ddiff2 = realArr2("ddiff2",dom.nCellMem,nScalar);
    	ddiff3 = realArr2("ddiff3",dom.nCellMem,nScalar);
    	diffc = realS2Arr("diffc",nScalar);
    #endif
		
		hphiSource = realArr2("hphiSource",dom.nCellMem,nScalar);
  }


	int readInitial(Domain &dom, const Parallel &par, const ScalarTransport &st){  
		//if(!st.initialMode.compare("file")){
			std::string fname;

			// this is the solution we are looking for
			//realArr buffer = Kokkos::subview(ade.hphi,Kokkos::ALL,iphi);
			// but since it is not working, let's do it in an inefficient and non elegant way
			realArr buffer = realArr("subview_buffer",dom.nCellMem);

			for(int iphi=0; iphi < nScalar; iphi++){
				fname = st.dir + st.scalarNames[iphi] + ".ini";
				#if SERGHEI_DEBUG_SCALAR_TRANSPORT
				std::cout << GGD << "Reading initial field for scalar " << iphi << std::endl;
				#endif
/*
				Kokkos::parallel_for("subview_buffer", dom.ncells , KOKKOS_CLASS_LAMBDA (int iGlob) {
					buffer(iGlob) = hphi(iGlob,iphi);
				});
*/
				if(!readRasterField(fname,dom,par,buffer)) return 0;
				Kokkos::parallel_for("subview_buffer", dom.nCellMem , KOKKOS_CLASS_LAMBDA (int iGlob) {
					hphi(iGlob,iphi) = buffer(iGlob);
				});
			}
		//}
		return 1;
	} 

  void initialise(Domain &dom, const Parallel &par, const realArr &h, const ScalarTransport &st){
    #if SERGHEI_DEBUG_SCALAR_TRANSPORT
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
		realArr phi0 = realArr("phi0",st.nScalar); 
  
		allocate(dom);

    // by default assumes initialMode == none
    bool isConstant=1;

  
	  if(!st.initialMode.compare("constant")){
      for(int iphi=0; iphi<nScalar; iphi++) phi0(iphi) = st.initialValue[iphi];
      //phi0=st.initialValue;
    }
   
    if(!st.initialMode.compare("file")){
			#if SERGHEI_DEBUG_SCALAR_TRANSPORT
      	std::cout << GGD << "Initialise scalar transport with file" << std::endl;
    	#endif
			readInitial(dom,par,st);
			isConstant=0;
			#if SERGHEI_DEBUG_SCALAR_TRANSPORT
      	std::cout << GOK << "Scalar transport file read completed" << std::endl;
    	#endif
		}
 
    // generic initialisation
    Kokkos::parallel_for("init_scalar", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
      int ii = dom.getIndex(iGlob);
			for(int iphi=0; iphi<nScalar; iphi++){
        if(isConstant){
					hphi(ii,iphi) = phi0(iphi)*h(ii);
				}else{
          real phi = hphi(ii,iphi);
          hphi(ii,iphi) = phi*h(ii);					
				}
        dhphi0(ii,iphi) = 0.0;
        dhphi1(ii,iphi) = 0.0;
				#if SERGHEI_SCALAR_DIFFUSION
					ddiff0(ii,iphi) = 0.0;
					ddiff1(ii,iphi) = 0.0;
					ddiff2(ii,iphi) = 0.0;
					ddiff3(ii,iphi) = 0.0;
				#endif	
				hphiSource(ii,iphi) = 0.0;
      }
    });

		#if SERGHEI_SCALAR_DIFFUSION
		for(int iphi=0; iphi<nScalar; iphi++){
			diffc(iphi)=st.diffc[iphi];
		}
		#endif

    std::cout << GOK "Scalar transport initialised" << std::endl;

  }



/////////////////////
KOKKOS_INLINE_FUNCTION void updateAndReset(int ii, real const &h, real const &dt, real const &dx ){
	//update contributionss (solutes + sediments)
	for(int iphi=0; iphi<nScalar; iphi++){
		//hphi(ii,iphi) -= dt * (dhphi0(ii,iphi)+dhphi1(ii,iphi))/dx - dt * hphiSource(ii,iphi);
		hphi(ii,iphi) -= dt * (dhphi0(ii,iphi)+dhphi1(ii,iphi))/dx;
		
		//reduction or remove. Should be in the order of machine accuracy
		if(h<TOL12 || hphi(ii,iphi)<TOL12){
			hphi(ii,iphi)=0.0;
		}

		//reset
		dhphi0(ii,iphi) = 0.0;
		dhphi1(ii,iphi) = 0.0;
		//hphiSource(ii,iphi) = 0.0;
	}

	//add and reset source terms (only solutes)
	/*for(int iphi=0; iphi<nScalar-nSed; iphi++){
		hphi(ii,iphi) += dt * hphiSource(ii,iphi);
		hphiSource(ii,iphi) = 0.0;
	}*/

}

/////////////////////
KOKKOS_INLINE_FUNCTION void upwinding(int id1, int id2, real h1, real h2, real numFlux){
	
	int id;
	real h;
	real netMassFlux;

	h=0.0;
	netMassFlux=0.0;
	if (numFlux>0.0){ //the equal sign does not affect the results since numFlux is zero
		h=h1;
		id=id1;
	}
	if (numFlux<0.0){
		h=h2;
		id=id2;
	}
	if(h>0.0){	
		for(int iphi=0; iphi<nScalar; iphi++){
			netMassFlux= numFlux*hphi(id,iphi)/h;
			
			//the flux is already negative for the second case so the contributions should be sent with the same sign as before
			dhphi0(id1,iphi) +=netMassFlux;
			dhphi1(id2,iphi) -=netMassFlux;
		}	
	}

}



#if SERGHEI_SCALAR_DIFFUSION

/////////////////////
KOKKOS_INLINE_FUNCTION void edgeDiffusion(const int &id1, const int &id2, const real &h1, const real &h2, Solver const &solver, real const &dt, real const &dx, real const &nx, real const &ny){

	real absun;
	real modU2;
	real numDiff;
	real frictionVel; 
	real kL, kT;
	real Dxx, Dxy, Dyx, Dyy; 
	real Dnn;
	real fluxDiff;
	real utilde;
	real vtilde;
	real htilde;
	real frictionSlope;
	real mu;
	real lk;
	real dnk;
	real area;


	utilde=solver.utilde;
	vtilde=solver.vtilde;
	htilde=solver.htilde;
	frictionSlope=solver.frictionSlope;

	absun = fabs(utilde*nx + vtilde*ny);
	modU2 = utilde*utilde+vtilde*vtilde;
	real u2=utilde*utilde;
	real v2=vtilde*vtilde;
	//numerical diffusion
	numDiff = 0.5*absun*(dx-absun*dt);
	
	frictionVel=sqrt(GRAV*htilde*frictionSlope);

	#if SERGHEI_ANALYTICAL_DIFFUSION
		if(fabs(frictionSlope)<TOL12){
			frictionVel=1.0/htilde;
		}
	#endif
	
  	for(int iphi=0; iphi<nScalar; iphi++){
		kL=diffc(iphi)(0)*htilde*frictionVel;
		kT=diffc(iphi)(1)*htilde*frictionVel;
		if(modU2>TOL12){
			Dxx=kL*u2/modU2 + kT*v2/modU2;
			Dyy=kT*u2/modU2 + kL*v2/modU2;
			Dxy=(kL-kT)*utilde*vtilde/modU2;
			Dyx=Dxy;

			Dnn=(Dxx*nx+Dxy*ny)*nx + (Dyx*nx+Dyy*ny)*ny;
			mu=max(Dnn-numDiff,0.0);
			
			lk=dx; //UCM. To be changed for adaptative mesh
			dnk=dx; //UCM. To be changed for adaptative mesh
			area = dx*dx; //UCM. To be changed for adaptative mesh
			fluxDiff=mu*htilde*lk/dnk;

			//ddiff0-4 match with the definition of neigbouring cells in the getNeighbours function
			if(nx>0.0){ //x-interfaces
				if(h1>0.0){
					ddiff1(id1,iphi)=fluxDiff/area/h1;
				}
				if(h2>0.0){
					ddiff0(id2,iphi)=fluxDiff/area/h2;
				}
			}else{ //y-interfaces
				if(h1>0.0){
					ddiff3(id1,iphi)=fluxDiff/area/h1;
				}
				if(h2>0.0){
					ddiff2(id2,iphi)=fluxDiff/area/h2;
				}
			}
		}
	}
}

	////////////////////////////////////////////////////////////////////////
	// MPI-AWARE DIFFUSION SUB-STEPPING API
	//
	// This three-step API enables proper MPI halo exchange between sub-steps,
	// ensuring mathematical consistency and bit-for-bit reproducibility across
	// different domain decompositions.
	//
	// Usage pattern (in TimeIntegrator):
	//   int nSteps = initializeDiffusion(h, dom);
	//   for (int iNs = 0; iNs < nSteps; iNs++) {
	//       real subDt = ...;
	//       performDiffusionSubstep(h, dom, subDt);
	//       if (iNs < nSteps-1 && dom.nsubdom > 1) {
	//           exch.exchangeMPIscalars(state, dom, par);
	//       }
	//   }
	//   finalizeDiffusion(h, dom);
	//
	////////////////////////////////////////////////////////////////////////

	/////////////////
	// Initialize diffusion sub-stepping
	//
	// Converts hphi → phi and computes the number of sub-steps needed
	// based on the diffusion CFL condition.
	//
	inline int initializeDiffusion(const realArr &h, const Domain &dom){
		real dt = dom.dt;

		// Convert hphi to phi (concentration form)
		Kokkos::parallel_for("hphiAuxiliary", dom.nCellMem, KOKKOS_CLASS_LAMBDA(int iGlob){
	     	for(int iphi=0; iphi<nScalar; iphi++){
				if(fabs(h(iGlob))>TOL12){
					hphi(iGlob,iphi) = hphi(iGlob,iphi)/h(iGlob);
				}else{
					hphi(iGlob,iphi) = 0.0;
				}
			}
		});

		// Compute diffusion time step
		diffDt = computediffDt(dom);
		nS = 1 + (int)(dt/diffDt);  // at least one sub-step
		currentSubStep = 0;

		return nS;
	}

	/////////////////
	// Perform one diffusion sub-step
	// subDt: time step for this sub-step
	inline void performDiffusionSubstep(const realArr &h, const Domain &dom, real subDt){
		// Store current phi in phiM for this sub-step
		Kokkos::parallel_for("updatephiM", dom.nCellMem, KOKKOS_CLASS_LAMBDA(int iGlob){
	     	for(int iphi=0; iphi<nScalar; iphi++){
				phiM(iGlob,iphi) = hphi(iGlob,iphi);
			}
		});

		// Compute diffusion for this sub-step (only computational cells)
		Kokkos::parallel_for("update_diffusion", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
		    int ii = dom.getIndex(iGlob);
			int neigh[4];
			dom.getNeighbours(ii,neigh);
	     	for(int iphi=0; iphi<nScalar; iphi++){
				hphi(ii,iphi) = phiM(ii,iphi) + subDt*(ddiff0(ii,iphi)*(phiM(neigh[0],iphi) - phiM(ii,iphi)) +  ddiff1(ii,iphi)*(phiM(neigh[1],iphi) - phiM(ii,iphi)) +  ddiff2(ii,iphi)*(phiM(neigh[2],iphi) - phiM(ii,iphi)) +  ddiff3(ii,iphi)*(phiM(neigh[3],iphi) - phiM(ii,iphi)));
			}
		});

		currentSubStep++;
	}

	/////////////////
	// Finalize diffusion: convert phi back to hphi
	inline void finalizeDiffusion(const realArr &h, const Domain &dom){
		Kokkos::parallel_for("updatehphi", dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
		    int ii = dom.getIndex(iGlob);
	     	for(int iphi=0; iphi<nScalar; iphi++){
				// phi was the updated value, now recover hphi
				hphi(ii,iphi) *= h(ii);
				ddiff0(ii,iphi) = 0.0;
				ddiff1(ii,iphi) = 0.0;
				ddiff2(ii,iphi) = 0.0;
				ddiff3(ii,iphi) = 0.0;
			}
		});
	}




/////////////////////
inline real computediffDt(const Domain &dom){

	real diffDt = 1.e7;

	Kokkos::parallel_reduce("reduceDiffDt",dom.nCell , KOKKOS_CLASS_LAMBDA (int iGlob, real &dt) {
    	int ii = dom.getIndex(iGlob);
	 	real totalDiff;
		dt=fmin(dt,1.e6);
		for(int iphi=0; iphi<nScalar; iphi++){
			totalDiff = ddiff0(ii,iphi) + ddiff1(ii,iphi)+ddiff2(ii,iphi) + ddiff3(ii,iphi);
	 		if(fabs(totalDiff)>TOL12){
				dt=fmin(dt,1.0/totalDiff);
			}
		}

	} , Kokkos::Min<real>(diffDt) );

	  Kokkos::fence();

	  real dtloc = diffDt;
	  int ierr = MPI_Allreduce(&dtloc, &diffDt, 1, MPI_DOUBLE , MPI_MIN, MPI_COMM_WORLD);
	
	return diffDt;

}

#endif

};

#endif
#endif
