#ifndef _DOMAIN_INTEGRATOR_H_
#define _DOMAIN_INTEGRATOR_H_

#include "State.h"
#include "SourceSink.h"
#include "Domain.h"
#include "Indexing.h"
#include "Sediment.h"

/*
// potential solution for custom reductions
namespace sample {  // namespace helps with name resolution in reduction identity 
  // template< class ScalarType, int N>
   template< class ScalarType>
   struct mass_type {
     ScalarType h;
     ScalarType inf;
     ScalarType rain;
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     mass_type() { 
       h=0.;
			 inf=0.;
			 rain=0.;
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     mass_type(const mass_type & rhs) { 
			h = rhs.h;
			inf = rhs.inf;
			rain = rhs.rain;
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     mass_type& operator += (const mass_type& src) {
			h += src.h;
			inf += src.inf;
			rain += src.rain;
       return *this;
     }
   };
   typedef mass_type<real> MassType;  // used to simplify code below
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< sample::MassType > {
      KOKKOS_FORCEINLINE_FUNCTION static sample::MassType sum() {
         return sample::MassType();
      }
   };
}
*/

class surfaceIntegrator {

  Kokkos::Timer timerFull, timer;

  public:

  // integrated variables
  real surfaceVolume ;     // surface water volume in domain [L^3] (local)
  real rainFlux ;   // total rain flux [L^3 / T] (local)
  real rainAccum = 0.;   // accumulated rainfall in simulation [L^3] (local)
  real infFlux ;    // total infiltration flux [L^3/T] (local)
  real infAccum = 0.;    // accumulated infiltration in simulation [L^3] (local)

  real surfaceVolumeG ;     // surface water volume in domain [L^3] (global)
  real rainFluxG ;   // total rain flux [L^3 / T] (global)
  real rainAccumG = 0.;   // accumulated rainfall in simulation [L^3] (global)
  real infFluxG ;    // total infiltration flux [L^3/T] (global)
  real infAccumG = 0.;    // accumulated infiltration in simulation [L^3] (global)
	
	real surfaceSolidVolume = 0.; // surface suspended sediment volume in domain [L^3] (local)
	real surfaceSolidVolumeG = 0.; // surface suspended sediment volume in domain [L^3] (global)
	real BedExchangeSolid = 0.; // bed exchange solid (only) volume in domain [L^3] (local)
	real BedExchangeSolidG = 0.; // bed exchange solid (only) volume in domain [L^3] (global)	
	real BedExchangeVol = 0.; // bed exchange volume in domain [L^3] (local)
	real BedExchangeVolG = 0.; // bed exchange volume in domain [L^3] (global)



  // pointers
  SourceSinkData *ss;
  State *state;
  Domain *dom;


  void initialize(State &state_, Domain &dom_, SourceSinkData &ss_){
    state = &state_;
    dom = &dom_;
    ss = &ss_;
  }

  void integrate(State const &state, Domain const &dom, SourceSinkData &ss){
    timerFull.reset();

    surfaceVolume = 0;
	
	if(dom.etime<TOL12){ //change by initial time when hotstart is implemented
	 	rainAccum=0.0;
	 	infAccum=0.0;
	}

	//sample::MassType mass;
	Kokkos::parallel_reduce( dom.nCell , KOKKOS_LAMBDA (int iGlob, real & hSum, real &rainSum, real& infSum) {
    	int ii = dom.getIndex(iGlob);
		if(!state.isnodata(ii)){
			real area = dom.cellArea();
        	hSum +=  state.h(ii) * area;
			if(dom.isRain){
				rainSum += ss.rainRate(ii) * area;
				#if SERGHEI_NETCDF_FORCING
				ss.rainAccum(ii) += ss.rainRate(ii)*dom.dt;
				#endif
			}
			if(ss.inf.model){
				real inffluxlocal = ss.inf.rate(ii) * area;
				infSum += inffluxlocal;
				ss.inf.infVol(ii) += inffluxlocal * dom.dt;
			}
		}
    } , Kokkos::Sum<real>(surfaceVolume) , Kokkos::Sum<real>(rainFlux), Kokkos::Sum<real>(infFlux));
	rainAccum += rainFlux * dom.dt;
	infAccum += infFlux * dom.dt;


  	#if SERGHEI_SUSPENDED_SEDIMENT
			surfaceSolidVolume = 0.0;
			Kokkos::parallel_reduce( dom.nCell , KOKKOS_LAMBDA (int iGlob, real &valUpdate) {
				int ii = dom.getIndex(iGlob);
				bool nodata=state.isnodata(ii);
				if(!nodata){
					for(int iphi=0;iphi<state.ade.nScalar;iphi++){
						valUpdate += state.ade.hphi(ii,iphi)*dom.cellArea();
					} 
				}	
			} , Kokkos::Sum<real>(surfaceSolidVolume) );
			Kokkos::fence();			
		#endif	
		
  	#if SERGHEI_SUSPENDED_SEDIMENT
		BedExchangeVol = 0.0;
		BedExchangeSolid = 0.0;
		Kokkos::parallel_reduce( dom.nCell , KOKKOS_LAMBDA (int iGlob, real &valUpdate1, real &valUpdate2) {
			int ii = dom.getIndex(iGlob);
			bool nodata=state.isnodata(ii);
			double zvol;
			if(!nodata){
				zvol = state.sediment.bedExchangeVol(ii)*dom.cellArea();
				valUpdate1 += zvol;
				valUpdate2 += state.sediment.bedConc*zvol; //1-xi will change by cells;
			}
		} , Kokkos::Sum<real>(BedExchangeVol), Kokkos::Sum<real>(BedExchangeSolid) );
		Kokkos::fence();
		#endif
	Kokkos::fence();

	
	surfaceVolumeG = surfaceVolume;
	rainFluxG = rainFlux;
	rainAccumG = rainAccum;
	infFluxG = infFlux;
	infAccumG = infAccum;
	
	if(dom.nsubdom > 1){
		timer.reset();
		MPI_Allreduce(&surfaceVolume, &surfaceVolumeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&rainFlux, &rainFluxG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&rainAccum, &rainAccumG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&infFlux, &infFluxG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&infAccum, &infAccumG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);

		#if SERGHEI_SUSPENDED_SEDIMENT
		surfaceSolidVolumeG=0.0;
		BedExchangeSolidG=0.0;
		BedExchangeVolG=0.0;
		MPI_Allreduce(&surfaceSolidVolume, &surfaceSolidVolumeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);	
		MPI_Allreduce(&BedExchangeSolid, &BedExchangeSolidG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);	
		MPI_Allreduce(&BedExchangeVol, &BedExchangeVolG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);	
		#endif
		MPI_Barrier(MPI_COMM_WORLD);
  		dom.timers.swe.integrate.mpi += timer.seconds();
	}
	dom.timers.swe.integrate.total += timerFull.seconds();
  }
};

class boundaryIntegrator{

Kokkos::Timer timer, timerFull;

public:

  int ncellsBC;

  real adjustedVolume ;  // boundary water volume in domain [L^3] adjusted (e.g. impose water depth) (local)
  real outflowDischarge; // boundary outflow discharge in domain [L^3/T] (local)
  real outflowAccumulated = 0.0; // boundary accumulated outflow volume in domain [L^3] (local)
  real inflowDischarge; // boundary inflow discharge in domain [L^3/T] (local)
  real inflowAccumulated = 0.0; // boundary accumulated inflow volume in domain [L^3] (local)

  real adjustedVolumeG ;  // boundary water volume in domain [L^3] adjusted (e.g. impose water depth) (global)
  real outflowDischargeG; // boundary outflow discharge in domain [L^3/T] (global)
  real outflowAccumulatedG = 0.0; // boundary accumulated outflow volume in domain [L^3] (global)
  real inflowDischargeG; // boundary inflow discharge in domain [L^3/T] (global)
  real inflowAccumulatedG = 0.0; // boundary accumulated inflow volume in domain [L^3] (global)

	#if SERGHEI_SUSPENDED_SEDIMENT
		real outflowSolidDischarge;
		real outflowSolidAccumulated = 0;
		real inflowSolidDischarge;
		real inflowSolidAccumulated = 0;

		real outflowSolidDischargeG;
		real outflowSolidAccumulatedG = 0;
		real inflowSolidDischargeG;
		real inflowSolidAccumulatedG = 0;		
	#endif		


  std::vector<ExtBC>* extbc;

  void initialize (std::vector<ExtBC> &extbc_)
  {
    extbc = &extbc_;
  }

  void integrate (std::vector<ExtBC> &extbc, Domain const &dom, int mode){
    timerFull.reset();

	 //mode is a flag to integrate extra mass or boundary flows

	 if(mode==0){

	 	adjustedVolume = 0.0;
		for (int i = 0; i < extbc.size(); i ++) {
			adjustedVolume += extbc[i].adjustedVolume;
		}
	 	adjustedVolumeG = adjustedVolume;

		if(dom.nsubdom > 1){		
			timer.reset();
			MPI_Allreduce(&adjustedVolume, &adjustedVolumeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			dom.timers.swe.integrate.mpi += timer.seconds();
		}

	 }else{

		 ncellsBC = 0;
		 int _ncellsBC;
		 inflowDischarge = 0.0;
		 outflowDischarge = 0.0;
     	inflowAccumulated= 0.0;
     	outflowAccumulated=0.0;

		#if SERGHEI_SUSPENDED_SEDIMENT
			real outflowSolidDischarge = 0.0;
			real outflowSolidAccumulated = 0.0;
			real inflowSolidDischarge = 0.0;
			real inflowSolidAccumulated = 0.0;
		#endif		 

		 // integrate over all the open external boundaries
		 // no MPI reduction is necessary, as they flows and volumes are already computed per open boundary in ExtBC::integrate
		 for (int i = 0; i < extbc.size(); i ++) {
			if(dom.nsubdom > 1){
				timer.reset();
				MPI_Allreduce(&(extbc[i].ncellsBC), &_ncellsBC, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				dom.timers.swe.integrate.mpi += timer.seconds();
			}

			ncellsBC += _ncellsBC;
			inflowDischarge += extbc[i].inflowDischarge;
			inflowAccumulated+= extbc[i].inflowAccumulated;
			outflowDischarge += extbc[i].outflowDischarge;
			outflowAccumulated+= extbc[i].outflowAccumulated;

			#if SERGHEI_SUSPENDED_SEDIMENT
				inflowSolidDischarge += extbc[i].inflowSolidDischarge;
				inflowSolidAccumulated+= extbc[i].inflowSolidAccumulated;
				outflowSolidDischarge += extbc[i].outflowSolidDischarge;
				outflowSolidAccumulated+= extbc[i].outflowSolidAccumulated;
			#endif

			#if SERGHEI_DEBUG_BOUNDARY
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\tinflowDischarge = " << extbc[i].inflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tinflowDischarge = " << inflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\toutflowDischarge = " << extbc[i].outflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\toutflowDischarge = " << outflowDischarge << std::endl;
				#if SERGHEI_SUSPENDED_SEDIMENT	
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\tinflowSolidDischarge = " << extbc[i].inflowSolidDischarge << std::endl;
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tinflowSolidDischarge = " << inflowSolidDischarge << std::endl;
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\toutflowSolidDischarge = " << extbc[i].outflowSolidDischarge << std::endl;
					std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\toutSolidflowDischarge = " << outflowSolidDischarge << std::endl;		
				#endif			
			#endif
		 }

		inflowDischargeG = inflowDischarge;
		outflowDischargeG = outflowDischarge;
		inflowAccumulatedG = inflowAccumulated;
		outflowAccumulatedG = outflowAccumulated;
		if(dom.nsubdom > 1){
			timer.reset();
			MPI_Allreduce(&inflowDischarge, &inflowDischargeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&outflowDischarge, &outflowDischargeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&inflowAccumulated, &inflowAccumulatedG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&outflowAccumulated, &outflowAccumulatedG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
	
			#if SERGHEI_SUSPENDED_SEDIMENT
			inflowSolidDischargeG=0.0;
			outflowSolidDischargeG=0.0;
			inflowSolidAccumulatedG=0.0;
			outflowSolidAccumulatedG=0.0;		
			MPI_Allreduce(&inflowSolidDischarge, &inflowSolidDischargeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&outflowSolidDischarge, &outflowSolidDischargeG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&inflowSolidAccumulated, &inflowSolidAccumulatedG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&outflowSolidAccumulated, &outflowSolidAccumulatedG, 1, SERGHEI_MPI_REAL , MPI_SUM, MPI_COMM_WORLD);
			#endif

			MPI_Barrier(MPI_COMM_WORLD);
			dom.timers.swe.integrate.mpi += timer.seconds();
		}
	 }
	dom.timers.swe.integrate.total += timerFull.seconds();
	}

};

#endif
