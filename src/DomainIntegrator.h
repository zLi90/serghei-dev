#ifndef _DOMAIN_INTEGRATOR_H_
#define _DOMAIN_INTEGRATOR_H_

#include "State.h"
#include "SourceSink.h"
#include "Domain.h"
#include "Indexing.h"

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
	rainFlux=0.0;
	infFlux=0.0;
	if(dom.etime<TOL12){ //change by initial time when hotstart is implemented
	 	rainAccum=0.0;
	 	infAccum=0.0;
	}

	// Copy host-side scalar values to local variables for GPU capture (by value)
	// Plain class members like int/bool are in host memory and need to be copied
	const int isRain_local = dom.isRain;
	const int infModel_local = ss.inf.model;
	const real area_local = dom.cellArea();
	const real dt_local = dom.dt;
	const int nx_local = dom.nx;

	// Copy Kokkos Views to local variables for proper GPU capture
	// Views are shallow copies, so this is efficient
	realArr h_view = state.h;
	boolArr isnodata_view = state.isnodata;

	// First reduction: always compute surface volume
	Kokkos::parallel_reduce( "integrate_surface_h", dom.nCell , KOKKOS_LAMBDA (int iGlob, real & hSum) {
		// Compute halo-extended index (same logic as dom.getIndex)
		int i = iGlob % nx_local;
		int j = iGlob / nx_local;
		int ii = (hc + j) * (nx_local + 2*hc) + hc + i;
		
		if(!isnodata_view(ii)){
			real h_val = h_view(ii);
			// Use Kokkos-compatible NaN/Inf checks (avoid std:: functions on GPU)
			// NaN check: val != val is true only for NaN
			bool h_valid = (h_val == h_val) && (h_val != HUGE_VAL) && (h_val != -HUGE_VAL) && (h_val >= 0.0);
			if(h_valid) {
        		hSum += h_val * area_local;
			}
		}
    } , Kokkos::Sum<real>(surfaceVolume));

	// Second reduction: rain flux (only if rain is enabled)
	if(isRain_local) {
		realArr rainRate_view = ss.rainRate;
		Kokkos::parallel_reduce( "integrate_surface_rain", dom.nCell , KOKKOS_LAMBDA (int iGlob, real &rainSum) {
			int i = iGlob % nx_local;
			int j = iGlob / nx_local;
			int ii = (hc + j) * (nx_local + 2*hc) + hc + i;
			
			if(!isnodata_view(ii)){
				real rain_val = rainRate_view(ii);
				bool rain_valid = (rain_val == rain_val) && (rain_val != HUGE_VAL) && (rain_val != -HUGE_VAL);
				if(rain_valid) {
					rainSum += rain_val * area_local;
				}
			}
		} , Kokkos::Sum<real>(rainFlux));
	}

	// Third reduction and update: infiltration (only if infiltration model is enabled)
	if(infModel_local){
		realArr infRate_view = ss.inf.rate;
		realArr infVol_view = ss.inf.infVol;
		
		Kokkos::parallel_reduce( "integrate_surface_inf", dom.nCell , KOKKOS_LAMBDA (int iGlob, real& infSum) {
			int i = iGlob % nx_local;
			int j = iGlob / nx_local;
			int ii = (hc + j) * (nx_local + 2*hc) + hc + i;
			
			if(!isnodata_view(ii)){
				real infrate = infRate_view(ii);
				bool inf_valid = (infrate == infrate) && (infrate != HUGE_VAL) && (infrate != -HUGE_VAL) && (infrate >= 0.0);
				if(inf_valid) {
					real inffluxlocal = infrate * area_local;
					infSum += inffluxlocal;
				}
			}
		} , Kokkos::Sum<real>(infFlux));

		// Separate parallel_for to update infVol (cannot write inside parallel_reduce)
		Kokkos::parallel_for( "update_infVol", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
			int i = iGlob % nx_local;
			int j = iGlob / nx_local;
			int ii = (hc + j) * (nx_local + 2*hc) + hc + i;
			
			if(!isnodata_view(ii)){
				real infrate = infRate_view(ii);
				bool inf_valid = (infrate == infrate) && (infrate != HUGE_VAL) && (infrate != -HUGE_VAL) && (infrate >= 0.0);
				if(inf_valid) {
					real inffluxlocal = infrate * area_local;
					infVol_view(ii) += inffluxlocal * dt_local;
				}
			}
		});
	}

	rainAccum += rainFlux * dom.dt;
	infAccum += infFlux * dom.dt;

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
			#if SERGHEI_DEBUG_BOUNDARY
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\tinflowDischarge = " << extbc[i].inflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tinflowDischarge = " << inflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tBC " << i << "\tinflowDischarge = " << extbc[i].outflowDischarge << std::endl;
				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\toutflowDischarge = " << outflowDischarge << std::endl;
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
			MPI_Barrier(MPI_COMM_WORLD);
			dom.timers.swe.integrate.mpi += timer.seconds();
		}
	 }
	dom.timers.swe.integrate.total += timerFull.seconds();
	}

};

#endif
