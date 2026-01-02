/* -*- mode: c++ -*- */

#ifndef _TIMEINTEGRATOR_H_
#define _TIMEINTEGRATOR_H_

#include <stdlib.h>
#include <math.h>

#include "define.h"
#include "Parallel.h"
#include "Domain.h"
#include "State.h"
#include "BC.h"
#include "Edges.h"
#include "Indexing.h"
#include "FileIO.h"
#include "SourceSink.h"

class TimeIntegrator {

  Edges edge;
  Kokkos::Timer timer,timermpi, timerSolve;

public :

  inline void stepForward(State &state, SourceSinkData &ss, std::vector<ExtBC> &extbc, Domain &dom, Exchange &exch, Parallel &par, FileIO &io) {
	timerSolve.reset();
		
	#if !SERGHEI_SWE_RE
	  computeDt(state,dom,io);
	#endif
	
	#if !SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION

		edge.computeDeltaStateSW(state, dom, exch, par);

		ss.ComputeSWSourceSink(state,dom);

	#endif

   	computeNewState(state, dom, ss);

	#if SERGHEI_VERTICAL_VELOCITY
		computeVerticalVelocity(state, dom, ss);
	#endif

	#if !SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION

		for (int k = 0; k < extbc.size(); k ++) { //should be done before the exchange (water depth might be modified).
			extbc[k].apply(state,dom);
		}

		exch.exchangeMPIh(state,dom,exch,par); //only neccesary to exchange the h (for wet-dry) but for the moment we exchange everything


		wetDryCorrection( state, dom);

		exch.exchangeMPIhuhv(state,dom,exch,par);//neccesary to exchange again because of the wet/dry correction


		for (int k = 0; k < extbc.size(); k ++) { //after getting the final values, the discharge is integrated at every BC. The reason for not doing this before is because the previous kernels could eventually modify the boundary cell values.
			extbc[k].integrate(state,dom);
		}
    #endif
	dom.timers.swe.solve += timerSolve.seconds();
	}

    inline void computeGwExchange(State &state , const Domain &dom) {
        // Note: state.qss is already distributed from gw.qss_gw in serghei.h before this call
        Kokkos::parallel_for( dom.nCell , KOKKOS_LAMBDA (int idom) {
            int ii = dom.getIndex(idom);
            state.h(ii) += state.qss(idom) * dom.dt;
            if (state.h(ii) < state.hmin)   {
                state.hu(ii) = 0.0;
                state.hv(ii) = 0.0;
                if(state.h(ii)<TOL12) {state.h(ii)=0.0;}
            }
        });
    }

	inline void computeNewState(State &state , const Domain &dom, const SourceSinkData &ss) {
		#if SERGHEI_DEBUG_WORKFLOW
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		timer.reset();
		Kokkos::parallel_for("computeNewState",dom.nCell , KOKKOS_LAMBDA (int iGlob) {
			int ii = dom.getIndex(iGlob);

			real z=state.z(ii);
			real hold=state.h(ii);
			real huold=state.hu(ii);
			real hvold=state.hv(ii);
			bool nodata=state.isnodata(ii);

			real hf,huf,hvf;
					int ncells =dom.nCellMem;
					#if SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION
				/* //SIMULATION WITHOUT HYDRODYNAMIC EVOLUTION*/
				hf = hold;
			#else
				real dx_inv = 1.0 / dom.dx();
				real flux_div = (state.dsw0(ii)+state.dsw1(ii)) * dx_inv;
				hf = hold - dom.dt * flux_div;
			#endif
			
			// Check for NaN or invalid values
			if(std::isnan(hf) || std::isinf(hf)) {
				hf = 0.0;
			}
			
			if (dom.isRain) {
				real rainAdd = ss.rainRate(ii)*dom.dt;
				// Check for NaN in rainfall
				if(std::isnan(rainAdd) || std::isinf(rainAdd)) {
					rainAdd = 0.0;
				}
				hf += rainAdd;
				// Negative rainfall represents evaporation
				if (hf < 0.0)	{hf = 0.0;}
			}

			if(ss.inf.model) {
				// Prevent division by very small dt
				if(dom.dt > 1e-10) {
					ss.inf.rate(ii)=min(ss.inf.rate(ii),hf/dom.dt); //correct infiltration arte according to the available water volume
				} else {
					ss.inf.rate(ii) = 0.0;
				}
				ss.inf.rate(ii)=max(ss.inf.rate(ii),0.0); //avoid negative (in the order of machine accuracy) infiltration rates
				
				// Check for NaN in infiltration rate
				if(std::isnan(ss.inf.rate(ii)) || std::isinf(ss.inf.rate(ii))) {
					ss.inf.rate(ii) = 0.0;
				}

			hf -= ss.inf.rate(ii)*dom.dt;
			}

			// Final check for NaN before assigning
			if(std::isnan(hf) || std::isinf(hf) || hf<TOL_MACHINE_ACCURACY || nodata){
				//reduction or remove. Should be in the order of machine accuracy
				hf=0.0;
			}
			if(hf<state.hmin){
				huf=0.0;
				hvf=0.0;
			}else{
				real dx_inv = 1.0 / dom.dx();
				real mx= huold - (state.dsw0(ii+ncells)+state.dsw1(ii+ncells))*dom.dt*dx_inv;
				real my= hvold - (state.dsw0(ii+2*ncells)+state.dsw1(ii+2*ncells))*dom.dt*dx_inv;
				
				// Check for NaN in momentum
				if(std::isnan(mx) || std::isinf(mx)) mx = 0.0;
				if(std::isnan(my) || std::isinf(my)) my = 0.0;
				
                // wind stress 
                if (dom.isWind == 1 && hf > dom.hwmin && hold > TOL12)    {
                    real absu = sqrt(mx*mx/hold/hold+my*my/hold/hold);
                    real coef = dom.dt * RHOA * dom.CwT / RHOW;
                    // omega, beta both in radians
                    real omega = (90.0 - ss.winddir(ii)) * PI/180.0;
                    real beta;
                    if (huold == 0)	{beta = omega-90.0*PI/180.0;}
                    else {beta = omega - atan(hvold/huold);}
                    real sigmax, sigmay;
                    sigmax = coef * cos(omega) * (ss.windspd(ii) - absu*cos(beta)) * (ss.windspd(ii) - absu*cos(beta));
                    sigmay = coef * sin(omega) * (ss.windspd(ii) - absu*cos(beta)) * (ss.windspd(ii) - absu*cos(beta));
                    if (huold != 0)	{mx += sigmax;}
                    if (hvold != 0) {my += sigmay;}
                }
				#if SERGHEI_POINTWISE_FRICTION
					real nsq= state.roughness(ii)*state.roughness(ii);
					real modM = 0.0;
					if(hold > TOL12) {
						modM=sqrt(mx*mx/hold/hold+my*my/hold/hold);
					}
					if(nsq>TOL12 && hold>=state.hmin && modM>TOL12 && hold > TOL12){
						real hold_cbrt = cbrt(hold);
						if(hold_cbrt > TOL12) {
							real tt=dom.dt*GRAV*nsq*modM/(hold*hold_cbrt);
							if(tt > TOL12) {
								real ff=sqrt(1.0+4.0*tt);
								huf=-0.5*(mx-mx*ff)/tt;
								hvf=-0.5*(my-my*ff)/tt;
							} else {
								huf=mx;
								hvf=my;
							}
						} else {
							huf=mx;
							hvf=my;
						}
					}else{
						huf=mx;
						hvf=my;
					}
				#else
					huf=mx;
					hvf=my;
				#endif
			}

			if(fabs(huf)<TOL_ZERO_MOMENTUM){
				huf=0.0;
			}
			if(fabs(hvf)<TOL_ZERO_MOMENTUM){
				hvf=0.0;
			}
			
			// Final validation before assignment
			if(std::isnan(hf) || std::isinf(hf) || hf < 0.0) {
				hf = 0.0;
				huf = 0.0;
				hvf = 0.0;
			}
			if(std::isnan(huf) || std::isinf(huf)) {
				huf = 0.0;
			}
			if(std::isnan(hvf) || std::isinf(hvf)) {
				hvf = 0.0;
			}

			state.h(ii)=hf;
			state.hu(ii)=huf;
			state.hv(ii)=hvf;

			//reset the contributions
			state.dsw0(ii)=0.0;
			state.dsw0(ii+ncells)=0.0;
			state.dsw0(ii+2*ncells)=0.0;
			state.dsw1(ii)=0.0;
			state.dsw1(ii+ncells)=0.0;
			state.dsw1(ii+2*ncells)=0.0;

			#if SERGHEI_MAXFLOOD
				if(hf > state.hMax(ii)){
					state.hMax(ii) = hf;
					state.time_hMax(ii) = dom.etime;
				}
				real mom = mysqrt(huf*huf+hvf*hvf);
				if(mom > state.momentumMax(ii)) state.momentumMax(ii) = mom;
			#endif

    	});

		dom.timers.swe.update.total += timer.seconds();
    }

	inline void dtMatchOutput(Domain &dom, const FileIO &io){
    	// correction to match output times
		//std::cout << GGD << dom.etime << "\t" << dom.etime+dom.dt << "\t" << io.numOut << "\t" << io.outFreq*io.numOut << "\t" << io.numOut*io.outFreq + dom.startTime << std::endl;
    	if (dom.etime + dom.dt > dom.startTime +  io.numOut*io.outFreq) dom.dt = io.numOut*io.outFreq + dom.startTime - dom.etime;
    	if (dom.etime + dom.dt > dom.endTime) { dom.dt = dom.endTime - dom.etime; }
		#if SERGHEI_DEBUG_DT
			std::cout << "time = " << dom.etime << "\tdt_cor = " << dom.dt << std::endl;
		#endif
	}
	
	inline void computeDt(State &state, Domain &dom, FileIO &io) {
		timer.reset();

		dom.dt = 1.e7;

		Kokkos::parallel_reduce("reduceDt",dom.nCell , KOKKOS_LAMBDA (int iGlob, real &dt) {
			int ii = dom.getIndex(iGlob);
			real h=state.h(ii);
			real hu=state.hu(ii);
			real hv=state.hv(ii);
			dt=min(dt,1.e6);
			if(h>TOL12){
				dt=min(dt,dom.dx()/(fabs(hu/h)+sqrt(GRAV*h)));
				dt=min(dt,dom.dx()/(fabs(hv/h)+sqrt(GRAV*h)));
			}
		} , Kokkos::Min<real>(dom.dt) );

		Kokkos::fence();

		real dtloc = dom.dt;
		if(dom.nsubdom > 1){
			timermpi.reset();
			MPI_Allreduce(&dtloc, &dom.dt, 1, SERGHEI_MPI_REAL , MPI_MIN, MPI_COMM_WORLD);
			dom.timers.swe.dt.mpi = timermpi.seconds();
		}

		dom.dt*=dom.cfl;

		#if SERGHEI_DEBUG_DT
			std::cout << "time = " << dom.etime << "\tdt_cfl = " << dom.dt << "\tdom.cfl = " << dom.cfl << std::endl;
		#endif
		if(dom.dt>1e5){ //it means that evertyhing is dry.
			//we impose a time step equivalent for h=1
			// this is to make sure we capture the start of the simulation
			dom.dt=dom.dx()/(1+sqrt(GRAV));		// dx() purposely used here to fail upon compilation when moving to adaptive mesh
			#if SERGHEI_DEBUG_DT
				std::cout << "time = " << dom.etime << "\tdt_dry = " << dom.dt << std::endl;
			#endif
		}
		dtMatchOutput(dom, io);
		dom.timers.swe.dt.total=timer.seconds();
	}


	inline void wetDryCorrection(State &state, Domain &dom) {
	#if SERGHEI_DEBUG_WORKFLOW
    	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	#endif
    timer.reset();

    	Kokkos::parallel_for("wetDryCorrection", dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
			int i, j;
      	unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
			if(i>hc-1 && i<dom.nx+hc && j>hc-1 && j<dom.ny+hc){ //computational domain without halo cells
				real hij=state.h(iGlob);
				real zij=state.z(iGlob);
				int isB=state.isBound(iGlob);
				if(hij >= state.hmin){
					if(((hij+zij< state.z(iGlob+1)) && state.h(iGlob+1)<TOL_WETDRY) || ((hij+zij<state.z(iGlob-1)) && state.h(iGlob-1)<TOL_WETDRY) || (isB==0 && state.isnodata(iGlob+1)) || (isB==0 && state.isnodata(iGlob-1) )){
						state.hu(iGlob)=0.0;
					}
					if(((hij+zij<state.z(iGlob+dom.nx+2*hc)) && state.h(iGlob+dom.nx+2*hc)<TOL_WETDRY)  || ((hij+zij<state.z(iGlob-(dom.nx+2*hc))) && state.h(iGlob-(dom.nx+2*hc))<TOL_WETDRY) || (isB==0 && state.isnodata(iGlob+dom.nx+2*hc)) || (isB==0 && state.isnodata(iGlob-(dom.nx+2*hc)))){
						state.hv(iGlob)=0.0;
					}
				}
			}

		});

    dom.timers.swe.wetdrycorr.total += timer.seconds();
  }

  #if SERGHEI_VERTICAL_VELOCITY
	inline void computeVerticalVelocity(State &state , const Domain &dom, const SourceSinkData &ss){
        Kokkos::parallel_for("computeBottomVelocityX", dom.nCellMem , KOKKOS_LAMBDA (int iGlob){
      		int i, j;
			int id1,id2;
      		unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
			if(i>0 && i<dom.nx+1 && j>0 && j<dom.ny+1){ //note the hc-2 (first valid halo-inner wall) 
				SArray<real,3> upwM, upwP;
				SArray<real,5> s1,s2; //3 sw variables plus z and roughness
			
				if(i==1){//Upwind
					id1=iGlob; //j*(dom.nx+2*hc)+i
					id2=j*(dom.nx+2*hc)+i+1;
				}else if(i==dom.nx_glob){//Downwind
					id1=iGlob-1; //j*(dom.nx+2*hc)+i
					id2=j*(dom.nx+2*hc)+i;
				}else{//Centered
					id1=iGlob-1;
					id2=j*(dom.nx+2*hc)+i+1;
				}

				s1(idH)=state.h(id1);
				s2(idH)=state.h(id2);
				
				bool nodata = state.isnodata(id1) || state.isnodata(id2);

				if((s1(idH)>0. || s2(idH)>0.) && !nodata){ //avoid dry-pair, nodata and boundary cells
					s1(idZ)=state.z(id1);
					s2(idZ)=state.z(id2);
					state.dZ_X(iGlob)=(s2(idZ)-s1(idZ))/(2*dom.dx());

					s1(idHU)=state.hu(id1);
					s2(idHU)=state.hu(id2);
					real u_1, u_2;
					if(s1(idH)<TOL6){
						u_1=0.0;
					}else{
						u_1=s1(idHU)/s1(idH);
					}	
					if(s2(idH)<TOL6){
						u_2=0.0;
					}else{
						u_2=s2(idHU)/s2(idH);
					}
					state.dU_X(iGlob)=(u_2-u_1)/(2*dom.dx());
				}else{
					state.dU_X(iGlob)=0.0;
					state.dZ_X(iGlob)=0.0;
				}
			}else{
				state.dU_X(iGlob)=0.0;
				state.dZ_X(iGlob)=0.0;
			}
		
		});
		Kokkos::parallel_for("computeBottomY",dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
      		int i, j;
			int id1,id2;
     		unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
			if(i>0 && i<dom.nx+1 && j>0  && j<dom.ny+1){ //note the hc-2 (first valid halo-inner wall)
				SArray<real,3> upwM, upwP;
				SArray<real,5> s1,s2; //3 sw variables plus z and roughness
			
				if(j==1){//Upwind
					id1=(j)*(dom.nx+2*hc)+i;
					id2=(j+1)*(dom.nx+2*hc)+i;
				}else if(j==dom.ny_glob){//Downwind
					id1=(j-1)*(dom.nx+2*hc)+i;
					id2=(j)*(dom.nx+2*hc)+i;
				}else{//Centered
					id1=(j-1)*(dom.nx+2*hc)+i;
					id2=(j+1)*(dom.nx+2*hc)+i;
				}
				

				s1(idH)=state.h(id1);
				s2(idH)=state.h(id2);

				bool nodata = state.isnodata(id1) || state.isnodata(id2);

				if((s1(idH)>0. || s2(idH)>0.) && !nodata){ //avoid dry-pair, nodata and boundary cells
					s1(idZ)=state.z(id1);
					s2(idZ)=state.z(id2);
					state.dZ_Y(iGlob)=(s2(idZ)-s1(idZ))/(2*dom.dx());

					s1(idHV)=state.hv(id1);
					s2(idHV)=state.hv(id2);
					real v_1, v_2;
					if(s1(idH)<TOL6){
						v_1=0.0;
					}else{
						v_1=s1(idHV)/s1(idH);
					}	
					if(s2(idH)<TOL6){
						v_2=0.0;
					}else{
						v_2=s2(idHV)/s2(idH);
					}
					state.dU_Y(iGlob)=(v_2-v_1)/(2*dom.dx());
				}else{
					state.dU_Y(iGlob)=0.0;
					state.dZ_Y(iGlob)=0.0;
				}
			}else{
				state.dU_Y(iGlob)=0.0;
				state.dZ_Y(iGlob)=0.0;
			}
		
		});
		Kokkos::parallel_for("computeW",dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
			bool nodata=state.isnodata(iGlob);
			int i,j;
			real h=state.h(iGlob);
			unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
			if(!nodata && h>0.){
				state.w(iGlob)=state.hu(iGlob)/h*state.dZ_X(iGlob)+state.hv(iGlob)/h*state.dZ_Y(iGlob)-h/2.0*state.dU_X(iGlob)-h/2.0*state.dU_Y(iGlob);
				if(state.w(iGlob)<=0.0){
					//printf("id:%d, i:%d, j:%d\n",iGlob, i, j);
				}
			}else{
				state.w(iGlob)=0.0;
			}
		});	
    }
	#endif


}; // end of TimeIntegrator class

#endif
