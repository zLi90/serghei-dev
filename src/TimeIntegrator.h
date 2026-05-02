/* -*- mode: c++ -*- */
#pragma once

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
	Kokkos::fence();
	timerSolve.reset();

	#if !SERGHEI_SWE_RE
	  computeDt(state,dom,io);
	#endif

	#if !SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION

		edge.computeDeltaStateSW(state, dom, exch, par);
		#if SERGHEI_SCALAR_TRANSPORT
			for (int k = 0; k < extbc.size(); k ++) { //should be done before the exchange (water depth might be modified).
				outletScalarFlux(state,extbc[k],dom);
			}
		#endif

		ss.ComputeSWSourceSink(state,dom);

	#endif

   	computeNewState(state, dom, ss, exch, par);

	#if SERGHEI_VERTICAL_VELOCITY
		computeVerticalVelocity(state, dom, ss);
	#endif

	#if !SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION

		for (int k = 0; k < extbc.size(); k ++) { //should be done before the exchange (water depth might be modified).
			extbc[k].apply(state,dom);
		}

		#if SERGHEI_SEDIMENT_TRANSPORT
		exch.exchangeMPIz(state,dom,par); //neccesary to exchange the z for wet-dry
		#endif

		exch.exchangeMPIh(state,dom,par); //neccesary to exchange the h (for wet-dry) but for the moment we exchange everything


		wetDryCorrection( state, dom);
		exch.exchangeMPIhuhv(state,dom,par);//neccesary to exchange again because of the wet/dry correction

		#if SERGHEI_SCALAR_TRANSPORT
		exch.exchangeMPIscalars(state,dom,par);
		#endif


		for (int k = 0; k < extbc.size(); k ++) { //after getting the final values, the discharge is integrated at every BC. The reason for not doing this before is because the previous kernels could eventually modify the boundary cell values.
			extbc[k].integrate(state,dom);
		}
    #endif
	Kokkos::fence();
	dom.timers.swe.solve += timerSolve.seconds();
	}

	inline void computeGwExchange(State &state , const Domain &dom) {
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

	inline void computeNewState(State &state , Domain &dom, const SourceSinkData &ss, Exchange &exch, Parallel &par) {
		#if SERGHEI_DEBUG_WORKFLOW
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		Kokkos::fence("computeNewState-start");
		timer.reset();
		Kokkos::parallel_for("computeNewState",dom.nCell , KOKKOS_LAMBDA (int iGlob) {
			int ii = dom.getIndex(iGlob);

		int ncells =dom.nCellMem;

		real zold=state.z(ii);
		real hold=state.h(ii);
		real huold=state.hu(ii);
		real hvold=state.hv(ii);
		bool nodata=state.isnodata(ii);

		real hf, huf, hvf;

		#if SERGHEI_SEDIMENT_TRANSPORT
		real zf;
		real dzf = 0.0;
		#endif

		#if SERGHEI_SEDIMENT_TRANSPORT || SERGHEI_EROSIVE_SHEAR
			real phi_total = 0.0;//phi_total is the total sediment that is transported in the cell
		#endif

		#if SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION
			hf = hold;
		#else
			#if SERGHEI_SEDIMENT_TRANSPORT && SERGHEI_UPWIND_BED
				zf = zold - dom.dt * (state.sediment.dz0(ii)+state.sediment.dz1(ii))/dom.dx();
				state.z(ii)=zf;

				//reset the contributions
				state.sediment.dz0(ii) = 0.0;
				state.sediment.dz1(ii) = 0.0;
			#endif


			hf = hold - dom.dt * (state.dsw0(ii)+state.dsw1(ii))/dom.dx();
		#endif

		if(dom.isRain) {
			hf += ss.rainRate(ii)*dom.dt;
			// Negative rainfall represents evaporation
			if (hf < 0.0) hf = 0.0;
		}
    if(ss.inf.model) {
	 		ss.inf.rate(ii)=min(ss.inf.rate(ii),hf/dom.dt); //correct infiltration arte according to the available water volume
			ss.inf.rate(ii)=max(ss.inf.rate(ii),0.0); //avoid negative (in the order of machine accuracy) infiltration rates

			hf -= ss.inf.rate(ii)*dom.dt;
		}

		if(hf<TOL_MACHINE_ACCURACY || nodata){
			//reduction or remove. Should be in the order of machine accuracy
			hf=0.0;
		}

		if(hf<state.hmin){
			huf=0.0;
			hvf=0.0;
		}else{
			real mx= huold - (state.dsw0(ii+ncells)+state.dsw1(ii+ncells))*dom.dt/dom.dx();
			real my= hvold - (state.dsw0(ii+2*ncells)+state.dsw1(ii+2*ncells))*dom.dt/dom.dx();

			// wind stress
			if (dom.isWind == 1 && hf > dom.hwmin)    {
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
			#if SERGHEI_POINTWISE_FRICTION == 1
				real modM=sqrt(mx*mx/hold/hold+my*my/hold/hold);
				if(nsq>TOL12 && hold>=state.hmin && modM>TOL12){
					real tt=dom.dt*GRAV*nsq*modM/(hold*cbrt(hold));
					real ff=sqrt(1.0+4.0*tt);
					huf=-0.5*(mx-mx*ff)/tt;
					hvf=-0.5*(my-my*ff)/tt;
				}
			#endif
			#if SERGHEI_POINTWISE_FRICTION == 2
				real modM=sqrt(mx*mx+my*my)/(hf*hf);
				if(modM>TOL_ZERO_MOMENTUM && nsq>TOL12 && dom.dt>TOL12){
					real tt = 2.*dom.dt*GRAV*nsq*modM/cbrt(hf);
					real ff = (sqrt(1.+2.*tt)-1.)/tt;
					huf = ff*mx;
					hvf = ff*my;
				}
			#endif
				else{
					huf=mx;
					hvf=my;
				}
			#else
				huf=mx;
				hvf=my;
			#endif

			if(fabs(huf)<TOL_ZERO_MOMENTUM){
				huf=0.0;
			}
			if(fabs(hvf)<TOL_ZERO_MOMENTUM){
				hvf=0.0;
			}
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

		#if SERGHEI_SCALAR_TRANSPORT
			//state.ade.timer.reset();
			state.ade.updateAndReset(ii, hf, dom.dt, dom.dx());
			//dom.timers.scalarUpdate+=state.ade.timer.seconds();
		#endif

		#if SERGHEI_SEDIMENT_TRANSPORT
			//state.sediment.timer.reset();
			state.sediment.updateAndReset(ii,dom,zf,hf,huf,hvf,state.ade,state.hmin,phi_total);
			state.h(ii)=hf;
			state.hu(ii)=huf;
			state.hv(ii)=hvf;
			#if SERGHEI_UPWIND_BED==0
				state.z(ii)=zf;
			#endif

			dzf=state.z(ii)-state.zini(ii);
			state.sediment.computeSuspendedSedimentExchange(ii,hf,huf,hvf,
				state.roughness(ii),
				state.ade,
				state.hmin,
				dom.dt,
				ss.rainRate(ii),
				dzf);
			//dom.timers.sedUpdate+=state.sediment.timer.seconds();
		#endif

		#if SERGHEI_EROSIVE_SHEAR
			state.shearAccum(ii) += computeShields(hold,huold,hvold,state.roughness(ii),state.hmin)*dom.dt;
			state.phiTot(ii) += phi_total*dom.dt;
		#endif

			#if SERGHEI_MAXFLOOD
				if(hf > state.hMax(ii)){
					state.hMax(ii) = hf;
					state.time_hMax(ii) = dom.etime;
				}
				real mom = mysqrt(huf*huf+hvf*hvf);
				if(mom > state.momentumMax(ii)) state.momentumMax(ii) = mom;
			#endif

    	});

		Kokkos::fence("computeNewState-end");
		dom.timers.swe.update.total += timer.seconds();

		#if SERGHEI_SCALAR_DIFFUSION
			// Use three-step diffusion with MPI exchange between sub-steps for consistency
			int numSubSteps = state.ade.initializeDiffusion(state.h,dom);
			for (int iNs=0; iNs<numSubSteps; iNs++){
				real subDt = state.ade.diffDt;
				if(iNs==numSubSteps-1){
					subDt = dom.dt - (numSubSteps-1)*state.ade.diffDt;
				}
				state.ade.performDiffusionSubstep(state.h,dom,subDt);
				// CRITICAL: Exchange halos after each sub-step (except the last)
				// This ensures halo cells contain updated values from neighboring ranks
				if(iNs < numSubSteps-1 && dom.nsubdom > 1){
					exch.exchangeMPIscalars(state,dom,par);
				}
			}
			state.ade.finalizeDiffusion(state.h,dom);
		#endif
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
		Kokkos::fence();
		timer.reset();

		dom.dt = 1.e7;

		#if SERGHEI_SWE_DRY_RUNOFF_START_DT || SERGHEI_DEBUG_DT
		real maxh=0.0;
		#endif

		Kokkos::parallel_reduce("reduceDt",dom.nCell , KOKKOS_LAMBDA (int iGlob, real &dt
			#if SERGHEI_SWE_DRY_RUNOFF_START_DT
			, real &upval
			#endif
			) {
			int ii = dom.getIndex(iGlob);

			real h=state.h(ii);
			real hu=state.hu(ii);
			real hv=state.hv(ii);
			dt=min(dt,1.e6);
			if(h>TOL12){
				dt=min(dt,dom.dx()/(fabs(hu/h)+sqrt(GRAV*h)));
				dt=min(dt,dom.dx()/(fabs(hv/h)+sqrt(GRAV*h)));

				#if SERGHEI_SEDIMENT_TRANSPORT && SERGHEI_UPWIND_BED
						real lambdab=0.0;
						real aux1;
						//east
						aux1 = state.sediment.getLambdaBcell(ii, state.z(ii), state.z(ii+1), dom.dx(), state.ade);
						lambdab=max(lambdab,fabs(aux1));
						//west
						aux1 = state.sediment.getLambdaBcell(ii, state.z(ii), state.z(ii-1), dom.dx(), state.ade);
						lambdab=max(lambdab,fabs(aux1));
						//north
						aux1 = state.sediment.getLambdaBcell(ii, state.z(ii), state.z(ii+dom.nx+2*dom.hc), dom.dx(), state.ade);
						lambdab=max(lambdab,fabs(aux1));
						//south
						aux1 = state.sediment.getLambdaBcell(ii, state.z(ii), state.z(ii-(dom.nx+2*dom.hc)), dom.dx(), state.ade);
						lambdab=max(lambdab,fabs(aux1));

						if(lambdab>0.0) dt=min(dt,dom.dx()/lambdab);
				#endif
				#if SERGHEI_SWE_DRY_RUNOFF_START_DT
				upval=max(upval,h);
				#endif
			}
		} , Kokkos::Min<real>(dom.dt)
			#if SERGHEI_SWE_DRY_RUNOFF_START_DT
		 	, Kokkos::Max<real>(maxh)
			#endif
		);
		Kokkos::fence();

		real dtloc = dom.dt;
	  if(dom.nsubdom > 1){
			timermpi.reset();
			MPI_Allreduce(&dtloc, &dom.dt, 1, SERGHEI_MPI_REAL , MPI_MIN, MPI_COMM_WORLD);
	  	#if SERGHEI_SWE_DRY_RUNOFF_START_DT
			real maxhloc = maxh;
			MPI_Allreduce(&maxhloc, &maxh, 1, SERGHEI_MPI_REAL , MPI_MAX, MPI_COMM_WORLD);
			#endif
			dom.timers.swe.dt.mpi = timermpi.seconds();
		}

	  dom.dt*=dom.cfl;
		#if SERGHEI_DEBUG_DT
			std::cout << "time = " << dom.etime
								<< "\tmaxh = " << maxh
								<< "\tdt_cfl = " << dom.dt
								<< "\tdom.cfl = " << dom.cfl << std::endl;
		#endif
		#if SERGHEI_SWE_DRY_RUNOFF_START_DT
			if(dom.isRain && maxh<state.hmin){ //runoff generation
				// if there is rain and the runoff has not started
				// we impose a max time step equivalent to critical flow Fr=1 with depth hmin
				// this is to make sure we capture the start of the rain
				real raindt = dom.cfl*dom.dx()/(2.*sqrt(GRAV*state.hmin));
				dom.dt=min(dom.dt,raindt);
				#if SERGHEI_DEBUG_DT
					std::cout << "time = " << dom.etime << "\tdt_rain = " << dom.dt << std::endl;
				#endif
			}
		#else
			if(dom.dt > 1E5 && dom.isRain) dom.dt=dom.dx()/(1+sqrt(GRAV));  // dx() purposely used here to fail upon compilation when moving to adaptive mesh
		#endif
		dtMatchOutput(dom, io);
		dom.timers.swe.dt.total=timer.seconds();
	}


	inline void wetDryCorrection(State &state, Domain &dom) {
	#if SERGHEI_DEBUG_WORKFLOW
  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	#endif
	Kokkos::fence();
    timer.reset();

    	Kokkos::parallel_for("wetDryCorrection", dom.nCellMem , KOKKOS_LAMBDA (int iGlob) {
			//int i, j;
      //unpackIndicesUniformGrid(iGlob,dom.ny+2*hc,dom.nx+2*hc,j,i);
			//if(i>hc-1 && i<dom.nx+hc && j>hc-1 && j<dom.ny+hc){ //computational domain without halo cells
      if(!dom.isHalo(iGlob)){
				real hij=state.h(iGlob);
				real zij=state.z(iGlob);
				int isB=state.isBound(iGlob);

				if(hij >= state.hmin){
					if(((hij+zij< state.z(iGlob+dom.di)) && state.h(iGlob+dom.di)<TOL_WETDRY) ||
             ((hij+zij<state.z(iGlob-dom.di)) && state.h(iGlob-dom.di)<TOL_WETDRY) ||
						 (isB==0 && state.isnodata(iGlob+dom.di)) || (isB==0 && state.isnodata(iGlob-dom.di) ) ||
             (state.isBound(iGlob+dom.di)==SERGHEI_BC_OUTER_REFLECTIVE_CELL) || (state.isBound(iGlob-dom.di)==SERGHEI_BC_OUTER_REFLECTIVE_CELL) ){
								state.hu(iGlob)=0.0;
					}
					if(((hij+zij<state.z(iGlob+dom.dj)) && state.h(iGlob+dom.dj)<TOL_WETDRY) ||
            ((hij+zij<state.z(iGlob-dom.dj)) && state.h(iGlob-dom.dj)<TOL_WETDRY) ||
						 (isB==0 && state.isnodata(iGlob+dom.dj)) || (isB==0 && state.isnodata(iGlob-dom.dj)) ||
						 (state.isBound(iGlob+dom.dj)==SERGHEI_BC_OUTER_REFLECTIVE_CELL) || (state.isBound(iGlob-dom.dj)==SERGHEI_BC_OUTER_REFLECTIVE_CELL) ){
								state.hv(iGlob)=0.0;
					}
				}
			}

		});

	Kokkos::fence();
    dom.timers.swe.wetdrycorr.total += timer.seconds();
  }

  #if SERGHEI_VERTICAL_VELOCITY
	inline void computeVerticalVelocity(State &state , const Domain &dom, const SourceSinkData &ss){
        Kokkos::parallel_for("computeBottomVelocityX", dom.nCellMem , KOKKOS_LAMBDA (int iGlob){
      		int i, j;
			int id1,id2;
      		unpackIndicesUniformGrid(iGlob,dom.ny+2*dom.hc,dom.nx+2*dom.hc,j,i);
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


#if SERGHEI_SCALAR_TRANSPORT
	//maintained here due to circular definitions between BC.h, State.h and ScalarTransport.h
	inline void outletScalarFlux(State &state, ExtBC &extbc, Domain &dom) {

		if(extbc.ncellsBC > 0){
			switch (extbc.bctype){
				//outlet boundary conditions
				case SWE_BC_CRITICAL:
				case SWE_BC_H_CONST:
				case SWE_BC_WSE_CONST:
				case SWE_BC_FREE_OUTFLOW:
				case SWE_BC_HZ_T_OUTLET:

					Kokkos::parallel_for("computeOutletScalarFluxes",extbc.ncellsBC, KOKKOS_LAMBDA (int iGlob){
						int ii = extbc.bcells[iGlob];
						real h=state.h(ii);
						if(h>=state.hmin) {
							real hu=state.hu(ii);
							real hv=state.hv(ii);
							real flux = (hu*extbc.normalx + hv*extbc.normaly)/h;
							real lout= fabs(extbc.normalx)+fabs(extbc.normaly); //dom.dx()*(fabs(extbc.normalx)+fabs(extbc.normaly))/dom.dx();
							for(int iphi=0; iphi<state.ade.nScalar; iphi++){
								state.ade.dhphi0(ii,iphi)+=state.ade.hphi(ii,iphi)*flux*lout;
							}
						}
					});

				break;
			}
		}

	}
#endif


}; // end of TimeIntegrator class
