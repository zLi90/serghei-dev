/* -*- mode: c++; c-default-style: "linux" -*- */

#include "stdlib.h"
#include <iostream>
#include <string>
#include "define.h"
#include "Domain.h"
#include "Parallel.h"
#include "Parser.h"
#include "State.h"
#include "BC.h"
#include "Initializer.h"
#include "TimeIntegrator.h"
#include "FileIO.h"
#include "Exchange.h"
#include "SourceSink.h"
#include "DomainIntegrator.h"
#include "Vegetation.h"
#include "ParticleTracking.h"
#include "tools.h"

#if SERGHEI_RE_MODEL
#include "GwDomain.h"
#include "GwFunction.h"
#include "GwInit.h"
#include "GwMPI.h"
#include "GwMatrix.h"
#include "GwState.h"
#include "GwSolver.h"
#include "GwIntegrator.h"
#endif

class SERGHEI{
public:
	Parallel       par;
	Initializer    init;
	FileIO				io;
	State               state;
	Domain              dom;
	#if SERGHEI_RE_MODEL
	GwState gw;
	GwDomain gdom;
	GwInit ginit;
	GwMPI gmpi;
	GwFunction gwf;
	SubsurfaceBoundaries	gbc;
	GwMatrix A;
	GwIntegrator gint;
	GwSolver<Kokkos::DefaultExecutionSpace> gsolver;
	#endif

	SourceSink      ss;
	//SourceSinkData      ss;

  Exchange            exch;
	ExternalBoundaries  ebc;

 private:
	Parser              parser;
	TimeIntegrator      tint;
	surfaceIntegrator   sint;
	boundaryIntegrator  bint;
	#if SERGHEI_TOOLS
		Observations obs;
	#endif
		
  #if SERGHEI_LPT
	ParticleTracker parTrack;
  #endif

	double oldVolume,newVolume, diffVolume;
	double accumDt=0.0;
  
  // Kokkos objects
  Kokkos::Timer timer;
  Kokkos::Timer timer_particles; 
  Kokkos::InitializationSettings kokkosSettings;

public:
	std::string inFolder, outFolder;

////////////// METHODS ///////////////
public:

	int start(int argc, char **argv){
		#if SERGHEI_DEBUG_WORKFLOW
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	  #endif

    
    init.initializeMPI( &argc , &argv , par );

    #ifdef KOKKOS_ENABLE_CUDA
		  kokkosSettings.set_device_id(par.myrank%par.nthreads);
	#else
		  if(par.nthreads!=0) kokkosSettings.set_num_threads(par.nthreads);
	#endif
	#if SERGHEI_DEBUG_KOKKOS_SETUP
		printKokkosInitArguments(par);
		#if KOKKOS_ENABLE_CUDA
			printKokkosCuda(args,par);
		#endif
	#endif

		#if SERGHEI_DEBUG_WORKFLOW
			std::cerr << GGD "Initialising Kokkos - rank " << par.myrank << std::endl;
		#endif

		Kokkos::initialize(kokkosSettings);

		#if SERGHEI_DEBUG_WORKFLOW
			std::cerr << GGD "Program instantiated, creating objects - rank " << par.myrank << std::endl;
		#endif

		// Initialize the model
		if(!init.initialize(state, ss.swss, ebc, dom, par, tint, sint, bint, parser, exch, io, inFolder, outFolder)){
			std::cerr << RERROR "Unable to start the simulation" << "\n";
			return 0;
		};
		#if SERGHEI_LPT
		  if(!parser.readParticles(inFolder,par,&parTrack)){
			std::cerr << RERROR "Unable to start the simulation because of LPT initialization files" << "\n";
		    return 0;
		  }
	      parTrack.initialiseParticles(dom, state);

	      if(io.outFormat==OUT_VTK){
			#if SERGHEI_PARTICLE_NO_OUTPUT
			#else
	                 io.outputIniParticle(dom, parTrack, outFolder);
			#endif
		  }
		  #ifdef SERGHEI_HAS_PNETCDF
		  else{
			io.outputInitParticlesNETCDF(parTrack,dom,par,outFolder);
		  }
		  #endif
		  std::cerr << GOK "LPT module has been initialized! Number of particles: " << parTrack.N_par << "\n" << std::endl;
		#endif


		// Initialize subsurface model if activated
		#if SERGHEI_RE_MODEL
		if (!ginit.initialize_gw(gw, gdom, state, dom, gbc, gmpi, gint, par, io, ss, inFolder, outFolder)) {
			std::cerr << RERROR "Unable to initialize the subsurface domain" << "\n"; return 0;
		};
		A.init(gdom);
		gsolver.init(A, gdom);
		if( par.masterproc){std::cerr << GOK "Subsurface Solver has been initialized! " << std::endl;}
		#endif

		#if SERGHEI_TOOLS
		if(!obs.readInputFiles(inFolder,par)) return 0;
		if(!obs.configure(dom,outFolder)) return 0;	// observations for surface domain
		//obs.printGauges(dom);
		obs.update(state,par,dom);
		if( par.masterproc){
			obs.writeLinesSamplingCoordinates(outFolder);
			obs.writeGauges(dom.etime);
			obs.writeLines(dom.etime);
		}
		#endif

		#if SERGHEI_DEBUG_WORKFLOW
		for (int k = 0; k < ebc.extbc.size(); k ++) {
			std::cout << GGD << GRAY << __FILE__ << ":" << __LINE__ << RESET << "\tExtBC[" << k << "]: " << ebc.extbc[k].bcvals(0) << ", " << ebc.extbc[k].bcvals(1) << ", " << ebc.extbc[k].bcvals(2) << std::endl;
		}
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "Initialisation finished, starting to run main loop" << std::endl;
		for(int i = 0; i < ebc.extbc.size(); i ++) {
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "ncellsBC for segment " << i << ": " << ebc.extbc[i].ncellsBC << "\n";
		}
		#if SERGHEI_DEBUG_BOUNDARY
		bint.integrate(ebc.extbc,dom,1);
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "ncellsBC (integrated) " << bint.ncellsBC << "\n";
		std::cout << GGD<< GRAY << __PRETTY_FUNCTION__ << RESET <<  "outflow discharge (integrated) " << bint.outflowDischarge << "\n";
		std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "outflow accumulated (integrated) " << bint.outflowAccumulated << std::endl;
		#endif
		#endif
		//integrator at the beginning or the simulation
		sint.integrate(state,dom,ss.swss);
		// Write initial time series data
		io.writeTimeSeriesIni(state,dom,par,ss.swss,sint,bint,ebc.extbc,outFolder);
		#if SERGHEI_RE_MODEL
		io.writeSubTimeSeriesIni(gdom, gint, par, outFolder);
		io.outputSubsurface(gw, gdom, par, outFolder);
		#endif
		// capture initialisation time
		dom.timers.swe.init.total = timer.seconds();

		if(par.masterproc) std::cout << GOK << "Initialisation complete. Initialisation time: " << dom.timers.swe.init.total << " [s]" << std::endl;
    return 1;
  }

	int compute(){
		#if SERGHEI_DEBUG_WORKFLOW
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		if (par.masterproc){
			std::cout << std::endl << GOK "SIMULATION STARTS" << std::endl;
			std::cout << BDASH << "Start time: " << dom.startTime << std::endl;
			std::cout << BDASH << "End time: " << dom.endTime << std::endl;
		}
		#if SERGHEI_RE_MODEL
		gdom.dt = gdom.dt_init;
		gdom.dtOld = gdom.dt_init;
		#if SERGHEI_SWE_RE
		// Set initial gw.dt based on dt_ratio
		// But first we need to compute initial sw.dt
		tint.computeDt(state,dom,io);
		// Use dt_ratio to set initial gw.dt
		gdom.dt = gdom.dt_ratio * dom.dt;
		if (gdom.dt > gdom.dt_max) {
			gdom.dt = gdom.dt_max;
		}
		// If gw.dt < sw.dt, synchronize sw.dt to gw.dt
		if (gdom.dt < dom.dt) {
			dom.dt = gdom.dt;
		}
		#else
		dom.dt = gdom.dt;
		#endif
		gdom.dtOld = gdom.dt;
		gdom.cg_iter = 0;
		#else
		tint.computeDt(state,dom,io);
		#endif

		// Main Time Loop
		while (dom.etime < dom.endTime) {
			//previous mass
			oldVolume=sint.surfaceVolumeG;
			bint.integrate(ebc.extbc,dom,1);//has to be called here (previous time step) with mode==1 (boundary flows)
			// run surface model
			#if SERGHEI_SWE_MODEL
			tint.stepForward(state, ss.swss, ebc.extbc, dom, exch, par, io);
			//std::cerr << GSTAR "TIME: " << dom.etime << " SW dt: " << dom.dt <<"\n";
			#else
			ss.swss.ComputeSWSourceSink(state, dom);
			#endif
			// run subsurface model
			#if SERGHEI_RE_MODEL
				#if SERGHEI_SWE_MODEL
				// If both surface and subsurface modules are on
				// Rainfall is first read by the surface module, then copy/aggregate to the subsurface
				if (gdom.isRain) {
					if (gdom.dxRatio == 1) {
						Kokkos::deep_copy(gdom.rainRate, ss.swss.rainRate);
					} else {
						// Aggregate rainfall from surface to subsurface
						Kokkos::parallel_for("aggregate_rainfall", gdom.nCell, KOKKOS_LAMBDA(int idom) {
							int ii, jj, kk;
							gdom.unpackIndices(idom, kk, jj, ii);
							if (kk == 0) {  // Only aggregate for top layer
								real rain_sum = 0.0;
								int n_valid = 0;
								int i_sw_start = ii * gdom.dxRatio;
								int j_sw_start = jj * gdom.dxRatio;
								int i_sw_end = (ii + 1) * gdom.dxRatio;
								int j_sw_end = (jj + 1) * gdom.dxRatio;
								
								for (int j_sw = j_sw_start; j_sw < j_sw_end && j_sw < dom.ny; j_sw++) {
									for (int i_sw = i_sw_start; i_sw < i_sw_end && i_sw < dom.nx; i_sw++) {
										int iGlobSW = dom.getIndex(j_sw * dom.nx + i_sw);
										if (!state.isnodata(iGlobSW)) {
											rain_sum += ss.swss.rainRate(iGlobSW);
											n_valid++;
										}
									}
								}
								if (n_valid > 0) {
									int iGlobGW = gdom.getHaloExtension(ii, jj, kk);
									gdom.rainRate(iGlobGW) = rain_sum / n_valid;
								} else {
									int iGlobGW = gdom.getHaloExtension(ii, jj, kk);
									gdom.rainRate(iGlobGW) = 0.0;
								}
							}
						});
					}
				}
				// Initialize qss_gw to zero for all subsurface cells
				Kokkos::parallel_for("init_qss_gw", gdom.nCell, KOKKOS_LAMBDA(int idom) {
					gw.qss_gw(idom) = 0.0;
				});
				
				// Aggregate surface water depth from surface to subsurface
				if (gdom.dxRatio == 1) {
					// Same resolution: direct copy (gw.hs and state.h have same size and indexing)
					Kokkos::deep_copy(gw.hs, state.h);
					// Also copy to hs_fine (same resolution, so same values)
					Kokkos::parallel_for("copy_hs_fine_dx1", dom.nCell, KOKKOS_LAMBDA(int idom) {
						int iGlobSW_halo = dom.getIndex(idom);
						if (!state.isnodata(iGlobSW_halo)) {
							gw.hs_fine(idom) = state.h(iGlobSW_halo);
						} else {
							gw.hs_fine(idom) = 0.0;
						}
					});
				} else {
					// Aggregate surface depth: average over wet cells only
					// First initialize all cells to zero
					Kokkos::deep_copy(gw.hs, 0.0);
					// Then aggregate for top layer physical cells
					Kokkos::parallel_for("aggregate_hs", gdom.nCell, KOKKOS_LAMBDA(int idom) {
						int ii, jj, kk;
						gdom.unpackIndices(idom, kk, jj, ii);
						if (kk == 0) {  // Only aggregate for top layer
							real h_sum = 0.0;
							int n_wet = 0;
							int i_sw_start = ii * gdom.dxRatio;
							int j_sw_start = jj * gdom.dxRatio;
							int i_sw_end = (ii + 1) * gdom.dxRatio;
							int j_sw_end = (jj + 1) * gdom.dxRatio;
							
							for (int j_sw = j_sw_start; j_sw < j_sw_end && j_sw < dom.ny; j_sw++) {
								for (int i_sw = i_sw_start; i_sw < i_sw_end && i_sw < dom.nx; i_sw++) {
									int iGlobSW = dom.getIndex(j_sw * dom.nx + i_sw);
									real h_sw = state.h(iGlobSW);
									if (h_sw > state.hmin && !state.isnodata(iGlobSW)) {
										h_sum += h_sw;
										n_wet++;
									}
								}
							}
							// Store aggregated value using halo extension index
							int iGlobGW = gdom.getHaloExtension(ii, jj, kk);
							if (n_wet > 0) {
								gw.hs(iGlobGW) = h_sum / n_wet;
							} else {
								gw.hs(iGlobGW) = 0.0;
							}
						}
					});
					
					// Populate fine-resolution surface depth (gw.hs_fine) for fine-resolution flux computation
					Kokkos::parallel_for("populate_hs_fine", dom.nCell, KOKKOS_LAMBDA(int idom) {
						int iGlobSW_halo = dom.getIndex(idom);
						if (!state.isnodata(iGlobSW_halo)) {
							gw.hs_fine(idom) = state.h(iGlobSW_halo);
						} else {
							gw.hs_fine(idom) = 0.0;
						}
					});
				}
				#endif
			// Surface-subsurface coupling (synchronous or asynchronous based on dt_ratio)
			// timer.reset();
			// For synchronous coupling (dt_ratio = 1): run once per surface step
			// For asynchronous coupling (dt_ratio > 1): run until subsurface catches up to or exceeds surface time
			if (gdom.dt_ratio == 1.0) {
				// Synchronous: run exactly once, ensure time sync
				gdom.etime = dom.etime;
				//	PC scheme
				if (gdom.gw_scheme == 1)	{
					gwf.pca_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
				}
				//	Modified Picard scheme
				else {
					gwf.picard_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
				}
			} else {
				// Asynchronous: run while subsurface can take a full step before surface time
				// Subsurface advances by gdom.dt (which is dt_ratio * dom.dt) per step
				// Condition: gdom.etime + gdom.dt <= dom.etime ensures we can take a full step
				while (gdom.etime + gdom.dt <= dom.etime) {
					//	PC scheme
					if (gdom.gw_scheme == 1)	{
						gwf.pca_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
					}
					//	Modified Picard scheme
					else {
						gwf.picard_solve<Kokkos::DefaultExecutionSpace>(gw, gdom, gbc.gwbc, A, gsolver, ss.gwss, gmpi, gint, par);
					}
					// Advance subsurface time after solver completes
					gdom.etime += gdom.dt;
				}
			}
			gdom.cg_iter += A.cg_iter;
			// gdom.timers.gw += timer.seconds();
				// surface-subsurface exchange
				#if SERGHEI_SWE_MODEL
					if (gdom.dxRatio == 1) {
						// Same resolution: direct copy from qss_gw to qss
						Kokkos::parallel_for("copy_qss_dx1", dom.nCell, KOKKOS_LAMBDA(int idom) {
							// When dxRatio=1, subsurface and surface have same grid
							// qss_gw is indexed by subsurface cell index (without halo)
							// state.qss is indexed by surface cell index (without halo)
							// They should match directly
							if (idom >= 0 && idom < gdom.nCell) {
								state.qss(idom) = gw.qss_gw(idom);
							} else {
								state.qss(idom) = 0.0;
							}
						});
					} else {
						// Fine-resolution exchange: copy from gw.qss_fine (computed in GwBC.h) to state.qss
						Kokkos::parallel_for("copy_qss_fine", dom.nCell, KOKKOS_LAMBDA(int idom) {
							// gw.qss_fine is indexed by surface cell physical index (same as state.qss)
							state.qss(idom) = gw.qss_fine(idom);
						});
					}
					tint.computeGwExchange(state , dom);
				#endif
			#endif

			oldVolume+=(bint.inflowDischargeG - bint.outflowDischargeG)*dom.dt; //Boundary fluxes with the new dt
			bint.integrate(ebc.extbc,dom,0);//called here with mode==0 (adjusted volume)
			oldVolume+=bint.adjustedVolumeG; //Some mass changes can occur through the boundaries
			sint.integrate(state,dom,ss.swss); //new mass after the new time step integration
			oldVolume+= (sint.rainFluxG-sint.infFluxG)*dom.dt; //after integrate, we have to sum the rain and inf mass
			newVolume=sint.surfaceVolumeG;

			if(fabs(oldVolume)>TOL12){
				diffVolume=(newVolume-oldVolume)/oldVolume*100.;
			}else{
				diffVolume=0.0;
			}
			
			dom.etime += dom.dt;
			dom.nIter++;
			dom.countIterDt++;
			accumDt+=dom.dt;		

			#if SERGHEI_LPT
			  parTrack.update(dom,state);
			#endif
			
			if (dom.nIter%io.nScreen==0 || fabs(dom.etime - dom.startTime - io.numOut*io.outFreq) <= TOL12) {
				if (par.masterproc) {
					std::cerr << std::fixed;
					std::cerr << GSTAR "TIME: " << dom.etime << " average dt: " << accumDt/dom.countIterDt <<"\n";
					std::cerr.precision(9);
					std::cerr << std::scientific;
					std::cerr << std::fixed;
					std::cerr.precision(12);
					#if SERGHEI_SWE_MODEL
					std::cerr << "     Surface Volume:\t" << newVolume <<"\n";
					std::cerr << "     Surface inflow: " << bint.inflowDischargeG <<"\n";
					std::cerr << "     Surface outflow Volume: " << bint.outflowDischargeG*dom.dt <<"\n";
					#if SERGHEI_RE_MODEL
					std::cerr << "     Exchange Volume: " << gint.Vexch_glob <<"\n";
					#endif
					#endif

					if(fabs(diffVolume)>TOL_MASS_ERROR){
						// std::cerr << YEXC "   Old Volume:\t" << oldVolume <<"\n";
						// std::cerr << YEXC "   New Volume:\t" << newVolume <<"\n";
						// std::cerr << YEXC "   Diff Volume:\t" << newVolume-oldVolume <<"\n";
						// std::cerr << YEXC "   Inflow Volume:\t" << bint.inflowDischargeG*dom.dt <<"\n";
						// std::cerr << YEXC "   Outflow Volume:\t" << bint.outflowDischargeG*dom.dt <<"\n";
						// std::cerr << YEXC "   Adjusted Volume:\t" << bint.adjustedVolumeG <<"\n";
						std::cerr << YEXC "   Rain Volume:\t" << sint.rainFluxG*dom.dt <<"\n";
						// std::cerr << YEXC "   Inf Volume:\t" << sint.infFluxG*dom.dt <<"\n";
						#if SERGHEI_DEBUG_MASS_CONS > 1
                            getchar();
                        #endif
					}
				}
				if(fabs(dom.etime - dom.startTime - io.numOut*io.outFreq) <= TOL12){
					
					#if SERGHEI_LPT==0
					  #if SERGHEI_SWE_MODEL
					  io.output(state, dom, ss.swss, par,outFolder);
					  #endif
					  #if SERGHEI_RE_MODEL
					  io.outputSubsurface(gw, gdom, par,outFolder);
					  #endif
					#else
					  io.output(state, dom, ss.swss, par,outFolder, parTrack);
					#endif

					if(par.masterproc) std::cerr << GIO "File " << io.numOut-1 << " written" << std::endl; //io.numOut already updated
				}
				if(par.masterproc) std::cerr << "-------------------------------------------------\n";
				dom.countIterDt=0;
				accumDt=0.0;


			}



			#if SERGHEI_PARTICLE_TRACKING
			parTrack.update(dom,state);
			#endif

			if (dom.etime >= io.numObs*io.obsFreq) {
				#if SERGHEI_TOOLS
				obs.update(state,par,dom);
				#endif
				io.writeTimeSeries(state,dom,par,sint,bint,ebc.extbc);
				#if SERGHEI_RE_MODEL
				io.writeSubsurfaceTimeSeries(gdom, gint);
				#endif
				if (par.masterproc){
					#if SERGHEI_TOOLS
					obs.write(dom);
					#endif
				}
			}

			// Unify dt for coupled simulations
			#if SERGHEI_SWE_RE
				tint.computeDt(state,dom,io);
				#if SERGHEI_RE_MODEL
				// Use dt_ratio to determine coupling mode
				// Calculate what gw.dt should be based on ratio
				real gdom_dt_target = gdom.dt_ratio * dom.dt;
				// Respect dt_max as upper bound
				if (gdom_dt_target > gdom.dt_max) {
					gdom_dt_target = gdom.dt_max;
				}
				// When gdom_dt_target >= dom.dt: use dt_ratio * dom.dt (asynchronous when dt_ratio > 1)
				// When gdom_dt_target < dom.dt: set dom.dt = gdom_dt_target (synchronize surface to subsurface)
				if (gdom_dt_target >= dom.dt) {
					gdom.dt = gdom_dt_target;
				} else {
					// If calculated gw.dt < sw.dt (shouldn't happen if dt_ratio >= 1, but handle for safety)
					gdom.dt = gdom_dt_target;
					dom.dt = gdom.dt;
				}
				// For synchronous coupling (dt_ratio = 1), ensure both use the same dt
				// This handles cases where solver may have adjusted gdom.dt
				if (gdom.dt_ratio == 1.0) {
					// Both should use the same dt - use the minimum to ensure stability
					if (dom.dt < gdom.dt) {
						gdom.dt = dom.dt;
					} else {
						dom.dt = gdom.dt;
					}
				}
				#endif
			#elif SERGHEI_RE_MODEL
				dom.dt = gdom.dt;
				tint.dtMatchOutput(dom,io);
				gdom.dt = dom.dt;
			#elif !SERGHEI_SWE_MODEL
        std::cout << RERROR << "Impossible configuration without SWE nor GW model" << std::endl;
				return 0;
			#endif
		} 		// end of time loop
		return 1;
	}

	int finalise(){
		#if SERGHEI_DEBUG_WORKFLOW
			std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
		#endif
		#if SERGHEI_LPT
		  io.writeParticleFile(dom,parTrack,par,outFolder);
		#endif
		dom.timers.total = timer.seconds();
		#if SERGHEI_RE_MODEL
		dom.timers.re = gdom.timers.re;
		dom.cg_iter = gdom.cg_iter;
		#endif
		if(par.masterproc){
			std::cerr << GOK "SIMULATION FINISHED\n";
			std::cerr << GOK "Time elapsed: " << dom.timers.total << std::endl;
		}
		
		dom.timers.closure();
		dom.timers.gather(par);
		dom.relative = dom.timers;
		dom.relative.computeRelative(dom.timers);
		dom.relative.gather(par);
		io.writeLogFile(dom, par, outFolder);
		io.closeOutputStreams();
		#if SERGHEI_TOOLS
			if(par.masterproc) obs.closeOutputStreams();
		#endif
		return 1;
	}


};
