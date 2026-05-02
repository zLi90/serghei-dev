#ifndef _SOURCESINK_H_
#define _SOURCESINK_H_

#include "define.h"
#include "SArray.h"
#include "State.h"
#include "GwDomain.h"
#include "GwState.h"
#include "atmForcing.h"

#define INF_NONE 0
#define INF_CONSTANT 1
#define INF_HORTON 2
#define INF_GREENAMPT 3

#ifndef SERGHEI_RAINFALL_POLYGONS
  #define SERGHEI_RAINFALL_POLYGONS 0
#endif


#if SERGHEI_RAINFALL_POLYGONS
KOKKOS_INLINE_FUNCTION int findTimeBlockByPolygons (const realArr raintime, const int idx0, const int idx1, real const &t){
  int ti=idx0; //current time index

  if(t >= raintime(idx1)){
    ti = idx1;
  }
  else{
    for(int ii=idx0; ii<idx1; ii++){

      if(t>=raintime(ii) && t<raintime(ii+1)) ti=ii;

    }
  }

  return(ti);
};

KOKKOS_INLINE_FUNCTION real interpolatePiecewiseByPolygons (const realArr rainvalue, int idx){
  return (rainvalue (idx));
};
#endif



KOKKOS_INLINE_FUNCTION void findTimeBlock (TimeSeries &ts, real const &t){
  if(t >= ts.time (ts.np - 1)){
    ts.timeIndex = ts.np - 1;
  }
  else{
	   if (t >= ts.time (ts.timeIndex + 1)) ts.timeIndex++;
  }
};

KOKKOS_INLINE_FUNCTION real interpolatePiecewise (TimeSeries const &ts, real const &t, int const spaceIndex ){
  return (ts.value (ts.np * spaceIndex + ts.timeIndex));
};

KOKKOS_INLINE_FUNCTION real interpolateLinear(TimeSeries &ts, real const &t){
  #if SERGHEI_DEBUG_WORKFLOW > 1
	std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
  #endif
  int ii,jj;
  findTimeBlock(ts,t);
  ii = ts.timeIndex;
  jj=ii+1;
  if(ii == ts.np - 1) jj = ii;
  real v = ts.value(ii) + (ts.value(jj) - ts.value(ii))/(ts.time(jj)-ts.time(ii))*(t-ts.time(ii));
  return(v);
};

KOKKOS_INLINE_FUNCTION real interpolateValues(TimeSeries &ts, real const &t, int icol){
  int ii,jj;
  findTimeBlock(ts,t);
  ii = ts.timeIndex;
  jj=ii+1;
  if(ii == ts.np - 1) jj = ii;
  real v = ts.values(ii,icol) + (ts.values(jj,icol) - ts.values(ii,icol))/(ts.time(jj)-ts.time(ii))*(t-ts.time(ii));
  return(v);
};





class ConstantInfiltration{

private:

  real _constCap;

public:

  ConstantInfiltration(real constCap) :  _constCap(constCap) {} 	 // constructor
  real operator()()
  {
    return _constCap;
  }

};


class InfiltrationModel{
public:
  static constexpr int spatialNone = 0;
  static constexpr int spatialRaster = 1;
  static constexpr int spatialSoilMap = 2;
  static constexpr int spatialLanduse = 3;
  static constexpr int spatialNetCDF = 4;
  int spatial = spatialRaster;
  MapClass soilmap;

private:
/*
  KOKKOS_INLINE_FUNCTION real horton(const int ii,const real dt) const
  {
    real t = infTime(ii) + dt;
    infTime(ii) = t;
    real infCap = fc + (f0-fc)*exp(-k * t);
    return(infCap);
  }
*/
  // TODO need to program GreenAmpt model
KOKKOS_INLINE_FUNCTION  real greenAmpt (const int ii,const real ) const{
    real infCap = 0.;
    return(infCap);
  }

  realArr infTime;

public:
    int model = -999;
    color nLabels = 0;
    realArr constCap ;
    // Horton
    realArr k;
    realArr fc;
    realArr f0;
    // Green-Ampt
    real ks = -999;
    real psi = -999;
    real dtheta = -999;

    real infDry = 1E-8;  // [L] threshold to consider dry for infiltration purposes

    // This is not a state variable which is why it is here and not in class State.
    // It is necessary for output and because it is a variable in the GreenAmpt model
    realArr infVol; // accumulated infiltration volume

    realArr rate;		    // infiltration rate, m/s
    realArr capacity;   // prescribed infiltration capacity, m/s
    colorArr infLabel;  // labels for heterogeneous infiltration

    // this is a function pointer which allows to redirect to the specific infiltration capacity function. the goal is to avoid evaluating which model to use every time step
    // real (InfiltrationModel::*capacity)(const int ii, const real dt) const;


    // Define infiltration capacity models
    /*
    real constant(const int ii, const real t) const{
    	return(constCap);
    }
    */


    void allocate(const Domain &dom){
        if(model){
          rate = realArr("rate",dom.nCellMem);
          infVol = realArr("infVol",dom.nCellMem);
          if(model == INF_HORTON) infTime = realArr("infTime",dom.nCellMem);
          if(spatial == spatialRaster || spatial == spatialNetCDF) capacity = realArr("capacity",dom.nCellMem);
        }
    }



  int checkModel(Parallel &par){
    int error = 0;

	  #if SERGHEI_DEBUG_INFILTRATION
	    std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET <<  "model: " << model << std::endl;
  	#endif
    switch(model){
      case INF_NONE:
		    if(par.masterproc) std::cerr << BDASH << "No infiltration capacity" << std::endl;
        break;
      case INF_CONSTANT:
      	// capacity = &InfiltrationModel::constant;
        if(par.masterproc) std::cerr << BDASH << "Constant infiltration capacity" << std::endl;
        for(int id=1; id<nLabels; id++){
        	if(constCap(id) < 0){
		  		  if(par.masterproc) std::cerr << RERROR << "Infiltration rate not found for constant infiltration model" << std::endl;
		        error++;
            }
        }
        break;
      case INF_HORTON:
      	// capacity = &InfiltrationModel::horton;
		if(par.masterproc) std::cerr << BDASH << "Horton infiltration capacity" << std::endl;
        for(int ii=0; ii<nLabels; ii++){
        	//std::cerr << ii << "\t" << k(ii) << "\t" << fc(ii) << "\t" << f0(ii) << std::endl;
          if(k(ii) < 0){
				    if(par.masterproc) std::cerr << RERROR << "Shape factor not found for Horton infiltration model" << std::endl;
            error++;
          }
          if(f0(ii) < 0){
				    if(par.masterproc) std::cerr << RERROR << "Initial infiltration capacity not found for Horton infiltration model" << std::endl;
            error++;
          }
          if(fc(ii) < 0){
        	  std::cerr << RERROR << "Asymptotic infiltration capacity not found for Horton infiltration model" << std::endl;
            error++;
          }
        }
        break;
      case INF_GREENAMPT:
         //			capacity = &InfiltrationModel::greenAmpt;
		    if(par.masterproc){
          std::cerr << BDASH << "Green-Ampt infiltration capacity" << std::endl;
          std::cerr << RERROR << "Not enabled yet" << std::endl;
		    }
        error++;
        if(ks < 0){
          std::cerr << RERROR << "Saturated hydraulic conductivity not found for Green-Ampt infiltration model" << std::endl;
          error++;
        }
        if(psi < 0){
          std::cerr << RERROR << "Average suction head not found for Green-Ampt infiltration model" << std::endl;
          error++;
        }
        if(dtheta < 0){
          std::cerr << RERROR << "Water content difference not found for Green-Ampt infiltration model" << std::endl;
         	error++;
        }
        break;
      default:
        std::cerr << RERROR << "Error processing data in infiltration.input using infiltration model " << model << "." << std::endl;
        error++;
        break;
      }
	  if(error > 0) return 0;
      return 1;
    }



    inline void ComputeInfiltrationCapacity(const Domain &dom){
      #if SERGHEI_DEBUG_WORKFLOW
        std::cerr << GGD << __PRETTY_FUNCTION__ << std::endl;
      #endif
        if(model){
                /*
                realArr &inf_p = rate;
                colorArr infLabel = this->infLabel;
                realArr constCap = this->constCap;
                */

                switch(model){
                    case INF_CONSTANT:
                      if(spatial == spatialNetCDF || spatial == spatialRaster){
                        Kokkos::deep_copy(rate,capacity);
                      }else{
                        Kokkos::parallel_for("inf_constant", dom.nCell, KOKKOS_CLASS_LAMBDA (int iGlob){
                            int ii = dom.getIndex(iGlob);
                            int id = infLabel(ii);
                            rate(ii) = constCap(id);
                        });
                      }
                      break;
                    case INF_HORTON:
                    /*
                        realArr fc = this->fc;
                        realArr f0 = this->f0;
                        realArr k = this->k;
                        realArr &infTime_p = infTime;
                    */
                        Kokkos::parallel_for("inf_horton", dom.nCell, KOKKOS_CLASS_LAMBDA (int iGlob){
                          int ii = dom.getIndex(iGlob);
                          int id = infLabel(ii);
                          real t = infTime(ii) + dom.dt;
                          infTime(ii) = t;
                          rate(ii) = fc(id) + (f0(id)-fc(id))*exp(-k(id) * t);
                        });
                        break;

                    }
            }
    }
};


class SourceSinkData{

public:
  #if SERGHEI_NETCDF_FORCING
  AtmosphericForcing aforcing;
  realArr rainAccum;
  #endif
    TimeSeries rain, evap, wind;
    InfiltrationModel inf;
    realArr rainRate, evapRate, windspd, winddir;

      intArr rainPol;

#if SERGHEI_RAINFALL_POLYGONS
  int nrainpol=0;
  int nrainpoints=0;
  std::vector<TimeSeries> rainSeries; //this can not be directly read from gpu kernels
                                      //we store rain data in raintimedata

  //store rain series in device memory
  intArr startIdxRain; //x nrainpol
  intArr npRain; //x nrainpol
  intArr timeIdx; //x nrainpol
  realArr raintime; //x nrainpoints
  realArr rainvalue; //x nrainpoints
#endif

  void initializeSW (Domain const &dom, State const &state, Parallel const &par){
      if(dom.isRain){
        #if SERGHEI_NETCDF_FORCING
          rainAccum = realArr("rainAccum",dom.nCellMem);
        #else
          rainRate  = realArr ("rainRate", dom.nCellMem);
          Kokkos::deep_copy(rainRate,0.0);
          #if SERGHEI_RAINFALL_POLYGONS
            rainPol  = intArr ("rainPol", dom.nCellMem);
            Kokkos::deep_copy(rainPol,-1);
            //assign rainPol index cell-by-cell
            for(int pp=0; pp<nrainpol; pp++){
              TimeSeries &rain = rainSeries[pp];
              find_raincells(dom,state,par,rain,pp);
            }
            //initialize rain series storage
            startIdxRain = intArr ("startIdxRain", nrainpol);
            npRain = intArr ("npRain", nrainpol);
            timeIdx = intArr ("timeIdx", nrainpol);
            Kokkos::parallel_for(nrainpol, KOKKOS_CLASS_LAMBDA(int iGlob){
              startIdxRain(iGlob) = 0;
              npRain(iGlob) = 0;
              timeIdx(iGlob) = 0;
            });

            raintime = realArr("raintime",nrainpoints);
            rainvalue = realArr("rainvalue",nrainpoints);
            Kokkos::parallel_for(nrainpoints, KOKKOS_CLASS_LAMBDA(int iGlob){
              raintime(iGlob) = 0.0;
              rainvalue(iGlob) = 0.0;
            });

            //store rain series
            int startposition=0;
            for(int pp=0; pp<nrainpol; pp++){
              TimeSeries &rain = rainSeries[pp];
              store_raintime(dom,rain,startposition,pp);
            }
          #endif
        #endif
      }
      if(dom.isEvap){evapRate  = realArr ("evapRate", dom.nCellMem);}
        if(dom.isWind){
          windspd = realArr("windspd", dom.nCellMem);
          winddir = realArr("winddir", dom.nCellMem);
        }
      if (inf.model)  {inf.allocate(dom);}
    }

  #if SERGHEI_RAINFALL_POLYGONS
	inline int find_raincells(Domain const &dom, State const &state, Parallel const &par, TimeSeries &rain, int pp){
    #if SERGHEI_DEBUG_WORKFLOW
    std::cerr << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif

    int nPoly=rain.nver;
    realArr &xPoly=rain.xPoly;
    realArr &yPoly=rain.yPoly;

    int count=0;
		for(int iGlob=0; iGlob<dom.nCell; iGlob++){
			int i,j;
			dom.unpackIndices(iGlob,j,i);
			int ii = dom.getHaloExtension(i,j);

			if(!state.isnodata(ii)){
        real xCoord = dom.xll + ( par.i_beg + i + 0.5) * dom.dxConst;
        real yCoord = dom.yll + dom.ny_glob*dom.dxConst - ( par.j_beg + j + 0.5) * dom.dxConst;
        if(geometry::isInsidePoly(nPoly,xPoly, yPoly, xCoord, yCoord)){
          rainPol(ii)=pp;
          count++;
        }
			}
		}
    #if SERGHEI_DEBUG_RAINFALL
      std::cerr << GGD "Rain polygon " << pp << " contains " << count << " cells" << std::endl;
    #endif
    printf("Rain polygon %d contains %d cells\n",pp,count);


    return(1);
  }

	inline int store_raintime(Domain const &dom, TimeSeries const &rain, int &startposition, int pp){
    #if SERGHEI_DEBUG_WORKFLOW
    std::cerr << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif

    //store start index for rain series
    startIdxRain(pp)=startposition;
    npRain(pp)=rain.np;
    timeIdx(pp)=startposition;

    //int count=0;
    int idx;
		for(int ii=0; ii<rain.np; ii++){
      idx=startposition+ii;
			raintime(idx) = rain.time(ii);
      rainvalue(idx) = rain.value(ii);
      //count++;
		}

    //next start position
    startposition += rain.np;

    return(1);
  }
  #endif

  #if SERGHEI_RAINFALL_POLYGONS
  inline void ComputeRainByPolygons (const State &state, const Domain &dom){
    #if SERGHEI_DEBUG_WORKFLOW
    std::cerr << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif

    if(dom.isRain){

      //select time index for each rain series
      Kokkos::parallel_for(nrainpol, KOKKOS_CLASS_LAMBDA(int pp){

        int idx0=timeIdx(pp); //current time index
        int idx1=startIdxRain(pp)+npRain(pp)-1;

        timeIdx(pp)=findTimeBlockByPolygons(raintime,idx0,idx1,dom.etime);

      });


      Kokkos::parallel_for("rain_interpolation",dom.nCell, KOKKOS_CLASS_LAMBDA (int iGlob){
        int ii = dom.getIndex(iGlob);
        real Irain = 0.0;

        if(!state.isnodata(ii)){

          int pp=rainPol(ii);
          if(pp>=0)
            Irain = interpolatePiecewiseByPolygons(rainvalue, timeIdx(pp));

          #if SERGHEI_DEBUG_RAINFALL
          std::cerr << GGD "raintime " << raintime(timeIdx(pp)) << std::endl;;
          std::cerr << GGD "rainvalue " << rainvalue(timeIdx(pp)) << std::endl;;
          #endif
        }

        rainRate(ii) = Irain;
      });

    }
	  #if SERGHEI_DEBUG_RAINFALL
   		std::cerr << GGD "-----------" << std::endl;
	  #endif

  }
  #endif









  inline void ComputeRain (const Domain &dom){
    if(dom.isRain){

	// ----------------------------------------------------------------------
	// get global values to map to the correct rain subdomain
	// ----------------------------------------------------------------------
	//int nx = dom.nx_glob; // computational cell number in x direction
  //int ny = dom.ny_glob; // computational cell number in y direction

	int rainx = rain.nx;  // rain subdomain number in x direction
	int rainy = rain.ny;  // rain subdomain number in y direction

	int intervalx = dom.nx / rainx; // approximate number of cells in
				    // a subdomain in x direction
	int intervaly = dom.ny / rainy; // approximate number of cells in
				    // a subdomain in y direction
	// ----------------------------------------------------------------------


	realArr &rr_p = rainRate;

  findTimeBlock(rain,dom.etime);
  TimeSeries rrain = rain;

   Kokkos::parallel_for("rain_interpolation",dom.nCell, KOKKOS_LAMBDA (int iGlob){
	    int ix;
	    int iy;

      dom.unpackIndices (iGlob, iy, ix);
      int ii = dom.getHaloExtension(ix,iy);

	    int _x = ix / intervalx;
	    int _y = iy / intervaly;

	    int rain_glob = _x + _y * rainx;

	    real rainValue = interpolatePiecewise(rrain, dom.etime, rain_glob);

	    rr_p(ii) = rainValue;

	    #if SERGHEI_DEBUG_RAINFALL
        Kokkos::printf("%s[DEBUG] %s%s _x: %lf _y: %lf ix: %d iy: %d, ~> rainfall: %lf \train_glob: $lf\n",GRAY,__PRETTY_FUNCTION__,RESET,_x,_y,ix,iy,rr_p(iGlob),rain_glob);
      #endif
    });

	/** basically the same as above but parallel for-ized. this
	    works with MPI. we may think about a switch that uses this
	    portion of code when compiled for CPU.

	Kokkos::parallel_for (dom.nCell, KOKKOS_LAMBDA (int iGlob)
			      {

				int ix; // global x coordinate
				int iy; // global y coordinate
				unpackIndices (iGlob, ny, nx, iy, ix);

				int _x = ix / intervalx;
				int _y = iy / intervaly;

				int rain_glob = _x + _y * rainx;

				real rainValue = rain.interpolate (dom.etime, rain_glob);

				int ii = getIndex (iGlob, dom);
				rr_p (ii) = rainValue;

			      });
	**/

      }
	  #if SERGHEI_DEBUG_RAINFALL
   		std::cerr << GGD "-----------" << std::endl;;
	  #endif
  }




  inline void ComputeEvap (const Domain &dom){
      if(dom.isEvap){
          realArr &rr_e = evapRate;
          findTimeBlock(evap,dom.etime);
          TimeSeries revap = evap;
          Kokkos::parallel_for("evap_interpolation",dom.nCell, KOKKOS_LAMBDA (int iGlob){
              int ix;
              int iy;
              dom.unpackIndices (iGlob, iy, ix);
              int ii = dom.getHaloExtension(ix,iy);
              int evap_glob = 0;
              real evapValue = interpolatePiecewise(revap, dom.etime, evap_glob);
              rr_e(ii) = evapValue;
          });
      }
  }


    inline void ComputeWind(const Domain &dom)    {
        if (dom.isWind)   {
            realArr &rr_w = windspd;
            realArr &rr_d = winddir;
            findTimeBlock(wind, dom.etime);
            TimeSeries rwind = wind;
            Kokkos::parallel_for("wind_interpolation", dom.nCell, KOKKOS_LAMBDA (int idom){
                int ix, iy;
                dom.unpackIndices (idom, iy, ix);
                int ii = dom.getHaloExtension(ix,iy);
                real spdValue = interpolatePiecewise(rwind, dom.etime, 0);
                rr_w(ii) = spdValue;
                real dirValue = interpolatePiecewise(rwind, dom.etime, 1);
                rr_d(ii) = dirValue;
            });
        }
    }


  inline void ComputeSWSourceSink(const State &state, const Domain &dom){
    Kokkos::fence();
    Kokkos::Timer timer;
    #if SERGHEI_DEBUG_WORKFLOW
    std::cerr << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    #if SERGHEI_RAINFALL_POLYGONS
      ComputeRainByPolygons(state,dom);
    #elif SERGHEI_NETCDF_FORCING
      aforcing.computeRain(dom,state,rainRate);
    #else
      ComputeRain(dom);
    #endif
      ComputeEvap(dom);
      ComputeWind(dom);
    inf.ComputeInfiltrationCapacity(dom);
    //no rate correction is necessary here beacuse the rate correction is done in ComputeNewState, according to the new water depth
   // timerRainInf += timer.seconds();
   Kokkos::fence();
   dom.timers.swe.ss.raininf += timer.seconds();
 }

};



// Subsurface source/sinks
#if SERGHEI_RE_MODEL
class GwSS{

public:
	// subsurface ss directions
	#define XPLUS 1
	#define XMINUS 2
	#define YPLUS 3
	#define YMINUS 4
	#define ZPLUS 5
	#define ZMINUS 6
	// The type of source/sink
    //  0 : Evapotranspiration (from PM equation)
    //  1 : Flux (constant or time series)
    //  2 : Head for drainage (constant or time series)

    int sstype;
    int direction; // direction of source/sink, only needed for drainage ss
    int ndepth;     // number of cells in the vertical direction of the polygon
	int ncellsIT = 0;	// number of internal source/sink cells
	intArr icells; //array of indexes of boundary cells
    realArr ssvals, ssdata;
    real Qinflow, Qoutflow, Cpipe;
    TimeSeries ts;
    TimeSeries evap, tran;

	// water stress and root distribution function
    real lai;
	real h1, h2, h3, h4;
	real px, py, pz, xs, ys, zs, xm, ym, zm;
	realArr coef_wat, coef_root;

	MPI_Comm comm;	// communicator for ranks associated to the BC

    void allocateGW (GwDomain const &gdom)  {
        ssdata = realArr ("ssdata", gdom.nCellMem);
        for (int idx = 0; idx < gdom.nCellMem; idx++)   {ssdata(idx) = 0.0;}
		if (sstype == 0)	{
			coef_wat = realArr("wat", ncellsIT);
			coef_root = realArr("root", ncellsIT);
		}
    }

    // find internal cells for applying source/sink conditions
	inline int find_icells(GwState &gw, std::string &id, GwDomain &gdom, Parallel &par, int nPoly, realArr &xPoly, realArr &yPoly, realArr &zPoly){
		int foundInSubdom; // to keep track of which subdomains are associated to this boundary
		std::vector<int> tmpicells; //array of indexes of internal cells
		std::vector<int> tmpgcells; //array of indexes of ghost cells
		std::vector<int> subdomains;	// keeps track of which subdomains are associated to the BC
		// Loop over the entire domain to find internal source/sink cells
        int kmax = 0, kmin = gdom.nz, idx = 0;
        real zm_min = -1e5, zm_max = 0.0;
		for (int kk = 0; kk < gdom.nz; kk++) {
			for (int jj = 0; jj < gdom.ny; jj++) {
				for (int ii = 0; ii < gdom.nx; ii++) {
					int iGlob = (gdom.hc+kk)*gdom.nxhc*gdom.nyhc + (gdom.hc+jj)*gdom.nxhc + ii + gdom.hc;
					foundInSubdom = -1;
		            real xCoord = gdom.xll + ( par.i_beg + ii + 0.5) * gdom.dx;
		            real yCoord = gdom.yll + gdom.ny_glob*gdom.dx - ( par.j_beg + jj + 0.5) * gdom.dx;
					real zCoord = gdom.z(iGlob);
		            if (geometry::isInsidePoly3D(nPoly, xPoly, yPoly, zPoly, xCoord, yCoord, zCoord)){
						tmpicells.push_back(iGlob);
                        if (kk > kmax)  {kmax = kk;}
                        if (kk < kmin)  {kmin = kk;}
                        if (zCoord < zm_max)    {zm_max = zCoord;}
                        if (zCoord > zm_min)    {zm_min = zCoord;}
		            }
				}
			}
		}
        // For now, only consider 1D(z) root distribution, 20240818
        xm = 0.0;   ym = 0.0;   zm = 0.0;
        if (kmax > kmin)    {ndepth = kmax - kmin;  zm = zm_min - zm_max;}
        else {ndepth = 1;}

		ncellsIT=int(tmpicells.size());
		if(ncellsIT > 0) foundInSubdom = par.myrank; // if at least one cell in this subdomain (rank) is in the BC, tag as found

		int ncells_all;
        MPI_Allreduce(&ncellsIT, &ncells_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		int *subdoms;
		subdoms = (int*) malloc(par.nranks * sizeof(int));
		MPI_Allgather(&foundInSubdom,1,MPI_INT,subdoms,1,MPI_INT,MPI_COMM_WORLD);
		for (int i=0; i<par.nranks; i++){
			if (subdoms[i] >= 0)	{subdomains.push_back(subdoms[i]);}
		}
		MPI_Group group, subgroup;
		MPI_Comm_group(MPI_COMM_WORLD,&group);
		MPI_Group_incl(group,subdomains.size(),subdomains.data(),&subgroup);
		MPI_Comm_create(MPI_COMM_WORLD,subgroup,&comm);
		//we need the total internal cells detected by all subdomain to launch an error otherwise
		if(ncells_all>0){
			icells=intArr("icells", ncellsIT);
			#ifdef KOKKOS_ENABLE_CUDA
				cudaMemcpyAsync( icells.data() , tmpicells.data() , ncellsIT*sizeof(int) , cudaMemcpyHostToDevice );
				cudaDeviceSynchronize();
			#else
				std::memcpy(icells.data(), tmpicells.data(), ncellsIT*sizeof(int));
			#endif
		}
		else{
			if(par.masterproc){
				std::cerr << RERROR << "No internal cells found for subsurface boundary with id '" << id << "'" << std::endl;
			}
			return 0;
		}
		return 1;
	}

	// get coefficients for root water uptake declining and root distribution
	void rootCoef(const GwState &gw, const GwDomain &gdom)	{
		Kokkos::parallel_for("root", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
			real c_wat = 1.0, c_root = 1.0, expo, x, y, z;
			int iGlob = icells[idx];
			real h = gw.h(iGlob,1);
			// get coef_wat
			if (h <= h1 && h > h2)	{c_wat = (h - h1) / (h2 - h1);}
			else if (h <= h3 && h > h4)	{c_wat = (h - h4) / (h3 - h4);}
			else if (h > h1 || h <= h4)	{c_wat = 0.0;}
			// get coordinates x, y, z
			x = gdom.x(iGlob);	y = gdom.y(iGlob);	z = gdom.depth(iGlob);
			// get coef_root
			//expo = px/(xm*myfabs(xs-x)) + py/(ym*myfabs(ys-y)) + pz/(zm*myfabs(zs-z));
			//c_root = (1.0-x/xm)*(1.0-y/ym)*(1.0-z/zm)*exp(-expo);
			expo = pz/(zm*myfabs(zs-z));
			c_root = (1.0-z/zm)*exp(-expo);
            if (c_root < 0.0)   {c_root = 0.0;}
			coef_wat(idx) = c_wat;
			coef_root(idx) = c_root;
		});
	}

    inline void applyMatSS(GwState &gw, GwDomain &gdom) {
        if (ncellsIT > 0)   {
        	// ET (Penman-Monteith)
        	if (sstype == 0)	{
        		real qt = -interpolateLinear(tran, gdom.etime);
                real qe = -interpolateLinear(evap, gdom.etime);
				// get root function coefficients
				rootCoef(gw, gdom);
                Kokkos::parallel_for("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        gw.coef(idom,7) += gdom.dt * qt * coef_wat(idx) * coef_root(idx) / gdom.dz(iGlob);
                        if (kk == 1)    {
                            gw.coef(idom,7) += gdom.dt * qe / gdom.dz(iGlob);
                        }
                });
        	}
            // Flux Source/Sink
            else if (sstype == 1)    {
                real qbc = interpolateLinear(ts, gdom.etime);
                Kokkos::parallel_for("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        gw.coef(idom,7) += gdom.dt * qbc;
                });
            }
            // Internal Drainage with Fixed Head
            // Note: For now, this only supports draining in the saturated zone
            else if (sstype == 2)   {
                // real hbc = interpolateLinear(ts, gdom.etime);
                real hbc = ssvals[0];
                if (direction == XPLUS || direction == XMINUS)	{
                	Kokkos::parallel_for("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        real flux = 0.0;
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                  		// get index of cells next to the source/sink
                        int jp = idom+1, jm = idom-1, kp = idom+gdom.nx*gdom.ny, km = idom-gdom.nx*gdom.ny;
                        // treat the source/sink as a pressure boundary
                        gw.coef(jp,7) -= Cpipe * gw.coef(idom,3) * hbc;
                        gw.coef(jm,7) -= Cpipe * gw.coef(idom,4) * hbc;
                        gw.coef(kp,7) -= Cpipe * gw.coef(idom,5) * hbc;
                        gw.coef(km,7) -= Cpipe * gw.coef(idom,6) * hbc;
                        // exclude the source/sink cell from linear system
                        gw.coef(idom,1) = 0.0; gw.coef(idom,3) = 0.0; gw.coef(idom,5) = 0.0;
                        gw.coef(idom,2) = 0.0; gw.coef(idom,4) = 0.0; gw.coef(idom,6) = 0.0;
                        gw.coef(jp,4) = 0.0;	gw.coef(jm,3) = 0.0;
                        gw.coef(kp,6) = 0.0;	gw.coef(km,5) = 0.0;
                	});
                }
                else if (direction == YPLUS || direction == YMINUS)	{
                	Kokkos::parallel_for("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        int ip = idom+1, im = idom-1, kp = idom+gdom.nx*gdom.ny, km = idom-gdom.nx*gdom.ny;
                        // Seepage as a fixed H condition
                        gw.coef(ip,7) -= Cpipe * gw.coef(idom,1) * hbc;
                        gw.coef(im,7) -= Cpipe * gw.coef(idom,2) * hbc;
                        gw.coef(kp,7) -= Cpipe * gw.coef(idom,5) * hbc;
                        gw.coef(km,7) -= Cpipe * gw.coef(idom,6) * hbc;

                        gw.coef(idom,1) = 0.0; gw.coef(idom,3) = 0.0; gw.coef(idom,5) = 0.0;
                        gw.coef(idom,2) = 0.0; gw.coef(idom,4) = 0.0; gw.coef(idom,6) = 0.0;
                        gw.coef(ip,2) = 0.0;	gw.coef(im,1) = 0.0;
                        gw.coef(kp,6) = 0.0;	gw.coef(km,5) = 0.0;
                	});
                }
                else 	{
                	Kokkos::parallel_for("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        real flux = 0.0;
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        // get index of cells next to the source/sink
                        int ip = idom+1, im = idom-1, jp = idom+1, jm = idom-1;
                        // treat the source/sink as a pressure boundary
                        gw.coef(jp,7) -= Cpipe * gw.coef(idom,3) * hbc;
                        gw.coef(jm,7) -= Cpipe * gw.coef(idom,4) * hbc;
                        gw.coef(ip,7) -= Cpipe * gw.coef(idom,1) * hbc;
                        gw.coef(im,7) -= Cpipe * gw.coef(idom,2) * hbc;
                        // exclude the source/sink cell from linear system
                        gw.coef(idom,1) = 0.0; gw.coef(idom,3) = 0.0; gw.coef(idom,5) = 0.0;
                        gw.coef(idom,2) = 0.0; gw.coef(idom,4) = 0.0; gw.coef(idom,6) = 0.0;
                        gw.coef(jp,4) = 0.0;	gw.coef(jm,3) = 0.0;
                        gw.coef(ip,2) = 0.0;	gw.coef(im,1) = 0.0;
                	});
                }

            }
        }
    }

    inline void applyWCSS(GwState &gw, GwDomain &gdom) {
    	if (ncellsIT > 0)   {
			Qoutflow = 0.0;
			Qinflow = 0.0;
    		// ET (Penman-Monteith)
        	if (sstype == 0)	{
        		real qt = -interpolateLinear(tran, gdom.etime);
                real qe = -interpolateLinear(evap, gdom.etime);
                Kokkos::parallel_reduce("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx, real &tmp){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        gw.wc(iGlob,1) += gdom.dt * qt * coef_wat(idx) * coef_root(idx) / gdom.dz(iGlob);
                        tmp += qt * coef_wat(idx) * coef_root(idx) * gdom.dx * gdom.dy;
                        if (kk == 1)    {
                            gw.wc(iGlob,1) += gdom.dt * qe / gdom.dz(iGlob);
                            tmp += qe * gdom.dx * gdom.dy;
                        }
				} , Kokkos::Sum<real>(Qoutflow) );
        	}
            // Flux Source/Sink
            else if (sstype == 1)    {
                real qbc = interpolateLinear(ts, gdom.etime);
                Kokkos::parallel_reduce("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx, real &tmp){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        gw.wc(iGlob,1) += gdom.dt * qbc;
                        if (qbc > 0)	{tmp += qbc * gdom.dt;}
                        else {tmp -= qbc * gdom.dt;}
                }, Kokkos::Sum<real>(Qinflow) );
            }
            else if (sstype == 2)   {
                // real hbc = interpolateLinear(ts, gdom.etime);
                real hbc = ssvals[0], flux;
                if (direction == XPLUS || direction == XMINUS)	{
                	Kokkos::parallel_reduce("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx, real &tmp){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        // Calculate the cumulative outflow
                        int jp = iGlob+gdom.nxhc, jm = iGlob-gdom.nxhc, kp = iGlob+gdom.nxhc*gdom.nyhc, km = iGlob-gdom.nxhc*gdom.nyhc;
                        if (hbc < gw.h(jp,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dz(iGlob) * gw.k(iGlob,1) * (hbc - gw.h(jp,1)) / gdom.dy;
                        }
                        if (hbc < gw.h(jm,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dz(iGlob) * gw.k(jm,1) * (hbc - gw.h(jm,1)) / gdom.dy;
                        }
                        if (hbc < gw.h(kp,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dy * gw.k(iGlob,2) * (hbc - gw.h(kp,1)) / gdom.dz(kp);
                        }
                        if (hbc < gw.h(km,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dy * gw.k(km,2) * (hbc - gw.h(km,1)) / gdom.dz(km);
                        }
                	}, Kokkos::Sum<real>(flux));
                }
                else if (direction == YPLUS || direction == YMINUS)	{
                	Kokkos::parallel_reduce("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx, real &tmp){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        // Only consider drainage under fully saturated condition
                        int ip = iGlob+1, im = iGlob-1, kp = iGlob+gdom.nxhc*gdom.nyhc, km = iGlob-gdom.nxhc*gdom.nyhc;
                        if (hbc < gw.h(ip,1))	{
                        	tmp += Cpipe * gdom.dy * gdom.dz(iGlob) * gw.k(iGlob,0) * (hbc - gw.h(ip,1)) / gdom.dx;
                        }
                        if (hbc < gw.h(im,1))	{
                        	tmp += Cpipe * gdom.dy * gdom.dz(iGlob) * gw.k(im,0) * (hbc - gw.h(im,1)) / gdom.dx;
                        }
                        if (hbc < gw.h(kp,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dy * gw.k(iGlob,2) * (hbc - gw.h(kp,1)) / gdom.dz(kp);
                        }
                        if (hbc < gw.h(km,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dy * gw.k(km,2) * (hbc - gw.h(km,1)) / gdom.dz(km);
                        }
                	}, Kokkos::Sum<real>(flux));
                }
                else	{
                	Kokkos::parallel_reduce("gw_et", ncellsIT, KOKKOS_CLASS_LAMBDA (int idx, real &tmp){
                        int ii, jj, kk, ivg, idom, iGlob = icells[idx];
                        gdom.unpackIndicesHalo(iGlob, kk, jj, ii);
                        idom = (kk-gdom.hc)*gdom.nx*gdom.ny + (jj-gdom.hc)*gdom.nx + ii - gdom.hc;
                        // Only consider drainage under fully saturated condition
                        int ip = iGlob+1, im = iGlob-1, jp = iGlob+gdom.nxhc, jm = iGlob-gdom.nxhc;
                        if (hbc < gw.h(ip,1))	{
                        	tmp += Cpipe * gdom.dy * gdom.dz(iGlob) * gw.k(iGlob,0) * (hbc - gw.h(ip,1)) / gdom.dx;
                        }
                        if (hbc < gw.h(im,1))	{
                        	tmp += Cpipe * gdom.dy * gdom.dz(iGlob) * gw.k(im,0) * (hbc - gw.h(im,1)) / gdom.dx;
                        }
                        if (hbc < gw.h(jp,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dz(iGlob) * gw.k(iGlob,1) * (hbc - gw.h(jp,1)) / gdom.dy;
                        }
                        if (hbc < gw.h(jm,1))	{
                        	tmp += Cpipe * gdom.dx * gdom.dz(iGlob) * gw.k(jm,1) * (hbc - gw.h(jm,1)) / gdom.dy;
                        }
                	}, Kokkos::Sum<real>(flux));
                }
                Qoutflow = -flux;
            }

        }
    }



};
#endif


class SourceSink{
public:
	std::vector<std::string> id;
    SourceSinkData swss;
    #if SERGHEI_RE_MODEL
    std::vector<GwSS> gwss;
    #endif
};

#endif
