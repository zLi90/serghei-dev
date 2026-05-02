#ifndef _TOOLS_
#define _TOOLS_

#include "State.h"
#include "Domain.h"
#include "geometry.h"
#include "ParticleTracking.h"

#ifndef SERGHEI_TOOLS
#define SERGHEI_TOOLS 1
#endif

#ifndef SERGHEI_DEBUG_TOOLS
#define SERGHEI_DEBUG_TOOLS 0
#endif
#ifndef SERGHEI_DEBUG_LINE_RESAMPLE
#define SERGHEI_DEBUG_LINE_RESAMPLE 0
#endif

#define N_VARS_WRITE 3
#define N_VARS_XS_WRITE 2
#define OBSLINE_MODE_STATE 0
#define OBSLINE_MODE_XS 1

#if SERGHEI_TOOLS

class ObservationGauge{
  // this class MUST have a default constructor/destructor because it is used in a Kokkos::View
  // here we have an implicit constructor/destructor (the compiler will create one)
  public:
    geometry::point x;
    swState sw;

    int ii; // to store the index which relates to the memory index count
    int ic; // to store the physical cell index
    int id; // to store the subdomain where to find the gauge

    KOKKOS_INLINE_FUNCTION void linkDomain(const Domain &dom){
      ii = dom.getIndexForPoint(x);
      ic = dom.getCellForPoint(x);
      id = dom.id;
    };

    KOKKOS_INLINE_FUNCTION void fetchSurfaceState(const State &state){
      if(ii < 0){  // gauge is undefined, therefore values should be zero, so they can be reduced with MPI_SUM
        sw.h = sw.hu = sw.hv = sw.z = 0.;
        #if SERGHEI_SCALAR_TRANSPORT
        sw.hconc=0;
        #endif        
      }else{
        sw.h = state.h(ii);
        sw.hu = state.hu(ii);
        sw.hv = state.hv(ii);
        sw.z = state.z(ii);
        #if SERGHEI_SCALAR_TRANSPORT
        real aux=0.0; 
        for(int iphi=0; iphi<state.ade.nScalar; iphi++){
          aux += state.ade.hphi(ii,iphi);
        }
        sw.hconc = aux;
        #endif        
      }
    };


    inline std::string printCoordinates(){
      std::stringstream ss;
      ss.precision(OUTPUT_PRECISION);
      for(int ii=0; ii<N_SPATIAL_DIM; ii++) ss << x(ii) << "\t";
      return (ss.str());
    };

};

#if defined(KOKKOS_ENABLE_CUDA)
typedef Kokkos::View<ObservationGauge *, Kokkos::LayoutRight,
                     Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>
    obsGaugeView;
#elif defined(KOKKOS_ENABLE_HIP)
typedef Kokkos::View<ObservationGauge *, Kokkos::LayoutRight,
                     Kokkos::Device<Kokkos::HIP, Kokkos::HIPManagedSpace>>
    obsGaugeView;
#elif defined(KOKKOS_ENABLE_SYCL)
typedef Kokkos::View<ObservationGauge *, Kokkos::LayoutRight,
                     Kokkos::Device<Kokkos::Experimental::SYCL,
                                    Kokkos::Experimental::SYCLSharedUSMSpace>>
    obsGaugeView;
#else
typedef Kokkos::View<ObservationGauge *, Kokkos::LayoutRight> obsGaugeView;
#endif

class ObservationLine{
  public:
    int Np;
    int Ng;
    geometry::point *p;
    obsGaugeView g;
    #if SERGHEI_LPT
      ParticleTracker particles;
    #endif
    real length;
    real ds;  // nominal sampling resolution
    real *s;  // distance from line origin to each segment point, along the trajectory
    real *sg; // distance from line origin to each gauge
    int mode; // mode of use of the observation line
    real swflow=0; // surface water flow rate (discharge) across line
    real swvol=0; // surface water accumulated volume across the line
    geometry::point *normal;  // normal vector to each line segment

    //Scalar and sediment transport
    real solflow=0; // solid rate (discharge) across line
    real solvol=0; // accumulated solid volume across the line  

    inline void resample(){
      length = 0;
      // calculate distance to all polyline vertices
      s = new real[Np];
      s[0] = 0.;
      for(int ip=0; ip<Np-1; ip++){
        real d = geometry::distance(p[ip],p[ip+1]);
        s[ip+1] = d+s[ip];
        length += d;
      }
      // temporary container
      geometry::point *q;
      int n = floor(length/ds)+Np;    // this is larger than required
      q = new geometry::point[n];
      geometry::point dv;
      int ig = 0;
      real ts;
      for(int ip=1; ip<Np; ip++){
        dv = p[ip] - p[ip-1];  // segment vector
        dv = dv.unit();
        ts = s[ip-1];   // distance to first segment point
        q[ig]=p[ip-1];  // coords of first segment point
        #if SERGHEI_DEBUG_LINE_RESAMPLE
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "ts= " << ts << std::endl;
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "q[" << ig << "]_x= " << q[ig](_X) << std::endl;
        #endif
        ig++;
        while(ts+ds < s[ip]){   // while in the same segment from p[ip-1] to p[ip]
          q[ig] = q[ig-1] + dv*ds;
          ts += ds;   // accumulate distance from segment start
          #if SERGHEI_DEBUG_LINE_RESAMPLE
          std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "ts= " << ts << std::endl;
          std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "q[" << ig << "]_x= " << q[ig](_X) << std::endl;
          #endif
          ig++;
        }
      }
      q[ig] = p[Np-1];
      ig++;
      Ng = ig;
      g = obsGaugeView("Lgauges",Ng);
      sg = new real[Ng];
      for(int ig=0; ig<Ng; ig++){
        g(ig).x = q[ig];
      }
      sg[0] = 0.;
      for(int ig=1; ig<Ng; ig++){
        sg[ig] = sg[ig-1] + geometry::distance(g(ig).x,g(ig-1).x);
      }
      #if SERGHEI_DEBUG_LINE_RESAMPLE
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "Ng = " << Ng << std::endl;
        for(int ig=0; ig<Ng; ig++) std::cout << ig << "\t" << g(ig).x(_X) << "," << g(ig).x(_Y) << std::endl;
        for(int ig=0; ig<Ng; ig++) std::cout << ig << "\t" << sg[ig] << std::endl;
      #endif
      delete[] q;
    };

    inline void getSegmentNormal(){
      normal = new geometry::point[Ng];
      for(int ig=0; ig<Ng-1; ig++){
        normal[ig]=geometry::normalToLineInPlane(g(ig).x,g(ig+1).x);
      }
    };

    inline void computeFlux(const real &dt){
      #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
      #endif
      real q  = 0;
      real ds = 0;
      swflow  = 0;
      solflow = 0;

      #if SERGHEI_SCALAR_TRANSPORT 
        real v  = 0;
      #endif

      for(int ig=0; ig<Ng-1; ig++){
        if(g(ig).sw.h>TOL_WETDRY){
          q = g(ig).sw.hu * normal[ig](0) + g(ig).sw.hv * normal[ig](1);
          ds = sg[ig+1]-sg[ig];

          swflow += q * ds;
          swvol += swflow*dt;

          #if SERGHEI_SCALAR_TRANSPORT 
            v = q / g(ig).sw.h; 
            solflow += g(ig).sw.hconc * v * ds;
            solvol += solflow*dt;
          #endif
        }
      }
    };

};


class Observations{
Kokkos::Timer timer;

public:
  int nGauge=0;
  int nLines=0;
  obsGaugeView gauges;
  std::ofstream gaugeOut;
  std::string gaugeFilename;
  std::ofstream *linesOut;
  std::string *lineFilenames;
  ObservationLine *lines;
  int nLineFiles=0;
  int n_vars_line_write; // to keep track of how many variables need to be written out

  std::string varNames[N_VARS_WRITE+N_VARS_XS_WRITE] = {"h", "hu", "hv", "Q", "V"};
  std::string varNamesGauges[N_VARS_WRITE+1]={"h","hu","hv","z"};

  int readInputFiles(std::string dir, Parallel const &par){
    if(!readGauges(dir,par)) return 0;
    if(!readLines(dir,par)) return 0;
    return 1;
  };

  // member functions
  int readGauges(std::string dir, Parallel const &par){
    std::ifstream fInStream(dir + "gauges.input");
    real tmp;
    nGauge = 0;
    if(fInStream.is_open()){
      fInStream.ignore(256,' ');
      fInStream >> nGauge;
      gauges = obsGaugeView("gauges",nGauge);  // allocate gauges
      for (int ig=0; ig<nGauge; ig++) {
        for (int kk=0; kk<N_SPATIAL_DIM; kk++) {
          if (!fInStream.fail() && !fInStream.eof()){
            fInStream >> tmp;
            gauges(ig).x(kk) = tmp;
          }else{
            std::cerr<<RERROR "Not enough gauge coordinates. Expected " << nGauge << " gauges, with " <<N_SPATIAL_DIM<< "coordinates." << std::endl;
				    return 0;
          }
        }
      }
      if(par.masterproc) std::cout << GOK << "Observation gauges read. Number of gauges: " << nGauge << std::endl;
    }else{
      if(par.masterproc) std::cout << BDASH << "Observation gauges file not found. Number of gauges set to zero." << std::endl;
    }
    return 1;
  };

  int readLines(std::string dir, Parallel const &par){
    std::ifstream fInStream(dir + "lines.input");
    real tmp;
    if(fInStream.is_open()){
      fInStream.ignore(256,' ');
      fInStream >> nLines;
      if(nLines <= 0){
        if(par.masterproc) std::cout << GOK << "No observation lines defined (" << nLines << ")" << std::endl;
        return 1;
      }
      lines = new ObservationLine[nLines];  // allocate lines
      for (int il=0; il<nLines; il++) {
        fInStream.ignore(256,' ');
        fInStream >> lines[il].ds;
        fInStream.ignore(256,' ');
        fInStream >> lines[il].mode;
        fInStream.ignore(256,' ');
        fInStream >> lines[il].Np;
        #if SERGHEI_DEBUG_TOOLS
          if(par.masterproc){
            std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "il=" << il << "\tds=" << lines[il].ds << "\tmode=" << lines[il].mode << "\tNp=" << lines[il].Np <<  std::endl;
          }
        #endif
        lines[il].p = new geometry::point[lines[il].Np];
        for (int ip=0; ip<lines[il].Np; ip++) {
          for (int kk=0; kk<N_SPATIAL_DIM; kk++) {
            if (!fInStream.fail() && !fInStream.eof()){
              fInStream >> tmp;
              lines[il].p[ip](kk) = tmp;
            }else{
              if(par.masterproc) std::cerr<<RERROR "Not enough coordinates for observation line" << il << ". Expected " << lines[il].Np << " points, with " <<N_SPATIAL_DIM<< "coordinates." << std::endl;
				      return 0;
            }
          }
        }
      }
        if(par.masterproc) std::cout << GOK << "Observation lines read. Number of lines: " << nLines << std::endl;
    }else{
      if(par.masterproc) std::cout << BDASH << "Observation lines file not found. Number of lines set to zero." << std::endl;
    }
    return 1;
  };

  void printGauges(const Domain &dom){
    std::cout << GGD << "OBSERVATION GAUGES" << std::endl;
    int i,j;
    std::cout << "gID\tx\ty\ti\tj\tiGlob\tii" << std::endl;
    for (int ig=0; ig<nGauge; ig++) {
      std::cout << ig;
      dom.unpackIndices(gauges(ig).ic,j,i);
      for (int kk=0; kk<N_SPATIAL_DIM; kk++) std::cout << "\t" << gauges(ig).x(kk);
      std::cout << "\t" << i << "\t" << j;
      std::cout << "\t" << gauges(ig).ic ;
      std::cout << "\t" << gauges(ig).ii << std::endl;
    }
    std::cout << GGD << "-----END-----" << std::endl;
  };

  int configure(const Domain &dom, const std::string outdir){
      if(!configureGauges(dom,outdir)) return 0;
      if(!configureLines(dom,outdir)) return 0;
      return 1;
  };

  int configureGauges(const Domain &dom, const std::string outdir){
    int err=0;
      #if SERGHEI_DEBUG_TOOLS
        std::cout <<GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tnGauge=" << nGauge << std::endl;
      #endif
    for(int ig=0; ig<nGauge; ig++){
      gauges(ig).linkDomain(dom);
      #if SERGHEI_DEBUG_TOOLS
        std::cout << GGD << GRAY <<__PRETTY_FUNCTION__ << RESET <<  "ig=" << ig << "\tii=" << gauges(ig).ii << "\tic=" << gauges(ig).ic << std::endl;
      #endif
    }
	if(nGauge){
    	err = subdomainGaugeReduction(gauges,nGauge,dom);
    	}

	 if(err) std::cout << RERROR << "Some gauges were not found" << std::endl;


    if(err) return 0;
    gaugeFilename = outdir + "gauges.out";
    int nVars = N_VARS_WRITE + 1;

    if(nGauge > 0){
      gaugeOut.open(gaugeFilename);
		  if (gaugeOut.is_open()){
        gaugeOut << "time\t";
        for(int ig=0; ig<nGauge; ig++){
          for(int iv=0; iv<nVars; iv++){
            gaugeOut << "G" << ig << "_" << varNamesGauges[iv] << "\t";
          }
        }
        gaugeOut << std::endl;
      }else{
        std::cerr << RERROR "Unable to create observation gauge output file " << gaugeFilename << std::endl;
        return 0;
      }
    }
    if(dom.id == SERGHEI_MASTERPROC) std::cout << GOK << "Observation gauges ready" << std::endl;
    return 1;
  };

  inline int subdomainGaugeReduction(obsGaugeView &gauges, int N, const Domain &dom){
    int *gii;
    int *gii_max;
    typedef struct{
      int val;
      int rank;
    } intPair;
    intPair *in, *gii_min;
    int err=0;

    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif

    gii = new int[N];
    gii_max = new int[N];
    gii_min = new intPair[N];
    in = new intPair[N];

    for(int ig=0; ig<N; ig++){
      ObservationGauge &g = gauges(ig);
      gii[ig] = g.ii;
      gii_max[ig] = -100;
      gii_min[ig].val = 0;
      gii_min[ig].rank = -1;
      in[ig].rank = dom.id;
      in[ig].val = g.ii;
      // std::cout << GGD << "dom = " << dom.id <<"\t\tig = "  << ig << "\t\tgii[ig] = " << gii[ig] << std::endl;
    }

    /* Gauges should be in one domain or another, which gives them a g.ii > 0
    in the subdomain corresponding to the MPI rank (dom.id)
    But they may be outside of the domain (g.ii == -1)
    or may be in two subdomains (due to arithmetic precision).
    We use two reductions two check.
    */

    // Reduce with MPI_MAX, to check if a gauge is found on any subdomain.
    // A gii_max[ig] > 0 means that gauge ig has been found in some subdomain.
    MPI_Allreduce(gii, gii_max, N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      // Reduce with MPI_MINLOC, to check if a gauge is found on more than one subdomain.
    // A gii_min[ig].val > 0 means that gauge ig has been found in more than one subdomain.
    // gii_min[ig].rank returns one of the subdomains which found the gauge.
    MPI_Allreduce(in, gii_min, N, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    for(int ig=0; ig<N; ig++){
      ObservationGauge &g = gauges(ig);
      // check if gauge ig has been found in any subdomain or has not been found at all
      if(dom.id==SERGHEI_MASTERPROC){
        if(gii_max[ig] < 0){
          std::cout << RERROR << "Observation gauge " << ig << " (" << g.x(_X) << "," << g.x(_Y) << ")"<< " is not in the domain " << std::endl;
          err++;
        }
      }

      // now we check if a gauge has been found in two subdomains
      if(gii_min[ig].val > 0){
        /*
        std::cout << GGD << "dom.id=" << dom.id << "\t" << ig ;
        std::cout << "\tgii_min="<< gii_min[ig].val << "\tgii_max=" << gii_max[ig] ;
        std::cout << "\tMIN in rank " << gii_min[ig].rank <<std::endl;
        */
        // if there is a duplicate gauge, we remove it from all ranks but one
        if(dom.id != gii_min[ig].rank) g.ii = -1;
      }
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    delete gii;
    delete gii_max;
    delete gii_min;
    delete in;

    return err;
  };

  int configureLines(const Domain &dom, const std::string outdir){
    int err=0;
    int mode_q=0;
    int mode_b=0;

    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif

    for(int il=0; il<nLines; il++){
      lines[il].resample();
      for(int ig=0; ig<lines[il].Ng; ig++){
        ObservationGauge &g = lines[il].g(ig);
        g.linkDomain(dom);
      }

      err = subdomainGaugeReduction(lines[il].g, lines[il].Ng, dom);

	#if SERGHEI_DEBUG_TOOLS
    	for(int ig=0; ig<lines[il].Ng; ig++){
		if(lines[il].g(ig).ic > 0) std::cout << GGD << "il=" << il << "\tig=" << ig << "\tindex =" << lines[il].g(ig).ii << "\tcell=" << lines[il].g(ig).ic << "\tdom.id" << dom.id << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

      if(err){
        std::cerr << RERROR << "Gauges not found in observation line " << il << std::endl;
      }
    }

    if(err) return 0;

    if(dom.id == SERGHEI_MASTERPROC){
      // check line modes and configure output
      for(int il=0; il<nLines; il++){
        if(lines[il].mode == OBSLINE_MODE_STATE) mode_b++;
        if(lines[il].mode == OBSLINE_MODE_XS) mode_q++;
      }

      // prepare the output files
      nLineFiles = mode_b*N_VARS_WRITE + mode_q;
      lineFilenames = new std::string[nLineFiles];

      int jj=0;
      int nvars=0;
      for(int il=0; il<nLines; il++){
        if(lines[il].mode == OBSLINE_MODE_XS){
            lines[il].getSegmentNormal();
        }

        if(lines[il].mode == OBSLINE_MODE_STATE){
          nvars=N_VARS_WRITE;
          for(int iv=0; iv<nvars; iv++){
            lineFilenames[jj] = outdir + "line" + std::to_string(il) + "_" + varNames[iv] + ".out";
            jj++;
          }
        }
        if(lines[il].mode == OBSLINE_MODE_XS){
          nvars=N_VARS_XS_WRITE;
          for(int iv=0; iv<nvars; iv++){
            lineFilenames[jj] = outdir + "line" + std::to_string(il) + "_xsQV" + ".out";
          }
          jj++;
        }
      }

      std::cout << GOK << "Line output files (" << jj << ") created" << std::endl;

      jj=0;
      linesOut = new std::ofstream[nLineFiles];

      for(int il=0; il<nLines; il++){
        if(lines[il].mode == OBSLINE_MODE_STATE){
          nvars=N_VARS_WRITE;
          for(int iv=0; iv<nvars; iv++){
            linesOut[jj].open(lineFilenames[jj]);
            if (linesOut[jj].is_open()){
              linesOut[jj] << "time\t";
              for(int ig=0; ig<lines[il].Ng; ig++){
                linesOut[jj] << lines[il].sg[ig] << "\t" ;
              }
              linesOut[jj] << std::endl;
              jj++;
            }else{
              std::cerr << RERROR "Unable to create observation line output file " << lineFilenames[il] << std::endl;
            return 0;
            }
          }
        }
        if(lines[il].mode == OBSLINE_MODE_XS){
          linesOut[jj].open(lineFilenames[jj]);
          if (linesOut[jj].is_open()){
            linesOut[jj] << "time \t";
            linesOut[jj] << "Q \t V \t";
            #if SERGHEI_SCALAR_TRANSPORT 
            linesOut[jj] << "QS \t VS \t";
            #endif            
            linesOut[jj] << std::endl;
            jj++;
          }else{
            std::cerr << RERROR "Unable to create observation line output file " << lineFilenames[il] << std::endl;
            return 0;
          }
        }
      }

      std::cout << GOK << "Observation lines ready" << std::endl;
    }

    return 1;
  };

  void writeGauges(const real &time){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
		if (gaugeOut.is_open()){
      std::cout.precision(OUTPUT_PRECISION);
      gaugeOut << time << "\t";
      for(int ig=0; ig<nGauge; ig++){
        gaugeOut << std::scientific << gauges(ig).sw.h << "\t" <<  gauges(ig).sw.hu << "\t" << gauges(ig).sw.hv << "\t" <<  gauges(ig).sw.z << "\t";
      }
      gaugeOut << std::endl;
    }
  };


  void writeLines(const real &time){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    int jj=0;
    for(int il=0; il<nLines; il++){
      if(lines[il].mode == OBSLINE_MODE_STATE ){
        int nvars=N_VARS_WRITE;
        for(int iv=0; iv<nvars; iv++){
          std::cout.precision(OUTPUT_PRECISION);
          linesOut[jj] << time << "\t";
          for(int ig=0; ig<lines[il].Ng; ig++){
            ObservationGauge &g = lines[il].g(ig);
            if(iv == 0) linesOut[jj] << std::scientific << g.sw.h << "\t";
            if(iv == 1) linesOut[jj] << std::scientific << g.sw.hu << "\t";
            if(iv == 2) linesOut[jj] << std::scientific << g.sw.hv << "\t";
          }
          linesOut[jj] << std::endl;
          jj++;
        }
      }
      if(lines[il].mode == OBSLINE_MODE_XS ){
        std::cout.precision(OUTPUT_PRECISION);
        linesOut[jj]  << std::scientific 
                      << time << "\t" 
                      << lines[il].swflow << "\t" 
                      << lines[il].swvol << "\t"
                      #if SERGHEI_SCALAR_TRANSPORT 
                      << lines[il].solflow << "\t" 
                      << lines[il].solvol << "\t" 
                      #endif
                      << std::endl;
        jj++;
      }
    }
  };

  void write(Domain const &dom){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    timer.reset();
    writeGauges(dom.etime);
    writeLines(dom.etime);
    dom.timers.swe.io.out += timer.seconds();
  }

  inline void writeLinesSamplingCoordinates(std::string outdir){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    for(int il=0; il<nLines; il++){
      std::string fname = outdir + "line" + std::to_string(il) + ".gout";
      std::ofstream fileout;
      fileout.open(fname);
      if (fileout.is_open()){
        fileout << "x" << "\ty" << "\tz" << "\td" << std::endl;
        std::cout.precision(OUTPUT_PRECISION);
        for(int ig=0; ig<lines[il].Ng; ig++){
          fileout << lines[il].g(ig).printCoordinates() << "\t" << lines[il].g(ig).sw.z << "\t" << lines[il].sg[ig] << std::endl;
        }
        fileout.close();
      }
    }
  };

  inline void gaugeStateReduction(obsGaugeView &gauges, const int N, const Parallel &par){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    // DCV: TODO this can be done more elegantly, by using an MPI struct reduction
    // This idea may also be more efficient than the current implementation
    //defineMPIswState(&structtype);
    //MPI_Op_create(sum_struct_ts, 1, &MPISUM_swState);
    //MPI_Reduce(local,global,lines[il].Ng,structtyp,MPISUM_swState,SERGHEI_MASTERPROC,MPI_COMM_WORLD);

    // perform reductions so that all data is in the line instance in the master process
    real *out,*in;
    out = new real[N];
    in = new real[N];

    for(int ig = 0; ig < N; ig++) in[ig] = gauges(ig).sw.h;
    MPI_Reduce(in, out, N, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC, MPI_COMM_WORLD);
    for(int ig = 0; ig < N; ig++){
      if(par.masterproc) gauges(ig).sw.h = out[ig];
      in[ig] = gauges(ig).sw.hu;
    }
    MPI_Reduce(in, out,N, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC, MPI_COMM_WORLD);
    for(int ig = 0; ig < N; ig++){
      if(par.masterproc) gauges(ig).sw.hu = out[ig];
      in[ig] = gauges(ig).sw.hv;
    }
    MPI_Reduce(in, out, N, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC, MPI_COMM_WORLD);
    for(int ig = 0; ig < N; ig++){
      if(par.masterproc) gauges(ig).sw.hv = out[ig];
	    in[ig] = gauges(ig).sw.z;
    }
    MPI_Reduce(in, out, N, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC, MPI_COMM_WORLD);
    for(int ig = 0; ig < N; ig++){
      if(par.masterproc) gauges(ig).sw.z = out[ig];
    }
    #if SERGHEI_SCALAR_TRANSPORT 
    for(int ig = 0; ig < N; ig++){
      in[ig] = gauges(ig).sw.hconc;
    }
    MPI_Reduce(in, out, N, SERGHEI_MPI_REAL, MPI_SUM, SERGHEI_MASTERPROC, MPI_COMM_WORLD);
    for(int ig = 0; ig < N; ig++){
      if(par.masterproc) gauges(ig).sw.hconc = out[ig];
    }    
    #endif 
    delete out,in;
  };

  inline void updateGauges(const State &state, const Parallel &par){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
  //  Kokkos::parallel_for("get_state_for_gauges",nGauge, KOKKOS_LAMBDA (int ig){
    for(int ig=0; ig <nGauge ; ig++)
      gauges(ig).fetchSurfaceState(state);
   // });
		if(nGauge){
    		gaugeStateReduction(gauges, nGauge, par);
  		}
  };

  inline void updateLines(const State &state, const Parallel &par, const real &dt){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    for(int il=0; il < nLines; il++){
      //  Kokkos::parallel_for("get_state_for_lines",nGauge, KOKKOS_LAMBDA (int ig){
      for(int ig=0; ig < lines[il].Ng ; ig++){
        lines[il].g(ig).fetchSurfaceState(state);
      }
      // });
      gaugeStateReduction(lines[il].g, lines[il].Ng, par);
      if(par.masterproc){
        if(lines[il].mode==OBSLINE_MODE_XS) lines[il].computeFlux(dt);
      }
    }
  };

  inline void update(const State &state, const Parallel &par, const Domain &dom){
    #if SERGHEI_DEBUG_TOOLS
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
      timer.reset();
      updateGauges(state,par);
      updateLines(state,par,dom.dt);
      dom.timers.swe.io.out += timer.seconds();

  };

  inline void closeOutputStreams(){
    gaugeOut.close();
    for(int jj=0; jj<nLineFiles; jj++){
      linesOut[jj].close();
    }
  };
};
#endif
#endif