#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include "define.h"
#include "Parallel.h"
#include "geometry.h"
#include "Indexing.h"
#include "timers.h"

class Domain {

public:

  Parallel *Par;  // pointer to par

  real cfl;
  double simLength=0;
  double startTime=0;
  double endTime=SERGHEI_NAN;  // intended to account for restarts
  double etime=0;

  real dt=0;

  SergheiTimers mutable timers, relative;

  //raster variables
  int nx_glob = 0;
  int ny_glob = 0;
  int nx = 0;         // physical number of cells in x-direction
  int ny = 0;         // physical number of cells in y-direction
  int nCellMem = 0;   // physical cells + halo cells
  int nCell = 0;    // physical cells
  int nCellValid = 0; // cells which have data
  double xll = 0; // southwest corner x-coordinate
  double yll = 0; // southwest corner y-coordinate
  #if SERGHEI_MESH_UNIFORM
  real dxConst;  // resolution
  #endif
  int i_beg, j_beg, i_end, j_end;

  //flags to see if the subdomain touch with either a East, West, South or North boundaries
  int iE=0;
  int iW=0;
  int iS=0;
  int iN=0;

  // global (reduced) variables
  real areaGlobal=0;

  long long nCellValidGlobal = 0;
  long long nCellGlobal = 0;

  // other variables
  int BCtype;
  int isRain = 0, isEvap = 0, isWind = 0;

  int nTimeSteps;
  int countIterDt;
  int cg_iter;

  real area, hwmin, CwT;
  int id;  // subdomain ID
  int nsubdom; // number of subdomains

  // halo cells (overlapping cells between domains for MPI)
  int hc = 1;

  bool rasterIndexing = true;


  realArr globalBuffer;

  geometry::point extent[2];

  void inline print(){
    std::cout << "DOMAIN PRINT" << std::endl;
    std::cout << "nx_glob = " << nx_glob << std::endl;
    std::cout << "ny_glob = " << ny_glob << std::endl;
    std::cout << "nx = " << nx << std::endl;
    std::cout << "ny = " << ny << std::endl;
    std::cout << "nCellMem = " << nCellMem << std::endl;
    std::cout << "nCell = " << nCell << std::endl;
    std::cout << "nCellValid = " << nCellValid << std::endl;
    std::cout << "dxConst = " << dxConst << std::endl;
    std::cout << "rasterIndexing = " << rasterIndexing << std::endl;
    std::cout << "hc = " << hc << std::endl;
    std::cout << "xll = " << xll << std::endl;
    std::cout << "yll = " << yll << std::endl;
    std::cerr << "extent[0] = (" << extent[0](_X) << "," << extent[0](_Y) << ") ";
    std::cerr << "extent[1] = (" << extent[1](_X) << "," << extent[1](_Y) << ") " << std::endl;
  }


  // this is purposely programmed to fail at compilation time if !SERGHEI_MESH_UNIFORM because the alternative is not implemented
  #if SERGHEI_MESH_UNIFORM
  int di;
  int dj;  // stride in i and j direction for the extended domain (including halo cells)

  KOKKOS_INLINE_FUNCTION real dx() const{
    return(dxConst);
  }
  #endif


  KOKKOS_INLINE_FUNCTION void domainArea(){
       area = cellArea()*nCellValid;
   };

  KOKKOS_INLINE_FUNCTION void unpackIndices(int const iGlob, int &j, int &i) const{
    unpackIndicesUniformGrid(iGlob,ny,nx,j,i);
  };

  // this is purposely programmed to fail at compilation time if !SERGHEI_MESH_UNIFORM because the alternative is not implemented
  #if SERGHEI_MESH_UNIFORM
  KOKKOS_INLINE_FUNCTION geometry::point getCellCenter(int i, int j) const{
    geometry::point p;
    p(_X) = extent[0](_X) + i*dxConst ;
    p(_Y) = extent[1](_Y) - j*dxConst ;
    return(p);
  #endif
  }

  KOKKOS_INLINE_FUNCTION geometry::point getCellCenter(int iGlob) const{
    int i,j;
    unpackIndices(iGlob,j,i);
    return(getCellCenter(i,j));
  }

  // this is purposely programmed to fail at compilation time if !SERGHEI_MESH_UNIFORM because the alternative is not implemented
  #if SERGHEI_MESH_UNIFORM
  KOKKOS_INLINE_FUNCTION real cellArea() const{
    return(dx()*dx());  // WARNING UCM
  }
  #endif


// this is purposely programmed to fail at compilation time if !SERGHEI_MESH_UNIFORM because the alternative is not implemented
#if SERGHEI_MESH_UNIFORM
  KOKKOS_INLINE_FUNCTION void getMatrixIndicesForPoint(const geometry::point &p, int &i, int &j) const{
    // get i,j coordinates of the cell containing the point
    // this only works on a raster-order grid, where j is zero at NORTH boundary
    i = floor((p(_X)-extent[0](_X))/dxConst);
    j = floor((extent[1](_Y)-p(_Y))/dxConst);
  };

  KOKKOS_INLINE_FUNCTION int getCellForPoint(const geometry::point &p) const{
    int i,j,iGlob;
    getMatrixIndicesForPoint(p,i,j);
    if(i < 0 || i >= nx || j < 0 || j >= ny){
      iGlob = -1;
    }else{
      iGlob = packIndicesUniformGrid(ny,nx,j,i);
    }
    // note: domain decomposition is handled by the i,j coords
    return(iGlob);
  };
  #endif

  KOKKOS_INLINE_FUNCTION int getHaloExtension(const int i, const int j) const {
    return( (hc+j)*(nx+2*hc)+hc+i ); //index for the extended domain (including halo cells)
  };

  KOKKOS_INLINE_FUNCTION int getIndex(int iGlob) const{
    int i,j;
    unpackIndices(iGlob,j,i);
    return ( getHaloExtension(i,j) );
  };

  KOKKOS_INLINE_FUNCTION int getSubdomainExtension(const Parallel &par, const int i, const int j) const{
    return( (par.j_beg+j)*nx_glob+par.i_beg+i ); //index for the subdomain (par.j_beg+j,par.i_beg+i)
  };

  KOKKOS_INLINE_FUNCTION int getIndexForPoint(const geometry::point &p) const{
    int i,j,ii;
    getMatrixIndicesForPoint(p,i,j);
    if(i < 0 || i >= nx || j < 0 || j >= ny){
      ii = -1;
    }else{
      ii  = getHaloExtension(i,j);
    }
    return(ii);
  };

  #if SERGHEI_MESH_UNIFORM
  KOKKOS_INLINE_FUNCTION void getNeighbours(const int iGlob, int* neigh) const{
		int i,j;
		unpackIndicesUniformGrid(iGlob,ny+2*hc,nx+2*hc,j,i);
		neigh[0]=iGlob-1;
		neigh[1]=iGlob+1;
		neigh[2]=(j-1)*(nx+2*hc)+i;
		neigh[3]=(j+1)*(nx+2*hc)+i;

  };

  KOKKOS_INLINE_FUNCTION bool isHalo(const int iGlob) const{
    int i,j;
    unpackIndicesUniformGrid(iGlob,ny+2*hc,nx+2*hc,j,i);
    return(i < hc || j < hc || i>=nx+hc || j>=ny+hc);
  }
  #endif


void initialiseSpatial(){
    #if SERGHEI_MESH_UNIFORM
      nCell = nx*ny;                    // physical cells in this subdomain
      nCellMem = (ny+2*hc)*(nx+2*hc);   // size of arrays (including halos)
      nCellGlobal = nx_glob * ny_glob;  // physical number of cells across all subdomains
      di = 1;                     // stride in i
      dj = nx+2*hc;                  // stride in j
    #endif
}

void initialise() {
   	// Initialize the time
   	nTimeSteps = 0;
		countIterDt=0;
    etime = startTime;
    if(std::isnan(endTime)){
      endTime = startTime + simLength;
    }else{
      startTime = endTime - simLength;
    }


    initialiseSpatial();

    globalBuffer = realArr("globalBuffer", nCellGlobal);

    if(id == 0) std::cout << GOK << "Domain initialised" << std::endl;
  };

  void getStatistics(){
    domainArea();
    Kokkos::fence();
    MPI_Allreduce(&area, &areaGlobal, 1, MPI_DOUBLE , MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nCellValid, &nCellValidGlobal, 1, MPI_INT , MPI_SUM, MPI_COMM_WORLD);
    std::cout << GOK << "Total physical computational cells: " << nCellValidGlobal << std::endl;
  };

  void inline get2Ddecomposition(const Parallel &par){
    double nper;
    nper = ((double) nx_glob)/par.nproc_x;
    i_beg = (long) round( nper* par.px    );
    i_end = (long) round( nper*(par.px+1) )-1;
    nper = ((double) ny_glob)/par.nproc_y;
    j_beg = (long) round( nper* par.py    );
    j_end = (long) round( nper*(par.py+1) )-1;

    nx = i_end - i_beg + 1;
    ny = j_end - j_beg + 1;
  }

    inline void getExtent(){
    // Determine my extent, point 0 is SW, point 1 is NE (standard cartesian)
    extent[0](_X) = xll + i_beg*dxConst;
    extent[0](_Y) = yll + ny_glob*dxConst - (j_end+1)*dxConst;
    extent[1](_X) = xll + (i_end+1)*dxConst;
    extent[1](_Y) = yll + ny_glob*dxConst - (j_beg)*dxConst;
    }

  int buildDomainDecomposition(Parallel &par) {

    int ierr=1;

    Par = &par;

    if (par.nranks != par.nproc_x*par.nproc_y) {
      std::cerr << RERROR "ERROR: nproc_x*nproc_y != nranks" << std::endl;
      std::cerr << RERROR << par.nproc_x << " " << par.nproc_y << " " << par.nranks << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //Get my x and y process grid ID
    par.px = par.myrank % par.nproc_x;
    par.py = par.myrank / par.nproc_x;

    // name subdomain
    id = par.myrank;

    //Get my beginning and ending global indices
    /*
    double nper;
    nper = ((double) nx_glob)/par.nproc_x;
    par.i_beg = (long) round( nper* par.px    );
    par.i_end = (long) round( nper*(par.px+1) )-1;
    nper = ((double) ny_glob)/par.nproc_y;
    par.j_beg = (long) round( nper* par.py    );
    par.j_end = (long) round( nper*(par.py+1) )-1;
    */

    // Determine my number of grid cells
    get2Ddecomposition(par);
    par.i_beg = i_beg;
    par.i_end = i_end;
    par.j_beg = j_beg;
    par.j_end = j_end;

    getExtent();


    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        int pxloc = par.px+i-1;
        if (pxloc < 0            ) pxloc = pxloc + par.nproc_x;
        if (pxloc > par.nproc_x-1) pxloc = pxloc - par.nproc_x;
        int pyloc = par.py+j-1;
        if (pyloc < 0            ) pyloc = pyloc + par.nproc_y;
        if (pyloc > par.nproc_y-1) pyloc = pyloc - par.nproc_y;
        par.neigh(j,i) = pyloc * par.nproc_x + pxloc;
      }
    }

  	//set topological boundaries
  	if(par.myrank % par.nproc_x ==0) iW=1; //west boundary of the full domain
	  if(par.myrank % par.nproc_x ==par.nproc_x-1) iE=1; //east boundary of the full domain
	  if(par.myrank / par.nproc_x ==0) iN=1; //north boundary of the full domain
	if(par.myrank / par.nproc_x ==par.nproc_y-1) iS=1; //south boundary of the full domain

    nsubdom = par.nranks;
    // Debug output for the parallel decomposition
    #if SERGHEI_DEBUG_PARALLEL_DECOMPOSITION
      for (int rr=0; rr < par.nranks; rr++) {
        if (rr == par.myrank) {
          std::cerr << GGD "Hello! My Rank is: " << par.myrank << "\n";
          std::cerr << GGD "My domain id is: " << id << std::endl;
          std::cerr << GGD "My proc grid ID is: " << par.px << " , " << par.py << "\n";
          std::cerr << GGD "I have: " << nx << " x " << ny << " grid cells" << "\n";
          std::cerr << GGD "I start at index: " << par.i_beg << " x " << par.j_beg << "\n";
          std::cerr << GGD << "My extent points are " ;
          std::cerr << "(" << extent[0](_X) << "," << extent[0](_Y) << ") ";
          std::cerr << "(" << extent[1](_X) << "," << extent[1](_Y) << ") " << std::endl;
          std::cerr << GGD "My neighbor matrix is:\n";
          for (int j = 2; j >= 0; j--) {
            for (int i = 0; i < 3; i++) {
              std::cerr << std::setw(6) << par.neigh(j,i) << " ";
            }
            std::cerr << "\n";
          }
          std::cerr << "\n";
          }
          ierr = MPI_Barrier(MPI_COMM_WORLD);
      }
      ierr = MPI_Barrier(MPI_COMM_WORLD);
    #endif
    if(par.masterproc) std::cout << GOK << "Domain decomposition" << std::endl;
	  return ierr;
	};



void fetchFieldFromGlobalBuffer(const Parallel &par, realArr &data) const {
  Kokkos::parallel_for("fetch_from_global_buffer", nCell , KOKKOS_CLASS_LAMBDA (int iGlob) {
 	  int i,j;
		unpackIndices(iGlob,j,i);
		int ii1 = getHaloExtension(i,j);
		int ii2 = getSubdomainExtension(par,i,j);
		data(ii1) = globalBuffer(ii2);
	});
}

int reorderViewToRasterIndexing(realArr &myView) const{
  if(rasterIndexing){
    if(id==0) std::cerr << RERROR << "Attempting to reorder a view on a domain which has already raster indexing" << std::endl;
    return SERGHEI_ERROR;
  }

  for (int j=0; j<ny/2; j++) {
    // Compute the corresponding row to swap with
    int ii0 = j * nx;                // Start index of row j
    int ii1 = (ny-1-j) * nx;         // Start index of corresponding row to swap

    // Swap row elements
    /*
    Kokkos::parallel_for("swapRows",nx,KOKKOS_CLASS_LAMBDA(int i){
      Kokkos::kokkos_swap(myView(ii0 + i), myView(ii1 + i));
    });
  */
    for(int i=0; i<nx; i++)
      Kokkos::kokkos_swap(myView(ii0 + i), myView(ii1 + i));
  }

  if(id==0) std::cout << GOK << "View reordered to raster indexing" << std::endl;
  return SERGHEI_OK;
}

template<typename T> void realToExtended(T &dataReal, T &data) const{
  Kokkos::parallel_for("realToExtended",nCell, KOKKOS_CLASS_LAMBDA(int iGlob) {
	  int ii = getIndex(iGlob);
    data(ii) = dataReal(iGlob);
  });
}


};


#endif
