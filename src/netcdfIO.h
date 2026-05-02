#pragma once

#include <mpi.h>
#include <pnetcdf.h>
#include "define.h"
#include "units.h"

std::string ncVarType(nc_type varType){
  std::string type;
  switch (varType) {
    case NC_BYTE:   type = "NC_BYTE"; break;
    case NC_CHAR:   type = "NC_CHAR"; break;
    case NC_SHORT:  type = "NC_SHORT"; break;
    case NC_INT:    type = "NC_INT"; break;
    case NC_FLOAT:  type = "NC_FLOAT"; break;
    case NC_DOUBLE: type = "NC_DOUBLE"; break;
    case NC_UBYTE:  type = "NC_UBYTE"; break;
    case NC_USHORT: type = "NC_USHORT"; break;
    case NC_UINT:   type = "NC_UINT"; break;
    case NC_INT64:  type = "NC_INT64"; break;
    case NC_UINT64: type = "NC_UINT64"; break;
    case NC_STRING: type = "NC_STRING"; break;
    default:        type = "UNKNOWN TYPE"; break;
  }

  return(type);
};

//Error reporting routine for the PNetCDF I/O
void ncwrap( int ierr , int line, int rank) {
  if (ierr != NC_NOERR) {
    std::cerr<< RERROR "Error code (" << ierr << ") from rank " << rank << " at " << __FILE__ << ":" << line << std::endl << RERROR << ncmpi_strerror(ierr) << std::endl;
    MPI_Abort(MPI_COMM_WORLD, ierr);
  }
}
void ncwrap( int ierr , int line) {
  if (ierr != NC_NOERR) {
    std::cerr<< NCERROR "Error code (" << ierr << ") at " << __FILE__ << ":" << line << std::endl << NCERROR << ncmpi_strerror(ierr) << std::endl;
    MPI_Abort(MPI_COMM_WORLD, ierr);
  }
}

void ncwrap(int ierr, std::string file, int line){
  if(ierr != NC_NOERR){
    std::cerr<< NCERROR "Error code (" << ierr << ") at " << file << ":" << line << std::endl << NCERROR << ncmpi_strerror(ierr) << std::endl;
    MPI_Abort(MPI_COMM_WORLD, ierr);
  }
}

class ncStream;

class ncVarStrings{
  public:
  std::string fname;
  std::string name;
  std::string units;
};

class ncVar{
  public:
  ncVarStrings *s;  // pointer to object containing strings
  int *fid;         // pointer to file id
  int id;           // var id
  int ndim = 0;         // number of variable dimensions
  real factor;      // unit conversion factor
  float nodata=SERGHEI_NAN;  // assume that the _FillValue is NaN if not specified
  int tStride=0;       // current netCDF stride in the time dimension
  nc_type type;         // data type
  bool allowUnitless = false;

  private:
  MPI_Offset *st;   // netCDF stride
  MPI_Offset *ct;   // netCDF count



  inline void setStrideCount(const Parallel &par, const Domain &dom){
    ct = new MPI_Offset[ndim];
    st = new MPI_Offset[ndim];
    if(ndim > 2){
      // This is the approach to read time-dependent variables
      st[0] = tStride; st[1] = dom.j_beg; st[2] = dom.i_beg;
      ct[0] = 1; ct[1] = dom.ny   ; ct[2] = dom.nx   ;
    }
    else{
      // Read a time-independent variable
      st[0] = dom.j_beg; st[1] = dom.i_beg;
      ct[0] = dom.ny   ; ct[1] = dom.nx;
    }
    if constexpr(SERGHEI_DEBUG_INPUT_NETCDF){
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
      std::cout << "ndim = " << ndim << std::endl;
      std::cout << "rank = " << par.myrank << "\tst : " << st[0] << "\t" << st[1];
      if(ndim > 2) std::cout << "\t" << st[2];
      std::cout << std::endl;
      std::cout << "rank = " << par.myrank << "\tct : " << ct[0] << "\t" << ct[1];
      if(ndim > 2) std::cout << "\t" << ct[2];
      std::cout << std::endl;
    }
  }

  public:

  inline void initialise(ncVarStrings &sp, const std::string &filename, const std::string vname){
    s = &sp;
    s->fname = filename;
    s->name = vname;
  }

  int inline findFieldData(const Parallel &par, const Domain &dom){
  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

  int ncerr = ncmpi_inq_varid(*fid,s->name.c_str(),&id);

  if(ncerr != NC_NOERR){
    if(ncerr == NC_ENOTVAR){  // if variable not found, return and deal with it one level up
      return ncerr;
      }
    else{
      ncwrap(ncerr,__LINE__,par.myrank);
    }
  }

  ncmpi_inq_vartype(*fid,id,&type);
  if(par.masterproc) std::cout << BDASH << "Variable " << GREEN << s->name << RESET << " has data type " <<  ncVarType(type) << std::endl;

  std::string attName = "_FillValue";

  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << ":" << __LINE__ << RESET << std::endl;

  ncerr=ncmpi_get_att_float(*fid,id,attName.c_str(),&nodata);
  if(ncerr==NC_ENOTATT){
    if(par.masterproc) std::cout << YEXC << "NetCDF _FillValue not found for variable " << CYAN << s->name << RESET << ". Assuming it is NaN" << std::endl;
  }else{
    ncwrap(ncerr,__LINE__,par.myrank);
  }
  if(par.masterproc) std::cout << BDASH << "No data value for variable " << GREEN << s->name << RESET << " is " << nodata << std::endl;


	if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tTarget NetCDF variable " << CYAN << s->name << RESET << " has index " << id << std::endl;

  ncwrap(ncmpi_inq_varndims(*fid, id, &ndim),__LINE__,par.myrank);

  MPI_Offset att_len;

  int retval = ncmpi_inq_attlen(*fid, id, "units", &att_len);

  if(retval == NC_NOERR) {
    std::vector<char> units_(att_len + 1); // +1 for null terminator
    ncwrap(ncmpi_get_att_text(*fid, id, "units", units_.data()),__LINE__);
    units_[att_len] = '\0'; // Null-terminate the string
    s->units = std::string(units_.data());
    if(par.masterproc) std::cout << BDASH << "Units for variable " << GREEN << s->name << RESET << ": " << s->units << std::endl;
  }else if(retval == NC_ENOTATT){
    // units attribute is absent; let the caller decide whether this is acceptable
    if(par.masterproc) std::cout << YEXC << "Units not found for variable " << s->name << std::endl;
  }else{
    ncwrap(retval,__LINE__,par.myrank);
  }
  return retval;
}

template<typename T> int readField(const Parallel &par, const Domain &dom, T &data){
  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

  if(par.masterproc) std::cout << std::endl << BDASH << "Reading NetCDF variable " << GREEN << s->name << RESET << " from " << CYAN << s->fname << RESET << std::endl;

  int varfound = findFieldData(par,dom);
  if(varfound != NC_NOERR){
    if(varfound == NC_ENOTATT && allowUnitless){}
    else{
      return varfound;
    }
  }

  setStrideCount(par,dom);
  // we use a buffer view, because of the order of coordinates
  if constexpr(std::is_same<T,realArr>::value) ncwrap(ncmpi_get_vara_double_all(*fid,id,st,ct,data.data()),__LINE__,par.myrank);
  if constexpr(std::is_same<T,ushortArr>::value) ncwrap(ncmpi_get_vara_ushort_all(*fid,id,st,ct,data.data()),__LINE__,par.myrank);
  if constexpr(std::is_same<T,ucharArr>::value) ncwrap(ncmpi_get_vara_uchar_all(*fid,id,st,ct,data.data()),__LINE__,par.myrank);
  if constexpr(std::is_same<T,intArr>::value) ncwrap(ncmpi_get_vara_int_all(*fid,id,st,ct,data.data()),__LINE__,par.myrank);

  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF){
    real maxval, minval;
    int count = ct[0]*ct[1];
    if(ndim > 2) count = (ct[1]*ct[2]);
    Kokkos::parallel_reduce("debug_netcdf",count,KOKKOS_LAMBDA(int ii, real &maxval, real &minval){
      maxval = max(data(ii),maxval);
      minval = min(data(ii),minval);
    }, Kokkos::Max<real>(maxval), Kokkos::Min<real>(minval));
    std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " Variable " << CYAN << s->name << RESET << " minimum: " << minval << " maximum: " << maxval << " from " << count << " read values."  << std::endl;
  }

	return NC_NOERR;
}

template<typename T> int readFieldExtended(const Parallel &par, const Domain &dom, T &data){
  T dataTemp = T("dataTemp",dom.nCell);
  int err=readField(par,dom,dataTemp);
  dom.realToExtended(dataTemp,data);
  if(err != NC_NOERR) return err;
  return err;
}

int read(const Parallel &par, Domain &dom, State &state) {
  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

  realArr data = realArr("ncdata",dom.nCell);
	int varfound = readField(par,dom,data);
  if(varfound != NC_NOERR){
    if(varfound == NC_ENOTVAR) return varfound;
    ncwrap(varfound,__LINE__);
  }

  int var = getIOvarID(s->name);

  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << s->name << " is internal var " << var << std::endl;

  if(var < 0 ){
		std::cerr << RERROR << "Internal variable " << CYAN << s->name << RESET << " not found." << std::endl;
		return NC_NOERR;
	}

  if(var == ioZ){
    real nodata_ = nodata;
    Kokkos::parallel_reduce("readNC_z_reduce",dom.nCell, KOKKOS_LAMBDA(int iGlob, int &nValid) {
		  int ii = dom.getIndex(iGlob);
	  	state.z(ii) = data(iGlob);
	  	if (state.z(ii) <= nodata_ || std::isnan(state.z(ii))) {
	   	 state.isnodata(ii) = true;
         state.z(ii) = NDTH+0.001;
	  	} else {
	   	 state.isnodata(ii) = false;
		 nValid++;
	  	}
	  }, Kokkos::Sum<int>(dom.nCellValid));
  }
  else{
  Kokkos::parallel_for("netCDFvariable",dom.nCell, KOKKOS_LAMBDA(int iGlob) {
	int ii = dom.getIndex(iGlob);
    if(!std::isnan(data[iGlob]) && !state.isnodata(ii)){// data[iGlob]=0;
        if(var == ioH) state.h(ii) = data[iGlob];
        if(var == ioU) state.hu(ii) = data[iGlob]*state.h(ii);
        if(var == ioV) state.hv(ii) = data[iGlob]*state.h(ii);
				if(var == ioR) state.roughness(ii) = data[iGlob];
    }
	});
}

  if(par.masterproc) std::cout << GOK << "NetCDF variable " << GREEN << BOLD << s->name << RESET << " read" << std::endl;
  return NC_NOERR;

}

};

class ncStreamStrings{
  public:
  std::string fname;
  ncVarStrings t, n, z, h, u, v, rain;
  ncVarStrings landuse, soilmap;
  ncVarStrings infRate;
};

class ncStream{
	public:
  ncStreamStrings *s;
  int nTime;

  ncVar varTime, varn, varz, varh, varu, varv, varRain;
  ncVar varInfRateCap;
  ncVar varSoilMap, varLandUse;

  private:
	int id;
	int ndims;
	int nvars;
	int ngatts;
	int unlimited;
	int dimids[3];
	int ndata;
  float nodata=SERGHEI_NAN;  // assume that the _FillValue is NaN if not specified


  void print_dimids(){
    std::cout << GGD << "dimids: " << dimids[0] << "\t" << dimids[1] << "\t" << dimids[2] << std::endl;
  }

  void inline getDimIds(const Parallel &par){
	  ncwrap(ncmpi_inq_dimid(id, "x", &dimids[2]),__LINE__,par.myrank);
	  ncwrap(ncmpi_inq_dimid(id, "y", &dimids[1]),__LINE__,par.myrank);
	  int err = ncmpi_inq_dimid(id, "t", &dimids[0]);
    if(err != NC_NOERR)  err = ncmpi_inq_dimid(id, "time", &dimids[0]);
    if(err != NC_NOERR) ncwrap(err,__LINE__,par.myrank);
  }

  public:

  ncStream(){
    if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

    // set pointer from variables to stream
    varTime.fid = &id;
    varn.fid = &id;
    varz.fid = &id;
    varh.fid = &id;
    varu.fid= &id;
    varv.fid = &id;
    varRain.fid = &id;
    varLandUse.fid = &id;
    varSoilMap.fid = &id;
    varInfRateCap.fid = &id;
    varLandUse.allowUnitless=true;
    varSoilMap.allowUnitless = true;
  }

  inline void initialise(ncStreamStrings &ncs, const std::string filename){
    s = &ncs;
    s->fname = filename;
  }
  inline void initialise(ncStreamStrings &ncs){
    s = &ncs;
  }

  inline void initialiseSurfaceVariables(){
    varz.initialise(s->z,s->fname,"z");
    varh.initialise(s->h,s->fname,"h");
    varu.initialise(s->u,s->fname,"u");
    varv.initialise(s->v,s->fname,"v");
    varn.initialise(s->n,s->fname,"roughness");
    varLandUse.initialise(s->landuse,s->fname,"landuse");
    varSoilMap.initialise(s->soilmap,s->fname,"soilmap");
    varInfRateCap.initialise(s->infRate,s->fname,"infRate");
  }

  inline void close(){
    ncwrap( ncmpi_close(id) , __LINE__);
  }




  int readNetCDFheader(const Parallel &par, Domain &dom, bool allowTime){
    if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    MPI_Offset dimsize;

    // open the netcdf file
    ncwrap(ncmpi_open(MPI_COMM_WORLD, s->fname.c_str(), NC_NOWRITE, MPI_INFO_NULL, &id),__LINE__,par.myrank);

    if (par.masterproc) std::cout << GOK << "Read header from " << CYAN <<  s->fname << RESET  << std::endl;

    ncwrap(ncmpi_inq(id, &ndims, &nvars, &ngatts, &unlimited),__LINE__,par.myrank);

    if constexpr(SERGHEI_DEBUG_INPUT_NETCDF){
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tNetCDF ndims: " << ndims << std::endl;
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tNetCDF nvars: " << nvars << std::endl;
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tNetCDF ngatts: " << ngatts << std::endl;
      std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "\tNetCDF unlimited: " << unlimited << std::endl;
    }

    // get the size of the domain
    getDimIds(par);

    if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) print_dimids();

    // read time dimension
    ncwrap(ncmpi_inq_dimlen(id, dimids[0], &dimsize),__LINE__,par.myrank);
    if(!allowTime && dimsize != 1){
      std::cerr << RERROR << "NetCDF input file " << s->fname << " has a time dimension of " << dimsize << " and it should be 1." << std::endl;
      std::cerr << RED << "Try using ncks to extract the time slice that you are interested, e.g.," << std::endl << "\tncks -d t,10,10,1 input.nc input.nc" << RESET << std::endl;
      return 0;
    }

    int varid;
    int err = ncmpi_inq_varid(id,"t",&varid);
    if(err != NC_NOERR) err = ncmpi_inq_varid(id,"time",&varid);
    if(err != NC_NOERR) ncwrap(err,__LINE__,par.myrank);

    MPI_Offset *index = new MPI_Offset[1];
    index[0] = 0;
    ncwrap(ncmpi_get_var1_double_all(id,varid,index,&dom.startTime),__LINE__,par.myrank);

    if(allowTime && par.masterproc) std::cout << BDASH << "Current time read from NetCDF input: " << dom.startTime << std::endl;

    // read x dimension
    ncwrap(ncmpi_inq_dimlen(id, dimids[2], &dimsize),__LINE__,par.myrank);
    dom.nx_glob = (int) dimsize;


    // read y dimension
    ncwrap(ncmpi_inq_dimlen(id, dimids[1], &dimsize),__LINE__,par.myrank);
    dom.ny_glob = (int) dimsize;
    if(par.masterproc) std::cout << GOK << "Domain dimensions: " << dom.nx_glob << " " << dom.ny_glob << std::endl;

    ndata = dom.nx_glob * dom.ny_glob;

    return 1;
  }

int readNetCDFcoordinates(const Parallel &par, Domain &dom){
  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
	int varid;
  char varname[NC_MAX_NAME+1];
  nc_type vtype;
  int var_ndims, var_natts;
  real dy;

	doubleArr data = realArr("databuffer",dom.nx_glob);

	ncwrap(ncmpi_inq_varid(id,"x",&varid),__LINE__,par.myrank);
  ncwrap(ncmpi_inq_var(id, varid, varname, &vtype, &var_ndims, dimids, &var_natts),__LINE__,par.myrank);

  ncwrap(ncmpi_get_var_double_all(id,varid,data.data()),__LINE__,par.myrank);

	// find the westmost corner, xll
	Kokkos::parallel_reduce( dom.nx_glob , KOKKOS_LAMBDA (int ii, double &x) {
       x = min(x,data(ii));
    } , Kokkos::Min<double>(dom.xll) );
	Kokkos::fence();

	dom.dxConst = data(1)-data(0);

	// the y-coordinates
	Kokkos::resize(data,dom.ny_glob);
	ncwrap(ncmpi_inq_varid(id,"y",&varid),__LINE__,par.myrank);
  ncwrap(ncmpi_get_var_double_all(id,varid,data.data()),__LINE__,par.myrank);
	// find the southmost corner, yll
	Kokkos::parallel_reduce( dom.ny_glob , KOKKOS_LAMBDA (int ii, double &y) {
       y =  min(y,data(ii));
    } , Kokkos::Min<double>(dom.yll) );

  dy = data(1) - data(0);

	Kokkos::fence();
	// find the cell vertex, instead of cell center
	dom.xll -= 0.5 * dom.dxConst;
	dom.yll -= 0.5 * dom.dxConst;

	if(par.masterproc){
    std::cout << GOK << "dx = " << dom.dx() << std::endl;
	  std::cout << GOK << "Domain extent : (" << dom.xll << ", " << dom.yll << ") (" << dom.xll + dom.dx() * dom.nx_glob << ", " << dom.yll + dom.dx() * dom.ny_glob<< ")" << std::endl;
    std::cout << GOK << "Coordinates read from " << CYAN << s->fname << RESET << std::endl;
  }

  // check if the coordinates match a raster/matrix indexing
  if(dom.dxConst < 0){ // usually not an issue, because raster indexing (west->east) matches Cartesian coordinates
    std::cerr << RERROR << "Order of x-coordinates in NetCDF file is incorrect." << std::endl;
    return 0;
  }
  if(dy > 0){ // may be an issue, because raster indexing is north-south, i.e., inverse of typical Cartesian coordinates
    dom.rasterIndexing = false;
  }

	return 1;
}


int readTime(const Parallel &par, Domain &dom, realArr &time){
  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

  int ncerr = varTime.findFieldData(par,dom);
  if(ncerr == NC_ENOTVAR){
    std::cerr << RERROR << "Variable '" << varTime.s->name << "' not found in atmforcing.nc" << std::endl;
    return SERGHEI_ERROR;
  }

  getDimIds(par);


  MPI_Offset dimsize;
	ncwrap(ncmpi_inq_dimlen(id, dimids[0], &dimsize),__LINE__,par.myrank);


  nTime = dimsize;
  MPI_Offset stt[1]; stt[0] = 0;
  MPI_Offset ctt[1]; ctt[0] = nTime;

  if constexpr(SERGHEI_DEBUG_INPUT_NETCDF) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << " nTimes = " << dimsize << std::endl;

  time = realArr("ncTime",nTime);
  ncwrap(ncmpi_get_vara_double_all(id,varTime.id,stt,ctt,time.data()),__LINE__,par.myrank);

  varTime.factor = Units::timeFactor(Units::parseTimeUnits(varTime.s->units));

  real factor = varTime.factor;
  Kokkos::parallel_for("convertTimeUnits", nTime, KOKKOS_LAMBDA(int ii){
    time(ii) *= factor;
  });

  if(par.masterproc) std::cout << GOK << "Read " << dimsize  << " time entries from NetCDF file " << CYAN << s->fname << RESET << std::endl;
  return SERGHEI_OK;
}


};

