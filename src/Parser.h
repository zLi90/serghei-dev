/* -*- mode: c++ -*- */
#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <sys/types.h>
#include <errno.h>
#include <sys/types.h>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include "define.h"
#include "string.h"
#include "Domain.h"
#include "FileIO.h"
#include "Parallel.h"
#include "SourceSink.h"
#include "rasterTools.h"
#include "Kokkos_StdAlgorithms.hpp"
#if SERGHEI_LPT
    #include "ParticleTracking.h"
#endif

class Parser {

class ParserLine{
public:
  std::string line;
  std::string key ;
  std::stringstream value;

  void lowercase(){
    std::for_each(line.begin(), line.end(), [](char & c) {
      c = ::tolower(c);
    });
  }

  void parse(){
    // make sure key and value are clean (in case of reuse)
    key.clear();
    value.clear();
     // Remove spaces and tabs from the line
     // line.erase (std::remove(line.begin(), line.end(), ' '), line.end());
     // line.erase (std::remove(line.begin(), line.end(), '\t'), line.end());

     // If the line isn't empty and doesn't begin with a comment specifier, split it based on the colon
     // normalize line: remove carriage returns
     line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

     // trim leading whitespace
     size_t first = line.find_first_not_of(" \t");
     if (first == std::string::npos) return; // line is empty or only whitespace

     std::string tmp = line.substr(first);

     // skip comment lines starting with //
     if (tmp.rfind("//", 0) == 0) return;

     // Find the colon; if none, treat as empty key (skip)
     size_t splitloc = tmp.find(':');
     if (splitloc == std::string::npos) return;

     // Store the key and value strings
     key = tmp.substr(0, splitloc);

     // Remove spaces and tabs from the key
     key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
     key.erase(std::remove(key.begin(), key.end(), '\t'), key.end());

     std::string val = tmp.substr(splitloc + 1);

     // Check for comments after values
     size_t splitter = val.find("//");
     std::string strloc = (splitter != std::string::npos) ? val.substr(0, splitter) : val;

     // trim value whitespace
     size_t vfirst = strloc.find_first_not_of(" \t");
     if (vfirst == std::string::npos) strloc.clear();
     else {
       size_t vlast = strloc.find_last_not_of(" \t");
       strloc = strloc.substr(vfirst, vlast - vfirst + 1);
     }

     // Transform the val into a string stream for convenience
     value.clear();
     value.str(strloc);
  }

  void print(){
     std::cout << GGD << "line: " << line << std::endl << "\tkey: " << key << "\tvalue: " << value.str() << std::endl;
   }
};

int checkValidOption(std::string mystring, std::set<std::string> myset){
  if( myset.count(mystring) != 0){
    return 1;
  }
  else{
    return 0;
  }
}
private:
    #if SERGHEI_LPT
    int nParticles=0;

    std::string Distribution;
    std::string Lifetime;
    std::string Resurrection;

    std::string Distribution1="random_coordinates";
    std::string Distribution2="random_cells";
    std::string Distribution3="random_polygon";
    std::string Distribution4="coordinates";

    std::string Lifetime1="random_lifetime";
    std::string Lifetime2="constant_lifetime";

    std::string Resurrection1="no_resurrection";
    std::string Resurrection2="resurrection";
    std::string Resurrection3="random_resurrection";
    std::string Resurrection4="resurrection_period";
    #endif

public:

  int readDimensions(Domain &dom, State &state, Parallel &par, FileIO &io){

  std::string fNameIn = io.inFolder;
	std::string tempStr;
 	tempStr = fNameIn + "parameters.input";
 	if(!readParamsFile(tempStr,dom,par,io)) return 0;
  #if SERGHEI_INPUT_NETCDF
    io.ncInputFilename = fNameIn + "input.nc";

    io.ncin.initialise(io.ncinS,io.ncInputFilename);
    //io.setNcFilename(tmpStr);
    if(par.masterproc) std::cout << BDASH << "Reading NetCDF input file " << CYAN << io.ncInputFilename << RESET << std::endl;
 	  if(!io.ncin.readNetCDFheader(par,dom,0)) return 0;
    if(!io.ncin.readNetCDFcoordinates(par,dom)) return 0;
  #else
	  tempStr = fNameIn + "dem.input";
 	  if(!readHeaderDEMFile(tempStr,dom,par));
  #endif

	return 1;

  }

  int readInputFiles (Domain &dom, State &state, SourceSinkData &ss,
		      ExternalBoundaries &ebc, Parallel &par, FileIO &io){

    std::string fNameIn = io.inFolder;
    int const Nfiles = 7;
    int ierr[Nfiles];
    std::string tempStr;

    for(int i=0; i<Nfiles; i++) ierr[i] = 1;

		#if SERGHEI_INPUT_NETCDF
      io.ncin.initialiseSurfaceVariables();
      if(par.masterproc) std::cout << BDASH << "Reading DTM from " << CYAN << io.ncin.s->fname << std::endl;
    	if(io.ncin.varz.read(par,dom,state) != NC_NOERR) return SERGHEI_ERROR;
      if(!Units::validateUnits("z",io.ncin.varz.s->units,"m")) return 0;
      if(par.masterproc) std::cout << GOK << "DTM ready" << std::endl;
		#else
			tempStr = fNameIn + "dem.input";
    	ierr[0] = readDEMFile(tempStr,dom,state,par);
		#endif

    #if SERGHEI_SWE_MODEL
      tempStr = fNameIn + "sw.input";
      ierr[1] = readSWFile(tempStr, dom, par, state, fNameIn, io);

      tempStr = fNameIn + "extbc.input";
      ierr[2] = readExtBCFile(tempStr, dom, ebc, par, state);

      io.infFile = fNameIn + "infiltration.input";
      ierr[3] = readInfiltrationFile(io, dom, ss.inf, par);

      std::cout << RED << __PRETTY_FUNCTION__ << ":: " << __LINE__ << RESET << std::endl;
      io.infMapFile = fNameIn +"infiltrationMap.input";
      ierr[4] = readInfiltrationMap(io, dom, ss.inf, par);

      std::cout << RED << __PRETTY_FUNCTION__ << ":: " << __LINE__ << RESET << std::endl;
    #if SERGHEI_NETCDF_FORCING
      io.aforcingS.fname = fNameIn + "atmforcing.nc";
      if(std::filesystem::exists(io.aforcingS.fname)){
      if(!ss.aforcing.initialise(par,dom,io.aforcingS,ss.rainRate)) return 0;
      }else{
        std::cerr << RERROR << "Atmospheric forcing " << CYAN << ss.aforcing.nc.s->fname << RESET << " file not found." << std::endl;
        return 0;
      }
    #else
      tempStr = fNameIn + "rainfall.input";
      #if SERGHEI_RAINFALL_POLYGONS
        ierr[5] = readRainByPolygons(tempStr, dom, ss, par);
      #else
        ierr[5] = readRainfallFile(tempStr, dom, ss.rain, par);
      #endif

      std::cout << RED << __PRETTY_FUNCTION__ << ":: " << __LINE__ << RESET << std::endl;
      // THIS CODE NEEDS TO BE DEPRECATED
      #ifdef _DEV_RAIN_
        if(par.masterproc){
        	int dim = ss.rain.nx * ss.rain.ny;
        	std::cerr << BDASH "rainfall partititioned in x-direction: "  << ss.rain.nx << "\n";
        	std::cerr << BDASH "rainfall partititioned in y-direction: "  << ss.rain.ny << "\n";

        	int t = 0;
        	int count = 0;
        	for (int i = 0; i < dim * ss.rain.np; i ++){
        	  std::cerr << BDASH "rainfall intensity (" << (t % dim) << ") nr. " << count << ": " << ss.rain.value(i) << "\n";
        	  t++;
        	  if ((t % dim) == 0)
        	  count ++;
        	}
        }
      #endif
    #endif

		#if SERGHEI_SCALAR_TRANSPORT
			io.conf.st.dir = fNameIn;
			if(!io.conf.st.read(par)) return 0;
			state.ade.addScalars(io.conf.st.nScalar);
		#endif

		#if SERGHEI_SEDIMENT_TRANSPORT
			io.conf.sedt.dir = fNameIn;
			if(!io.conf.sedt.read(par)) return 0;
			state.sediment.addSeds(io.conf.sedt.nSed, state.ade);
			if(!io.conf.sedt.addSedToScalarTransport(io.conf.st)) return 0;
		#endif
    #endif
    
	tempStr = fNameIn + "wind.input";
    ierr[6] = readWindFile(tempStr, dom, ss.wind, par);

        for (int i = 0; i < Nfiles; i++){
          if (!ierr[i]){
            if(par.masterproc) std::cerr << RERROR << "Terminating due to input error (look for error above)." << std::endl;
            return 0;
          }
        }

    Kokkos::fence();
    std::cout << GOK << "Input files reading completed " << std::endl;
    return 1;

  }

  int readParamsFile(std::string fNameIn, Domain &dom, Parallel &par, FileIO &io) {

    // Initialize all read-in values to -999
    dom.simLength = -999;
    dom.endTime = SERGHEI_NAN;
    dom.cfl       = -999;
    par.nproc_x   = -999;
    par.nproc_y   = -999;
    io.outFreq    = -999;
    io.obsFreq    = -999;
    std::string strAux;

    // Read in colon-separated key: value file line by line
    std::ifstream fInStream(fNameIn);
    std::string line;
    ParserLine pline;

    if (fInStream.is_open()){
      while (std::getline(fInStream, line)) {
    	pline.line = line;
    	pline.parse();
	//pline.print();

	// If the line was valid and a key is stored
	if(!pline.key.empty()){
	  // Match the key, and store the value
	  if (!strcmp("simLength", pline.key.c_str())) { pline.value >> dom.simLength; }
    else if (!strcmp("endTime", pline.key.c_str())) {pline.value >> dom.endTime;}
	  else if (!strcmp("cfl", pline.key.c_str())) { pline.value >> dom.cfl; }
	  else if (!strcmp("parNx", pline.key.c_str())) { pline.value >> par.nproc_x; }
	  else if (!strcmp("parNy", pline.key.c_str())) { pline.value >> par.nproc_y; }
	  else if (!strcmp("outFreq", pline.key.c_str())) { pline.value >> io.outFreq; }
	  else if (!strcmp("obsFreq", pline.key.c_str())) { pline.value >> io.obsFreq; }
	  else if (!strcmp("nScreen", pline.key.c_str())) { pline.value >> io.nScreen; }
	  else if (!strcmp("outFormat", pline.key.c_str())) { pline.value >> strAux; handleOutFormat(strAux,io,fNameIn,par); }
	  else if (!strcmp("BCtype", pline.key.c_str())) { pline.value >> strAux; handleBCtype(strAux,dom,fNameIn,par); }
    else if (!strcmp("writeRain", pline.key.c_str())) { pline.value >> io.writeRain;}
    else if (!strcmp("writeLandUse", pline.key.c_str())) { pline.value >> io.writeLandUse;}
    else if (!strcmp("writeSoilMap", pline.key.c_str())) { pline.value >> io.writeSoilMap;}
    else if (!strcmp("writeRoughness", pline.key.c_str())) { pline.value >> io.writeRoughness;}
    else if (!strcmp("writeRainAccum", pline.key.c_str())){
      if constexpr(SERGHEI_NETCDF_FORCING){
        pline.value >> io.writeRainA;
      }else{
        std::cout << YEXC << "writeRainAccum setting ignored because SERGHEI was not compiled with SERGHEI_NETCDF_FORCING on" << std::endl;
      }
    }
    else if (!strcmp("writeInfParameters", pline.key.c_str())) { pline.value >> io.writeInfParameters;}
	  else {
	  	 if(par.masterproc){
	    std::cerr << RERROR "key " << pline.key << " not understood in file " << fNameIn << "\n";
	    exit(-1);
		 }
	  }
	}
      }

    }else{
	 	if(par.masterproc){
      std::cerr<< RERROR "Unable to open " << fNameIn << "\n";
      return 0;
		}
    }


    // Test to make sure all values were initialized

    if (dom.simLength == -999) { if (par.masterproc) std::cerr << RERROR "key " << "simLength" << " not set.\n"; exit(-1); }
    if(dom.simLength == -999 && isnan(dom.endTime)){
      if (par.masterproc) std::cerr << RERROR "Neither simLength nor endTime were set. One must be set." << std::endl;
      return 0;
    }
    if (dom.cfl       == -999) { if (par.masterproc) std::cerr << RERROR "key " << "cfl"       << " not set.\n"; exit(-1); }
    if (par.nproc_x   == -999) { if (par.masterproc) std::cerr << RERROR "key " << "parNx"     << " not set.\n"; exit(-1); }
    if (par.nproc_y   == -999) { if (par.masterproc) std::cerr << RERROR "key " << "parNy"     << " not set.\n"; exit(-1); }
    if (io.outFreq    == -999) { if (par.masterproc) std::cerr << RERROR "key " << "outFreq"   << " not set.\n"; exit(-1); }
    if (io.obsFreq    == -999) { if (par.masterproc) std::cerr << RERROR "key " << "obsFreq"   << " not set.\n"; exit(-1); }
    if (io.outFormat  == -999) { if (par.masterproc) std::cerr << RERROR "key " << "outFormat" << " not set.\n"; exit(-1); }
    if (dom.BCtype    == -999) { if (par.masterproc) std::cerr << RERROR "key " << "BCtype"    << " not set.\n"; exit(-1); }

    // Print out the values
    if (par.masterproc) {
      std::cerr << BDASH "simLength: "  << dom.simLength << "\n";
      if (isnan(dom.endTime)){
        std::cerr << BDASH << "endTime: not set, will be calculated from simLength" << std::endl;
      }
      else{
        std::cerr << BDASH "endTime: "  << dom.endTime << std::endl;
      }
      std::cerr << BDASH "cfl: "        << dom.cfl       << "\n";
      std::cerr << BDASH "parNx: "      << par.nproc_x   << "\n";
      std::cerr << BDASH "parNy: "      << par.nproc_y   << "\n";
      std::cerr << BDASH "outFreq: "    << io.outFreq    << "\n";
      std::cerr << BDASH "obsFreq: "    << io.obsFreq    << "\n";
      std::cerr << BDASH "outFormat: "  << io.outFormat  << "\n";
      std::cerr << BDASH "BCtype: "     << dom.BCtype    << "\n";
    }

    if (par.masterproc){
      std::cerr << GOK "Parameters read\n";
    }

    return 1;

  }


 int readHeaderDEMFile(std::string fNameIn, Domain &dom, Parallel &par) {

    // Initialize all read-in values to -999 except for NODATA, which usually takes this value
	 dom.nx_glob 	= -999;
    dom.ny_glob  	= -999;
	 dom.xll       = -999;
    dom.yll   		= -999;
	 dom.dxConst   		= -999;

	 real nodata	= 123456789;
    std::string line;
    std::ifstream fInStream(fNameIn);

   std::string str;
	if (fInStream.is_open()){
    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> dom.nx_glob;

    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> dom.ny_glob;

    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> dom.xll;

    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> dom.yll;

    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> dom.dxConst;

    std::getline(fInStream,str,' ');
		std::getline(fInStream,str);
    std::stringstream(str) >> nodata;

	}else{
		if (par.masterproc) {
			std::cerr<<RERROR "Unable to open " << fNameIn << "\n";
			return 0;
		}
	}

    // Print out the values
    if (par.masterproc) {
      std::cerr << BDASH "nx_glob: " 	<< dom.nx_glob 	<< "\n";
      std::cerr << BDASH "ny_glob: "<< dom.ny_glob 	<< "\n";
      std::cerr << BDASH "xll: "    << dom.xll 	<< "\n";
      std::cerr << BDASH "yll: "    << dom.yll  << "\n";
      std::cerr << BDASH "dx: "     << dom.dxConst   << "\n";
    }

    // Test to make sure all values were initialized
	 //
    if (dom.nx_glob 		== -999) { if (par.masterproc) std::cerr << RERROR "" << "ncols" << " not set." << std::endl ; exit(-1); }
    if (dom.ny_glob 		== -999) { if (par.masterproc) std::cerr << RERROR "" << "nrows"       << " not set." << std::endl; exit(-1); }
    if (dom.xll      == -999) { if (par.masterproc) std::cerr << RERROR "" << "xll"   << " not set." << std::endl; exit(-1); }
    if (dom.yll      == -999) { if (par.masterproc) std::cerr << RERROR "" << "yll"   << " not set." << std::endl; exit(-1); }
    if (dom.dxConst       == -999) { if (par.masterproc) std::cerr << RERROR "" << "dx"   << " not set." << std::endl; exit(-1); }

    if(par.masterproc) std::cout << GOK << "Read DEM raster header." << std::endl;
	 return 1;

  }


  int readDEMFile(std::string fNameIn, Domain &dom, State &state, Parallel &par) {

    //Initialize isnodata in all memory cells
    Kokkos::deep_copy(state.isnodata, true);

		if(!readRasterField(fNameIn, dom, par, state.z)) return 0;

		Kokkos::parallel_reduce("init_z", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &ncell) {
			int ii = dom.getIndex(iGlob);
			if(isnan(state.z(ii))){
        state.z(ii) = SERGHEI_NAN;
			}else{
        state.isnodata(ii) = false;
        ncell++;
			}
		}, Kokkos::Sum<int>(dom.nCellValid));

		if (par.masterproc){
			std::cerr<< GOK "DEM ready" << std::endl;
		}
		return 1;
  }

  int readRoughnessFile(std::string fNameIn, Domain const &dom, State &state, Parallel &par) {

		int found = readRasterField(fNameIn, dom, par, state.roughness);

		if(!found){
			if (par.masterproc){
				std::cerr << YEXC << fNameIn << " not found" << std::endl;
				std::cerr << BDASH "A constant roughness is set " << std::endl;
			}
		}

		int err=0;
		Kokkos::parallel_reduce("init_roughness", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &err) {
 			int ii = dom.getIndex(iGlob);

			if(state.isnodata(ii)){
				state.roughness(ii) = NAN;
			}else{
        if(state.roughness(ii) < 0.0 ) err++;
      }
		}, Kokkos::Sum<int>(err));

		if(err > 0){
			if(par.masterproc){
				std::cerr << RERROR "There are negative roughness values in " << fNameIn << std::endl;
				return 0;
			}
		}

		if (par.masterproc) std::cout << GOK "Roughness set" << std::endl;

	 	return 1;
	}



	int readHiniFile(std::string fNameIn, Domain const &dom, State &state, Parallel &par) {

		int found = readRasterField(fNameIn, dom, par, state.h);

		if(!found){
			if (par.masterproc){
				std::cerr << YEXC << fNameIn << " not found" << std::endl;
				std::cerr << BDASH "A dry domain is set " << std::endl;
			}
		}

		int err=0;
		Kokkos::parallel_reduce("init_h", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &err) {
 			int ii = dom.getIndex(iGlob);
			if(state.h(ii) < 0.0 ) err++;
			if(state.isnodata(ii)) state.h(ii)=0.0;
		}, Kokkos::Sum<int>(err));

		if(err > 0){
			if(par.masterproc){
				std::cerr<< RERROR "There are negative depth values in " << fNameIn << std::endl;
				return 0;
			}
		}

		if (par.masterproc) std::cerr<<GOK "Initial water depth set" << std::endl;

	 	return 1;

	}


	int readUiniFile(std::string fNameIn, Domain const &dom, State &state, Parallel &par) {

		int found;
		const real constVel=0.0;

		found = readRasterField(fNameIn, dom, par, state.hu);
		if(!found){
			if (par.masterproc){
				std::cerr << YEXC << fNameIn << " not found" << std::endl;
				std::cerr << BDASH "A constant value of "<< constVel << " set for initial x-velocity" << std::endl;
			}
		}

		int err=0;
		Kokkos::parallel_reduce("init_hu", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &err) {
 			int ii = dom.getIndex(iGlob);
			if(!found) state.hu(ii) = constVel;
			if(state.isnodata(ii)){
				state.hu(ii) = NAN;
			}else{
				state.hu(ii) *= state.h(ii);
			}
		}, Kokkos::Sum<int>(err));

		if (par.masterproc) std::cerr<<GOK "Initial x-velocity (u) set" << std::endl;

	 	return 1;

	}

	int readViniFile(std::string fNameIn, Domain const &dom, State &state, Parallel &par) {

		int found;
		const real constVel=0.0;

		found = readRasterField(fNameIn, dom, par, state.hv);
		if(!found){
			if (par.masterproc){
				std::cerr << YEXC << fNameIn << " not found" << std::endl;
				std::cerr << BDASH "A constant value of "<< constVel << " set for initial y-velocity" << std::endl;
			}
		}

		int err=0;
		Kokkos::parallel_reduce("init_hu", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &err) {
 			int ii = dom.getIndex(iGlob);
			if(!found) state.hv(ii) = constVel;
			if(state.isnodata(ii)){
				state.hv(ii) = NAN;
			}else{
				state.hv(ii) *= state.h(ii);
			}
		}, Kokkos::Sum<int>(err));

		if (par.masterproc) std::cerr<<GOK "Initial y-velocity (v) set" << std::endl;

	 	return 1;

	}

// Reads infiltration data files
int readInfiltrationFile(FileIO &io, Domain &dom, InfiltrationModel &inf, Parallel &par){
  std::string modelName;
  std::string spatialInput;
  std::ifstream fInStream(io.infFile);
  std::string line;

  ParserLine pline;

  int jk,jfc,jf0;
  jk = jfc = jf0 = 1;

  if (fInStream.is_open()){
    while (std::getline(fInStream, line)) {
      pline.line = line;
      pline.lowercase();
      pline.parse();

      // If the line was valid and a key is stored
      if(!pline.key.empty()){
        // Match the key, and store the value
        if(!strcmp("infmodel",pline.key.c_str())){
          int check=0;
          pline.value >> modelName ;
          if(!modelName.compare("none")){
            inf.model = INF_NONE;
            check++;
          }
          if(!modelName.compare("constant")){
            inf.model = INF_CONSTANT;
            check++;
          }
          if(!modelName.compare("horton")){
            inf.model = INF_HORTON;
            check++;
          }
          if(!modelName.compare("greenampt")){
            inf.model = INF_GREENAMPT;
            check++;
          }
          #if SERGHEI_DEBUG_INFILTRATION
          std::cout << GGD << "modelName: " << modelName << std::endl;
          std::cout << GGD << "model: " << inf.model << std::endl;
          #endif
          if(check < 1){
            if(par.masterproc){
              std::cerr << RERROR << "Invalid infiltration model" << std::endl;
              return 0;
            }
          }else{
            if(par.masterproc)
            std::cout << GOK << "Infiltration model set to " << modelName << std::endl;
          }
        }

        if(!strcmp("spatialinput",pline.key.c_str())){
          pline.value >> spatialInput;
          if constexpr(SERGHEI_MAPCLASS){
            if(!spatialInput.compare("soilmap")){
              inf.spatial = inf.spatialSoilMap;
              if(par.masterproc) std::cout << BDASH << "Infiltration spatial input: soil map" << std::endl;
            }
            if(!spatialInput.compare("landuse")){
              inf.spatial = inf.spatialLanduse;
              if(par.masterproc) std::cout << BDASH << "Infiltration spatial input: land use map" << std::endl;
            }
          }
          if constexpr(SERGHEI_INPUT_NETCDF){
            if(!spatialInput.compare("netcdf")){
              inf.spatial = inf.spatialNetCDF;
              if(par.masterproc) std::cout << BDASH << "Infiltration spatial input: netcdf" << std::endl;
            }
          }
          if(!spatialInput.compare("raster")){
            inf.spatial = inf.spatialRaster;
            std::cout << BDASH << "Infiltration spatial input: raster" << std::endl;
            if constexpr(SERGHEI_MAPCLASS){
              if(par.masterproc) std::cerr << RERROR << "Invalid infiltration spatial input with SERGHEI_MAPCLASS." << std::endl;
              return SERGHEI_ERROR;
            }
          }
        }

        if(!strcmp("infiltrationclasses",pline.key.c_str())){pline.value >> inf.nLabels ;}
      }
    }
  }
  else{
    inf.model = INF_NONE;
    if (par.masterproc){
      std::cout << YEXC << io.infFile << " not found" << std::endl;
      std::cout << BDASH "Impervious domain set" << std::endl;
    }
    modelName = "none";
    return 1;
  }

  // now that the headers have been read, allocate stuff
  if(inf.model){

    #if SERGHEI_MAPCLASS
    //if constexpr(SERGHEI_MAPCLASS){
      if(inf.spatial == inf.spatialSoilMap) inf.soilmap.use = true;
			if(!readMapClasses(par,io,inf.soilmap,"soilmap.input")) return SERGHEI_ERROR;
      inf.nLabels = inf.soilmap.maxid; 
      inf.infLabel = colorArr( "infLabel", dom.nCellMem);
      return SERGHEI_OK;
    //}
    #endif

    if(inf.spatial != inf.spatialNetCDF){
      if(inf.nLabels <= 0){
        if(par.masterproc) std::cerr << RERROR "Number of infiltration classes must be larger than 0" << std::endl;
        return SERGHEI_ERROR;
      }
      inf.nLabels++;  // to account for label value 0 as impervious
      // allocate the infiltration map
      if(SERGHEI_DEBUG_INFILTRATION) std::cout << GGD << "Allocating infiltration label array with " << inf.nLabels << " labels." << std::endl;
      inf.infLabel = colorArr( "infLabel", dom.nCellMem);
      // TODO parallelisation
      if(inf.nLabels == 2){
        for(int ii=0; ii<dom.nCellMem; ii++){
          inf.infLabel(ii) = 1;
        }
      }

      if(inf.nLabels > 2){
        if(inf.spatial == inf.spatialNone){
          if(par.masterproc){
            std::cerr << RERROR << "More than a single infiltration class specified in infiltration.input, but no valid 'spatialInput' key provided.";
            std::cerr << "You provided the key " << RED << spatialInput << ". If this seems valid, check that it matches your compilation options." << std::endl;
          }
          return SERGHEI_ERROR;
        }
      }
    }

    if constexpr(!SERGHEI_MAPCLASS){
      if(inf.model == INF_CONSTANT){
        inf.constCap = realArr("constCap",inf.nLabels);
        inf.constCap(0) = 0.;
        for(int ii=1; ii<inf.nLabels; ii++) inf.constCap(ii) = NO_DATA;
      }

      if(inf.model == INF_HORTON){
        inf.k = realArr("k",inf.nLabels);
        inf.fc = realArr("fc",inf.nLabels);
        inf.f0 = realArr("f0",inf.nLabels);
        inf.k(0) = inf.fc(0) = inf.f0(0) = 0.;
        for(int ii=1; ii<inf.nLabels; ii++){
          inf.k(ii) = NO_DATA;
          inf.fc(ii) = NO_DATA;
          inf.f0(ii) = NO_DATA;
        }
      }

      // now read the data
      fInStream.clear();
      fInStream.seekg(0);
      if (fInStream.is_open()){
        while (std::getline(fInStream, line)) {
          pline.line = line;
          //std::cout << pline.line << std::endl;
          pline.lowercase();
          pline.parse();

          // If the line was valid and a key is stored
          if(!pline.key.empty()){
            // Match the key, and store the value
            if(inf.model == INF_CONSTANT){
              if(!strcmp("rate",pline.key.c_str())){
                pline.value >> inf.constCap(jfc)  ; jfc++;
              }
            }
            if(inf.model != INF_CONSTANT && inf.model != INF_NONE){
              if(inf.spatial == inf.spatialNetCDF){
                if(par.masterproc) std::cerr << RERROR << "Infiltration rate input via NetCDF is only valid for constant infiltration model." << std::endl;
                return SERGHEI_ERROR;
              }
            }
            // Horton model
            if(inf.model == INF_HORTON){
              if(!strcmp("k",pline.key.c_str())){pline.value >> inf.k(jk); jk++;}
              if(!strcmp("fc",pline.key.c_str())){ pline.value >> inf.fc(jfc); jfc++;}
              if(!strcmp("f0",pline.key.c_str())){ pline.value >> inf.f0(jf0); jf0++;}
            }
            if(inf.model == INF_GREENAMPT){
              // Green-Ampt model
              if(!strcmp("ks",pline.key.c_str())){ pline.value >> inf.ks  ; }
              if(!strcmp("psi",pline.key.c_str())){ pline.value >> inf.psi  ; }
              if(!strcmp("dtheta",pline.key.c_str())){ pline.value >> inf.dtheta  ;}
            }
          }
        }
      }
    }
    // at this point, everything has been read

    // rate unit conversion from mm/s -> m/s
    for(int ii=0; ii<inf.nLabels; ii++){
      if(inf.model == INF_CONSTANT) inf.constCap(ii) /= 1000.;
      if(inf.model == INF_HORTON){
        inf.fc(ii) /= 1000.;
        inf.f0(ii) /= 1000.;
      }
    }

    if(!inf.checkModel(par)){
      if(par.masterproc){
        std::cerr << RERROR << "Error when assigning infiltration model input." << std::endl;
        return 0;
      }
    }

    if (par.masterproc) std::cout << BDASH << "Infiltration classes: " << inf.nLabels << std::endl;
  }

  if(par.masterproc)  std::cout << GOK << "Infiltration model set" << std::endl;
  std::cout << RED << __PRETTY_FUNCTION__ << ":: " << __LINE__ << RESET << std::endl;
  return SERGHEI_OK;

}


/* Reads rainfall data file */
inline int readRainfallFile (std::string fNameIn, Domain &dom, TimeSeries &rain, Parallel &par){
  if constexpr(SERGHEI_DEBUG_WORKFLOW) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

    // TODO modify this reader to use a parsing strategy

    std::ifstream fInStream (fNameIn);
    std::string line;
    std::string tunits;
    std::string runits;

    real tfactor;
    real rfactor;
    int isok = 0; // flag to check if procedure completed as expected

    if (fInStream.is_open ()){
	     dom.isRain = 1;
	     rain.timeIndex = 0;

	     fInStream.ignore (256, ' ');
       fInStream >> tunits;
	     fInStream.ignore (256, ' ');
	     fInStream >> runits;
	     fInStream.ignore (256, ' ');
	     fInStream >> rain.np;
	     fInStream.ignore (256, ' ');
	     fInStream >> rain.nx;
	     fInStream.ignore (256, ' ');
	     fInStream >> rain.ny;

	     rain.time  = realArr ("raintime", rain.np);
	     rain.value = realArr ("rain",     rain.nx * rain.ny * rain.np);

	// ---------------------------------------------------------------------------
	// internally, the entire solver uses meters and seconds,
	// therefore, everything needs to be converted
	// ---------------------------------------------------------------------------

    if(!tunits.compare ("h")){
	    tfactor = 3600.0;  // hours to seconds
	    isok = 1;
	   }

	  if(!tunits.compare ("s")){
	    tfactor = 1.0;
	    isok = 1;
	  }

	  if(!isok){
	  	if(par.masterproc){
	    	std::cerr << RERROR "Invalid time units specified in rainfall file" << std::endl;
		  }
	    return 0;
	  }

	isok = 0;

	if (!runits.compare ("mm/h"))
	  {
	    rfactor = 0.001 / 3600.0; // mm/h to m/s
	    isok = 1;
	  }

	if (!runits.compare ("mm/s"))
	  {
	    rfactor = 0.001; // mm/s to m/s
	    isok = 1;
	  }

	if (!isok){
	  if(par.masterproc) std::cerr << RERROR "Invalid rainfall units specified in rainfall file" << std::endl;
	  return 0;
	}

	// ---------------------------------------------------------------------------
	// the units have been converted at this point
	// ---------------------------------------------------------------------------

	// ---------------------------------------------------------------------------
	/*

	   Read in the rain data values. the data is in row-first
	   order, meaning that the following raster values

	   +-----+-----+-----+-----+
	   | 0,0 | 1,0 | 2,0 | 3,0 |
	   +-----+-----+-----+-----+
	   | 1,0 | 1,1 | 2,1 | 3,1 |
	   +-----+-----+-----+-----+
	   | 2,0 | 1,2 | 2,2 | 3,2 |
	   +-----+-----+-----+-----+

	   The data (nx = 4, ny = 3) are given in the rain input file
	   as columns in the following order:

	   0,0 - 1,0 - 2,0 - 3,0 - 1,0 - 1,1 - 2,1 - 3,1 - 2,0 - 1,2 - 2,2 - 3,2

	   In the following loop, we map these values onto a 1D array
	   that appends the time series for each partition in the
	   order given above. The first np entries of the array belong
	   to the partition 0,0 and the next np entries belong to 1,0
	   and so on.

	   This means that the first value of the first column, that
	   maps to ti = 0 in the partition (0,0) is stored in
	   value(0). The first value of the second column that is the
	   value in partition (1,0) is stored in value(np). The first
	   value of the third column is stored in value(2 * np). The
	   first value of the nth column is stored in value((n-1) * np)

	   The nth value of the first column, corresponding to ti=n-1
	   is stored in value(n-1). The nth value of the second column
	   is stored in (2 * np + n - 1).

	   The nth value of the kth column is stored in:
	   (k - 1) * np + n - 1.

	*/
	// ---------------------------------------------------------------------------

	 for (int i = 0; i < rain.np; i++){
	    if (!fInStream.fail () && !fInStream.eof ()){
		      fInStream >> rain.time (i);
          rain.time (i) *= tfactor;
          for (int j = 0; j < rain.nx * rain.ny; j ++){
		          fInStream >> rain.value (i + j * rain.np);
		          rain.value (i + j * rain.np) *= rfactor;
//              std::cout << GGD << " t = " << rain.time(i) << "\tr = " <<  rain.value(i + j*rain.np) << std::endl;
		      }
      }
      else{
			if(par.masterproc){
		      std::cerr << RERROR "Error reading rainfall file" << std::endl;
		      return 0;
			}
	     }
	  }

	  fInStream.close();
  }
  else{
	   dom.isRain = 0;
  }

  if(par.masterproc) std::cerr << GOK "Rainfall set" << std::endl;

  return 1;
}
  /* Reads wind data file */
  inline int readWindFile (std::string fNameIn, Domain &dom, TimeSeries &wind, Parallel &par){
    std::ifstream fInStream (fNameIn);
    std::string line;

    if (fInStream.is_open ()){
	     dom.isWind = 1;
	     wind.timeIndex = 0;
	     fInStream.ignore (256, ' ');
	     fInStream >> wind.np;
	     fInStream.ignore (256, ' ');
	     fInStream >> dom.CwT;
	     fInStream.ignore (256, ' ');
	     fInStream >> dom.hwmin;
	     wind.time  = realArr ("wind", wind.np);
	     wind.value = realArr ("wind", 2 * wind.np);

         for (int i = 0; i < wind.np; i++)  {
             if (!fInStream.fail () && !fInStream.eof ()){
                 fInStream >> wind.time(i);
                 fInStream >> wind.value (i);
                 fInStream >> wind.value (i + wind.np);
             }
             else {
                 if (par.masterproc)    {
                     std::cerr << RERROR "Error reading wind file\n";   return 0;
                 }
             }
         }
         fInStream.close();
     }
     else{
         dom.isWind = 0;
     }
     if (par.masterproc) std::cerr << GOK "Wind set" << std::endl;

     return 1;
}

#if SERGHEI_RAINFALL_POLYGONS
inline int readRainByPolygons (std::string fNameIn, Domain &dom, SourceSinkData &ss, Parallel &par){

  // TODO modify this reader to use a parsing strategy

  std::string dir;
  dir = fNameIn.substr(0, fNameIn.length() - 14); // 14 chars equivalent to "rainfall.input" to get the dir

  std::ifstream fInStream (fNameIn);
  std::string tunits;
  std::string runits;
  std::string polfilename;
  std::string line;

  real tfactor;
  real rfactor;
  int isok; // flag to check if procedure completed as expected

  dom.isRain = 0; //Rain is inactive by default

  if (fInStream.is_open ()){
    fInStream.ignore (256, ' ');
    fInStream >> tunits;
    fInStream.ignore (256, ' ');
    fInStream >> runits;

    // ---------------------------------------------------------------------------
    // internally, the entire solver uses meters and seconds,
    // therefore, everything needs to be converted
    // ---------------------------------------------------------------------------
    isok = 0;
    if(!tunits.compare ("h")){
      tfactor = 3600.0;  // hours to seconds
      isok = 1;
    }else if(!tunits.compare ("s")){
      tfactor = 1.0;
      isok = 1;
    }
    if(!isok){
      if(par.masterproc){
        std::cerr << RERROR "Invalid time units specified in rainfall file" << std::endl;
        return 0;
      }
    }

    isok = 0;
    if(!runits.compare ("mm/h")){
      rfactor = 0.001 / 3600.0; // mm/h to m/s
      isok = 1;
    }else if (!runits.compare ("mm/s")){
      rfactor = 0.001; // mm/s to m/s
      isok = 1;
    }
    if(!isok){
      if(par.masterproc){
        std::cerr << RERROR "Invalid rainfall units specified in rainfall file" << std::endl;
        return 0;
      }
    }
    // ---------------------------------------------------------------------------

    //Read rain polygons
    fInStream.ignore (256, ' ');
    fInStream >> ss.nrainpol;

    int npol=ss.nrainpol;
    #if SERGHEI_DEBUG_RAINFALL
      std::cerr << GGD "Rain pol " << npol << std::endl;
    #endif

    if(npol>0){
      ///////////////////////////////////////// ACTIVATE RAIN
      dom.isRain = 1;
      ///////////////////////////////////////////////////////
      ss.rainSeries.resize(npol);

      int countpoints=0;
      for(int nr=0; nr<npol; nr++){
        TimeSeries &rain = ss.rainSeries[nr];

        fInStream >> line;
        polfilename = dir + line;
        #if SERGHEI_DEBUG_RAINFALL
          std::cerr << GGD "Rain file " << polfilename << std::endl;
        #endif

        std::ifstream fpolInStream (polfilename);
        if(fpolInStream.is_open ()){
          //---------------- Read polygon
          fpolInStream.ignore (256, ' ');
          fpolInStream >> rain.nver;

          int nvertex=rain.nver;
          rain.initPolygon(nvertex);

          for (int i = 0; i < nvertex; i++){
            if (!fpolInStream.fail () && !fpolInStream.eof ()){
              fpolInStream >> rain.xPoly(i) >> rain.yPoly(i);
              #if SERGHEI_DEBUG_RAINFALL
                std::cerr << GGD "xver: " << rain.xPoly(i) << " - yver: " << rain.yPoly(i) << std::endl;
              #endif
            }else{
              if(par.masterproc){
              std::cerr << RERROR "Error reading file " << polfilename << "\n";
              return 0;
              }
            }
          }

          //---------------- Read time series
          fpolInStream.ignore (256, ' ');
          fpolInStream >> rain.np;

          int np=rain.np;
          rain.initialise(np);
          countpoints+=np;

          for (int i = 0; i < np; i++){
            if (!fpolInStream.fail () && !fpolInStream.eof ()){
              fpolInStream >> rain.time(i);
              rain.time(i) *= tfactor;

              fpolInStream >> rain.value(i);
              rain.value(i) *= rfactor;

              #if SERGHEI_DEBUG_RAINFALL
                std::cerr << GGD "t rain: " << rain.time(i) << " - h rain: " << rain.value(i) << std::endl;
              #endif
            }else{
              if(par.masterproc){
              std::cerr << RERROR "Error reading file " << polfilename << "\n";
              return 0;
              }
            }
          }
          fpolInStream.close();
        }else{
          if(par.masterproc){
          std::cerr << RERROR "Rain polygon file " << polfilename << " not found\n";
          return 0;
          }
        }
      }
      ss.nrainpoints = countpoints;
    }

  }
  fInStream.close();

  if (par.masterproc) std::cerr << GOK "Rainfall set\n";
  return 1;
}
#endif



  int readExtBCFile(std::string fNameIn, Domain const &dom, ExternalBoundaries &ebc, Parallel &par, State &state) {
		#if SERGHEI_DEBUG_WORKFLOW
  	  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif
    std::ifstream fInStream(fNameIn);
    std::string dir;
    std::vector<std::string> polygonFile;
    std::vector<std::string> fullPathPoly;
    std::vector<std::string> hydrographFile;

    std::string file_header;
    std::string line;
    ParserLine pline;

    int nPoly;



    dir = fNameIn.substr(0, fNameIn.length() - 11); // 11 chars equivalent to "extbc.input" to get the dir

    int ibc = -2;
    int bccount = 0;
    int bccountFound = 0;
    if(fInStream.is_open()){
      while (std::getline(fInStream, line)) {
        pline.line = line;
        pline.parse();
				#if SERGHEI_DEBUG_BOUNDARY
				pline.print();
				#endif
        if(!pline.key.empty()){
          // we should read the number of boundaries here
	  			if (!strcmp("bccount", pline.key.c_str())){
	    			pline.value >> bccount;
            bccountFound = 1;
            if(bccount < 1){
              std::cout << YEXC << "extbc.input indicates zero external boundaries." << std::endl;
              return 1;
            }
						#if SERGHEI_DEBUG_BOUNDARY
	    				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET;
	    				std::cout << "Resizing boundary vector to " << bccount << " boundaries."  << std::endl;
	    				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET;
	    				std::cout << "ibc = " << ibc << " boundaries."  << std::endl;
						#endif
            ebc.extbc.resize(bccount);
            ebc.id.resize(bccount);
            polygonFile.resize(bccount);
            fullPathPoly.resize(bccount);
            hydrographFile.resize(bccount);
            ibc++;    // ibc should be set to -1
      			#if SERGHEI_DEBUG_BOUNDARY
      				std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET;
      				std::cout << "Resized to " << bccount << std::endl;
      			#endif
	  			}
          else if(!strcmp("id",pline.key.c_str())){
            ibc++;
            if(bccount > 0 && ibc >= 0) pline.value >> ebc.id[ibc];
          }
          else if(!strcmp("bctype", pline.key.c_str()) && ibc >=0 ) {if(bccount > 0) pline.value >> ebc.extbc[ibc].bctype;}
          else if(!strcmp("polygon", pline.key.c_str()) && ibc >=0 ) {if(bccount > 0) pline.value >> polygonFile[ibc];}
          else if (!strcmp("direction", pline.key.c_str()) && ibc >=0 ) {if(bccount > 0) pline.value >> ebc.extbc[ibc].normalx >> ebc.extbc[ibc].normaly;}
          else if(!strcmp("bcvals", pline.key.c_str()) && ibc >=0 ){
              if(bccount > 0 ){
                ebc.extbc[ibc].bcvals = realArr("bcvals", 3);
                pline.value >> ebc.extbc[ibc].bcvals(0) >> ebc.extbc[ibc].bcvals(1) >> ebc.extbc[ibc].bcvals(2);
              }
          }
          else if(!strcmp("hydrograph",pline.key.c_str()) && ibc >=0){
            pline.value >> hydrographFile[ibc];
          }
          else if(ibc >= 0){
			 	       if(par.masterproc){
					            std::cerr << RERROR << "In extbc.input: Key " << pline.key << " not understood." << std::endl;
					            return 0;
				       }
          }
          else if(ibc < -1){
           if(par.masterproc){
              std::cerr << RERROR << "No boundaries defined in extbc.input, number of boundaries not defined, or 'id' key not found." << std::endl;
              return 0;
               }
          }
        }
      } //end while
      fInStream.close();
      if(!bccountFound){
			     std::cerr << RERROR << "Number of boundaries not defined in extbc.input. Please define 'bccount'" << std::endl;
					 return 0;
      }
    }
    else{
	 	if(par.masterproc)
    	std::cerr << YEXC << "extbc.input not found. Default boundaries used." << std::endl;
    }

    #if SERGHEI_DEBUG_BOUNDARY
      for (int k = 0; k < ebc.extbc.size(); k++) {
        std::cerr << GGD "Boundary type for " << k << "th boundary: " << ebc.BoundaryTypes[ebc.extbc[k].bctype] << " (" << ebc.extbc[k].bctype << ")." << std::endl;
        std::cerr << GGD << "Direction for " << k << "th boundary: (" << ebc.extbc[k].normalx << "," << ebc.extbc[k].normaly << ")" << std::endl;
        std::cerr << GGD << "Polygon file for " << k << "th boundary: " << polygonFile[k] << std::endl;
        std::cerr << GGD << "Hydrograph file for " << k << "th boundary: " << hydrographFile[k] << std::endl;
      }
    #endif
    if(par.masterproc){
      if(ibc+1 < ebc.extbc.size()){
			 if(par.masterproc){

        std::cerr << RERROR << "Expected " << ebc.extbc.size() << " boundary condition input blocks, but only found " << ibc+1 << " in extbc.input." << std::endl;
		  	return 0;
		  }
      }
      for (int k = 0; k < ebc.extbc.size(); k++) {
			if(par.masterproc){
        std::cout << GOK <<  "External boundary " << k << " set to " << ebc.BoundaryTypes[ebc.extbc[k].bctype] << " (" << ebc.extbc[k].bctype << ")"<< std::endl;
		  }
      }
    }
    // normalise BC normal vectors by magnitude
    for (int k = 0; k < ebc.extbc.size(); k++) {
      real mod = sqrt(ebc.extbc[k].normalx*ebc.extbc[k].normalx +ebc.extbc[k].normaly*ebc.extbc[k].normaly);
      ebc.extbc[k].normalx /= mod;
      ebc.extbc[k].normaly /= mod;
    }

    // read polygon files
    for (int k = 0; k < polygonFile.size(); k ++) {
      fullPathPoly[k] = dir + polygonFile[k];
      std::ifstream fPoly(fullPathPoly[k]);
      // read in kth polygon
      if (fPoly.is_open()) {
        fPoly.ignore(256,' ');
        fPoly >> nPoly;
        realArr xPoly=realArr( "xPoly" , nPoly );
        realArr yPoly=realArr( "yPoly" , nPoly );
        for (int i=0; i<nPoly; i++) {
          if (!fPoly.fail() && !fPoly.eof()) {
            fPoly >> xPoly(i) >> yPoly(i);
            #if SERGHEI_DEBUG_BOUNDARY
              std::cout << GGD << "extbc polygon " << k << ". Point " << i << "/" << nPoly << "\t" << xPoly(i) << "\t" << yPoly(i) << std::endl;
            #endif
          }
          else{
			 	       if(par.masterproc){
					            std::cerr<< RERROR "Error reading boundary polygon file " << k << ": " << fullPathPoly[k] << std::endl;
					            return 0;
				       }
          }
        }
        if(!ebc.extbc[k].find_bcells(state, ebc.id[k], dom, par, nPoly, xPoly, yPoly)) return 0;
      }
      else{
        if(par.masterproc){
          std::cerr<< RERROR "Polygon file " << k << ": " << fullPathPoly[k] << " not found." << std::endl;
        	return 0;
		    }
      }
      fPoly.close();
	  } // end for read in of the kth polygon


      // read hydrograph files
      for (int k = 0; k < hydrographFile.size(); k ++) {
        std::string fname = dir + hydrographFile[k];
        std::ifstream fHydro(fname);
        int ndata=0;

				// read in kth polygon
				int readHydro=0;
				switch(ebc.extbc[k].bctype){
					case SWE_BC_Q_T:
					case SWE_BC_HZ_T_INLET:
					case SWE_BC_HZ_T_OUTLET:
						readHydro = 1;
						break;
				}
				if(readHydro){
					#if SERGHEI_DEBUG_BOUNDARY
						std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "Reading BC hydrograph file " << k << ": " << fname << std::endl;
					#endif
					if(fHydro.is_open()) {
            fHydro >> file_header >> ndata;
						if (ndata > 0){
							ebc.extbc[k].hydrograph.initialise(ndata);
						}
						for(int i=0; i<ndata; i++) {
							if (!fHydro.fail() && !fHydro.eof()) {
								fHydro >> ebc.extbc[k].hydrograph.time(i) >> ebc.extbc[k].hydrograph.value(i);
								#if SERGHEI_DEBUG_BOUNDARY
									std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << ebc.extbc[k].hydrograph.time(i) << "\t" << ebc.extbc[k].hydrograph.value(i) << std::endl;
								#endif
							}
							else{
								if(par.masterproc){
									std::cerr<< RERROR "Error reading hydrograph file for boundary " << k << ": " << hydrographFile[k] << std::endl;
									return 0;
								}
							}
						} // end for ndata
						fHydro.close();
					}else{
						if(par.masterproc){
							std::cerr << RERROR "Error opening hydrograph file " << fname << std::endl;
							return 0;
						}
					}
				}
      } // end for hydrogaph files

/*
	 for (int k = 0; k < ebc.extbc.size(); k ++){
			if(ebc.extbc[k].bctype == SWE_BC_Q_T) ebc.extbc[k].computeLength(dom);
	 }
*/


	 for (int k = 0; k < ebc.extbc.size(); k ++) {
		  int value=0;
		  switch (ebc.extbc[k].bctype){
			case SWE_BC_CRITICAL:
			case SWE_BC_H_CONST:
			case SWE_BC_WSE_CONST:
			case SWE_BC_FREE_OUTFLOW:
			case SWE_BC_HZ_T_OUTLET:
				value=-1*(k+1);
				break;

			case SWE_BC_Q_CONST:
			case SWE_BC_HZ_T_INLET:
			case SWE_BC_Q_T:
				value=k+1;
			break;

		  default:
			 std::cerr << RERROR " Boundary type: " << ebc.extbc[k].bctype << " not recognised." << std::endl;
			 exit(EXIT_FAILURE);

			}

			ebc.extbc[k].setIsBound(state,value);

		}

  if(par.masterproc) std::cout << GOK << "External boundary file parsed and boundaries set" << std::endl;
  return 1;
};


  int readSWFile(std::string fNameIn, Domain &dom, Parallel &par, State &state, std::string fDirIn, FileIO &io){
	  if constexpr(SERGHEI_DEBUG_WORKFLOW) std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;

    std::ifstream fInStream(fNameIn);
    std::string line;
    ParserLine pline;
    std::string tempStr;

    io.sw.initialMode="dry";
    io.sw.frictionModel="none";

    if (fInStream.is_open()){
      while (std::getline(fInStream, line)) {
        pline.line = line;
	      pline.lowercase();
        pline.parse();

        // If the line was valid and a key is stored
	      if(!pline.key.empty()){
          // Match the key, and store the value
          if(!strcmp("initialmode",pline.key.c_str())){ pline.value >> io.sw.initialMode; }
	        if(!strcmp("initialvalue",pline.key.c_str())){ pline.value >> io.sw.initialValue ; }
          if(!strcmp("friction",pline.key.c_str())){ pline.value >> io.sw.frictionModel  ; }
          if(!strcmp("roughness",pline.key.c_str())){ pline.value >> io.sw.roughnessInput  ; }
          if(!strcmp("drydepth",pline.key.c_str())){ pline.value >> io.sw.hmin  ; }
        }
      }
    }else{
      if (par.masterproc) std::cerr << RERROR "File " << fNameIn << " not found" << std::endl;
      return 0;
    }

		fInStream.close();
    // Initialise roughness
    if(!checkValidOption(io.sw.frictionModel, io.sw.frictionModels)){
	    if(par.masterproc){
        std::cerr << RERROR "Invalid friction model. Please correct sw.input" << std::endl;
        return 0;
		  }
    }

    // assign dry threshold water depth to zero velocities
    #if SERGHEI_SWE_DRY_RUNOFF_START_DT
    if(io.sw.hmin <= 0){
    #else
    if(io.sw.hmin < 0){
    #endif
      std::cerr << RERROR "Invalid or missing runoff depth tolerance" << std::endl;
      return 0;
    }
    state.hmin = io.sw.hmin;

    bool consistency=1;
    bool keyWordFound=0;

    if(io.sw.frictionModel.compare("none") != 0){
      // consistency check
      switch(SERGHEI_FRICTION_MODEL){
        case SERGHEI_FRICTION_MANNING:
          if(io.sw.frictionModel.compare("manning")) consistency=0 ;
          break;
        case SERGHEI_FRICTION_DARCYWEISBACH:
          if(io.sw.frictionModel.compare("darcyweisbach")) consistency=0;
          break;
        case SERGHEI_FRICTION_CHEZY:
          if(io.sw.frictionModel.compare("chezy")) consistency=0;
          break;
      }
      if(!consistency){
        std::cerr << RERROR "Friction model in .sw file is inconsistent with SERGHEI_FRICTION_MODEL compilation flag" << std::endl;
        return 0;
      }

      io.sw.roughness  = atof(io.sw.roughnessInput.c_str());
      if constexpr(SERGHEI_INPUT_NETCDF){
        if(!io.sw.roughnessInput.compare("netcdf")){
          keyWordFound = 1;
          int varfound=1;
          if(par.masterproc) std::cout << BDASH << "Reading hydraulic roughness from " << CYAN << io.ncin.s->fname << std::endl;
          varfound = io.ncin.varn.read(par,dom,state);
          if(varfound == NC_ENOTVAR){
            if(par.masterproc) std::cout << RERROR << "Roughness not found in NetCDF input file " << CYAN << io.ncin.s->fname << RESET << std::endl;
            return SERGHEI_ERROR;
          }
        }
      }else{
        if(!io.sw.roughnessInput.compare("file")){
          keyWordFound = 1;
          tempStr = fDirIn + "roughness.input";
          if(!readRoughnessFile(tempStr,dom,state,par)) return 0;
        }
      }
      if constexpr(SERGHEI_MAPCLASS){
        if(!io.sw.roughnessInput.compare("landuse")){
          keyWordFound = 1;
          if(par.masterproc) std::cout << BDASH << "Roughness will be assigned from land use type." << std::endl;
          io.readLandUse = 1; // so that the land use gets read elsewhere
          state.landuse.use = true;
        }
      }
      if(!keyWordFound){
        if(io.sw.roughness < 0 ){
		  	  if(par.masterproc){
            std::cerr << RERROR "Negative friction coefficient. Please correct sw.input" << std::endl;
			 	    return 0;
			    }
        }
	      real roughness = io.sw.roughness;
        Kokkos::parallel_for("init_roughness", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
          int ii = dom.getIndex(iGlob);
          state.roughness(ii) = roughness;
        });
        if (par.masterproc){
          std::cout << BDASH "Friction coefficient set constant to " << roughness << std::endl;
        }
      }
    }


    if(!checkValidOption(io.sw.initialMode, io.sw.initialModes)){
	 	  if(par.masterproc){
        std::cerr << RERROR "Invalid initial SW mode. Please correct sw.input" << std::endl;
        return 0;
		  }
    }

    if(!io.sw.initialMode.compare("file")){
      tempStr = fDirIn + "hini.input";
      if(!readHiniFile(tempStr,dom,state,par)) return 0;
      tempStr = fDirIn + "uini.input";
      if(!readUiniFile(tempStr,dom,state,par)) return 0;
      tempStr = fDirIn + "vini.input";
	    if(!readViniFile(tempStr,dom,state,par)) return 0;
    }
		#if SERGHEI_INPUT_NETCDF
    else if(!io.sw.initialMode.compare("netcdf")){
      if(par.masterproc) std::cout << BDASH << "Reading initial surface state from " << CYAN << io.ncin.s->fname << std::endl;
      int varfound=1;

      varfound = io.ncin.varh.read(par,dom,state);
      // if variable not found, set to zero (nothing needs to be done, as variables are initialised to zero at allocation)
      if(varfound != NC_NOERR){
        if(varfound == NC_ENOTVAR){
          if(par.masterproc) std::cout << YEXC << "Water depth (" << YELLOW << "h" << RESET << ") not found in NetCDF input file. Domain will be set dry" << std::endl;
        }else{
          return SERGHEI_ERROR;
        }
      }

      varfound = io.ncin.varu.read(par,dom,state);
      if(varfound != NC_NOERR){
        if(varfound == NC_ENOTVAR){
          if(par.masterproc) std::cout << YEXC << "x-velocity (" << YELLOW << "u" << RESET << ") not found in NetCDF input file. It will be set to zero" << std::endl;
        }else{
          return SERGHEI_ERROR;
        }
      }

      varfound = io.ncin.varv.read(par,dom,state);
      if(varfound != NC_NOERR){
        if(varfound == NC_ENOTVAR){
          if(par.masterproc) std::cout << YEXC << "y-velocity (" << YELLOW << "v" << RESET ") not found in NetCDF input file. It will be set to zero" << std::endl;
        }else{
          return SERGHEI_ERROR;
        }
      }

      int err;
		  Kokkos::parallel_reduce("validate_init", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &hzero) {
        int ii = dom.getIndex(iGlob);
        hzero=0;
        if(state.isnodata(ii)){
          state.h(ii) = state.hu(ii) = state.hv(ii) = 0.;
        }
        else{
          if(state.h(ii) < 0.) hzero++;
        }
		  }, Kokkos::Sum<int>(err));
      if(err){
        std::cerr << RERROR << "There are " << err << " cells with negative depths in initial condition in NetCDF file" << std::endl;
        return 0;
      }
    }
		#endif
    else{
      if(!io.sw.initialMode.compare("dry")) io.sw.initialValue = 0.;
      real initialValue = io.sw.initialValue;
      Kokkos::parallel_for("set_init_dry", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        state.h(ii) = initialValue;
        state.hu(ii) = state.hv(ii) = 0.;
      });
    }

    if(!io.sw.initialMode.compare("h+z")){
      Kokkos::parallel_for("set_init_h+z", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        state.h(ii) -= state.z(ii);
        if(state.h(ii) < 0) state.h(ii) = 0.;
      });
    }

		#if SERGHEI_EROSIVE_SHEAR
	    Kokkos::parallel_for(dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
        int ii = dom.getIndex(iGlob);
        state.shearAccum(ii) = 0.0;
        state.phiTot(ii) = 0.0;
			});
		#endif

		#if SERGHEI_SEDIMENT_TRANSPORT
	    Kokkos::parallel_for(dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
        int ii = dom.getIndex(iGlob);
        state.zini(ii) = state.z(ii);
			});
		#endif

		if (par.masterproc){
			std::cerr<<GOK "Shallow water initial condition set" << std::endl;
			std::cerr<<GOK "Roughness and friction set" << std::endl;
		}

    return 1;
  }





  int createOutputDir(std::string &fNameOut) {

	if (mkdir(fNameOut.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
	{
    if(fNameOut[fNameOut.length()-1] != '/'){
      fNameOut += "/";
    }
		 if( errno == EEXIST ) {
			 // already exists
			 std::cerr << RERROR "Output folder already exists. Please delete it before continue. Error:" << strerror(errno) << std::endl;
			 	return 0;
		 } else {
			 // something else
			  std::cerr << RERROR "Cannot create output folder. Error:" << strerror(errno) << std::endl;
		 		return 0;
		 }
	}

	std::cerr << GOK "Output folder created\n";

	return 1;

  }


  void handleOutFormat(std::string &str, FileIO &io, std::string &fNameIn, Parallel &par) {
    // TODO this could be handled with ParserLine::parse
    size_t splitloc = str.find("//",0);
    std::string strloc;
    if (splitloc != std::string::npos){
      strloc = str.substr(0,splitloc);
    } else {
      strloc = str;
    }

    if      ( !strcmp(strloc.c_str(),"NETCDF") ) { io.outFormat = OUT_NETCDF; }
    else if ( !strcmp(strloc.c_str(),"VTK"  ) ) { io.outFormat  = OUT_VTK ; }
    else if ( !strcmp(strloc.c_str(),"BIN"  ) ) { io.outFormat  = OUT_BIN ; }
    else  {
	 	if(par.masterproc){

      std::cerr << RERROR "unrecognized outFormat " << strloc.c_str() << " in file " << fNameIn << "\n";
      exit(-1);
		}
    }
  }

  void handleBCtype(std::string &str, Domain &dom, std::string &fNameIn, Parallel &par) {
    size_t splitloc = str.find("//",0);
    std::string strloc;
    if (splitloc != std::string::npos){
      strloc = str.substr(0,splitloc);
    } else {
      strloc = str;
    }
    if      ( !strcmp(strloc.c_str(),"PERIODIC") ) { dom.BCtype = SWE_BC_PERIODIC; }
    else if ( !strcmp(strloc.c_str(),"REFLECTIVE"  ) ) { dom.BCtype  = SWE_BC_REFLECTIVE ; }
    else if ( !strcmp(strloc.c_str(),"TRANSMISSIVE"  ) ) { dom.BCtype  = SWE_BC_TRANSMISSIVE ; }
    else  {
	 	if(par.masterproc){
        std::cerr << RERROR "unrecognized BCtype " << strloc.c_str() << " in file " << fNameIn << "\n";
        exit(-1);
      }
      }
    }


    #if SERGHEI_MAPCLASS
      int readMapClasses(const Parallel &par, FileIO &io, MapClass &map, std::string filename){
        if(!map.use) return SERGHEI_OK;
        intArr tmpid;
        realArr tmpval;

        std::string fname = io.inFolder + filename;
        std::ifstream fMap(fname);
        if(fMap.is_open()){
          fMap.ignore(256,' ');
          fMap >> map.nClass;

          tmpid = intArr("tmpTableID",map.nClass);
          tmpval = realArr("tmpTableVal",map.nClass);

          for (int i=0; i<map.nClass; i++) {
            if (!fMap.fail() && !fMap.eof()) {
              fMap >> tmpid(i) >> tmpval(i);
              if(tmpid(i) > map.maxid) map.maxid=tmpid(i);
            }
          }
        }else{
          if(par.masterproc) std::cerr << RERROR << "Map classes file " << CYAN << fname << RESET << " not found." << std::endl;
          return SERGHEI_ERROR;
        }

        map.table = realArr("mapTable",map.maxid+1);
        Kokkos::deep_copy(map.table, SERGHEI_NAN);


        Kokkos::parallel_for("createMapTable",map.nClass, KOKKOS_LAMBDA(int i){
          map.table(tmpid(i)) = tmpval(i);
        });

          for(int i=0; i<map.nClass; i++){
            map.table(tmpid(i)) = tmpval(i);
            if constexpr(SERGHEI_DEBUG_MAPCLASS) std::cout << i << "\ttmpid " << tmpid(i) << "\ttmpval " << tmpval(i)  << std::endl;
          }
          if constexpr(SERGHEI_DEBUG_MAPCLASS) for(int i=0; i<map.maxid+1; i++) std::cout << i << "\ttable " << map.table(i) << std::endl;

        if(par.masterproc) std::cout << GOK << "Map classes (" << map.nClass << ") with maximum class ID " << map.maxid << " read from " << CYAN << fname << RESET << std::endl;
        return SERGHEI_OK;
      }


    int readSoilMap(const Parallel &par, FileIO &io, Domain &dom, SourceSinkData &ss, State &state){
      if(!ss.inf.soilmap.use) return SERGHEI_OK;

      if(par.masterproc) std::cout << BDASH << "Reading soil map from " << CYAN << io.ncin.s->fname << std::endl;
        int varfound=1;

        varfound = io.ncin.varSoilMap.readField(par,dom,ss.inf.soilmap.mapClass);
        if(varfound != NC_NOERR){
          if(varfound == NC_ENOTVAR && par.masterproc){
            std::cout << YEXC << "Soil class  (" << YELLOW << "soilID" << RESET << ") not found in NetCDF input file." << std::endl;
          }
          return SERGHEI_ERROR;
        }

        if(!ss.inf.soilmap.validate(dom,state.isnodata)) return SERGHEI_ERROR;
        if(par.masterproc) std::cout << GOK << "Soil map and infiltration classes validated" << std::endl << std::endl;
        return SERGHEI_OK;
      }


      int readLandUseClasses(const Parallel &par, FileIO &io, State &state){
        if(!state.landuse.use) return SERGHEI_OK;
        intArr tmpid;
        realArr tmpval;

        std::string fname = io.inFolder + "landuse.input";
        std::ifstream fLandUse(fname);
        if(fLandUse.is_open()){
          fLandUse.ignore(256,' ');
          fLandUse >> state.landuse.nClass;

          tmpid = intArr("tmpTableID",state.landuse.nClass);
          tmpval = realArr("tmpTableVal",state.landuse.nClass);

          for (int i=0; i<state.landuse.nClass; i++) {
            if (!fLandUse.fail() && !fLandUse.eof()) {
              fLandUse >> tmpid(i) >> tmpval(i);
              if(tmpid(i) > state.landuse.maxid) state.landuse.maxid=tmpid(i);
            }
          }
        }else{
          if(par.masterproc) std::cerr << RERROR << "Land use classes file " << CYAN << fname << RESET << " not found." << std::endl;
          return SERGHEI_ERROR;
        }

        state.landuse.table = realArr("landUseTable",state.landuse.maxid+1);
        Kokkos::deep_copy(state.landuse.table, SERGHEI_NAN);


        Kokkos::parallel_for("createLandUseTable",state.landuse.nClass, KOKKOS_LAMBDA(int i){
          state.landuse.table(tmpid(i)) = tmpval(i);
        });
        /*
        for(int i=0; i<state.landuse.nClass; i++){
          state.landuse.table(tmpid(i)) = tmpval(i);
          std::cout << i << "\ttmpid " << tmpid(i) << "\ttmpval " << tmpval(i)  << std::endl;
        }
        for(int i=0; i<state.landuse.maxid+1; i++){
          std::cout << i << "\ttable " << state.landuse.table(i) << std::endl;
        }
        */

        if(par.masterproc) std::cout << GOK << "Land use classes (" << state.landuse.nClass << ") with maximum class ID " << state.landuse.maxid << " read from " << CYAN << fname << RESET << std::endl;
        return SERGHEI_OK;
      }



      int readLandUse(const Parallel &par, FileIO &io, Domain &dom, State &state){

        if(!io.readLandUse) return SERGHEI_OK;
        if(!state.landuse.use) return SERGHEI_OK;

        if(par.masterproc) std::cout << BDASH << "Reading land use map from " << CYAN << io.ncin.s->fname << std::endl;

        int varfound=1;

        varfound = io.ncin.varLandUse.readField(par,dom,state.landuse.mapClass);
        if(varfound != NC_NOERR){
          if(varfound == NC_ENOTVAR && par.masterproc){
            std::cout << YEXC << "Land use class  (" << YELLOW << "landuse" << RESET << ") not found in NetCDF input file. " << std::endl;
          }
          return SERGHEI_ERROR;
        }

        if(!state.landuse.validate(dom,state.isnodata)) return SERGHEI_ERROR;
        if(par.masterproc) std::cout << GOK << "Land use map and classes validated" << std::endl << std::endl;

        return SERGHEI_OK;
      }

    #endif

    #if SERGHEI_INPUT_NETCDF
    int readInfiltrationRateNetCDF(FileIO &io, Domain const &dom, InfiltrationModel &inf, Parallel const &par){
	  if(inf.model == INF_NONE) return SERGHEI_OK;
      if(inf.spatial != inf.spatialNetCDF) return SERGHEI_OK;
      if(par.masterproc) std::cout << BDASH << "Reading infiltration map from " << CYAN << io.ncInputFilename << RESET << std::endl;

      int varfound = io.ncin.varInfRateCap.readFieldExtended(par,dom,inf.capacity);
      if(varfound != NC_NOERR){
        if(varfound == NC_ENOTVAR && par.masterproc){
          std::cout << YEXC << "Infiltration rate/capacity (" << YELLOW << "infRate" << RESET << ") not found in NetCDF input file." << std::endl;
        }
        return SERGHEI_ERROR;
      }
      io.ncin.varInfRateCap.factor = Units::rainFactor(io.ncin.varInfRateCap.s->units);
      real factor = io.ncin.varInfRateCap.factor;
      Kokkos::parallel_for("infRateFactor",dom.nCellMem, KOKKOS_LAMBDA (int ii){
        inf.capacity(ii) *= factor;
      });
      Kokkos::fence();
      return SERGHEI_OK;
    }
    #endif

    int readInfiltrationMap(FileIO const &io, Domain const &dom, InfiltrationModel &inf, Parallel &par) {
      if(inf.spatial != inf.spatialRaster) return SERGHEI_OK;


      #if SERGHEI_DEBUG_INFILTRATION
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "inf.Model = " << inf.model << "\tinf.nLabels = " << inf.nLabels << std::endl;
      #endif
      if(!inf.model) return 1;
      if(inf.nLabels == 1) return 1; // only impervious

      realArr tmpVar=realArr("var", dom.ny_glob*dom.nx_glob );
      std::ifstream fInStream(io.infMapFile);
      std::string line;

    int tnx,tny;
    real txll,tyll,tdx;
    real nodata;

    tnx=-999;
    tny=-999;
    txll=-999;
    tyll=-999;
    tdx=-999;
    nodata = 123456789;

    int ndata=dom.ny_glob*dom.nx_glob;

    if (fInStream.is_open()){
      fInStream.ignore(256,' ');
      fInStream >> tnx;
      fInStream.ignore(256,' ');
      fInStream >> tny;
      fInStream.ignore(256,' ');
      fInStream >> txll;
      fInStream.ignore(256,' ');
      fInStream >> tyll;
      fInStream.ignore(256,' ');
      fInStream >> tdx;
      fInStream.ignore(256,' ');
      fInStream >> nodata;
      #if SERGHEI_DEBUG_INFILTRATION
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET;
        std::cout << "NCOLS " << tnx << " ? " << dom.nx_glob << std::endl;
        std::cout << "NROWS " << tny << " ? " << dom.ny_glob << std::endl;
        std::cout << "XLLCORNER " << txll << " ? " << dom.xll << std::endl;
        std::cout << "YLLCORNER " << tyll << " ? " << dom.yll << std::endl;
        std::cout << "DX " << tdx << " ? " << dom.dxConst << std::endl;
      #endif

      //compare the values t* with the DEM file just to check if we are using the same values, otherwise error
      if(dom.ny_glob !=tny || dom.nx_glob !=tnx || dom.xll !=txll || dom.yll !=tyll || dom.dxConst !=tdx){
        if(par.masterproc){
        std::cerr<<RERROR "Infiltration map file header does not match DEM file header.\n";
        return 0;
        }
      }

      real tmp;
        for (int ii=0; ii<ndata; ii++) {
        if (!fInStream.fail() && !fInStream.eof()){
          fInStream >> tmp;
          if(tmp<0.0){
            if(par.masterproc){
            std::cerr<<RERROR "There are negative values. Data should only be positive integers.\n";
            return 0;
            }
          }
          tmpVar(ii)=tmp;
        }else{
          if(par.masterproc){
            std::cerr<<RERROR "Error reading infiltration map. Not enough data\n";
          }
          return 0;
        }
      }

      fInStream.close();

    Kokkos::parallel_for("init_infLabel", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
      int i,j;
      dom.unpackIndices(iGlob,j,i);
      int ii1 = dom.getHaloExtension(i,j);
      int ii2 = dom.getSubdomainExtension(par,i,j);
      inf.infLabel(ii1)=tmpVar(ii2);
    });


    }
    else{
      if(par.masterproc){
        if(inf.nLabels <= 2){ // 1 for the pervious type, 1 for the impervious type, therefore 2
          std::cerr << YEXC "File " << io.infMapFile << " not found. Setting infiltration parameters for the only pervious infiltration class homogenously for all the domain." << std::endl;
        }
        else{
          std::cerr << RERROR "File " << io.infMapFile << " not found" << std::endl;
          return 0;
        }
    }
    }

    if (par.masterproc){
      std::cerr<<GOK "Infiltration map read\n";
    }

    return 1;

    }

  #if SERGHEI_LPT

    int readParticles(std::string dir, Parallel const &par, ParticleTracker* particles){
      std::ifstream fInParticles(dir + "particlesIni.input");
      std::string line;
      ParserLine pline;

      if (fInParticles.is_open()){
        while (std::getline(fInParticles, line)) {
          pline.line = line;
          pline.parse();

          // If the line was valid and a key is stored
          if(!pline.key.empty()){
          // Match the key, and store the value
            if (!strcmp("nParticles", pline.key.c_str())){
              pline.value >> nParticles;//Number of Particles
              if(nParticles <= 0){
                if(par.masterproc) std::cerr<<RERROR "No particles defined correctly" << std::endl;
                  return 0;
              }else{
                  particles->N_par=nParticles;
              }
            }else if (!strcmp("initialDist", pline.key.c_str())){
              pline.value >> Distribution;//Initial distribution for particles
              if(Distribution==Distribution1){
                particles->initialDist=100;//Random coordinates
              }else{
                if(Distribution==Distribution2){
                  particles->initialDist=101;//Random cells, coordinates in the center of the cell
                }else{
                  if(Distribution==Distribution3){
                    particles->initialDist=102;//Random coordinates in a polygon
                  }else{
                    if(Distribution==Distribution4){
                      particles->initialDist=103;//Coordinates provided by the user
                    }else{
                      if(par.masterproc) std::cerr<<RERROR "No correct initial distribution." << std::endl;
                      return 0;
                    }
                  }
                }
              }
              if(particles->initialDist==SERGHEI_PARTICLE_RANDOM_POLYGON){
                fInParticles.ignore(256,' ');
                fInParticles >> particles->numberPolygons;//Number of polygons: 1 or 2
                if(particles->numberPolygons==1){//1 polygon
                  fInParticles.ignore(256,' ');
                  fInParticles >> particles->Npoints;
                  if(particles->Npoints>0){
                    particles->initialisePolygon();
                    for(int ip1=0;ip1<particles->Npoints;ip1++){
                      if (!fInParticles.fail() && !fInParticles.eof()){
                        fInParticles >> particles->pointsXPolygon(ip1);
                        fInParticles >> particles->pointsYPolygon(ip1);
                      }else{
                        if(par.masterproc) std::cerr<<RERROR "Not enough coordinates for observation line" << ip1 << ". Expected " << particles->Npoints << " points, with " <<N_SPATIAL_DIM<< "coordinates." << std::endl;
                        return 0;
                      }
                    }
                  }else{
                    if(par.masterproc) std::cerr<<RERROR "The number of points for the polygon must be an positive integer." << std::endl;
                    return 0;
                  }
                }else if(particles->numberPolygons==2){//2 polygons
                  fInParticles.ignore(256,' ');
                  fInParticles >> particles->Npoints;
                  fInParticles.ignore(256,' ');
                  fInParticles >> particles->Npoints2;
                  if(particles->Npoints>0){
                    particles->initialisePolygon();
                    for(int ip1=0;ip1<particles->Npoints;ip1++){
                      if (!fInParticles.fail() && !fInParticles.eof()){
                        fInParticles >> particles->pointsXPolygon(ip1);
                        fInParticles >> particles->pointsYPolygon(ip1);
                      }else{
                        if(par.masterproc) std::cerr<<RERROR "Not enough coordinates for observation line" << ip1 << ". Expected " << particles->Npoints << " points, with " <<N_SPATIAL_DIM<< "coordinates." << std::endl;
                        return 0;
                      }
                    }
                  }else{
                    if(par.masterproc) std::cerr<<RERROR "The number of points for the polygon must be an positive integer." << std::endl;
                    return 0;
                  }
                  if(particles->Npoints2>0){
                    for(int ip2=0;ip2<particles->Npoints2;ip2++){
                      if (!fInParticles.fail() && !fInParticles.eof()){
                        fInParticles >> particles->pointsXPolygon2(ip2);
                        fInParticles >> particles->pointsYPolygon2(ip2);
                      }else{
                        if(par.masterproc) std::cerr<<RERROR "Not enough coordinates for observation line" << ip2 << ". Expected " << particles->Npoints << " points, with " <<N_SPATIAL_DIM<< "coordinates." << std::endl;
                        return 0;
                      }
                    }
                  }else{
                    if(par.masterproc) std::cerr<<RERROR "The number of points for the polygon must be an positive integer." << std::endl;
                    return 0;
                  }
                }else{
                  if(par.masterproc) std::cerr<<RERROR "The number of polygons must be 1 or 2." << std::endl;
                  return 0;
                }
              }
              if(particles->initialDist==SERGHEI_PARTICLE_COORDINATES){
                particles->initialiseParticlesCoordinates();
                for(int ipp=0;ipp<particles->N_par;ipp++){
                  if (!fInParticles.fail() && !fInParticles.eof()){//Reading the coordinates of the particles
                    fInParticles >> particles->particles(ipp).x(_X);
                    fInParticles >> particles->particles(ipp).x(_Y);
                    fInParticles >> particles->particles(ipp).z;
                  }else{
                    if(par.masterproc) std::cerr<<RERROR "Not enough coordinates for" << particles->N_par << " particles with" <<N_SPATIAL_DIM<< "dimensions." << std::endl;
                    return 0;
                  }
                }
              }
            }else if (!strcmp("initialLifetime", pline.key.c_str())){
              pline.value >> Lifetime;
              if(Lifetime==Lifetime1){
                particles->initialLifetime=32;//Random life time for each particle
              }else{
                if(Lifetime==Lifetime2){
                  particles->initialLifetime=33;//Constant life time for all particles, provided by the user
                }else{
                  if(par.masterproc) std::cerr<<RERROR "No correct initial lifetime." << std::endl;
                  return 0;
                }
              }
            }else if(particles->initialLifetime == SERGHEI_PARTICLE_CONSTANT_LIFETIME && !strcmp("Lifetime", pline.key.c_str())){
              pline.value >>  particles->lifetimeIni;
              if(particles->lifetimeIni<0.0){
                if(par.masterproc) std::cerr<<RERROR "No correct lifetime value, it must be a positive real number." << std::endl;
                return 0;
              }
            }else if (!strcmp("LifetimeOption", pline.key.c_str())){
              pline.value >> Resurrection;
              if(Resurrection==Resurrection1){
                particles->LifetimeOption=11;//no resurrection
              }else{
                if(Resurrection==Resurrection2){
                  particles->LifetimeOption=12;//resurrection in the upstream boundary condition cells, with random coordinates
                }else{
                  if(Resurrection==Resurrection3){
                    particles->LifetimeOption=13;//resurrection in random coordinates
                  }else{
                    if(Resurrection==Resurrection4){
                      particles->LifetimeOption=14;//resurrection in the upstream boundary condition cells, each a specific period of time
                    }else{
                      if(par.masterproc) std::cerr<<RERROR "No correct lifetime option." << std::endl;
                      return 0;
                    }
                  }
                }
              }
            }
            //Nothing more to read
            else{
              if(par.masterproc){
                std::cerr << RERROR "key " << pline.key << " not understood in file " << dir << "\n";
                exit(-1);
              }
            }
          }
        }
      }else{
        if(par.masterproc){
        std::cerr<< RERROR "Unable to open " << dir << "\n";
          return 0;
        }
      }
      return 1;
    }
  #endif

  };




