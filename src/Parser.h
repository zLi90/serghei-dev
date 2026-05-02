/* -*- mode: c++ -*- */

#ifndef _PARSER_H_
#define _PARSER_H_

#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <sys/types.h>
#include <errno.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "define.h"
#include "string.h"
#include "Domain.h"
#include "FileIO.h"
#include "Parallel.h"
#include "SourceSink.h"
#include "rasterTools.h"
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
     if (!line.empty() && line.find("//",0) != 0) {

       // Find the colon
       uint splitloc = line.find(':',0);

       // Store the key and value strings
       key   = line.substr(0,splitloc);

       // Remove spaces and tabs from the key
       key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
       key.erase(std::remove(key.begin(), key.end(), '\t'), key.end());
       std::string val = line.substr(splitloc+1,line.length()-splitloc);

       // Check for comments after values
       size_t splitter = val.find("//",0);
       std::string strloc;
       if (splitter != std::string::npos){
         strloc = val.substr(0,splitter);
       } else {
         strloc = val;
       }

       // Transform the val into a string stream for convenience
       value.clear();
       value.str(strloc);
     }
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

  int readDimensions(std::string fNameIn, Domain &dom, State &state, Parallel &par, FileIO &io){

	std::string tempStr;
 	tempStr = fNameIn + "parameters.input";
 	if(!readParamsFile(tempStr,dom,par,io)) return 0;
  #if SERGHEI_INPUT_NETCDF
	  io.ncin.fname = fNameIn + "input.nc";
 	  if(!io.readNetCDFheader(par,io.ncin,dom)) return 0;
    if(!io.readNetCDFcoordinates(par,io.ncin,dom)) return 0;
  #else
	  tempStr = fNameIn + "dem.input";
 	  if(!readHeaderDEMFile(tempStr,dom,par));
  #endif

	return 1;

  }

  int readInputFiles (std::string fNameIn, Domain &dom, State &state, SourceSinkData &ss,
		      ExternalBoundaries &ebc, Parallel &par, FileIO &io)
  {

    int const Nfiles = 6;
    int ierr[Nfiles];
    std::string fNames[Nfiles];
    std::string tempStr;

		#if SERGHEI_INPUT_NETCDF
			tempStr = fNameIn + "input.nc";
      if(par.masterproc) std::cout << BDASH << "Reading variables from NetCDF file" << std::endl;
    	if(!io.readNetCDFvariable(par,dom,state,io.ncin,"z")){
        if(par.masterproc) std::cout << RERROR << tempStr << " not found" << std::endl;
        return 0;
      };
      ierr[0] = 1; fNames[0] = tempStr;
		#else
			tempStr = fNameIn + "dem.input";
    	ierr[0] = readDEMFile(tempStr,dom,state,par); fNames[0] = tempStr;
		#endif

        #if SERGHEI_SWE_MODEL
        tempStr = fNameIn + "sw.input";
        ierr[1] = readSWFile(tempStr, dom, par, state, fNameIn, io); fNames[1] = tempStr;

        tempStr = fNameIn + "rainfall.input";
        ierr[2] = readRainfallFile(tempStr, dom, ss.rain, par); fNames[2] = tempStr;

        #ifdef _DEV_RAIN_
            if (par.masterproc)
              {
        	int dim = ss.rain.nx * ss.rain.ny;
        	std::cerr << BDASH "rainfall partititioned in x-direction: "  << ss.rain.nx << "\n";
        	std::cerr << BDASH "rainfall partititioned in y-direction: "  << ss.rain.ny << "\n";

        	int t = 0;
        	int count = 0;
        	for (int i = 0; i < dim * ss.rain.np; i ++)
        	  {
        	    std::cerr << BDASH "rainfall intensity (" << (t % dim) << ") nr. " << count << ": " << ss.rain.value(i) << "\n";
        	    t ++;
        	    if ((t % dim) == 0)
        	      count ++;
        	  }
              }
        #endif

        tempStr = fNameIn + "extbc.input";
        ierr[3] = readExtBCFile(tempStr, dom, ebc, par, state); fNames[3] = tempStr;

        tempStr = fNameIn + "infiltration.input";
        ierr[4] = readInfiltrationFile(tempStr, dom, ss.inf, par); fNames[4] = tempStr;

        tempStr = fNameIn + "infiltrationMap.input";
        ierr[5] = readInfiltrationMap(tempStr, dom, ss.inf, par); fNames[5] = tempStr;

        tempStr = fNameIn + "wind.input";
        ierr[6] = readWindFile(tempStr, dom, ss.wind, par);

        for (int i = 0; i < Nfiles; i++){
          if (!ierr[i]){
            if(par.masterproc) std::cerr << RERROR << "Terminating. File " << fNames[i] << " not found or had errors" << std::endl;
            return 0;
          }
        }
        #endif

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

		if(!readRasterField(fNameIn, dom, par, state.z)) return 0;

		Kokkos::parallel_reduce("init_z", dom.nCell , KOKKOS_LAMBDA (int iGlob, int &ncell) {
			int ii = dom.getIndex(iGlob);

			if(isnan(state.z(ii))){
				state.isnodata(ii) = true;
				state.z(ii) = NDTH + 0.001;
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
int readInfiltrationFile(std::string fNameIn, Domain &dom, InfiltrationModel &inf, Parallel &par){
  std::string modelName;
  // Read in colon-separated key: value file line by line
  std::ifstream fInStream(fNameIn);
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

        if(!strcmp("infiltrationclasses",pline.key.c_str())){pline.value >> inf.nLabels ;}
      }
    }
  }
  else{
    inf.model = INF_NONE;
    if (par.masterproc){
      std::cout << YEXC << fNameIn << " not found" << std::endl;
      std::cout << BDASH "Impervious domain set" << std::endl;
    }
    modelName = "none";
    return 1;
  }

  // now that the headers have been read, allocate stuff
  if(inf.model){
    if(inf.nLabels <= 0){
      if(par.masterproc){
        std::cerr << RERROR "Number of infiltration classes must be larger than 0" << std::endl;
        return 0;
      }
    }
    inf.nLabels++;  // to account for label value 0 as impervious
    // allocate the infiltration map
    inf.infLabel = intArr( "infLabel", dom.nCellMem);
    // TODO parallelisation
    if(inf.nLabels == 2){
      for(int ii=0; ii<dom.nCellMem; ii++){
        inf.infLabel(ii) = 1;
      }
    }
  }

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

  // at this point, everything has been read

  // rate unit conversion from mm/s -> m/s
  for(int ii=0; ii<inf.nLabels; ii++){
    if(inf.model == INF_CONSTANT) inf.constCap(ii) /= 1000.;
    if(inf.model == INF_HORTON){
      inf.fc(ii) /= 1000.;
      inf.f0(ii) /= 1000.;
    }
  }

  if(!inf.assignModel(par)){
    if(par.masterproc){
      std::cerr << RERROR << "Error when assigning infiltration model input." << std::endl;
      return 0;
    }
  }

  if (par.masterproc){
		std::cout << BDASH << "Infiltration classes: " << inf.nLabels << std::endl;
    std::cout << GOK << "Infiltration model set" << std::endl;
  }

  return 1;

}


  /* Reads rainfall data file */
  inline int readRainfallFile (std::string fNameIn, Domain &dom, TimeSeries &rain, Parallel &par)
  {
	#if SERGHEI_DEBUG_WORKFLOW
  	  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    #endif

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
	   	 return 0;
		 }
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

	if (!isok)
	  {
	  	 if(par.masterproc){
	    std::cerr << RERROR "Invalid rainfall units specified in rainfall file" << std::endl;
	    return 0;
		 }
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
		      std::cerr << RERROR "Error reading rainfall file\n";
		      return 0;
			}
	     }
	  }

	  fInStream.close();
  }
  else{
	   dom.isRain = 0;
  }

  if (par.masterproc) std::cerr << GOK "Rainfall set\n";

    return 1;

}


  /* Reads wind data file */
  inline int readWindFile (std::string fNameIn, Domain &dom, TimeSeries &wind, Parallel &par)
  {
    std::ifstream fInStream (fNameIn);
    std::string line;
    printf("Reading wind file: %s\n", fNameIn.c_str());
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


  int readExtBCFile(std::string fNameIn, Domain &dom, ExternalBoundaries &ebc, Parallel &par, State &state) {
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
	    			ibc++;	// ibc should be set to -1
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
          if(ibc < -1){
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

	Kokkos::parallel_for("init_isBound", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
		  state.isBound(ii)=0;
    });


	 for (int k = 0; k < ebc.extbc.size(); k ++) {
		  int value;
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

		 for(int iGlob=0;iGlob<ebc.extbc[k].ncellsBC;iGlob++){
	 		int ii=ebc.extbc[k].bcells[iGlob];
			state.isBound(ii)=value;
		}

	}

  if(par.masterproc) std::cout << GOK << "External boundary file parsed and boundaries set" << std::endl;
  return 1;
};


  int readSWFile(std::string fNameIn, Domain &dom, Parallel &par, State &state, std::string fDirIn, FileIO &io){
	#if SERGHEI_DEBUG_WORKFLOW
  	  std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
    	#endif
    std::ifstream fInStream(fNameIn);
    std::string line;
    ParserLine pline;
    ShallowWater sw;
    std::string tempStr;


    sw.initialMode="dry";
    sw.frictionModel="none";

    if (fInStream.is_open()){
      while (std::getline(fInStream, line)) {
        pline.line = line;
	      pline.lowercase();
        pline.parse();

        // If the line was valid and a key is stored
	      if(!pline.key.empty()){
          // Match the key, and store the value
          if(!strcmp("initialmode",pline.key.c_str())){ pline.value >> sw.initialMode; }
	        if(!strcmp("initialvalue",pline.key.c_str())){ pline.value >> sw.initialValue ; }
          if(!strcmp("friction",pline.key.c_str())){ pline.value >> sw.frictionModel  ; }
          if(!strcmp("roughness",pline.key.c_str())){ pline.value >> sw.roughnessInput  ; }
          if(!strcmp("drydepth",pline.key.c_str())){ pline.value >> sw.hmin  ; }
        }
      }
    }else{
      if (par.masterproc){
         std::cerr << RERROR "File " << fNameIn << " not found" << std::endl;
         return 0;
      }
    }

    // Initialise roughness
    if(!checkValidOption(sw.frictionModel, sw.frictionModels)){
	    if(par.masterproc){
        std::cerr << RERROR "Invalid friction model. Please correct sw.input" << std::endl;
        return 0;
		  }
    }

    // assign dry threshold water depth to zero velocities
    if(sw.hmin < 0){
      std::cerr << RERROR "Invalid or missing dry depth tolerance" << std::endl;
      return 0;
    }
    state.hmin = sw.hmin;

    int consistency=1;
    if(sw.frictionModel.compare("none") != 0){
      // consistency check
      switch(SERGHEI_FRICTION_MODEL){
        case SERGHEI_FRICTION_MANNING:
          if(sw.frictionModel.compare("manning")) consistency=0 ;
          break;
        case SERGHEI_FRICTION_DARCYWEISBACH:
          if(sw.frictionModel.compare("darcyweisbach")) consistency=0;
          break;
        case SERGHEI_FRICTION_CHEZY:
          if(sw.frictionModel.compare("chezy")) consistency=0;
          break;
      }
      if(!consistency){
        std::cerr << RERROR "Friction model in .sw file is inconsistent with SERGHEI_FRICTION_MODEL compilation flag" << std::endl;
        return 0;
      }
      sw.roughness  = atof(sw.roughnessInput.c_str());
      #if SERGHEI_INPUT_NETCDF
      if(!sw.roughnessInput.compare("netcdf")){
        int varfound=1;
        varfound = io.readNetCDFvariable(par,dom,state,io.ncin,"roughness");
        // if variable not found, set to zero (nothing needs to be done, as variables are initialised to zero at allocation)
        if(varfound == NC_ENOTVAR && par.masterproc) std::cout << YEXC << "Roughness not found in NetCDF input file. Value set to zero" << std::endl;
        if(!varfound) return 0;
      }
      #else
      if(!sw.roughnessInput.compare("file")){
        tempStr = fDirIn + "roughness.input";
 	      if(!readRoughnessFile(tempStr,dom,state,par)) return 0;
      }
      #endif
      else{
        if(sw.roughness < 0 ){
		  	  if(par.masterproc){
          std::cerr << RERROR "Negative friction coefficient. Please correc sw.input" << std::endl;
			 	return 0;
			 }
        }
	real roughness = sw.roughness;
        Kokkos::parallel_for("init_roughness", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
          int ii = dom.getIndex(iGlob);
          state.roughness(ii) = roughness;
        });
        if (par.masterproc){
          std::cout << BDASH "Friction coefficient set constant to " << roughness << std::endl;
        }
      }
    }


    if(!checkValidOption(sw.initialMode, sw.initialModes)){
	 	if(par.masterproc){
      std::cerr << RERROR "Invalid initial SW mode. Please correct sw.input" << std::endl;
      return 0;
		}
    }

    if(!sw.initialMode.compare("file")){
      tempStr = fDirIn + "hini.input";
      if(!readHiniFile(tempStr,dom,state,par)) return 0;
      tempStr = fDirIn + "uini.input";
      if(!readUiniFile(tempStr,dom,state,par)) return 0;
      tempStr = fDirIn + "vini.input";
	    if(!readViniFile(tempStr,dom,state,par)) return 0;
    }
		#if SERGHEI_INPUT_NETCDF
    else if(!sw.initialMode.compare("netcdf")){
      int varfound=1;
      varfound = io.readNetCDFvariable(par,dom,state,io.ncin,"h");
      // if variable not found, set to zero (nothing needs to be done, as variables are initialised to zero at allocation)
      if(varfound == NC_ENOTVAR && par.masterproc) std::cout << YEXC << "Water depth not found in NetCDF input file. Domain will be set dry" << std::endl;
      if(!varfound) return 0;

      varfound = io.readNetCDFvariable(par,dom,state,io.ncin,"u");
      if(varfound == NC_ENOTVAR && par.masterproc) std::cout << YEXC << "x-velocity (u) not found in NetCDF input file. u will be set to zero" << std::endl;
      if(!varfound) return 0;
      
      varfound = io.readNetCDFvariable(par,dom,state,io.ncin,"v");
      if(varfound == NC_ENOTVAR && par.masterproc) std::cout << YEXC << "y-velocity (v) not found in NetCDF input file. v will be set to zero" << std::endl;
      if(!varfound) return 0;

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
      if(!sw.initialMode.compare("dry")) sw.initialValue = 0.;
      real initialValue = sw.initialValue;
      Kokkos::parallel_for("set_init_dry", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        state.h(ii) = initialValue;
        state.hu(ii) = state.hv(ii) = 0.;
      });
    }

    if(!sw.initialMode.compare("h+z")){
      Kokkos::parallel_for("set_init_h+z", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        state.h(ii) -= state.z(ii);
        if(state.h(ii) < 0) state.h(ii) = 0.;
      });
    }


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

    if      ( !strcmp(strloc.c_str(),"NETCDF") ) { 
      #ifdef SERGHEI_HAS_PNETCDF
        io.outFormat = OUT_NETCDF; 
      #else
        if(par.masterproc){
          std::cerr << YEXC << "NETCDF output requested but PNETCDF is not available." << std::endl;
          std::cerr << BDASH << "Falling back to VTK output format." << std::endl;
        }
        io.outFormat = OUT_VTK;
      #endif
    }
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


  int readInfiltrationMap(std::string fNameIn, Domain const &dom, InfiltrationModel &inf, Parallel &par) {
		#if SERGHEI_DEBUG_INFILTRATION
	    std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << "inf.Model = " << inf.model << "\tinf.nLabels = " << inf.nLabels << std::endl;
		#endif
    if(!inf.model) return 1;
    if(inf.nLabels == 1) return 1; // only impervious

  	realArr tmpVar=realArr("var", dom.ny_glob*dom.nx_glob );
    std::ifstream fInStream(fNameIn);
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
				std::cerr << YEXC "File " << fNameIn << " not found. Setting infiltration parameters for the only pervious infiltration class homogenously for all the domain." << std::endl;
			}
			else{
    		std::cerr << RERROR "File " << fNameIn << " not found" << std::endl;
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

#endif
