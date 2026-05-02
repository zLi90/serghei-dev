#ifndef __RASTER_H__
#define __RASTER_H__

#include <string>
#include "define.h"
#include "Domain.h"
#include "Parallel.h"

int readRasterField(const std::string fNameIn, const Domain &dom, const Parallel &par, realArr &data) {
  if(par.masterproc) std::cout << BDASH << "Reading raster file " << fNameIn << std::endl;

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

  std::string str;
	if (fInStream.is_open()){
    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> tnx;

    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> tny;

    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> txll;

    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> tyll;

    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> tdx;

    std::getline(fInStream,str,' ');
    std::getline(fInStream,str);
    std::stringstream(str) >> nodata;

		//std::cout << GGD << tnx << "\t" << tny  <<"\t" << txll << "\t" <<tyll<< "\t" << tdx<< "\t" << nodata << std::endl;
		//compare the values t* with the DEM raster size, to check if we are using the same values, otherwise error
		if(dom.ny_glob !=tny || dom.nx_glob !=tnx || dom.xll !=txll || dom.yll !=tyll || dom.dxConst !=tdx){
			if(par.masterproc){
				std::cerr << RERROR "Parameters in " << fNameIn << " don't match DEM file parameters." << std::endl;
      }
			return 0;
		}

		real tmp;
		int flagnodata=0;

		//real *buffer = new real(dom.nCellGlobal);

    for (int ii=0; ii < dom.nCellGlobal; ii++) {
	//	Kokkos::parallel_for("readRaster", Kokkos::Serial(),dom.nCellGlobal, KOKKOS_LAMBDA(int ii){
			if (!fInStream.fail() && !fInStream.eof()){
				fInStream >> tmp;
				//std::cout << GGD << ii << "/" << dom.nCellGlobal << "\t" << tmp << std::endl;
				if(tmp - nodata < TOL12 && tmp - nodata > TOL12NEG){
					if(!flagnodata){
						if(par.masterproc){
							std::cerr<< YEXC "There is some no-data in your file\n";
						}
						flagnodata = 1;
					}
					dom.globalBuffer(ii) = NAN; //we set no data values to NAN
					//buffer[ii] = NAN;
				}else{
					dom.globalBuffer(ii) = tmp;
					//buffer[ii] = tmp;
				}
			}else{
		    if(par.masterproc){
          std::cerr<< RERROR "Error reading " << fNameIn << ". Not enough data. Last index read: " << ii << std::endl;
				    return 0;
        }
			}	
	//	})
  	}
		fInStream.close();
	}
	else{
		if (par.masterproc) std::cerr << YEXC "Unable to open " << fNameIn << std::endl;
		return 0;
	}

/*
	Kokkos::parallel_for("rasterBuffer", dom.nCellGlobal, KOKKOS_LAMBDA(int ii) {
		dom.globalBuffer(ii) = buffer[ii];
	});
*/
	if(par.masterproc) std::cerr << GOK << "Read file " << fNameIn << std::endl;
	/*
  Kokkos::parallel_for("fetch_from_global_buffer", dom.nCell , KOKKOS_LAMBDA (int iGlob) {
 	  int i,j;
		dom.unpackIndices(iGlob,j,i);
		int ii1 = dom.getHaloExtension(i,j);
		int ii2 = dom.getSubdomainExtension(par,i,j);
		data(ii1) = dom.globalBuffer(ii2);
	});
	*/
	dom.fetchFieldFromGlobalBuffer(par,data);

	if(par.masterproc) std::cerr << GOK << "Distributed data from raster" << std::endl;
 	return 1;
}

#endif

