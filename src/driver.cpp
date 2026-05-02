/* -*- mode: c++; c-default-style: "linux" -*- */

#include "serghei.h"

int main(int argc, char** argv) {

	// Read command line arguments
	 if (argc != 4){
	  std::cerr << RERROR "The program is run as: ./nprogram inputFolder/ outputFolder/ Nthreads" << std::endl;
		  return 0;
	}

	{
	SERGHEI serghei;
	
	serghei.io.inFolder = argv[1];
	serghei.io.outFolder = argv[2];

	serghei.par.nthreads = atoi(argv[3]);

	if(!serghei.start(argc, argv)) return 1;
	if(!serghei.compute()) return 1;
	if(!serghei.finalise()) return 1;

	} // scope guard required to ensure serghei destructor is called

	Kokkos::finalize();
  MPI_Finalize();

	return 0;	// success should return zero
}