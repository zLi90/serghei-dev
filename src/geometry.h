#ifndef _GEOMETRY_
#define _GEOMETRY_

#include "SArray.h"

#define _X 0
#define _Y 1
#define _Z 2

#define _SOUTH 0
#define _EAST 1
#define _NORTH 2
#define _WEST 3
#define N_SPATIAL_DIM 2
#define N_CELL_EDGES 4


namespace geometry{
  typedef SArray<real,N_SPATIAL_DIM> point;

  typedef SArray<real,3> point3D;//ParticleTracking

  KOKKOS_INLINE_FUNCTION real distance(const point &p, const point &q){
    point d;
    real ss=0.;
    for(int i=0; i<N_SPATIAL_DIM; i++){
      d(i) = p(i) - q(i);
      ss += d(i)*d(i);
    }
    return(mysqrt(ss));
  };

  inline point crossProduct(const point &a, const point &b){
    point c;
    c(0) = a(1)*b(2) - a(2)*b(1);
    c(1) = a(2)*b(0) - a(0)*b(2);
    c(2) = a(0)*b(1) - a(1)*b(0);
    return(c);
  };

  inline geometry::point normalToLineInPlane(const point &a, const point &b){
    point c;
    c(0) = a(1) - b(1);
    c(1) = b(0) - a(0);
    return(c);
  };


	//returns 1 if the coordinate is inside, 0 otherwise
	int isInsidePoly(int np, realArr &xPoly, realArr &yPoly, real &xCoord, real &yCoord) {
		int count;
		int i;
		double xInter;
		real x1,x2,y1,y2;

		count=0;
		for (i=0;i<np;i++){
			x1= xPoly(i);
			y1= yPoly(i);
			x2= xPoly((i+1)%np);
			y2= yPoly((i+1)%np);
			if ((yCoord > fmin(y1,y2)) && (yCoord <= fmax(y1,y2)) && (xCoord <= fmax(x1,x2))) {
			  if (y1 != y2) {
				 xInter = x1 + (yCoord-y1)*(x2-x1)/(y2-y1);
				 if (x1 == x2 || xCoord <= xInter){
						count++;
				 }
			  }
			}
		}

		if(count % 2 == 0){
			return 0;
		}else{
			return 1;
		}
	}

    int isInsidePoly3D(int np, realArr &xPoly, realArr &yPoly, realArr &zPoly, real &xCoord, real &yCoord, real &zCoord) {
		int count;
		int i;
		double xInter;
		real x1,x2,y1,y2,z1,z2;

		count=0;
		for (i=0;i<np;i++){
			x1= xPoly(i);
			y1= yPoly(i);
            z1 = zPoly(i);
			x2= xPoly((i+1)%np);
			y2= yPoly((i+1)%np);
            z2 = zPoly((i+1)%np);
			if ((yCoord > fmin(y1,y2)) && (yCoord <= fmax(y1,y2)) && (xCoord <= fmax(x1,x2))) {
			  if (y1 != y2) {
				 xInter = x1 + (yCoord-y1)*(x2-x1)/(y2-y1);
				 if (x1 == x2 || xCoord <= xInter){
						count++;
				 }
			  }
			}
		}
		if    (count % 2 == 0)    {return 0;}
        else    {
            // check if inside zPoly
            if (zCoord > fmin(z1,z2) && zCoord <= fmax(z1,z2) && z1 != z2)  {
                return 1;
            }
            else    {
                return 0;
            }
		}
	}


};
#endif
