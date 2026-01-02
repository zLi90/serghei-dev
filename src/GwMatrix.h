/*
	The sparse matrix
*/
#ifndef _GW_MATRIX_H_
#define _GW_MATRIX_H_

#include "const.h"
#include "define.h"
#include "SArray.h"
#include "Indexing.h"
#include "GwDomain.h"

class GwMatrix {

public:
	int nrow, ncol, nnz, nx, ny, nz, cg_iter;
	intArr ptr, ind, ptrT, indT;
	realArr val, diag, rhs, x, valT, lt, ut;
	// views for cg solver
	realArr r, z, p, q;

	// initialize
	// GwMatrix(GwDomain &gdom)	{
	void init(GwDomain &gdom)	{
		int ii, jj, kk, ndom = gdom.nCell;
		nx = gdom.nx;	ny = gdom.ny;	nz = gdom.nz;
		nrow = ndom;
		ncol = ndom;
		// Number of non-zeros for 1D-z, 2D-xz, 3D-xyz simulations
		if (nx == 1 & ny == 1 & nz > 1)	{
			nnz = nrow + (nz-2)*2 + 2;
		}
		else if (nx > 1 & ny == 1 & nz > 1)	{
			nnz = nrow + 4*2 + ((nx-2)*2 + (nz-2)*2)*3 + (nx-2)*(nz-2)*4;
		}
		else if (nx > 1 & ny > 1 & nz > 1)	{
			nnz = nrow + 3*8 + 4*((nx-2)*4 + (ny-2)*4 + (nz-2)*4) +
		    	5*((nx-2)*(nz-2)*2 + (nx-2)*(ny-2)*2 + (ny-2)*(nz-2)*2) +
		    	6*((nx-2)*(ny-2)*(nz-2));
		}
		else	{printf("ERROR : Domain must be 1D-z, 2D-xz or 3D-xyz!\n");}
		ptr = intArr("ptr", nrow+1);		ind = intArr("ind", nnz);
		val = realArr("val", nnz);			diag = realArr("diag", nrow);
		rhs = realArr("rhs", nrow);			x = realArr("x", nrow);
		r = realArr("r", nrow);				z = realArr("z", nrow);
		p = realArr("p", nrow);				q = realArr("q", nrow);
		// Get ptr for the CRS matrix
		for (int idx = 0; idx < nrow+1; idx++)	{
			gdom.unpackIndices(idx, kk, jj, ii);
			// gdom.unpackIndicesGw(idx, nz, ny, nx, kk, jj, ii);
			if (idx == nrow) {ptr(idx) = nnz;}
			else	{ptr(idx) = get_irow(ii, jj, kk, nx, ny, nz);}
		}

	}

	// get starting index
	inline int get_irow(int i, int j, int k, int nx, int ny, int nz) {
		int nrowi = 0, nlayer1 = 0, nlayerk = 0, ncol1 = 0, ncolj = 0, irow;
	    if (k == 0)   {
	        if (j == 0)   {
	            // 3D
	            if (ny > 1) {if (i > 0)   {nrowi = 4 + (i-1)*5;}}
	            // 2D x-z
	            else {if (i > 0)   {nrowi = 3 + (i-1)*4;}}
	        }
	        else {
	            // 3D
	            ncol1 = 4*2 + 5*(nx-2);
	            ncolj = (j-1)*5*2 + (j-1)*6*(nx-2);
	            if (j == ny-1)   {if (i > 0)   {nrowi = 4 + (i-1)*5;}}
	            else {if (i > 0)   {nrowi = 5 + (i-1)*6;}}
	        }
	    }
	    else {
	        if (nx > 1) {
	            // 3D
	            if (ny > 1) {
	                nlayer1 = 4*4 + 5*2*(nx-2+ny-2) + 6*(nx-2)*(ny-2);
	                nlayerk = (k-1)*(5*4 + 6*2*(nx-2+ny-2) + 7*(nx-2)*(ny-2));
	            }
	            // 2D
	            else {
	                nlayer1 = 3*2 + 4*(nx-2);
	                nlayerk = (k-1)*(4*2 + 5*(nx-2));
	            }
	        }
	        else {
	            // 1D
	            nlayer1 = 2;    nlayerk = (k-1)*3;
	        }
	        if (k == nz-1)    {
	            if (j == 0)   {
	                if (ny > 1) {if (i > 0)   {nrowi = 4 + (i-1)*5;}}
	                else {if (i > 0)   {nrowi = 3 + (i-1)*4;}}
	            }
	            else {
	                ncol1 = 4*2 + 5*(nx-2);
	                ncolj = (j-1)*5*2 + (j-1)*6*(nx-2);
	                if (j == ny-1)   {if (i > 0)   {nrowi = 4 + (i-1)*5;}}
	                else {if (i > 0)   {nrowi = 5 + (i-1)*6;}}
	            }
	        }
	        else {
	            if (j == 0)   {
	                if (ny > 1) {if (i > 0)   {nrowi = 5 + (i-1)*6;}}
	                else {if (i > 0)   {nrowi = 4 + (i-1)*5;}}
	            }
	            else {
	                ncol1 = 5*2 + 6*(nx-2);
	                ncolj = (j-1)*6*2 + (j-1)*7*(nx-2);
	                if (j == ny-1)   {if (i > 0)   {nrowi = 5 + (i-1)*6;}}
	                else {if (i > 0)   {nrowi = 6 + (i-1)*7;}}
	            }
	        }

	    }
	    irow = nlayer1 + nlayerk + ncol1 + ncolj + nrowi;
	    return irow;

	}


};

#endif
