/*
	Solvers for the Ax=b system
*/
#ifndef _GW_SOLVER_H_
#define _GW_SOLVER_H_

#include "KokkosKernels_config.h"
#include "KokkosSparse_pcg.hpp"

#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosKernels_default_types.hpp"
#include <iostream>

#include "GwDomain.h"
#include "GwMatrix.h"

template <typename execution_space>
class GwSolver {

	typedef typename KokkosSparse::CrsMatrix<default_scalar, default_lno_t, execution_space, void, default_size_type> crsMat_t;
	typedef typename crsMat_t::StaticCrsGraphType graph_t;
	typedef typename graph_t::row_map_type::non_const_type lno_view_t;
	typedef typename graph_t::entries_type::non_const_type   lno_nnz_view_t;
	typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
	typedef typename scalar_view_t::value_type scalar_t;
	typedef typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type::value_type size_type;
	typedef typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type::value_type lno_t;


public:
	int iter, iter_max, gsteps, nrow, nnz;
	real eps, eps_min;

	lno_view_t ptr;
	lno_nnz_view_t ind;
	scalar_view_t val, rhs, vecx, diag;

	inline void init(GwMatrix &A, GwDomain &gdom)	{
		nrow = A.nrow;	nnz = A.nnz;
		iter_max = gdom.cg_iter;
		gsteps = 20;
		iter_max = iter_max * gsteps;
		eps_min = gdom.cg_tol;
		#if SERGHEI_KOKKOSKERNELS_SOLVER
		ptr = lno_view_t("ptr", A.nrow+1);
		ind = lno_nnz_view_t("ind", A.nnz);
		val = scalar_view_t("val", A.nnz);
		rhs = scalar_view_t("rhs", A.nrow);
		vecx = scalar_view_t("vecx", A.nrow);
		diag = scalar_view_t("diag", A.nrow);
		Kokkos::deep_copy (ptr, A.ptr);
		#endif
	}

	/*
		----------------------------------------------------------
		----------------------------------------------------------
							Iterative Solvers
		----------------------------------------------------------
		----------------------------------------------------------
	*/

	/*
		PCG from KokkosKernels
	*/
	// Top-level PCG solver
	void kkpcg(GwMatrix &A)
	{
		bool usePreconditioner = 1;
		const unsigned cg_iteration_limit = iter_max;
		const double   cg_iteration_tolerance     = eps_min ;

		decompose(A);
		Kokkos::deep_copy (ind, A.ind);
		Kokkos::deep_copy (val, A.val);
		Kokkos::deep_copy (rhs, A.rhs);
		Kokkos::deep_copy (diag, A.diag);

		crsMat_t matA = crsMat_t("matA", A.nrow, A.nrow, A.nnz, val, ptr, ind);

		// initialize KokkosKernels solver
		KokkosKernels::Experimental::Example::CGSolveResult cg_result ;
		typedef KokkosKernels::Experimental::KokkosKernelsHandle
		  < size_type, lno_t, scalar_t, execution_space, execution_space, execution_space > KernelHandle;
		KernelHandle kh;
		kh.create_gs_handle();
		//Kokkos::Impl::Timer timer1;
		KokkosKernels::Experimental::Example::pcgsolve(kh, matA, rhs, vecx, diag
		    , cg_iteration_limit, cg_iteration_tolerance, &cg_result, usePreconditioner);
		Kokkos::fence();
		A.cg_iter = cg_result.iteration;
		//solve_time = timer1.seconds();
		//std::cout  << "DEFAULT SOLVE: " << "(P)CG_NUM_ITER = [" << cg_result.iteration << "], " << "RESIDUAL = [" << cg_result.norm_res << "]"<< std::endl ;
		kh.destroy_gs_handle();

		Kokkos::deep_copy (A.x, vecx);
	}

	/*
    	CG Solver
    */
    int cg(GwMatrix &A, GwDomain &gdom)	{
		real rho, rhoOld, alpha, beta;
		rhoOld = 0.0;
		decompose(A);
		// initialize residual
		get_residual(A);
		iter = 0;	eps = 1.0;
		while (iter < iter_max & eps > eps_min)	{
			precJACO(A);
			rho = dot(A.r, A.z, A.nrow);
			if (iter == 0)	{
				beta = 0.0;
				Kokkos::deep_copy(A.p, A.z);
			}
			else {
				beta = rho / rhoOld;
				kxpy(A.p, beta, A.p, A.z, A.nrow);
			}
			mul_MV(A.q, A, A.p);
			alpha = rho / dot(A.q, A.p, A.nrow);
			update_X(A, alpha);
			kxpy(A.r, -alpha, A.q, A.r, A.nrow);
			rhoOld = rho;
			eps = pow(dot(A.r, A.r, A.nrow),0.5);
			iter += 1;
    	}
		// std::cerr << "      >> CG solver converges in " << iter << " iterations! eps = " << eps <<"\n";
    	return iter;
    }

	/*
    	Jacobi Preconditioner
    */
    void precJACO(GwMatrix A)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idom) {
			A.z(idom) = A.r(idom) * A.diag(idom);
		});
    }

	/*
		Reset A.x
	*/
	void reset(realArr x, int n)	{
		Kokkos::parallel_for(n , KOKKOS_LAMBDA(int idom) {x(idom) = 0.0;});
	}

	/*
		Get residual
	*/
	void get_residual(GwMatrix &A)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idom) {
			int icol;
			A.r(idom) = A.rhs(idom);
			for (icol = A.ptr(idom); icol < A.ptr(idom+1); icol++)	{
				A.r(idom) -= A.val(icol) * A.x(A.ind(icol));
			}
		});
	}

	/*
		Dot product of two vectors
	*/
	real dot(realArr v1, realArr v2, int n)	{
		real out;
		Kokkos::parallel_reduce( n , KOKKOS_LAMBDA (int idx, real &prod) {
			prod += v1(idx) * v2(idx);
		} , out);
		return out;
	}

	/*
		kX+Y
	*/
	void kxpy(realArr out, real k, realArr x, realArr y, int n)	{
		Kokkos::parallel_for( n , KOKKOS_LAMBDA(int idx) {
			out(idx) = k * x(idx) + y(idx);
		});
	}

	/*
		Matrix - Vector Multiplication
	*/
	void mul_MV(realArr out, GwMatrix A, realArr x)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			int icol;
			out(idx) = 0.0;
			for (icol = A.ptr(idx); icol < A.ptr(idx+1); icol++)	{
				out(idx) += A.val(icol) * x(A.ind(icol));
			}
		});
	}

	/*
		Update solution
	*/
	void update_X(GwMatrix A, real alpha)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			A.x(idx) += alpha * A.p(idx);
		});
	}

	/*
		Get diagonal of Matrix
	*/
	void decompose(GwMatrix A)	{
		Kokkos::parallel_for(A.nrow , KOKKOS_LAMBDA(int idx) {
			int icol;
			A.diag(idx) = 0.0;
			for (icol = A.ptr(idx); icol < A.ptr(idx+1); icol++)	{
				if (A.ind(icol) == idx)	{A.diag(idx) = 1.0 / A.val(icol);}
			}
		});
	}

};
#endif
