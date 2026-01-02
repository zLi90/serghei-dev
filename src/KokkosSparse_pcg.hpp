//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Modified by Zhi Li, 2024-01-06
//@HEADER

#ifndef KOKKOS_EXAMPLE_CG_SOLVE
#define KOKKOS_EXAMPLE_CG_SOLVE

#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>

#include <iostream>
#include "KokkosKernels_Handle.hpp"
#include <KokkosSparse_spmv.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_gauss_seidel.hpp>
#include <KokkosSparse_sor_sequential_impl.hpp>
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//#define KK_TICTOCPRINT
namespace KokkosKernels {
namespace Experimental {
namespace Example {

struct CGSolveResult {
  size_t iteration;
  double iter_time;
  double matvec_time;
  double norm_res;
  double precond_time;
  double precond_init_time;
};


template <typename KernelHandle_t, typename crsMatrix_t, typename y_vector_t,
          typename x_vector_t, typename d_vector_t>
void pcgsolve(KernelHandle_t &kh, const crsMatrix_t &crsMat,
              const y_vector_t &y_vector, x_vector_t x_vector, d_vector_t &diag,
              const size_t maximum_iteration = 200,
              const double tolerance = std::numeric_limits<double>::epsilon(),
              CGSolveResult *result = 0, bool use_prec = true) {
    using namespace KokkosSparse;
    using namespace KokkosSparse::Experimental;
    using size_type = typename KernelHandle_t::size_type;
    using nnz_lno_t = typename KernelHandle_t::nnz_lno_t;
    using Space     = typename KernelHandle_t::HandleExecSpace;
    static_assert(
        std::is_same<double, typename KernelHandle_t::nnz_scalar_t>::value,
        "The PCG performance test only works with scalar = double.");

    const nnz_lno_t count_total = crsMat.numRows();

    size_t iteration         = 0;
    double iter_time         = 0;
    double matvec_time       = 0;
    double norm_res          = 0;
    double precond_time      = 0;
    double precond_init_time = 0;

    Kokkos::Timer wall_clock;
    Kokkos::Timer timer;

    // Need input vector to matvec to be owned + received
    y_vector_t pAll("cg::p", count_total);

    y_vector_t p =
      Kokkos::subview(pAll, std::pair<size_t, size_t>(0, count_total));
    y_vector_t r("cg::r", count_total);
    y_vector_t Ap("cg::Ap", count_total);

    /* r = b - A * x ; */
    /* p  = x       */ Kokkos::deep_copy(p, x_vector);

    /* Ap = A * p   */ KokkosSparse::spmv("N", 1, crsMat, pAll, 0, Ap);

    /* r  = Ap       */ Kokkos::deep_copy(r, Ap);

    /* r = b - r   */ KokkosBlas::axpby(1.0, y_vector, -1.0, r);

    /* p  = r       */ Kokkos::deep_copy(p, r);

    double old_rdot = KokkosBlas::dot(r, r);
    norm_res        = sqrt(old_rdot);

    y_vector_t z("pcg::z", count_total);

    double precond_old_rdot = 1;
    int apply_count = 1;

    if (use_prec)   {
    	auto ptrHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     crsMat.graph.row_map);
  		auto indHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     crsMat.graph.entries);
  		auto valHost =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), crsMat.values);
  Kokkos::View<double *, Kokkos::HostSpace> diagHost;
    	auto xHost =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_vector);
  		auto yHost =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_vector);
      	z = y_vector_t("pcg::z", count_total);
    	gauss_seidel_numeric(&kh, count_total, count_total, crsMat.graph.row_map,
                           crsMat.graph.entries, crsMat.values);

      Space().fence();

      precond_init_time += timer.seconds();
      Space().fence();
      timer.reset();

      symmetric_gauss_seidel_apply(&kh, count_total, count_total,
                                   crsMat.graph.row_map, crsMat.graph.entries,
                                   crsMat.values, z, r, true, true, 1.0,
                                   apply_count);

      Space().fence();
      
    
        //KokkosBlas::mult(0.0, z, 1.0, r, diag);
        precond_time += timer.seconds();
        precond_old_rdot = KokkosBlas::dot(r, z);
        Kokkos::deep_copy(p, z);
    }

    iteration = 0;

    while (tolerance < norm_res && iteration < maximum_iteration) {
        timer.reset();
        /* Ap = A * p   */ KokkosSparse::spmv("N", 1, crsMat, pAll, 0, Ap);
        Space().fence();
        matvec_time += timer.seconds();

        /* pAp_dot = dot(Ap , p ) */ const double pAp_dot = KokkosBlas::dot(p, Ap);

        double alpha = 0;
        if (use_prec) {
            alpha = precond_old_rdot / pAp_dot;
        }
        else {
            alpha = old_rdot / pAp_dot;
        }
        /* x +=  alpha * p ;  */ KokkosBlas::axpby(alpha, p, 1.0, x_vector);

        /* r += -alpha * Ap ; */ KokkosBlas::axpby(-alpha, Ap, 1.0, r);

        const double r_dot = KokkosBlas::dot(r, r);
        const double beta_original = r_dot / old_rdot;
        double precond_r_dot       = 1;
        double precond_beta        = 1;

        if (use_prec)   {
        	Space().fence();
        	timer.reset();
        	symmetric_gauss_seidel_apply(&kh, count_total, count_total,
                                     crsMat.graph.row_map, crsMat.graph.entries,
                                     crsMat.values, z, r, true, true, 1.0,
                                     apply_count);
                                     
        
            //KokkosBlas::mult(0.0, z, 1.0, r, diag);
            precond_time += timer.seconds();
            precond_r_dot = KokkosBlas::dot(r, z);
            precond_beta  = precond_r_dot / precond_old_rdot;
        }

        double beta = 1;
        if (!use_prec) {
            beta = beta_original;
            /* p = r + beta * p ; */ KokkosBlas::axpby(1.0, r, beta, p);
        }
        else {
            beta = precond_beta;
            KokkosBlas::axpby(1.0, z, beta, p);
        }

        norm_res         = sqrt(old_rdot = r_dot);
        precond_old_rdot = precond_r_dot;

        ++iteration;
    }

    Space().fence();
    iter_time = wall_clock.seconds();

    if (0 != result) {
        result->iteration         = iteration;
        result->iter_time         = iter_time;
        result->matvec_time       = matvec_time;
        result->norm_res          = norm_res;
        result->precond_time      = precond_time;
        result->precond_init_time = precond_init_time;
    }
}

}  // namespace Example
}  // namespace Experimental
}  // namespace KokkosKernels
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_EXAMPLE_CG_SOLVE */
