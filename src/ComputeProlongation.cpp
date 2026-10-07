
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeProlongation.cpp

 HPCG routine
 */

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation.hpp"
#include <cassert>
#include <riscv_vector.h>

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/

int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {


  assert(Af.mgData != 0);

  const local_int_t nc = Af.mgData->rc->localLength;
  const local_int_t * const f2c = Af.mgData->f2cOperator;
  const double * const xcv = Af.mgData->xc->values;
  double * const xfv = xf.values;

  #ifndef HPCG_NO_OPENMP
    #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        local_int_t chunk = nc / nthreads;
        local_int_t start = tid * chunk;
        local_int_t end = (tid == nthreads - 1) ? nc : start + chunk;

        for (local_int_t i = start; i < end; ) {
          size_t vl = __riscv_vsetvl_e64m1(end - i);

          vfloat64m1_t v_xc = __riscv_vle64_v_f64m1(xcv + i, vl);

    #if defined(HPCG_INDEX_64)
          vuint64m1_t idx = __riscv_vle64_v_u64m1((const uint64_t *)(f2c + i), vl);
    #else
          vuint32mf2_t idx32 = __riscv_vle32_v_u32mf2((const uint32_t *)(f2c + i), vl);
          vuint64m1_t idx = __riscv_vzext_vf2_u64m1(idx32, vl);
    #endif

          idx = __riscv_vsll_vx_u64m1(idx, 3, vl);

          vfloat64m1_t v_old = __riscv_vluxei64_v_f64m1(xfv, idx, vl);
          vfloat64m1_t v_new = __riscv_vfadd_vv_f64m1(v_old, v_xc, vl);

          __riscv_vsuxei64_v_f64m1(xfv, idx, v_new, vl);

          i += vl;
        }
      }
  #else
    for (local_int_t i = 0; i < nc; ) {
      size_t vl = __riscv_vsetvl_e64m1(nc - i);

      vfloat64m1_t v_xc = __riscv_vle64_v_f64m1(xcv + i, vl);

      #if defined(HPCG_INDEX_64)
          vuint64m1_t idx = __riscv_vle64_v_u64m1((const uint64_t *)(f2c + i), vl);
      #else
          vuint32mf2_t idx32 = __riscv_vle32_v_u32mf2((const uint32_t *)(f2c + i), vl);
          vuint64m1_t idx = __riscv_vzext_vf2_u64m1(idx32, vl);
      #endif

      idx = __riscv_vsll_vx_u64m1(idx, 3, vl);

      vfloat64m1_t v_old = __riscv_vluxei64_v_f64m1(xfv, idx, vl);
      vfloat64m1_t v_new = __riscv_vfadd_vv_f64m1(v_old, v_xc, vl);

      __riscv_vsuxei64_v_f64m1(xfv, idx, v_new, vl);

      i += vl;
    }
  #endif
  return 0;
}
