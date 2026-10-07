
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
// #include "ComputeSPMV_ref.hpp"
#include <stdio.h>
#include <riscv_vector.h>
#include <omp.h>
#include <assert.h>
#include <iostream>

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  // A.isSpmvOptimized = false;
  // return ComputeSPMV_ref(A, x, y);

  A.isSpmvOptimized = true;
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

  #ifndef HPCG_NO_MPI
    ExchangeHalo(A, x, 0);
  #endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  #ifndef HPCG_NO_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double *cur_vals = A.matrixValues[i];
    const local_int_t *cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (local_int_t colid = 0; colid < cur_nnz; ) {
      size_t vl = __riscv_vsetvl_e64m1(cur_nnz - colid);

      vfloat64m1_t va = __riscv_vle64_v_f64m1(cur_vals + colid, vl);

      #if defined(HPCG_INDEX_64)
        vuint64m1_t idx64 = __riscv_vle64_v_u64m1((const uint64_t *)(cur_inds + colid), vl);
      #else
        vuint32mf2_t idx32 = __riscv_vle32_v_u32mf2((const uint32_t *)(cur_inds + colid), vl);
        vuint64m1_t idx64 = __riscv_vzext_vf2_u64m1(idx32, vl);
      #endif

      idx64 = __riscv_vsll_vx_u64m1(idx64, 3, vl);

      vfloat64m1_t vx = __riscv_vluxei64_v_f64m1(xv, idx64, vl);
      vfloat64m1_t vprod = __riscv_vfmul_vv_f64m1(va, vx, vl);

      vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
      vfloat64m1_t red = __riscv_vfredosum_vs_f64m1_f64m1(vprod, zero, vl);

      sum += __riscv_vfmv_f_s_f64m1_f64(red);

      colid += vl;
    }

    yv[i] = sum;
  }

  return 0;
}
