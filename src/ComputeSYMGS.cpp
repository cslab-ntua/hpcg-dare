
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
// #include "ComputeSYMGS_ref.hpp"
#include <cassert>
#include <cstdint>
#include <riscv_vector.h>

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  // return ComputeSYMGS_ref(A, r, x);

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  #ifndef HPCG_NO_MPI
    ExchangeHalo(A,x, 0);
  #endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (local_int_t j = 0; j < currentNumberOfNonzeros; ) {
      size_t vl = __riscv_vsetvl_e64m1(currentNumberOfNonzeros - j);
      vfloat64m1_t va =__riscv_vle64_v_f64m1(currentValues + j, vl);
      vuint64m1_t idx64;
      if (sizeof(local_int_t) == 8) {
        idx64 = __riscv_vle64_v_u64m1((const uint64_t *)(currentColIndices + j), vl);
      } else {
        vuint32mf2_t idx32 =__riscv_vle32_v_u32mf2((const uint32_t *)(currentColIndices + j), vl);
        idx64 = __riscv_vzext_vf2_u64m1(idx32, vl);
      }
      idx64 = __riscv_vsll_vx_u64m1(idx64, 3, vl);
      vfloat64m1_t vx =__riscv_vluxei64_v_f64m1(xv, idx64, vl);
      vfloat64m1_t vprod =__riscv_vfmul_vv_f64m1(va, vx, vl);
      vfloat64m1_t zero =__riscv_vfmv_v_f_f64m1(0.0, vl);
      vfloat64m1_t red = __riscv_vfredusum_vs_f64m1_f64m1(vprod, zero, vl);
      sum -= __riscv_vfmv_f_s_f64m1_f64(red);
      j += vl;
    }
    xv[i] = (sum+(xv[i]*currentDiagonal))/currentDiagonal;
  }

  // Now the back sweep.
  for (local_int_t i = nrow; i-- > 0; ) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value
    for (local_int_t j = 0; j < currentNumberOfNonzeros; ) {
        size_t vl = __riscv_vsetvl_e64m1(currentNumberOfNonzeros - j);
        vfloat64m1_t va =__riscv_vle64_v_f64m1(currentValues + j, vl);
        vuint64m1_t idx64;
        if (sizeof(local_int_t) == 8) {
            idx64 = __riscv_vle64_v_u64m1((const uint64_t *)(currentColIndices + j), vl);
        } else {
            vuint32mf2_t idx32 =__riscv_vle32_v_u32mf2((const uint32_t *)(currentColIndices + j), vl);
            idx64 = __riscv_vzext_vf2_u64m1(idx32, vl);
        }
        idx64 = __riscv_vsll_vx_u64m1(idx64, 3, vl);
        vfloat64m1_t vx =__riscv_vluxei64_v_f64m1(xv, idx64, vl);
        vfloat64m1_t vprod =__riscv_vfmul_vv_f64m1(va, vx, vl);
        vfloat64m1_t zero =__riscv_vfmv_v_f_f64m1(0.0, vl);
        vfloat64m1_t red = __riscv_vfredusum_vs_f64m1_f64m1(vprod, zero, vl);
        sum -= __riscv_vfmv_f_s_f64m1_f64(red);
        j += vl;
    }
    xv[i] = (sum+(xv[i]*currentDiagonal))/currentDiagonal;
  }

  return 0;

}
