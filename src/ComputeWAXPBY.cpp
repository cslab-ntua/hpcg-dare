
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
// #include "ComputeWAXPBY_ref.hpp"
#include <stdio.h>
#include <riscv_vector.h>
#include <omp.h>
#include <assert.h>

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  // isOptimized = false;
  // return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);

  isOptimized = true;
  assert(x.localLength >= n); 
  assert(y.localLength >= n);

  const double * xv = x.values;
  const double * yv = y.values;
  double * wv = w.values;

  #ifndef HPCG_NO_OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Work distribution
        size_t len = (size_t)n;
        size_t chunk = len / nthreads;
        size_t start = tid * chunk;
        size_t end = (tid == nthreads - 1) ? len : start + chunk;

        if (start < end) {
            size_t local_data_num = end - start;
            const double *lx = xv + start;
            const double *ly = yv + start;
            double *lw = wv + start;

            for (size_t vl; local_data_num > 0; local_data_num -= vl) {
                vl = __riscv_vsetvl_e64m1(local_data_num);
                vfloat64m1_t v_x = __riscv_vle64_v_f64m1(lx, vl);
                vfloat64m1_t v_y = __riscv_vle64_v_f64m1(ly, vl);
                vfloat64m1_t v_res;

                if (alpha == 1.0) {
                    v_res = __riscv_vfmacc_vf_f64m1(v_x, beta, v_y, vl);
                } else if (beta == 1.0) {
                    v_res = __riscv_vfmacc_vf_f64m1(v_y, alpha, v_x, vl);
                } else {
                    vfloat64m1_t v_tmp = __riscv_vfmul_vf_f64m1(v_x, alpha, vl);
                    v_res = __riscv_vfmacc_vf_f64m1(v_tmp, beta, v_y, vl);
                }

                __riscv_vse64_v_f64m1(lw, v_res, vl);
                lx += vl; ly += vl; lw += vl;
            }
        }
    }
  #else
    local_int_t data_num = n;
    const double *lx = xv;
    const double *ly = yv;
    double *lw = wv;

    for (size_t vl; data_num > 0; data_num -= vl) {
        vl = __riscv_vsetvl_e64m1(data_num);
        vfloat64m1_t v_x = __riscv_vle64_v_f64m1(lx, vl);
        vfloat64m1_t v_y = __riscv_vle64_v_f64m1(ly, vl);
        vfloat64m1_t v_res;

        if (alpha == 1.0) {
            v_res = __riscv_vfmacc_vf_f64m1(v_x, beta, v_y, vl);
        } else if (beta == 1.0) {
            v_res = __riscv_vfmacc_vf_f64m1(v_y, alpha, v_x, vl);
        } else {
            vfloat64m1_t v_tmp = __riscv_vfmul_vf_f64m1(v_x, alpha, vl);
            v_res = __riscv_vfmacc_vf_f64m1(v_tmp, beta, v_y, vl);
        }

        __riscv_vse64_v_f64m1(lw, v_res, vl);
        lx += vl; ly += vl; lw += vl;
    }
  #endif

  
  return 0;
}
