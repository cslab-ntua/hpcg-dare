
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
// #include "ComputeDotProduct_ref.hpp"
#include <riscv_vector.h>
#include <omp.h>
#include <assert.h>

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  // isOptimized = false;
  // return ComputeDotProduct_ref(n, x, y, result, time_allreduce);

  isOptimized = true;
  double * xv = x.values;
  double * yv = y.values;
  double local_result = 0.0;

  #ifdef HPCG_NO_OPENMP  
    if (yv==xv) {
      // for (local_int_t i=0; i<n; i++)
      //   local_result += xv[i]*xv[i];

      // We use a vector register to accumulate partial sums to stay in the vector unit
      // v_sum[0] will eventually hold our result
      vfloat64m1_t v_sum = __riscv_vfmv_s_f_f64m1(0.0, __riscv_vsetvl_e64m1(1));
      
      local_int_t data_num = n;
      local_int_t cnt = 0;
      for (size_t vl; data_num > 0; data_num -= vl) {
          // vsetvli t0, a2, e64, m1
          vl = __riscv_vsetvl_e64m1(data_num);

          // vle64.v load operations
          vfloat64m1_t v_x = __riscv_vle64_v_f64m1(xv, vl);

          // vfmacc.vv (Multiply-Accumulate)
          // This is more efficient than vmul + vadd separately
          // It calculates: v_acc = (v_x * v_y) + v_acc
          // For a simple dot product, multiply then reduce:
          vfloat64m1_t v_prod = __riscv_vfmul_vv_f64m1(v_x, v_x, vl);

          // vfredusum.vs (Floating-point unordered reduction sum)
          // Unordered is the fastest one (https://fprox.substack.com/p/risc-v-vector-reduction-operations)
          // v_sum = v_prod + v_sum
          v_sum = __riscv_vfredusum_vs_f64m1_f64m1(v_prod, v_sum, vl);
          double tmp_result = __riscv_vfmv_f_s_f64m1_f64(v_sum);

          // printf(">>> (NEW)tmp_result after cnt=%lld: %lf (xv[0] = %lf, xv[1] = %lf, xv[2] = %lf, xv[3] = %lf)\n", (long long)cnt, tmp_result, xv[0], xv[1], xv[2], xv[3]);

          xv += vl;
          yv += vl;
      }
      // vfmv.f.s: Move the first element of the vector register to a scalar double
      local_result = __riscv_vfmv_f_s_f64m1_f64(v_sum);
      // printf("(NOOPENMP) yv==xv\tn = %lld, local_result = %lf\n", (long long)n, local_result);
    } else {
      // for (local_int_t i=0; i<n; i++)
      //   local_result += xv[i]*yv[i];

      vfloat64m1_t v_sum = __riscv_vfmv_s_f_f64m1(0.0, __riscv_vsetvl_e64m1(1));
      local_int_t data_num = n;
      for (size_t vl; data_num > 0; data_num -= vl) {
          vl = __riscv_vsetvl_e64m1(data_num);

          vfloat64m1_t v_x = __riscv_vle64_v_f64m1(xv, vl);
          vfloat64m1_t v_y = __riscv_vle64_v_f64m1(yv, vl);

          vfloat64m1_t v_prod = __riscv_vfmul_vv_f64m1(v_x, v_y, vl);

          v_sum = __riscv_vfredusum_vs_f64m1_f64m1(v_prod, v_sum, vl);

          xv += vl;
          yv += vl;
      }
      local_result = __riscv_vfmv_f_s_f64m1_f64(v_sum);
      // printf("(NOOPENMP) yv!=xv\tn = %lld, local_result = %lf\n", (long long)n, local_result);
    }

  #else // ! HPCG_NO_OPENMP
    if (yv==xv) {
      #pragma omp parallel reduction(+:local_result)
      {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        // Simple static work distribution
        size_t len = (size_t)n;
        size_t chunk = len / nthreads;
        size_t start = tid * chunk;
        size_t end = (tid == nthreads - 1) ? len : start + chunk;
        
        if (start < end) {
            size_t local_data_num = end - start;
            const double *local_x = xv + start;
            
            vfloat64m1_t v_acc = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvl_e64m1(local_data_num));
            
            for (size_t vl; local_data_num > 0; local_data_num -= vl) {
                vl = __riscv_vsetvl_e64m1(local_data_num);
                vfloat64m1_t v_x = __riscv_vle64_v_f64m1(local_x, vl);
                v_acc = __riscv_vfmacc_vv_f64m1(v_acc, v_x, v_x, vl);
                local_x += vl;
            }
            vfloat64m1_t v_zero = __riscv_vfmv_s_f_f64m1(0.0, __riscv_vsetvl_e64m1(1));
            vfloat64m1_t v_res = __riscv_vfredusum_vs_f64m1_f64m1(v_acc, v_zero, __riscv_vsetvl_e64m1(-1));
            local_result += __riscv_vfmv_f_s_f64m1_f64(v_res);
        }
      }
      // printf("(OPENMP) yv==xv\tn = %lld, local_result = %lf\n", (long long)n, local_result);
    } else {
      #pragma omp parallel reduction(+:local_result)
      {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        // Simple static work distribution
        size_t len = (size_t)n;
        size_t chunk = len / nthreads;
        size_t start = tid * chunk;
        size_t end = (tid == nthreads - 1) ? len : start + chunk;
        
        if (start < end) {
            size_t local_data_num = end - start;
            const double *local_x = xv + start;
            const double *local_y = yv + start;
            
            vfloat64m1_t v_acc = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvl_e64m1(local_data_num));
            
            for (size_t vl; local_data_num > 0; local_data_num -= vl) {
                vl = __riscv_vsetvl_e64m1(local_data_num);
                vfloat64m1_t v_x = __riscv_vle64_v_f64m1(local_x, vl);
                vfloat64m1_t v_y = __riscv_vle64_v_f64m1(local_y, vl);
                v_acc = __riscv_vfmacc_vv_f64m1(v_acc, v_x, v_y, vl);
                local_x += vl;
                local_y += vl;
            }
            vfloat64m1_t v_zero = __riscv_vfmv_s_f_f64m1(0.0, __riscv_vsetvl_e64m1(1));
            vfloat64m1_t v_res = __riscv_vfredusum_vs_f64m1_f64m1(v_acc, v_zero, __riscv_vsetvl_e64m1(-1));
            local_result += __riscv_vfmv_f_s_f64m1_f64(v_res);
        }
      }
      // printf("(OPENMP) yv!=xv\tn = %lld, local_result = %lf\n", (long long)n, local_result);
    }

  #endif

  #ifndef HPCG_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    double t0 = mytimer();
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
        MPI_COMM_WORLD);
    result = global_result;
    time_allreduce += mytimer() - t0;
  #else
    time_allreduce += 0.0;
    result = local_result;
  #endif

  return 0;

}
