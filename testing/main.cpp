// Run from testing/: make -j && make run
// Single-process numerical check: reference calls, RVV calls, comparisons.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <sdv_tracing.h>
#include "hpcg.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem_ref.hpp"
#include "SetupHalo_ref.hpp"
#include "GenerateCoarseProblem.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"

// Required by the existing HPCG sources linked by the Makefile.
std::ofstream HPCG_fout;

// Save a copy, because later MG calls overwrite the multigrid work vectors.
std::vector<double> snapshot(const Vector &v) {
  return std::vector<double>(v.values, v.values + v.localLength);
}

// Non-finite results fail. Small floating-point rounding differences are allowed.
bool compare(const char *name, const std::vector<double> &ref,
             const std::vector<double> &vec, int ref_status, int vec_status) {
  const double atol = 1e-12, rtol = 1e-10;
  bool pass = ref_status == 0 && vec_status == 0 && ref.size() == vec.size();
  double max_error = 0.0;
  for (size_t i = 0; i < std::min(ref.size(), vec.size()); ++i) {
    if (!std::isfinite(ref[i]) || !std::isfinite(vec[i])) {
      pass = false;
      max_error = std::numeric_limits<double>::infinity();
    } else {
      double error = std::abs(vec[i] - ref[i]);
      max_error = std::max(max_error, error);
      if (error > atol + rtol * std::abs(ref[i])) pass = false;
    }
  }
  std::cout << (pass ? "PASS " : "FAIL ") << name
            << " max_abs_error=" << max_error << '\n';
  return pass;
}

int main(int argc, char **argv) {
  trace_init();
  trace_disable(); // Exclude argument parsing, setup, and reference calls.
  // Accept the existing Makefile's arguments: nx ny nz threads output_directory.
  // The last argument is accepted for compatibility; outputs are saved in memory.
  int nx = 8, ny = 8, nz = 8, threads = 1;
  if (argc != 1 && argc != 6) {
    std::cerr << "Usage: " << argv[0] << " [nx ny nz threads output_directory]\n";
    return 2;
  }
  if (argc == 6) {
    try {
      for (int i = 1; i <= 4; ++i) {
        size_t used = 0;
        int value = std::stoi(argv[i], &used);
        if (argv[i][used] != '\0' || value <= 0) throw 0;
        if (i == 1) nx = value;
        if (i == 2) ny = value;
        if (i == 3) nz = value;
        if (i == 4) threads = value;
      }
    } catch (...) {
      std::cerr << "Grid dimensions and threads must be positive integers\n";
      return 2;
    }
  }
  // One coarse grid is sufficient to call restriction/prolongation and MG.
  if (nx % 2 || ny % 2 || nz % 2 ||
      static_cast<long long>(nx) * ny > std::numeric_limits<int>::max() / nz / 27) {
    std::cerr << "Use positive even grid dimensions within the 32-bit index range\n";
    return 2;
  }
#ifndef HPCG_NO_OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#else
  if (threads != 1) {
    std::cerr << "Serial build requires threads=1\n";
    return 2;
  }
#endif

  // 1. SETUP: both sets of calls share the same matrix and original inputs.
  Geometry *geometry = new Geometry;
  GenerateGeometry(1, 0, threads, 0, 0, 0, nx, ny, nz, 1, 1, 1, geometry);
  SparseMatrix A;
  InitializeSparseMatrix(A, geometry);
  Vector b, initial, exact;
  GenerateProblem_ref(A, &b, &initial, &exact);
  SetupHalo_ref(A);
  GenerateCoarseProblem(A);
  const local_int_t n = A.localNumberOfRows;
  const local_int_t ncol = A.localNumberOfColumns;

  Vector x, y, out_ref, out_vec;
  InitializeVector(x, ncol);
  InitializeVector(y, ncol);
  InitializeVector(out_ref, ncol);
  InitializeVector(out_vec, ncol);
  for (local_int_t i = 0; i < ncol; ++i) {
    x.values[i] = std::sin(0.17 * (i + 1));
    y.values[i] = std::cos(0.31 * (i + 1));
  }
  // Standalone restriction/prolongation inputs. These must be restored because
  // MG overwrites A.mgData->Axf, rc, and xc.
  const std::vector<double> axf_input = snapshot(x);
  std::vector<double> xc_input(A.mgData->xc->localLength);
  for (size_t i = 0; i < xc_input.size(); ++i) xc_input[i] = 0.1 * (i + 1);

  // 2. ALL REFERENCE CALLS FIRST. Save each result immediately in memory.
  std::cout << "Reference kernels\n";
  double dot_ref = 0.0, time_ref = 0.0;
  int dot_ref_status = ComputeDotProduct_ref(n, x, y, dot_ref, time_ref);

  ZeroVector(out_ref);
  int wax_ref_status = ComputeWAXPBY_ref(n, 0.3, x, -0.8, y, out_ref);
  const std::vector<double> wax_ref = snapshot(out_ref);

  ZeroVector(out_ref);
  int spmv_ref_status = ComputeSPMV_ref(A, x, out_ref);
  const std::vector<double> spmv_ref = snapshot(out_ref);

  CopyVector(x, out_ref); // SYMGS modifies its initial solution.
  int symgs_ref_status = ComputeSYMGS_ref(A, b, out_ref);
  const std::vector<double> symgs_ref = snapshot(out_ref);

  std::copy(axf_input.begin(), axf_input.end(), A.mgData->Axf->values);
  ZeroVector(*A.mgData->rc);
  int restriction_ref_status = ComputeRestriction_ref(A, b);
  const std::vector<double> restriction_ref = snapshot(*A.mgData->rc);

  std::copy(xc_input.begin(), xc_input.end(), A.mgData->xc->values);
  CopyVector(x, out_ref); // Prolongation adds into the existing fine vector.
  int prolongation_ref_status = ComputeProlongation_ref(A, out_ref);
  const std::vector<double> prolongation_ref = snapshot(out_ref);

  ZeroVector(*A.mgData->rc);
  ZeroVector(*A.mgData->xc);
  ZeroVector(*A.mgData->Axf);
  ZeroVector(out_ref);
  int mg_ref_status = ComputeMG_ref(A, b, out_ref);
  const std::vector<double> mg_ref = snapshot(out_ref);

  // 3. ALL VECTORISED CALLS SECOND, using the same original inputs.
  std::cout << "Vectorised kernels\n";
  bool optimized = true;
  double dot_vec = 0.0, time_vec = 0.0;
  trace_enable();
  trace_begin_region("DotProduct");
  int dot_vec_status = ComputeDotProduct(n, x, y, dot_vec, time_vec, optimized);
  trace_end_region("DotProduct");
  trace_disable();

  ZeroVector(out_vec);
  trace_enable();
  trace_begin_region("WAXPBY");
  int wax_vec_status = ComputeWAXPBY(n, 0.3, x, -0.8, y, out_vec, optimized);
  trace_end_region("WAXPBY");
  trace_disable();
  const std::vector<double> wax_vec = snapshot(out_vec);

  ZeroVector(out_vec);
  trace_enable();
  trace_begin_region("SPMV");
  int spmv_vec_status = ComputeSPMV(A, x, out_vec);
  trace_end_region("SPMV");
  trace_disable();
  const std::vector<double> spmv_vec = snapshot(out_vec);

  CopyVector(x, out_vec);
  trace_enable();
  trace_begin_region("SYMGS");
  int symgs_vec_status = ComputeSYMGS(A, b, out_vec);
  trace_end_region("SYMGS");
  trace_disable();
  const std::vector<double> symgs_vec = snapshot(out_vec);

  std::copy(axf_input.begin(), axf_input.end(), A.mgData->Axf->values);
  ZeroVector(*A.mgData->rc);
  trace_enable();
  trace_begin_region("Restriction");
  int restriction_vec_status = ComputeRestriction(A, b);
  trace_end_region("Restriction");
  trace_disable();
  const std::vector<double> restriction_vec = snapshot(*A.mgData->rc);

  std::copy(xc_input.begin(), xc_input.end(), A.mgData->xc->values);
  CopyVector(x, out_vec);
  trace_enable();
  trace_begin_region("Prolongation");
  int prolongation_vec_status = ComputeProlongation(A, out_vec);
  trace_end_region("Prolongation");
  trace_disable();
  const std::vector<double> prolongation_vec = snapshot(out_vec);

  ZeroVector(*A.mgData->rc);
  ZeroVector(*A.mgData->xc);
  ZeroVector(*A.mgData->Axf);
  ZeroVector(out_vec);
  trace_enable();
  trace_begin_region("MG");
  int mg_vec_status = ComputeMG(A, b, out_vec);
  trace_end_region("MG");
  trace_disable();
  const std::vector<double> mg_vec = snapshot(out_vec);

  // 4. COMPARE THE SAVED RESULTS.
  int failures = 0;
  failures += !compare("DotProduct", {dot_ref}, {dot_vec}, dot_ref_status, dot_vec_status);
  failures += !compare("WAXPBY", wax_ref, wax_vec, wax_ref_status, wax_vec_status);
  failures += !compare("SPMV", spmv_ref, spmv_vec, spmv_ref_status, spmv_vec_status);
  failures += !compare("SYMGS", symgs_ref, symgs_vec, symgs_ref_status, symgs_vec_status);
  failures += !compare("Restriction", restriction_ref, restriction_vec, restriction_ref_status, restriction_vec_status);
  failures += !compare("Prolongation", prolongation_ref, prolongation_vec, prolongation_ref_status, prolongation_vec_status);
  failures += !compare("MG", mg_ref, mg_vec, mg_ref_status, mg_vec_status);
  std::cout << "7 comparisons, " << failures << " failures\n";

  DeleteVector(x); DeleteVector(y);
  DeleteVector(out_ref); DeleteVector(out_vec);
  DeleteVector(b); DeleteVector(initial); DeleteVector(exact);
  DeleteMatrix(A);
  return failures ? 1 : 0;
}

