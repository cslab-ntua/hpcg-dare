#!/bin/bash
# --- Execution / Emulation Mode ---
# Options: "qemu", "rave", "banana" (baremetal)
MODE="qemu"

# Define the absolute path to the main HPCG directory
HPCG_ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- QEMU Emulation Settings ---
QEMU_BIN="${QEMU_BIN:-${HPCG_ROOT}/../../vec_sdv1/build/qemu-riscv64}"
if [ ! -f "$QEMU_BIN" ] && [ -f "/workspace/vec_sdv1/build/qemu-riscv64" ]; then
    QEMU_BIN="/workspace/vec_sdv1/build/qemu-riscv64"
fi
QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"
QEMU_CPU="${QEMU_CPU:-max,vlen=1024}"

# --- Environment & Configuration Setup ---
if [ "$MODE" = "qemu" ]; then
    BUILD_CONFIG="QEMU_GCC_OMP"
elif [ "$MODE" = "rave" ]; then
    module load llvm/cross/EPI-development
    module load rave/development/EPI
    module load sdv_trace/development
    BUILD_CONFIG="CLANG_OMP"
elif [ "$MODE" = "banana" ]; then
    module load llvm/EPI-development
    BUILD_CONFIG="CLANG_OMP"
fi

# THREADS_TO_TEST=(1 2 4 8)
THREADS_TO_TEST=(1)
MAX_THREADS=8 # Used for GOMP_CPU_AFFINITY

echo "========================================="
echo "  Starting HPCG Build and Test Script"
echo "  Mode: $MODE"
echo "  Root Directory: $HPCG_ROOT"
echo "========================================="

# 1. Reset and Build
# We execute these commands from the project root directory
cd "$HPCG_ROOT" || { echo "Error: Could not enter HPCG_ROOT."; exit 1; }

echo "--- 1. Cleaning and Configuring ---"
rm -rf build
mkdir -p build
cd build || { echo "Error: Could not enter 'build' directory."; exit 1; }

echo "Running configure with $BUILD_CONFIG..."
../configure "$BUILD_CONFIG"

echo "--- 2. Building Project (make -j) ---"
# Using the number of available cores for parallel compilation
# You can set a fixed number like 'make -j8' if preferred
make -j

# Check if the executable was successfully created
if [ ! -f bin/xhpcg ]; then
    echo "ERROR: HPCG executable 'bin/xhpcg' was not found after building."
    exit 1
fi

echo "Build successful."

# 3. Running Benchmarks
cd bin || { echo "Error: Could not enter 'bin' directory."; exit 1; }

echo "--- 3. Running Benchmarks with OMP_NUM_THREADS: ${THREADS_TO_TEST[@]} ---"

# Set the CPU affinity to cover the maximum threads we will use (0-7 for 8 threads)
export GOMP_CPU_AFFINITY="0-$(($MAX_THREADS - 1))"
echo "GOMP_CPU_AFFINITY set to: $GOMP_CPU_AFFINITY"

# Loop through the desired thread counts
for threads in "${THREADS_TO_TEST[@]}"; do
    echo "----------------------------------------"
    echo "  Running xhpcg with OMP_NUM_THREADS=$threads"
    
    # Set the OMP environment variable
    export OMP_NUM_THREADS="$threads"
    
    # Run the benchmark
    if [ "$MODE" = "qemu" ]; then
        { time "$QEMU_BIN" -L "$QEMU_LD_PREFIX" -cpu "$QEMU_CPU" ./xhpcg ; } 2>&1 | tee "hpcg_run_${threads}_threads.log"
    elif [ "$MODE" = "rave" ]; then
        { time trace_rave_1_0 ./xhpcg ; } 2>&1 | tee "hpcg_run_${threads}_threads.log"
    else
        { time ./xhpcg ; } 2>&1 | tee "hpcg_run_${threads}_threads.log"
    fi

    echo "  Completed run with $threads threads."
done

grep "^GFLOP/s Summary" hpcg*.txt

echo "========================================="
echo "  Script finished. Results are in the bin directory."
echo "========================================="