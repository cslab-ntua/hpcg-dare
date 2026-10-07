#!/bin/bash
# Banana
module load llvm/EPI-development

# Pioneer
# module load llvm/EPI-0.7-development

# RAVE emulation
# module load llvm/cross/EPI-development
# module load rave/development/EPI
# module load sdv_trace/development

# --- Configuration ---
BUILD_CONFIG="CLANG_OMP"
# BUILD_CONFIG="CLANG_OMP_PIONEER"
# THREADS_TO_TEST=(1 2 4 8)
THREADS_TO_TEST=(8)
# THREADS_TO_TEST=(1)
MAX_THREADS=8 # Used for GOMP_CPU_AFFINITY

# Define the absolute path to the main HPCG directory
# This ensures the script works regardless of where it is called from.
HPCG_ROOT="$(dirname "$0")"

echo "========================================="
echo "  Starting HPCG Build and Test Script"
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
    
    # Run the benchmark and time it
    # The 'time' command outputs to stderr, so we redirect stderr to a log file 
    # and also to the console (using tee) for easier viewing.
    # The actual output of ./xhpcg goes to stdout.
    { time ./xhpcg ; } 2>&1 | tee "hpcg_run_${threads}_threads.log"
    # { time trace_rave_1_0 ./xhpcg ; } 2>&1 | tee "hpcg_run_${threads}_threads.log"
    # time rave ./xhpcg 2>&1

    echo "  Completed run with $threads threads."
done

grep "^GFLOP/s Summary" hpcg*.txt

echo "========================================="
echo "  Script finished. Results are in the bin directory."
echo "========================================="