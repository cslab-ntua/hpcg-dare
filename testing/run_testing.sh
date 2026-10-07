#!/bin/bash
RAVE_EMULATION=1
SDV_TRACING=1

# Configuration options (can be overridden by env variables)
RAVE_EMULATION=${RAVE_EMULATION:-0}
SDV_TRACING=${SDV_TRACING:-0}
OPENMP=${OPENMP:-1}
THREADS=${THREADS:-1}

# Define root
TESTING_ROOT="$(dirname "$0")"

echo "========================================="
echo "  Starting Testing Build and Run Script"
echo "  RAVE_EMULATION: $RAVE_EMULATION"
echo "  SDV_TRACING: $SDV_TRACING"
echo "========================================="

# 1. Load correct modules
module purge || true

if [ "$RAVE_EMULATION" -eq 1 ]; then
    echo "Loading modules for RAVE Emulation..."
    module load llvm/cross/EPI-development
    module load rave/development/EPI
    if [ "$SDV_TRACING" -eq 1 ]; then
        echo "Loading modules for SDV Tracing..."
        module load sdv_trace/development
    fi
else
    echo "Loading standard RISC-V toolchain (Banana)..."
    module load llvm/EPI-development
fi

cd "$TESTING_ROOT" || { echo "Error: Could not enter TESTING_ROOT."; exit 1; }

# 2. Build
echo "--- 1. Building ---"
make clean
# Override compiler to clang++ since we loaded llvm modules
# Also adjust SCALAR_FLAGS for clang++
make -j all \
    CXX=clang++ \
    SCALAR_FLAGS='-fno-vectorize -fno-slp-vectorize' \
    RAVE_EMULATION=$RAVE_EMULATION \
    SDV_TRACING=$SDV_TRACING \
    OPENMP=$OPENMP

# 3. Run
echo "--- 2. Running ---"
# Set target executable path
TARGET="build_make_omp${OPENMP}/verify_kernels"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Executable '$TARGET' not found."
    exit 1
fi

# Define test sizes
SIZES=(
    # "8 8 8"
    "16 8 8"
    # "16 16 16"
    # "32 32 32"
    # "16 8 24"
    # "64 64 64"
)

for size in "${SIZES[@]}"; do
    read -r nx ny nz <<< "$size"

    out_dir="results_test_${nx}_${ny}_${nz}"
    echo "----------------------------------------"
    echo "  Running size: $nx x $ny x $nz"
    echo "----------------------------------------"

    if [ "$RAVE_EMULATION" -eq 1 ]; then
        if [ "$SDV_TRACING" -eq 1 ]; then
            echo "Running with trace_rave_1_0..."
            trace_rave_1_0 ./$TARGET $nx $ny $nz $THREADS $out_dir
        else
            echo "Running with rave..."
            rave ./$TARGET $nx $ny $nz $THREADS $out_dir
        fi
    else
        echo "Running natively..."
        ./$TARGET $nx $ny $nz $THREADS $out_dir
    fi
done

echo "========================================="
echo "  Script finished."
echo "========================================="
