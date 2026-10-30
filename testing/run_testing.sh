#!/bin/bash

# Configuration options (can be overridden by env variables)
# MODE="${MODE:-qemu}" # "qemu", "rave", "banana"
MODE="rave"
OPENMP=${OPENMP:-1}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Define root
TESTING_ROOT="$(cd "$(dirname "$0")" && pwd)"

QEMU_BIN="${QEMU_BIN:-${TESTING_ROOT}/../../../vec_sdv1/build/qemu-riscv64}"
if [ ! -f "$QEMU_BIN" ] && [ -f "/workspace/vec_sdv1/build/qemu-riscv64" ]; then
    QEMU_BIN="/workspace/vec_sdv1/build/qemu-riscv64"
fi
QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"
QEMU_CPU="${QEMU_CPU:-max,vlen=1024}"

echo "========================================="
echo "  Starting Testing Build and Run Script"
echo "  Mode: $MODE"
echo "========================================="

# 1. Load modules if needed
if [ "$MODE" = "rave" ]; then
    module purge || true
    echo "Loading modules for RAVE Emulation..."
    module load llvm/cross/EPI-development
    module load rave/development/EPI
    module load sdv_trace/development
elif [ "$MODE" = "banana" ]; then
    module purge || true
    echo "Loading standard RISC-V toolchain (Banana)..."
    module load llvm/EPI-development
fi

cd "$TESTING_ROOT" || { echo "Error: Could not enter TESTING_ROOT."; exit 1; }

# 2. Build
echo "--- 1. Building ---"
make clean

if [ "$MODE" = "qemu" ]; then
    make -j all \
        OCT_QEMU_EMULATION=1 \
        CXX=riscv64-linux-gnu-g++ \
        OPENMP=$OPENMP
else
    make -j all \
        OCT_QEMU_EMULATION=0 \
        CXX=clang++ \
        SCALAR_FLAGS='-fno-vectorize -fno-slp-vectorize' \
        RAVE_EMULATION=1 \
        SDV_TRACING=1 \
        OPENMP=$OPENMP
fi

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
    # "16 8 8"
    # "16 16 16"
    # "32 32 32"
    # "16 8 24"
    # "64 64 64"
    # "8 8 8"
    # "6 6 6"
    # "10 10 10"
    "16 8 8"
)

for size in "${SIZES[@]}"; do
    read -r nx ny nz <<< "$size"

    out_dir="results_test_${nx}_${ny}_${nz}"
    echo "----------------------------------------"
    echo "  Running size: $nx x $ny x $nz"
    echo "----------------------------------------"

    if [ "$MODE" = "qemu" ]; then
        echo "Running under QEMU emulation..."
        "$QEMU_BIN" -L "$QEMU_LD_PREFIX" -cpu "$QEMU_CPU" ./$TARGET $nx $ny $nz $OMP_NUM_THREADS $out_dir
    elif [ "$MODE" = "rave" ]; then
        echo "Running with rave..."
        rave ./$TARGET $nx $ny $nz $OMP_NUM_THREADS $out_dir
        # trace_rave_1_0 ./$TARGET $nx $ny $nz $OMP_NUM_THREADS $out_dir
    else
        echo "Running natively..."
        ./$TARGET $nx $ny $nz $OMP_NUM_THREADS $out_dir
    fi
done

echo "========================================="
echo "  Script finished."
echo "========================================="
