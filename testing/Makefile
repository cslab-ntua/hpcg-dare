# Run from testing/: make -j; make run
# Real RVV kernels: build/run on a compatible RISC-V machine.
# CXX = g++
CXX          = clang++

# RVV_FLAGS ?= -march=rv64gcv -mabi=lp64d
RVV_FLAGS ?= -mcpu=avispado -mepi

CXXFLAGS ?= -O3
# GCC defaults. For clang++, override with:
# SCALAR_FLAGS='-fno-vectorize -fno-slp-vectorize'
SCALAR_FLAGS ?= -fno-tree-vectorize -fno-tree-slp-vectorize
OPENMP ?= 1
ifeq ($(OPENMP),1)
OMP_FLAGS := -fopenmp
THREADS ?= 4
else ifeq ($(OPENMP),0)
OMP_DEFS := -DHPCG_NO_OPENMP
THREADS ?= 1
else
$(error OPENMP must be 0 or 1)
endif
BUILD_DIR := build_make_omp$(OPENMP)
TARGET := $(BUILD_DIR)/verify_kernels
OUTPUT_DIR ?= results
NX ?= 8
NY ?= 8
NZ ?= 8

SOURCES := ComputeDotProduct ComputeDotProduct_ref ComputeWAXPBY ComputeWAXPBY_ref \
 ComputeSPMV ComputeSPMV_ref ComputeSYMGS ComputeSYMGS_ref \
 ComputeRestriction ComputeRestriction_ref ComputeProlongation \
 ComputeProlongation_ref ComputeMG ComputeMG_ref CG CG_ref \
 GenerateGeometry GenerateProblem GenerateProblem_ref SetupHalo SetupHalo_ref \
 GenerateCoarseProblem ComputeOptimalShapeXYZ MixedBaseCounter mytimer
OBJECTS := $(BUILD_DIR)/main.o $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(SOURCES)))
REFERENCE_NAMES := ComputeDotProduct_ref ComputeWAXPBY_ref ComputeSPMV_ref \
 ComputeSYMGS_ref ComputeRestriction_ref ComputeProlongation_ref ComputeMG_ref CG_ref
REFERENCE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(REFERENCE_NAMES)))
EPI_WORKAROUNDS = -fno-lto -fmacro-backtrace-limit=0 -flax-vector-conversions \
                  -Wno-vla-cxx-extension -Xclang -target-feature -Xclang +does-not-implement-vszext \
                  -Xclang -target-feature -Xclang +does-not-implement-tu
# COMMON_FLAGS = $(RVV_FLAGS) -std=c++11 $(OMP_FLAGS) -fno-fast-math -ffp-contract=off
COMMON_FLAGS = $(RVV_FLAGS) -std=c++11 $(OMP_FLAGS) -ffast-math -fno-slp-vectorize $(EPI_WORKAROUNDS)

# INCLUDE_FLAGS = -I../src -DHPCG_NO_MPI $(OMP_DEFS)
INCLUDE_FLAGS = -I../src -DHPCG_NO_MPI -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_DEBUG $(OMP_DEFS)

ifeq ($(RAVE_EMULATION), 1)
ifeq ($(SDV_TRACING), 1)
COMMON_FLAGS += -DSDV_TRACING $(SDV_TRACE_INCL)
LDFLAGS += $(SDV_TRACE_C_LINK)
endif
endif

.PHONY: all run test clean
all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

# Disable automatic vectorization in the reference oracle. Explicit RVV
# restriction/prolongation in those same source files remain active.
$(REFERENCE_OBJECTS): EXTRA_FLAGS = $(SCALAR_FLAGS)

$(BUILD_DIR)/main.o: main.cpp Makefile | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDE_FLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(SCALAR_FLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: ../src/%.cpp Makefile | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDE_FLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(EXTRA_FLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET) $(NX) $(NY) $(NZ) $(THREADS) $(OUTPUT_DIR)

test: $(TARGET)
	./$(TARGET) 8 8 8 1 results_make_threads1
ifeq ($(OPENMP),1)
	./$(TARGET) 8 8 8 4 results_make_threads4
	./$(TARGET) 16 8 24 3 results_make_rectangular
endif

clean:
	$(RM) $(OBJECTS) $(OBJECTS:.o=.d) $(TARGET)

-include $(OBJECTS:.o=.d)

