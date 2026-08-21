# Multi-thread GMMA FlashAttention operator.

TESTCASE ?= fa_2d_unroll_gmma

Sq ?= 512
Skv ?= 512
Tm ?= 16
Tk ?= 16

# Element data type for the Q/K/V and stored O tiles. The attention score (W)
# and output-accumulator tiles are always FP32 (see fa_2d_unroll_gmma.hpp).
# Supported: float, __bf16, __half.
DTYPE ?= float

ifeq ($(filter $(DTYPE),float __bf16 __half),)
$(error Unsupported DTYPE '$(DTYPE)'; supported: float, __bf16, __half)
endif

DEFINES += -DDTYPE=$(DTYPE)
DEFINES += -DTsq=$(Sq)
DEFINES += -DTskv=$(Skv)
DEFINES += -DTm=$(Tm)
DEFINES += -DTk=$(Tk)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp

# Keep the default (FP32) ELF name unchanged for backward compatibility;
# non-default dtypes get a _DType<dtype> suffix so binaries don't collide.
ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf
else
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_DType$(DTYPE).elf
endif

include ../../../common/Makefile.common
