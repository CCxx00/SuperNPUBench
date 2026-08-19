DType ?= __half
G_A ?= 1
G_R ?= 8192
TILE_A ?= 1
TILE_R ?= 1024
POW_R ?= 4096

ifeq ($(TESTCASE), rms_norm_binary)
DEFINES += -DDType=$(DType)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType).elf
endif

ifeq ($(TESTCASE), rms_norm_binary_static)
DEFINES += -DDType=$(DType)
DEFINES += -DG_A=$(G_A) -DG_R=$(G_R) -DTILE_A=$(TILE_A) -DTILE_R=$(TILE_R)
DEFINES += -DPOW_R=$(POW_R)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_tA$(TILE_A)_tR$(TILE_R)_powR$(POW_R).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
