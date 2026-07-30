# Multi-thread vector operators.

TESTCASE ?= vec

TileRows ?= 16
TileCols ?= 16

PE_NUM := 4
TILE_ELEM_LIMIT := 8192
TILE_SHAPE_VALID := $(shell \
	test $(TileRows) -gt 0 && test $(TileCols) -gt 0 && \
	test $$(( $(TileRows) % $(PE_NUM) )) -eq 0 && \
	test $$(( ($(TileRows) / $(PE_NUM)) * $(TileCols) )) -lt $(TILE_ELEM_LIMIT) && \
	echo yes)

ifneq ($(TILE_SHAPE_VALID),yes)
$(error Invalid vector tile shape: TileRows must be divisible by 4 and (TileRows/4)*TileCols must be < 8192 elements; got TileRows=$(TileRows), TileCols=$(TileCols))
endif

DEFINES += -DTileRows=$(TileRows)
DEFINES += -DTileCols=$(TileCols)

ifeq ($(TESTCASE),trowsum)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/trowsum.cpp
TARGET = $(ELF_HEAD)_trowsum_Rows$(TileRows)_Cols$(TileCols).elf
else ifneq ($(filter $(TESTCASE),vec tadd),)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/tadd.cpp
TARGET = $(ELF_HEAD)_Rows$(TileRows)_Cols$(TileCols).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected vec, tadd, or trowsum)
endif

include ../../../common/Makefile.common
