# Multi-thread vector-add operator.

TESTCASE ?= vec

TileRows ?= 16
TileCols ?= 16

DEFINES += -DTileRows=$(TileRows)
DEFINES += -DTileCols=$(TileCols)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/tadd.cpp
TARGET = $(ELF_HEAD)_Rows$(TileRows)_Cols$(TileCols).elf

include ../../../common/Makefile.common
