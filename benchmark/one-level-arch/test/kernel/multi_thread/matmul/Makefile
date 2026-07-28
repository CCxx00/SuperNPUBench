# Multi-thread GMMA matmul operator.

TESTCASE ?= matmul

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 64
tN ?= 128
tK ?= 64

DEFINES += -DBatch=$(B)
DEFINES += -DglobM=$(M)
DEFINES += -DglobN=$(N)
DEFINES += -DglobK=$(K)
DEFINES += -DtilM=$(tM)
DEFINES += -DtilN=$(tN)
DEFINES += -DtilK=$(tK)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul.cpp
TARGET = $(ELF_HEAD)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

include ../../../common/Makefile.common
