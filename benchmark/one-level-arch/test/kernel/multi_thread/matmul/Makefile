# Multi-thread GMMA matmul operator.

TESTCASE ?= matmul

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 64
tN ?= 64
tK ?= 64

TILE_ELEM_LIMIT := 8192
TILE_SHAPE_VALID := $(shell \
	test $(tM) -gt 0 && test $(tN) -gt 0 && test $(tK) -gt 0 && \
	test $$(( $(tM) * $(tK) )) -lt $(TILE_ELEM_LIMIT) && \
	test $$(( $(tK) * $(tN) )) -lt $(TILE_ELEM_LIMIT) && \
	test $$(( $(tM) * $(tN) )) -lt $(TILE_ELEM_LIMIT) && \
	echo yes)

ifneq ($(TILE_SHAPE_VALID),yes)
$(error Invalid matmul tile shape: require tM*tK, tK*tN, and tM*tN to each be < 8192 elements; got tM=$(tM), tN=$(tN), tK=$(tK))
endif

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
