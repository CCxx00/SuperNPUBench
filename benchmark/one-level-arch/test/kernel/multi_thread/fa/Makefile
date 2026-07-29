# Multi-thread GMMA FlashAttention operator.

TESTCASE ?= fa_2d_unroll_gmma

Sq ?= 512
Skv ?= 512
Tm ?= 32
Tk ?= 32

TILE_ELEM_LIMIT := 8192
HEAD_DIM := 128
TILE_SHAPE_VALID := $(shell \
	test $(Tm) -gt 0 && test $(Tk) -gt 0 && \
	test $$(( $(Tm) * $(HEAD_DIM) )) -lt $(TILE_ELEM_LIMIT) && \
	test $$(( $(Tm) * $(Tk) )) -lt $(TILE_ELEM_LIMIT) && \
	test $$(( $(Tk) * $(HEAD_DIM) )) -lt $(TILE_ELEM_LIMIT) && \
	echo yes)

ifneq ($(TILE_SHAPE_VALID),yes)
$(error Invalid FA tile shape: require Tm*128, Tm*Tk, and Tk*128 to each be < 8192 elements; got Tm=$(Tm), Tk=$(Tk))
endif

DEFINES += -DTsq=$(Sq)
DEFINES += -DTskv=$(Skv)
DEFINES += -DTm=$(Tm)
DEFINES += -DTk=$(Tk)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf

include ../../../common/Makefile.common
