# Multi-thread GMMA FlashAttention operator.

TESTCASE ?= fa_2d_unroll_gmma

Sq ?= 512
Skv ?= 512
Tm ?= 128
Tk ?= 128

DEFINES += -DTsq=$(Sq)
DEFINES += -DTskv=$(Skv)
DEFINES += -DTm=$(Tm)
DEFINES += -DTk=$(Tk)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf

include ../../../common/Makefile.common
