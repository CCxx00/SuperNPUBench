# flashMLA dense decode tileop testcase

ifeq ($(TESTCASE), flashMLA)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/flashMLA.cpp
   Sq ?= 64
   QHeadPerHK ?= 1
   NumBlocks ?= 2
   MaxBlocksPerSeq ?= 2
   Dk ?= 512
   Dv ?= 512
   DChunk ?= 128
   VChunk ?= 128
   Tm ?= 16
   Tk ?= 16
   PageBlockSize ?= 64
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DQHeadPerHK=$(QHeadPerHK)
   DEFINES += -DNumBlocks=$(NumBlocks)
   DEFINES += -DMaxBlocksPerSeq=$(MaxBlocksPerSeq)
   DEFINES += -DDk=$(Dk)
   DEFINES += -DDv=$(Dv)
   DEFINES += -DDChunk=$(DChunk)
   DEFINES += -DVChunk=$(VChunk)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DPageBlockSize=$(PageBlockSize)
   ifeq ($(debug_tile), on)
      DEFINES += -DFLASHMLA_DEBUG_TILE
   endif
   ifeq ($(debug_bin), on)
      DEFINES += -DFLASHMLA_DEBUG_BIN
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_QHeadPerHK$(QHeadPerHK)_NumBlocks$(NumBlocks)_Dk$(Dk)_Dv$(Dv)_DChunk$(DChunk)_VChunk$(VChunk)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), tmatmul_unit)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/tmatmul_unit.cpp
   M ?= 16
   N ?= 16
   K ?= 128
   DEFINES += -DgM=$(M)
   DEFINES += -DgN=$(N)
   DEFINES += -DgK=$(K)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_M$(M)_N$(N)_K$(K).elf
endif

ifeq ($(TESTCASE), tadd_accum_unit)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/tadd_accum_unit.cpp
   TileRows ?= 16
   TileCols ?= 16
   DEFINES += -DTileRows=$(TileRows)
   DEFINES += -DTileCols=$(TileCols)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Rows$(TileRows)_Cols$(TileCols).elf
endif

ifeq ($(TESTCASE), tmatmul_acc_unit)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/tmatmul_acc_unit.cpp
   M ?= 16
   N ?= 16
   K ?= 128
   DChunk ?= 64
   Rows ?= 4
   DEFINES += -DgM=$(M)
   DEFINES += -DgN=$(N)
   DEFINES += -DgK=$(K)
   DEFINES += -DgDChunk=$(DChunk)
   DEFINES += -DgRows=$(Rows)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_M$(M)_N$(N)_K$(K)_DChunk$(DChunk)_Rows$(Rows).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
