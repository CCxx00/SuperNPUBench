# Conv2d 1x1 RowMajor dynamic kernel test Makefile
#
# Runtime parameters via Conv2dParam struct, GM shapes use RowMajor<-1,-1>.
# Tile storage dims (tM, tN, tK) still compile-time via template.
#
# Usage:
#   make TYPE=FP32 IN_H=4 IN_W=4 IN_C=16 OUT_C=16 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d_rm_dyn \
#        CC_OPTS="-DRES_CHECK -DENABLE_BINARY_OUTPUT -include /tmp/conv2d_rm_dyn_chk_dir.h"
#   make TYPE=FP16 IN_H=4 IN_W=4 IN_C=16 OUT_C=16 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d_rm_dyn \
#        CC_OPTS="-DRES_CHECK -DENABLE_BINARY_OUTPUT -include /tmp/conv2d_rm_dyn_chk_dir.h"

ifeq ($(TYPE), FP32)
    SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d_rm_dyn.cpp
    DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
               -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
    DEFINES += -DCONV_FP32
    TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK).elf
endif

ifeq ($(TYPE), FP16)
    SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d_rm_dyn.cpp
    DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
               -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
    DEFINES += -DCONV_FP16
    TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
