#!/bin/bash
# SuperNPUBench — 精简测试运行脚本（gfrun + gfsim）
# 用法: bash run_smoke.sh /path/to/SuperScalarModel

SSM=${1:-/Users/liyi/Documents/GitHub/SuperScalarModel}
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/output/kernel

if [ ! -x "$SSM/bin/gfrun" ]; then
    echo "ERROR: gfrun not found at $SSM/bin/gfrun"
    echo "Usage: bash run_smoke.sh /path/to/SuperScalarModel"
    exit 1
fi

PASS=0; FAIL=0; SKIP=0
GFSIM_PASS=0; GFSIM_FAIL=0

echo "=========================================="
echo "  SuperNPUBench Smoke Test — 运行"
echo "  gfrun: $SSM/bin/gfrun"
echo "  gfsim: $SSM/bin/gfsim"
echo "=========================================="

find_elf() {
    local pattern=$1
    find "$BENCH" -name "*.elf" -type f | grep -i "$pattern" | head -1
}

run_test() {
    local name=$1
    local elf=$2

    echo ""
    echo "--- $name ---"

    if [ -z "$elf" ] || [ ! -f "$elf" ]; then
        echo "  ⚠ ELF not found, skipping"
        SKIP=$((SKIP+1))
        return
    fi

    # gfrun (功能验证)
    local gfrun_out
    gfrun_out=$("$SSM/bin/gfrun" -f "$elf" 2>&1)
    if echo "$gfrun_out" | grep -q "R2 = 0"; then
        local blocks insts
        blocks=$(echo "$gfrun_out" | grep "Block number" | tail -1 | grep -oE '[0-9]+' | head -1)
        insts=$(echo "$gfrun_out" | grep "Inst number" | tail -1 | grep -oE '[0-9]+' | head -1)
        echo "  ✓ gfrun PASS (blocks=$blocks, insts=$insts)"
        PASS=$((PASS+1))
    else
        echo "  ✗ gfrun FAIL"
        FAIL=$((FAIL+1))
    fi

    # gfsim (时序验证) — 超时 60s 自动跳过
    local gfsim_out gfsim_cycles
    gfsim_out=$(timeout 60 "$SSM/bin/gfsim" -f "$elf" 2>&1)
    if [ $? -eq 0 ]; then
        gfsim_cycles=$(echo "$gfsim_out" | grep "Total Cycles\.\." | head -1 | grep -oE '[0-9]+$')
        if [ -n "$gfsim_cycles" ]; then
            echo "  ✓ gfsim PASS (cycles=$gfsim_cycles)"
            GFSIM_PASS=$((GFSIM_PASS+1))
        else
            echo "  ✗ gfsim FAIL (no cycle output)"
            GFSIM_FAIL=$((GFSIM_FAIL+1))
        fi
    else
        echo "  ⚠ gfsim TIMEOUT/CRASH"
        GFSIM_FAIL=$((GFSIM_FAIL+1))
    fi
}

# --- 1. matmul ---
run_test "matmul FP32 256×256" "$(find_elf "matmul_MASK_MASK_FP32_M256.*tM32_tN32_tK32")"

# --- 2. sfa ---
run_test "sfa Sq256" "$(find_elf "sfa_Sq256")"

# --- 3. flashMLA ---
run_test "flashMLA Sq64" "$(find_elf "flashMLA_Sq64")"

# --- 4. gelu ---
run_test "gelu bf16" "$(find_elf "gelu")"

# --- 5. reducesum_col ---
run_test "reducesum_col" "$(find_elf "reducesum_col")"

# --- 6. reducemax_row ---
run_test "reducemax_row" "$(find_elf "reducemax_row")"

# --- 7. broadcast vec_07 ---
run_test "broadcast vec_07" "$(find_elf "broadcast_vec_07")"

# --- 8. concat gather ---
run_test "concat gather" "$(find_elf "concat_gather.*int32")"

# --- 9. transpose 2D ---
run_test "transpose 2D" "$(find_elf "transpose.*IN1476")"

# --- 10. gather ---
run_test "gather" "$(find_elf "gather_DType__fp32")"

# --- 11. control hashtable ---
run_test "hashtable_lookup" "$(find_elf "hashtable.*col256.*debug_off")"

# --- 12. sort topk ---
run_test "topk" "$(find_elf "topk.elf")"

# --- 13. multi_thread/matmul ---
run_test "multi_thread/matmul" "$(find_elf "matmul_GMMA")"

# --- 14. multi_thread/vec ---
run_test "multi_thread/vec" "$(find_elf "tadd")"

# --- 15. multi_thread/fa ---
run_test "multi_thread/fa" "$(find_elf "fa_2d_unroll_gmma")"

echo ""
echo "=========================================="
echo "  Smoke Test 完成"
echo "  gfrun:  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "  gfsim:  PASS=$GFSIM_PASS  FAIL=$GFSIM_FAIL"
echo "=========================================="
