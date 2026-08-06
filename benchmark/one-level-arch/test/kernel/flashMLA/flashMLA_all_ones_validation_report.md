# flashMLA 全 1 输入验证与问题定位报告

## 验证配置

使用全 1 输入验证 flashMLA tileop 算子：

```bash
make TESTCASE=flashMLA Sq=16 QHeadPerHK=1 NumBlocks=1 MaxBlocksPerSeq=1 \
  Dk=128 Dv=16 DChunk=64 VChunk=16 Tm=16 Tk=16 PageBlockSize=64 \
  debug_bin=on res_check=on \
  COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
```

运行器：

```bash
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun -t 1 -f \
  /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/flashMLA/elf/kernel_flashMLA/flashMLA_Sq16_QHeadPerHK1_NumBlocks1_Dk128_Dv16_DChunk64_VChunk16_Tm16_Tk16.elf
```

对比目录：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/compare/flashMLA_Sq16_QHeadPerHK1_NumBlocks1_Dk128_Dv16_DChunk64_VChunk16_Tm16_Tk16
```

## 当前结论

全 1 输入下当前 **已通过**。

本次使用拉取远端最新代码后重新构建的：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun
```

重新验证，`gfrun` 正常退出，最终输出与 reference 完全一致。

| 输出 | 期望 | 实际 | 结论 |
|---|---:|---:|---|
| `res.bin` | `1.0` | `1.0` | pass |
| `dbg_o.bin` | `1.0` | `1.0` | pass |
| `lse.bin` | `64` | `64` | pass |

## 分阶段数值对比

在 `Dk=128, DChunk=64, Tk=16, PageBlockSize=64` 下，每个 KV page 有 4 个 sub-block。
全 1 输入时，理论上：

- QK raw score = `128`
- QK scaled score = `128 / sqrt(128) = 11.313708`
- softmax probability = `1 / 64 = 0.015625`
- 每个 sub-block 的 `PV = prob @ V = 16 * 0.015625 = 0.25`
- 4 个 sub-block 累加后 `O = 1.0`

实际阶段 dump：

| sub | RawScore | Score | K dchunk means | Prob | V mean | PV mean | O mean |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | `128` | `11.313708` | `[1.0, 1.0]` | `0.015625` | `1.0` | `0.25` | `0.25` |
| 1 | `128` | `11.313708` | `[1.0, 1.0]` | `0.015625` | `1.0` | `0.25` | `0.5` |
| 2 | `128` | `11.313708` | `[1.0, 1.0]` | `0.015625` | `1.0` | `0.25` | `0.75` |
| 3 | `128` | `11.313708` | `[1.0, 1.0]` | `0.015625` | `1.0` | `0.25` | `1.0` |

这说明：

- `K` 的两个 DChunk 都由 `TLOAD` 正确读成了全 1。
- `V` 的 `TLOAD` 也是正确的。
- `TADD(tO, tO, tPV)` 连续累加不是主因；最小单测 4 次累加 `0.25` 可以得到全 1。
- QK raw score、softmax probability、PV 和 O 累加全部对齐 reference。

## 已做修正

本次复测没有发现新的不一致，因此没有继续修改 `flashMLA.cpp`。当前通过依赖的是前面已完成的两类修正：

- flashMLA kernel 侧规避复杂控制流中的 `TMATMUL_ACC` 累加链路，改为每个 DChunk 独立 `TMATMUL + ACCCVT`，再用 `TADD` 累加 partial score。
- gfrun simulator 侧修复 compact `[Rows,1]` row-vector tile 在 TEPL/TMA 路径中的 shape 和 store stride 处理。

### gfrun TEPL row-wise op

在 gfrun 本地工程中补充并修正了：

- `TROWEXPANDSUB`
- `TROWEXPANDMUL`
- `TROWSUM/TROWMAX` 的 row-vector compact 输出
- `TADD/TSUB/TMUL/TMAX/TEXP/TRECIP/TEXPANDS` 对 `[Rows,1]` row-vector 的 compact 处理

涉及文件：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/emulator/SoftCore.h
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/emulator/engine/TEPLEngine.cpp
```

这些修正后，first sub 的 `QK/exp/prob/V/PV` 已经可以和 reference 对齐。

### flashMLA QK 累加结构

原先 QK 的 DChunk 累加使用 accumulator 链：

```cpp
if (d_block == 0) {
    TMATMUL(tScoreAcc, tQ, tK);
} else {
    TMATMUL_ACC(tScoreAcc, tQ, tK);
}
```

在 flashMLA 的复杂控制流中，`Dk=128, DChunk=64` 时 sub1/sub3 曾出现 raw score 只有 `64` 的问题。为绕开这条不稳定的 ACC 跨 cube 指令累加路径，将 QK 改成每个 DChunk 单独 `TMATMUL`，再用 vector `TADD` 合并 partial score：

```cpp
TMATMUL(tScoreAcc0, tQ0, tK0);
ACCCVT(tScore, tScoreAcc0);
for (int d_block = 1; d_block < Db; ++d_block) {
    TMATMUL(tPartialAcc, tQ, tK);
    ACCCVT(tPartial, tPartialAcc);
    TADD(tScore, tScore, tPartial);
}
```

该改动后，所有 sub 的 raw score 都恢复为 `128`。

### gfrun row-vector TSTORE

`tSum`/LSE 的 tile 是 `[Rows,1]` compact row-vector。gfrun 原先在 `TSTORE(NORM)` 时仍按 physical tile width 读，导致只写出前两行。已在：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/emulator/engine/TMAEngine.cpp
```

中对 `LayOut::NORM && validCol == 1` 做 compact store 处理：

```cpp
tCopy.totalCol = 1;
tCopy.totalRow = tCopy.validRow;
```

修复后 `lse.bin` 全部为 `64`。

## 进一步定位

新增了最小 `TMATMUL_ACC` 单测：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/flashMLA/src/tmatmul_acc_unit.cpp
```

命令：

```bash
make TESTCASE=tmatmul_acc_unit M=16 N=16 K=128 DChunk=64 Rows=4 res_check=on \
  COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
```

全 1 输入下结果：

```text
row 0 mean = 128
row 1 mean = 128
row 2 mean = 128
row 3 mean = 128
```

说明 `TMATMUL + TMATMUL_ACC` 在简单控制流中可以正确工作。

## 当前剩余问题判断

当前全 1 case 已通过。保留的工程风险是：复杂控制流中的 `TMATMUL_ACC` 链路仍需进一步单独确认；flashMLA 目前通过改写为 `TMATMUL + ACCCVT + TADD` 避免依赖该路径。

## 验证流程建议

1. 先跑 `tmatmul_acc_unit`，确认基础 `TMATMUL + TMATMUL_ACC` 正常。
2. 再跑 flashMLA 全 1 debug case，检查 `debug_compare.log` 中每个 sub 的：
   - `RawScore`
   - `K dchunk means`
   - `Prob`
   - `PV`
   - `O`
3. 若 `K dchunk means` 都为 `1.0`，但 RawScore 为 `64`，继续沿反汇编 CFG 检查该 sub 的 `TMATMUL_ACC` block 是否被正确执行。
4. 修复后已达到：
   - 4 个 sub 的 `RawScore` 都为 `128`
   - 4 个 sub 的 `Prob` 都为 `0.015625`
   - 4 个 sub 的 `PV` 都为 `0.25`
   - final `O` 和 `res.bin` 都为 `1.0`
   - `lse.bin` 全部为 `64`
