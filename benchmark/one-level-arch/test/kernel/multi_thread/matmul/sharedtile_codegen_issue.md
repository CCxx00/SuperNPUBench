# [LinxV5] C++ `SharedTile` 内联汇编丢失 `C.B.IOS S#n` 操作数，并导致 MC 目标文件生成崩溃

## 问题概述

使用 C++ TileOP API 调用 `TMOV_L2S_PUBLISH`，并将返回的
`SharedTile<TileRight<...>>` 作为 `TMATMUL` 的矩阵 B 时，编译器在生成
目标文件的过程中崩溃。

C++ 模板检查、类型检查和 LLVM IR 生成均能正常完成。但是，将同一份
LLVM IR 生成为汇编文本后，可以看到两条 `C.B.IOS` 指令的 Shared
寄存器操作数为空，没有输出预期的 `S#0`。继续生成目标文件时，
`LinxV5MCCodeEmitter::getImmOpValue()` 收到无法处理的表达式，最终触发
断言。

该问题可以在不包含 `TMATMUL_ACC`、`TMATMUL_ACC_FIXP`、Global Memory
访问、循环以及 SuperNPUBench 代码的最小用例中稳定复现。

## 环境信息

- 主机：Apple M1，arm64
- 操作系统：macOS 26.5.2，Darwin 25.5.0
- Clang/LLVM：15.0.4，启用了 assertions
- 编译目标：`linx64v5-unknown-linux-musl`
- LLVM 分支：`dev-llvm15_56`
- LLVM commit：`5a1be738c3136a31c3dbfd8b41428721d5eb4623`
  - commit 信息：`[LinxV5] Accept non-last B.IOT destination forms`
- Linx-TileOP-API 分支：`linx`
- Linx-TileOP-API commit：
  `4ef8a112789fd5373dd9feb054108a6ec1cb11a0`
- 工具链目录：
  `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build`

编译器版本：

```text
clang version 15.0.4
(linx64v5-musl-local 5a1be738c3136a31c3dbfd8b41428721d5eb4623)
Target: linx64v5-unknown-linux-musl
Thread model: posix
```

## 最小复现代码

将下面的代码保存为 `/tmp/shared_probe.cpp`：

```cpp
#include <common/pto_tileop.hpp>

using namespace pto;

using A = TileLeft<float, 16, 16>;
using B = TileRight<float, 16, 16>;
using C = Tile<Location::Vec, float, 16, 16, BLayout::RowMajor>;
using SharedB = SharedTile<B>;

void shared_probe(C &c, A &a, B &b) {
    SharedB shared_b = TMOV_L2S_PUBLISH(b);
    TMATMUL(c, a, shared_b);
}
```

## 复现步骤

先设置以下路径：

```bash
TOOLCHAIN=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
PTO_INCLUDE=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/include
```

编译最小用例：

```bash
"${TOOLCHAIN}/clang++" \
  -c -mlxbc -fenable-matrix -O2 -mcpu=janus \
  -mllvm -enable-all-vector-as-tilereg=true \
  -mllvm -linxv5-enable-clock-hand-opt=false \
  -std=c++20 \
  -I"${PTO_INCLUDE}" \
  /tmp/shared_probe.cpp \
  -o /tmp/shared_probe.o
```

## 实际结果

编译命令返回非零状态，没有生成 `/tmp/shared_probe.o`。

```text
Assertion failed:
(FixupKind != LinxV5::fixup_linxv5_invalid && "Unhandled expression!"),
function getImmOpValue,
file LinxV5MCCodeEmitter.cpp,
line 900.

1. <eof> parser at end of file
2. Code generation
3. Running pass 'Function Pass Manager' on module '/tmp/shared_probe.cpp'.
4. Running pass 'LinxV5 Assembly Printer' on function 'shared_probe(...)'
```

## 预期结果

代码应当能够正常编译并生成目标文件。`TMOV_L2S_PUBLISH` 返回的 Shared
寄存器应当被正确分配，并在两条 `C.B.IOS` 中输出，例如：

```asm
BSTART.TLSU TMOV.L2S.PUBLISH, FP32
C.B.IOS S#0
B.IOT ..., mask=1111, TSize=4, last

BSTART.CUBE TMATMUL, FP32
C.B.IOS S#0
B.IOT ...
```

## 汇编文本验证

LLVM IR 可以正常生成：

```bash
"${TOOLCHAIN}/clang++" \
  -S -emit-llvm \
  -mlxbc -fenable-matrix -O2 -mcpu=janus \
  -mllvm -enable-all-vector-as-tilereg=true \
  -mllvm -linxv5-enable-clock-hand-opt=false \
  -std=c++20 \
  -I"${PTO_INCLUDE}" \
  /tmp/shared_probe.cpp \
  -o /tmp/shared_probe.ll
```

LLVM IR 中可以看到 `Sr` 约束和 `S` modifier 均被保留：

```text
"C.B.IOS ${0:S}\0A...", "=@2Sr,@2Tr,..."
```

使用 `llc` 生成汇编文本：

```bash
"${TOOLCHAIN}/llc" \
  -mtriple=linx64v5 \
  -mcpu=janus \
  -enable-all-vector-as-tilereg=true \
  -linxv5-enable-clock-hand-opt=false \
  -filetype=asm \
  /tmp/shared_probe.ll \
  -o /tmp/shared_probe.s
```

该步骤能够完成，但生成的汇编中 `C.B.IOS` 操作数为空：

```asm
BSTART.TLSU TMOV.L2S.PUBLISH, FP32
C.B.IOS
B.IOT ..., mask=1111, TSize=4, last

BSTART.CUBE TMATMUL, FP32
C.B.IOS
B.IOT ...
```

因此，问题发生在 LLVM IR 到 LinxV5 汇编的 operand 打印过程中。生成
文本汇编时只表现为缺少操作数；使用 integrated assembler 生成目标文件
时，缺少操作数的 `C.B.IOS` 进一步导致 MC 编码器断言失败。

## 初步原因分析

TileOP API 中的 Shared 操作使用 `Sr` 内联汇编约束，并通过目标相关的
`S` modifier 输出 Shared 寄存器：

```cpp
asm volatile(
    "BSTART.TLSU TMOV.L2S.PUBLISH, %c[DataType]\n"
    "C.B.IOS %S[Shared]\n"
    "B.IOT %[src], mask=%c[PEMask], TSize=%c[TileSize], last\n"
    : [Shared] "=Sr"(result.handle_ref())
    : /* ... */);
```

`LinxV5AsmPrinter::PrintAsmOperand()` 中已经存在用于输出 Shared
寄存器的代码：

```cpp
case 'S':
  if (!MO.isReg())
    return true;
  OS << "S#" << STI->getRegisterInfo()->getEncodingValue(MO.getReg());
  return false;
```

但是，该函数会先调用通用的 `AsmPrinter::PrintAsmOperand()`：

```cpp
if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
  return false;
```

如果通用 printer 返回“已处理”，函数会直接返回，不再进入 LinxV5
自己的 `case 'S'`。结合最终生成的空 `C.B.IOS`，初步判断是通用 printer
提前消费了 `S` modifier，但没有输出 LinxV5 Shared 寄存器，导致
目标相关的 `case 'S'` 没有执行。

后续 integrated assembler 收到的是缺少 `S#n` 的 `C.B.IOS`。MC
目标文件生成阶段将该异常操作数作为无法识别的表达式处理，最终到达：

```cpp
assert(FixupKind != LinxV5::fixup_linxv5_invalid &&
       "Unhandled expression!");
```

相关源码位置：

- `llvm/lib/Target/LinxV5/LinxV5AsmPrinter.cpp`
  - `LinxV5AsmPrinter::PrintAsmOperand()`
- `llvm/lib/Target/LinxV5/MCTargetDesc/LinxV5MCCodeEmitter.cpp`
  - `LinxV5MCCodeEmitter::getImmOpValue()`
- `Linx-TileOP-API/include/jcore/template_asm.hpp`
  - `TMOV_L2S_PUBLISH()`
  - Shared-right `TMATMUL`

工具链已有的测试
`llvm/test/CodeGen/LinxV5/v5-shared-register-allocation.ll` 可以正确生成
`S#0`。但该测试通过 LLVM intrinsic 和目标 pseudo 指令完成 lowering，
没有覆盖 C++ TileOP API 的 `%S` 内联汇编 modifier 路径，因此无法发现
本问题。

## 建议修复方式

建议在调用通用 `AsmPrinter::PrintAsmOperand()` 之前，优先处理 LinxV5
目标相关的 `S` modifier；或者改用不会被通用 printer 提前消费的
modifier。

参考修改方式：

```cpp
bool LinxV5AsmPrinter::PrintAsmOperand(const MachineInstr *MI,
                                      unsigned OpNo,
                                      const char *ExtraCode,
                                      raw_ostream &OS) {
  const MachineOperand &MO = MI->getOperand(OpNo);

  if (ExtraCode && ExtraCode[0] == 'S' && ExtraCode[1] == '\0') {
    if (!MO.isReg())
      return true;
    OS << "S#"
       << STI->getRegisterInfo()->getEncodingValue(MO.getReg());
    return false;
  }

  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  // 保留其他目标相关的 operand 处理逻辑。
}
```

同时建议新增一条回归测试：

1. 在 LLVM IR 中构造同时使用 `Sr` 约束和 `${0:S}` modifier 的内联汇编。
2. 使用 `llc -filetype=obj` 编译。
3. 使用 `llvm-objdump` 检查 `C.B.IOS S#0`。
4. 同时覆盖 `TMOV.L2S.PUBLISH` 和 Shared-right `TMATMUL`。

现有基于 Shared intrinsic 的测试仍应保留，但它不能替代 C++ 内联汇编
路径的回归测试。

## 当前规避方式

当前工具链中没有可用的 C++ `SharedTile` 规避方案。将矩阵 B 保持为
普通的本地 `TileRight` 可以避免本次编译器崩溃，但无法使用 Shared
存储和 Shared-right `TMATMUL`。
