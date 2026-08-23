# [Linx-TileOP-API] SharedTile 16KB/32KB 容量检查与 B.IOS TSize 编码错误

## Issue 摘要

使用 Linx-TileOP-API 编译 4 PE 协作算子时，`SharedTile` 需要表达整个
Core 共享的 16KB 或 32KB Tile。原 API 在容量检查和指令编码两个层面仍
沿用了单 PE Local Tile 的规则，导致 32KB Shared Tile 先被模板
`static_assert` 拒绝；放开检查后，又无法生成正确的 `B.IOS TSize=7`。

该问题的根本原因是 API 混用了两套不同的 Tile 容量编码：

- Local Tile 通过 `B.IOT` 表达，容量为 128B 至 8KB；
- Shared Tile 通过 `B.IOS` 表达，容量为 512B 至 32KB；
- 两者虽然都使用 3-bit TSize，但数值与容量的对应关系不同。

修复后，Shared Tile 拥有独立的容量合法性检查和 TSize 映射，32KB
Shared `TLOAD` 可以正确编译为 `B.IOS ... ->S#n<32KB>`。

## 影响范围

- 仓库：`linx-toolchain-build/src/Linx-TileOP-API`
- 工作分支：`temp/shared-32kb-debug`
- 涉及文件：
  - `include/jcore/type.hpp`
  - `include/jcore/template_asm.hpp`
- 涉及 API：
  - `TLOAD(..., SharedTile<...>)`
  - `TLOAD<...>() -> SharedTile<...>`
  - `TMOV_L2S_INSERT`
  - `TMOV_L2S_PUBLISH`
  - Shared `TSTORE`/`TSTORE.SPART` 的容量合法性检查

## 复现配置

以 multi-thread matmul 为例：

```text
M=256, N=256, K=256
tM=64, tN=64, tK=128
dtype=FP32
PE 数量=4
```

对应 Tile 容量：

| Tile | Shape | 容量 | 类型 |
|---|---:|---:|---|
| A | `64 x 128` | 32KB | `SharedTile<TileLeft<...>>` |
| B | `128 x 64` | 32KB | `SharedTile<TileRight<...>>` |
| C（单 PE） | `16 x 64` | 4KB | Local Tile |

Shared A/B 是整个 Core 可见的 32KB Tile，4 个 PE 共同参与后续矩阵计算；
它们不能按单 PE Local Tile 的 8KB 上限检查。

## 原始报错

### 第一阶段：Shared Tile 被 Local Tile 上限拒绝

原代码对 Shared 目的 `TLOAD` 使用：

```cpp
static_assert(
    tile_type_traits<shp_dtype>::IsValidActiveSize,
    "TLOAD Shared dst logical Tile size must be 128 B..8 KB (TSize=1..7)");
```

`IsValidActiveSize` 是普通 Local Tile 的检查，只允许 128B 至 8KB。
因此 16KB/32KB `SharedTile` 在 C++ 模板实例化阶段直接失败，典型错误为：

```text
static assertion failed:
TLOAD Shared dst logical Tile size must be 128 B..8 KB (TSize=1..7)
```

同样的问题也存在于 Shared `TSTORE`、`TSTORE.SPART`、
`TMOV_L2S_INSERT` 和 `TMOV_L2S_PUBLISH`。

### 第二阶段：放开容量检查后仍使用错误的 TSize 编码

最初增加 `IsValidSharedActiveSize` 后，32KB Tile 可以通过模板检查，但
汇编模板仍然使用：

```cpp
[TileSize] "i"(tile_type_traits<shp_dtype>::TilesizeCode)
```

`TilesizeCode` 来自 `mapBytesToEnum()`，只定义了 Local `B.IOT` 的
128B 至 8KB 映射。32KB 会落入 `__tilesize_unknown`，因此通过检查并不
等于可以正确编码。

与此同时，Shared B.IOS 的输出仍使用普通 Tile 的 `%Z` modifier：

```cpp
"B.IOS mask=%c[PEMask], ->%S[Shared]<%Z[TileSize]>"
```

这条路径仍按 B.IOT 容量枚举解释 TileSize，无法正确表达 B.IOS 的
16KB/32KB 编码。也就是说，问题不是只需要放宽一个 `static_assert`，
还必须把 Shared Tile 的 TSize 生成路径完全独立出来。

## 根因

### B.IOT 容量编码

Local Tile 使用 `B.IOT`，API 中的 `TilesizeCode` 对应：

| Local 容量 | B.IOT TSize |
|---:|---:|
| 128B | 1 |
| 256B | 2 |
| 512B | 3 |
| 1KB | 4 |
| 2KB | 5 |
| 4KB | 6 |
| 8KB | 7 |

### B.IOS 容量编码

Shared Tile 使用 `B.IOS`，其 TSize 应当对应：

| Shared 容量 | B.IOS TSize |
|---:|---:|
| 512B | 1 |
| 1KB | 2 |
| 2KB | 3 |
| 4KB | 4 |
| 8KB | 5 |
| 16KB | 6 |
| 32KB | 7 |

例如数值 7 在 B.IOT 中表示 8KB，在 B.IOS 中则表示 32KB。原 API 用
同一个 `TilesizeCode` 和 `%Z` modifier 处理两者，因此容量检查、立即数
生成和汇编显示无法同时正确。

## 修复内容

### 1. 增加 Shared Tile 专用容量映射

在 `include/jcore/type.hpp` 中新增：

```cpp
static constexpr int mapBytesToSharedEnum(std::size_t b) {
  return
    b == 512   ? 1 :
    b == 1024  ? 2 :
    b == 2048  ? 3 :
    b == 4096  ? 4 :
    b == 8192  ? 5 :
    b == 16384 ? 6 :
    b == 32768 ? 7 :
    __tilesize_unknown;
}

static constexpr int SharedTilesizeCode =
    mapBytesToSharedEnum(PETileBytes);
```

这样 Local Tile 继续使用 `TilesizeCode`，Shared Tile 使用
`SharedTilesizeCode`，不再共享同一容量枚举。

### 2. 修正 Shared Tile 合法性检查

`IsValidSharedActiveSize` 改为基于 Shared 编码判断：

```cpp
static constexpr bool IsValidSharedActiveSize =
    SharedTilesizeCode >= 1 && SharedTilesizeCode <= 7;
```

合法范围严格限制为：

```text
512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB
```

非 2 的幂或超出范围的容量仍在编译期拒绝。

### 3. Shared 目的指令改用 SharedTilesizeCode

在 `include/jcore/template_asm.hpp` 中，以下指令的目标容量参数由
`TilesizeCode` 替换为 `SharedTilesizeCode`：

- 两个 Shared `TLOAD` 重载；
- `TMOV_L2S_INSERT`；
- `TMOV_L2S_PUBLISH`。

修改前：

```cpp
[TileSize] "i"(tile_type_traits<shp_dtype>::TilesizeCode)
```

修改后：

```cpp
[TileSize] "i"(tile_type_traits<shp_dtype>::SharedTilesizeCode)
```

### 4. B.IOS 容量改为直接输出编码值

修改前：

```cpp
"B.IOS mask=%c[PEMask], ->%S[Shared]<%Z[TileSize]>"
```

修改后：

```cpp
"B.IOS mask=%c[PEMask], ->%S[Shared]<%c[TileSize]>"
```

`%c` 直接输出 B.IOS 的立即数编码，例如 32KB 输出 7，由汇编器按照
B.IOS 语义解析为 `<32KB>`，不再经过 Local Tile 的 `%Z` 映射。

## 修复结果

修复后，上述 32KB Shared A/B 配置可以完成编译、链接和反汇编。关键
反汇编结果为：

```asm
BSTART.TLSU TLOAD, FP32
B.IOS mask=0001, ->S0<32KB>
B.IOR [a1,a4],[]

BSTART.TLSU TLOAD, FP32
B.IOS mask=0001, ->S1<32KB>
B.IOR [a2,a4],[]

BSTART.CUBE TMATMUL, FP32
B.IOS S0, mask=1111
B.IOS S1, mask=1111
B.IOT mask=1111, last, ->t<4KB>
```

结果说明：

- Shared `TLOAD` 目的寄存器正确生成为 `S0/S1`；
- 32KB 容量正确进入 B.IOS，而不是 unknown 或 8KB；
- Shared A/B 可以继续作为 `TMATMUL` 的输入；
- Local C 仍使用原有 B.IOT 容量映射，未受到 Shared 映射修改的影响。

## TileOP API 自测

在 `Linx-TileOP-API` 仓库执行：

```bash
make check
```

结果：

```text
Ran 17 tests
OK
C++20 compatibility syntax check: PASS
git diff --check: PASS
```

## 修改影响

- 只改变 Shared 目的指令的容量检查与 TSize 编码。
- 普通 Local Tile 的 `TilesizeCode`、128B 至 8KB 上限及 B.IOT 生成保持
  不变。
- 512B 至 8KB Shared Tile 的编码也改为使用 B.IOS 正确映射。例如
  Shared 8KB 从 B.IOT 语义的编码 7 调整为 B.IOS 语义的编码 5。
- 新增对 16KB 和 32KB Shared Tile 的完整编译表达能力。
- 非法 Shared Tile 容量仍由 `static_assert` 在编译期阻止。

## 建议补充的回归测试

1. 分别实例化 512B、1KB、2KB、4KB、8KB、16KB、32KB Shared
   `TLOAD`，检查 B.IOS TSize 依次为 1 至 7。
2. 为非 2 的幂容量、低于 512B 和高于 32KB 的 Shared Tile 增加编译失败
   测试。
3. 覆盖返回 `SharedTile` 和传入 `SharedTile &dst` 两种 `TLOAD` 重载。
4. 覆盖 `TMOV_L2S_INSERT` 与 `TMOV_L2S_PUBLISH`。
5. 增加 Local/Shared 同容量对照测试，防止再次误用同一 TSize 映射。

## 验收标准

- 512B 至 32KB 的七种合法 Shared Tile 均可编译；
- 反汇编中的 B.IOS TSize 与 Shared 编码表一致；
- 32KB Shared Tile 显示为 `S#n<32KB>`；
- Local B.IOT 的容量编码没有变化；
- `make check` 全部通过。
