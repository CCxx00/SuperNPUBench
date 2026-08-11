# PTO ISA Tile 指令集整理

> 基于 [`pto-spec`](https://github.com/PTO-ISA/pto-spec) 仓库 `docs/tile/` 与 `asl/tile/` 形式化规范整理。覆盖 PTO ISA 0.58.0 的 109 条 tile 操作。每条指令在主表中带有内文锚点,可在附录 B 索引中点击跳转。

**快速跳转**: [附录 B 助记符索引](#附录-b-助记符索引) | [概览](#概览)

## 概览

- **tile 操作总数**: 109 条
- **七大数据类**:

- <a id="elementwise-tile-tile"></a>[`elementwise-tile-tile`](#elementwise-tile-tile): 25 条 — Tile 与 Tile 逐元素运算(arithmetic/logical/transcendental/format-conversion)。两个源 tile 输入,一个目标 tile。
- <a id="tile-scalar-and-immediate"></a>[`tile-scalar-and-immediate`](#tile-scalar-and-immediate): 15 条 — Tile 与标量/立即数的逐元素运算。一个源 tile 加一个标量/立即数输入,后缀 S 区分(如 TADD vs TADDS)。
- <a id="reduce-and-expand"></a>[`reduce-and-expand`](#reduce-and-expand): 28 条 — 归约(row/column reduction)与扩展(row/column expansion)。归约把 tile 压缩为 1D 结果;扩展把 1D 广播到 2D。
- <a id="memory-and-data-movement"></a>[`memory-and-data-movement`](#memory-and-data-movement): 9 条 — Tile 与主存间的数据搬运,含规则 TLOAD/TSTORE/TPREFETCH、不规则 MGATHER/MSCATTER、PE 间 GMOV。
- <a id="matrix-and-matrix-vector"></a>[`matrix-and-matrix-vector`](#matrix-and-matrix-vector): 12 条 — 矩阵-矩阵和矩阵-向量乘法族(GEMM/GEMV),含 MX(混合精度)、ACC(累加)、BIAS(偏置融合)变体。
- <a id="layout-and-rearrangement"></a>[`layout-and-rearrangement`](#layout-and-rearrangement): 7 条 — Tile 布局重排与初始化,含转置、拼接、插入、抽取、im2col、填充、移动。
- <a id="irregular-and-complex"></a>[`irregular-and-complex`](#irregular-and-complex): 13 条 — 不规则/复杂操作,含量化、直方图、三角化、排序/归并排序、scatter/gather、partition。

- **四种执行引擎**:

  - `VEC` (向量引擎): 35 条
  - `SFU` (标量功能单元): 52 条
  - `CUBE` (Cube 立方体引擎): 12 条
  - `TLSU` (Tile Load/Store 单元): 10 条

- **编码载体**: TEPL 二进制 carrier(对应 `BSTART.VEC`/`BSTART.SFU`/`C.BSTART`),CUBE 走 Local C/D 立方体编码
- **tile 寄存器**: 64 个 flat T/U/M/N tiles,128 字节 CELL,B.IOT 分配 128B–8KiB
- **bundle 模型**: 每条 tile 指令必须包裹在 `BSTART.<engine> <MNEMONIC>, DataType` / `B.DIM` / `B.IOT` / `BSTOP` 之间

## 每条指令的查阅入口

| 字段 | 含义 |
| --- | --- |
| 助记符 | 操作名(如 `TADD`),带内文锚点 |
| 摘要 | 来自 `PTO-INSTRUCTION` JSON `summary` 字段 |
| 引擎 | VEC/SFU/CUBE/TLSU |
| 选择子 | TEPL `selector`(3 位十六进制)或 CUBE `function`/`mode` |
| 处理器 | ASL `semantic_handler` |
| 操作数 | `destination0/source0/source1` 等角色 |
| ASL / Doc | 指向 pto-spec 仓库的规范源与镜像页(绝对 URL) |

---

## <a id="elementwise-tile-tile-section"></a>elementwise-tile-tile

Tile 与 Tile 逐元素运算(arithmetic/logical/transcendental/format-conversion)。两个源 tile 输入,一个目标 tile。

### <a id="elementwise-tile-tile-arithmetic"></a>arithmetic

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tadd"></a>`TADD` | Apply elementwise addition to the two source Tiles. | VEC | 0x000 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TADD.md) |
| <a id="tfma"></a>`TFMA` | Compute a fused elementwise left-times-right plus addend result. | VEC | 0x01C |  | `TFMA` | destination0=destination, source0=multiplicand-left, source1=multiplicand-right, source2=addend | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TFMA.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TFMA.md) |
| <a id="tmax"></a>`TMAX` | Apply elementwise maximum selection to the two source Tiles. | VEC | 0x00B |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMAX.md) |
| <a id="tmin"></a>`TMIN` | Apply elementwise minimum selection to the two source Tiles. | VEC | 0x00C |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMIN.md) |
| <a id="tmul"></a>`TMUL` | Apply elementwise multiplication to the two source Tiles. | VEC | 0x002 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMUL.md) |
| <a id="tsub"></a>`TSUB` | Apply elementwise subtraction to the two source Tiles. | VEC | 0x001 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TSUB.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TSUB.md) |

### <a id="elementwise-tile-tile-format-conversion"></a>format-conversion

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tcvt"></a>`TCVT` | Convert source elements to the destination data type under rounding and saturation controls. | VEC | 0x01B |  | `TCVT` | destination0=destination, source0=source, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/format-conversion/TCVT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/format-conversion/TCVT.md) |

### <a id="elementwise-tile-tile-logical"></a>logical

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tabs"></a>`TABS` | Apply elementwise absolute value to the source Tile. | VEC | 0x00F |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TABS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TABS.md) |
| <a id="tand"></a>`TAND` | Apply elementwise bitwise AND to the two source Tiles. | VEC | 0x006 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TAND.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TAND.md) |
| <a id="tcmp"></a>`TCMP` | Apply elementwise comparison to the two source Tiles. | VEC | 0x00D |  | `ExecuteTileCompare` | destination0=destination, source0=source-left, source1=source-right, comparison=comparison | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TCMP.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TCMP.md) |
| <a id="tneg"></a>`TNEG` | Apply elementwise arithmetic negation to the source Tile. | VEC | 0x011 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TNEG.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TNEG.md) |
| <a id="tnot"></a>`TNOT` | Apply elementwise bitwise complement to the source Tile. | VEC | 0x010 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TNOT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TNOT.md) |
| <a id="tor"></a>`TOR` | Apply elementwise bitwise OR to the two source Tiles. | VEC | 0x007 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TOR.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TOR.md) |
| <a id="trelu"></a>`TRELU` | Apply elementwise rectified-linear activation to the source Tile. | VEC | 0x017 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TRELU.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TRELU.md) |
| <a id="tsel"></a>`TSEL` | Select each destination element from the true or false source under the mask Tile. | VEC | 0x01A |  | `ExecuteTileSelect` | destination0=destination, source0=mask, source1=source-true, source2=source-false | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TSEL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TSEL.md) |
| <a id="tshl"></a>`TSHL` | Apply elementwise left shift to the two source Tiles. | VEC | 0x009 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TSHL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TSHL.md) |
| <a id="tshr"></a>`TSHR` | Apply elementwise right shift to the two source Tiles. | VEC | 0x00A |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TSHR.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TSHR.md) |
| <a id="txor"></a>`TXOR` | Apply elementwise bitwise XOR to the two source Tiles. | VEC | 0x008 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/logical/TXOR.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/logical/TXOR.md) |

### <a id="elementwise-tile-tile-transcendental"></a>transcendental

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tdiv"></a>`TDIV` | Apply elementwise division to the two source Tiles. | VEC | 0x003 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TDIV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TDIV.md) |
| <a id="texp"></a>`TEXP` | Apply elementwise exponential to the source Tile. | SFU | 0x012 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TEXP.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TEXP.md) |
| <a id="tlog"></a>`TLOG` | Apply elementwise logarithm to the source Tile. | SFU | 0x013 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TLOG.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TLOG.md) |
| <a id="trecip"></a>`TRECIP` | Apply elementwise reciprocal to the source Tile. | SFU | 0x014 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TRECIP.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TRECIP.md) |
| <a id="trem"></a>`TREM` | Apply elementwise remainder to the two source Tiles. | VEC | 0x004 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TREM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TREM.md) |
| <a id="trsqrt"></a>`TRSQRT` | Apply elementwise reciprocal square root to the source Tile. | SFU | 0x016 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TRSQRT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TRSQRT.md) |
| <a id="tsqrt"></a>`TSQRT` | Apply elementwise square root to the source Tile. | SFU | 0x015 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TSQRT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TSQRT.md) |

## <a id="tile-scalar-and-immediate-section"></a>tile-scalar-and-immediate

Tile 与标量/立即数的逐元素运算。一个源 tile 加一个标量/立即数输入,后缀 S 区分(如 TADD vs TADDS)。

### <a id="tile-scalar-and-immediate-arithmetic"></a>arithmetic

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tadds"></a>`TADDS` | Apply elementwise addition between the source Tile and bound scalar. | VEC | 0x020 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TADDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TADDS.md) |
| <a id="tdivs"></a>`TDIVS` | Apply elementwise division between the source Tile and bound scalar. | VEC | 0x023 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TDIVS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TDIVS.md) |
| <a id="tmaxs"></a>`TMAXS` | Apply elementwise maximum selection between the source Tile and bound scalar. | VEC | 0x02B |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMAXS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMAXS.md) |
| <a id="tmins"></a>`TMINS` | Apply elementwise minimum selection between the source Tile and bound scalar. | VEC | 0x02C |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMINS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMINS.md) |
| <a id="tmuls"></a>`TMULS` | Apply elementwise multiplication between the source Tile and bound scalar. | VEC | 0x022 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMULS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMULS.md) |
| <a id="trems"></a>`TREMS` | Apply elementwise remainder between the source Tile and bound scalar. | VEC | 0x024 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TREMS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TREMS.md) |
| <a id="tsubs"></a>`TSUBS` | Apply elementwise subtraction between the source Tile and bound scalar. | VEC | 0x021 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TSUBS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TSUBS.md) |

### <a id="tile-scalar-and-immediate-initialization"></a>initialization

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="texpands"></a>`TEXPANDS` | Fill the destination Tile by expanding the bound scalar value. | VEC | 0x03B |  | `ExecuteTileFillScalar` | destination0=destination, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/initialization/TEXPANDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/initialization/TEXPANDS.md) |

### <a id="tile-scalar-and-immediate-logical"></a>logical

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tands"></a>`TANDS` | Apply elementwise bitwise AND between the source Tile and bound scalar. | VEC | 0x026 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TANDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TANDS.md) |
| <a id="tcmps"></a>`TCMPS` | Apply elementwise comparison between the source Tile and bound scalar. | VEC | 0x02D |  | `ExecuteTileCompareScalar` | destination0=destination, source0=source, scalar0=scalar, comparison=comparison | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TCMPS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TCMPS.md) |
| <a id="tors"></a>`TORS` | Apply elementwise bitwise OR between the source Tile and bound scalar. | VEC | 0x027 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TORS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TORS.md) |
| <a id="tsels"></a>`TSELS` | Select each destination element from the Tile source or scalar alternative under the mask Tile. | VEC | 0x03A |  | `ExecuteTileSelectScalar` | destination0=destination, source0=mask, source1=source-true, scalar0=scalar-false | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSELS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSELS.md) |
| <a id="tshls"></a>`TSHLS` | Apply elementwise left shift between the source Tile and bound scalar. | VEC | 0x029 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSHLS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSHLS.md) |
| <a id="tshrs"></a>`TSHRS` | Apply elementwise right shift between the source Tile and bound scalar. | VEC | 0x02A |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSHRS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSHRS.md) |
| <a id="txors"></a>`TXORS` | Apply elementwise bitwise XOR between the source Tile and bound scalar. | VEC | 0x028 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TXORS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TXORS.md) |

## <a id="reduce-and-expand-section"></a>reduce-and-expand

归约(row/column reduction)与扩展(row/column expansion)。归约把 tile 压缩为 1D 结果;扩展把 1D 广播到 2D。

### <a id="reduce-and-expand-column-expansion"></a>column-expansion

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tcolexpand"></a>`TCOLEXPAND` | Apply broadcast while expanding the bound col vector across the source Tile. | SFU | 0x054 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPAND.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPAND.md) |
| <a id="tcolexpandadd"></a>`TCOLEXPANDADD` | Apply addition while expanding the bound col vector across the source Tile. | SFU | 0x055 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDADD.md) |
| <a id="tcolexpanddiv"></a>`TCOLEXPANDDIV` | Apply division while expanding the bound col vector across the source Tile. | SFU | 0x058 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDDIV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDDIV.md) |
| <a id="tcolexpandexpdif"></a>`TCOLEXPANDEXPDIF` | Apply exponential difference while expanding the bound col vector across the source Tile. | SFU | 0x05B |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDEXPDIF.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDEXPDIF.md) |
| <a id="tcolexpandmax"></a>`TCOLEXPANDMAX` | Apply maximum selection while expanding the bound col vector across the source Tile. | SFU | 0x059 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDMAX.md) |
| <a id="tcolexpandmin"></a>`TCOLEXPANDMIN` | Apply minimum selection while expanding the bound col vector across the source Tile. | SFU | 0x05A |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDMIN.md) |
| <a id="tcolexpandmul"></a>`TCOLEXPANDMUL` | Apply multiplication while expanding the bound col vector across the source Tile. | SFU | 0x057 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDMUL.md) |
| <a id="tcolexpandsub"></a>`TCOLEXPANDSUB` | Apply subtraction while expanding the bound col vector across the source Tile. | SFU | 0x056 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-expansion/TCOLEXPANDSUB.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-expansion/TCOLEXPANDSUB.md) |

### <a id="reduce-and-expand-column-reduction"></a>column-reduction

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tcolargmax"></a>`TCOLARGMAX` | Reduce each source col to its maximum index. | SFU | 0x05C |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLARGMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLARGMAX.md) |
| <a id="tcolargmin"></a>`TCOLARGMIN` | Reduce each source col to its minimum index. | SFU | 0x05D |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLARGMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLARGMIN.md) |
| <a id="tcolmax"></a>`TCOLMAX` | Reduce each source col to its maximum. | SFU | 0x051 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLMAX.md) |
| <a id="tcolmin"></a>`TCOLMIN` | Reduce each source col to its minimum. | SFU | 0x052 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLMIN.md) |
| <a id="tcolprod"></a>`TCOLPROD` | Reduce each source col to its product. | SFU | 0x053 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLPROD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLPROD.md) |
| <a id="tcolsum"></a>`TCOLSUM` | Reduce each source col to its sum. | SFU | 0x050 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLSUM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLSUM.md) |

### <a id="reduce-and-expand-row-expansion"></a>row-expansion

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="trowexpand"></a>`TROWEXPAND` | Apply broadcast while expanding the bound row vector across the source Tile. | SFU | 0x044 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPAND.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPAND.md) |
| <a id="trowexpandadd"></a>`TROWEXPANDADD` | Apply addition while expanding the bound row vector across the source Tile. | SFU | 0x045 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDADD.md) |
| <a id="trowexpanddiv"></a>`TROWEXPANDDIV` | Apply division while expanding the bound row vector across the source Tile. | SFU | 0x048 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDDIV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDDIV.md) |
| <a id="trowexpandexpdif"></a>`TROWEXPANDEXPDIF` | Apply exponential difference while expanding the bound row vector across the source Tile. | SFU | 0x04B |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDEXPDIF.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDEXPDIF.md) |
| <a id="trowexpandmax"></a>`TROWEXPANDMAX` | Apply maximum selection while expanding the bound row vector across the source Tile. | SFU | 0x049 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDMAX.md) |
| <a id="trowexpandmin"></a>`TROWEXPANDMIN` | Apply minimum selection while expanding the bound row vector across the source Tile. | SFU | 0x04A |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDMIN.md) |
| <a id="trowexpandmul"></a>`TROWEXPANDMUL` | Apply multiplication while expanding the bound row vector across the source Tile. | SFU | 0x047 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDMUL.md) |
| <a id="trowexpandsub"></a>`TROWEXPANDSUB` | Apply subtraction while expanding the bound row vector across the source Tile. | SFU | 0x046 |  | `ExecuteTileExpand` | destination0=destination, source0=source, source1=broadcast-source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-expansion/TROWEXPANDSUB.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-expansion/TROWEXPANDSUB.md) |

### <a id="reduce-and-expand-row-reduction"></a>row-reduction

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="trowargmax"></a>`TROWARGMAX` | Reduce each source row to its maximum index. | SFU | 0x04C |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWARGMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWARGMAX.md) |
| <a id="trowargmin"></a>`TROWARGMIN` | Reduce each source row to its minimum index. | SFU | 0x04D |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWARGMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWARGMIN.md) |
| <a id="trowmax"></a>`TROWMAX` | Reduce each source row to its maximum. | SFU | 0x041 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWMAX.md) |
| <a id="trowmin"></a>`TROWMIN` | Reduce each source row to its minimum. | SFU | 0x042 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWMIN.md) |
| <a id="trowprod"></a>`TROWPROD` | Reduce each source row to its product. | SFU | 0x043 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWPROD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWPROD.md) |
| <a id="trowsum"></a>`TROWSUM` | Reduce each source row to its sum. | SFU | 0x040 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWSUM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWSUM.md) |

## <a id="memory-and-data-movement-section"></a>memory-and-data-movement

Tile 与主存间的数据搬运,含规则 TLOAD/TSTORE/TPREFETCH、不规则 MGATHER/MSCATTER、PE 间 GMOV。

### <a id="memory-and-data-movement-irregular"></a>irregular

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="mgather"></a>`MGATHER` | Gather GM elements at Tile-provided indices into the destination. | TLSU |  |  | `MGATHER` | destination0=destination, address=base-address, source0=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER.md) |
| <a id="mgather_cas"></a>`MGATHER_CAS` | Atomically compare and conditionally replace GM elements at Tile-provided indices. | TLSU |  |  | `MGATHER_CAS` | destination0=destination, address=base-address, source0=indices, source1=expected, source2=replacement | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER_CAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER_CAS.md) |
| <a id="mgather_mask"></a>`MGATHER_MASK` | Gather masked GM elements at Tile-provided indices into the destination. | TLSU |  |  | `MGATHER_MASK` | destination0=destination, address=base-address, source0=indices, source1=mask | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER_MASK.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER_MASK.md) |
| <a id="mscatter"></a>`MSCATTER` | Scatter source Tile elements to GM addresses selected by Tile indices. | TLSU |  |  | `MSCATTER` | address=base-address, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MSCATTER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MSCATTER.md) |
| <a id="mscatter_mask"></a>`MSCATTER_MASK` | Scatter masked source elements to GM addresses selected by Tile indices. | TLSU |  |  | `MSCATTER_MASK` | address=base-address, source0=source, source1=indices, source2=mask | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MSCATTER_MASK.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MSCATTER_MASK.md) |

### <a id="memory-and-data-movement-pe-movement"></a>pe-movement

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="gmov"></a>`GMOV` | Copy the resolved peer-PE Tile fragment selected by the bound peer TID. | TLSU |  |  | `GMOV` | destination0=destination, source0=resolved-peer-source, scalar0=peer-tid | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/pe-movement/GMOV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/pe-movement/GMOV.md) |

### <a id="memory-and-data-movement-regular"></a>regular

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tload"></a>`TLOAD` | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. | TLSU |  |  | `TLOAD` | destination0=destination, address=base-address, scalar0=row-stride-elements | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TLOAD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TLOAD.md) |
| <a id="tprefetch"></a>`TPREFETCH` | Prefetch the requested GM byte range without producing a Tile destination. | TLSU |  |  | `TPREFETCH` | address=base-address, byte_count=byte-count | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TPREFETCH.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TPREFETCH.md) |
| <a id="tstore"></a>`TSTORE` | Store the valid Tile rectangle to GM using the encoded base and logical row stride. | TLSU |  |  | `TSTORE` | address=base-address, scalar0=row-stride-elements, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TSTORE.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TSTORE.md) |

## <a id="matrix-and-matrix-vector-section"></a>matrix-and-matrix-vector

矩阵-矩阵和矩阵-向量乘法族(GEMM/GEMV),含 MX(混合精度)、ACC(累加)、BIAS(偏置融合)变体。

### <a id="matrix-and-matrix-vector-matrix-matrix"></a>matrix-matrix

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tmatmul"></a>`TMATMUL` | Multiply the left and right matrices into the destination. | CUBE |  | 0/None | `TMATMUL` | destination0=destination, source0=left, source1=right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.md) |
| <a id="tmatmul_acc"></a>`TMATMUL_ACC` | Multiply matrices and accumulate into the supplied accumulator Tile. | CUBE |  | 2/None | `TMATMUL_ACC` | destination0=destination, source0=accumulator, source1=left, source2=right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_ACC.md) |
| <a id="tmatmul_bias"></a>`TMATMUL_BIAS` | Multiply matrices and add the bias Tile into the destination. | CUBE |  | 1/None | `TMATMUL_BIAS` | destination0=destination, source0=left, source1=right, source2=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_BIAS.md) |
| <a id="tmatmul_mx"></a>`TMATMUL_MX` | Multiply matrices using row and column scale Tiles. | CUBE |  | 4/None | `TMATMUL_MX` | destination0=destination, source0=left, source1=row-scale, source2=right, source3=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX.md) |
| <a id="tmatmul_mx_acc"></a>`TMATMUL_MX_ACC` | Multiply scaled matrices and accumulate into the supplied accumulator Tile. | CUBE |  | 6/None | `TMATMUL_MX_ACC` | destination0=destination, source0=accumulator, source1=left, source2=row-scale, source3=right, source4=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_ACC.md) |
| <a id="tmatmul_mx_bias"></a>`TMATMUL_MX_BIAS` | Multiply scaled matrices and add the bias Tile. | CUBE |  | 5/None | `TMATMUL_MX_BIAS` | destination0=destination, source0=left, source1=row-scale, source2=right, source3=column-scale, source4=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_BIAS.md) |

### <a id="matrix-and-matrix-vector-matrix-vector"></a>matrix-vector

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tgemv"></a>`TGEMV` | Multiply the matrix by the vector into the destination. | CUBE |  | 16/None | `TGEMV` | destination0=destination, source0=matrix, source1=vector | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV.md) |
| <a id="tgemv_acc"></a>`TGEMV_ACC` | Multiply the matrix by the vector and accumulate into the supplied Tile. | CUBE |  | 18/None | `TGEMV_ACC` | destination0=destination, source0=accumulator, source1=matrix, source2=vector | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_ACC.md) |
| <a id="tgemv_bias"></a>`TGEMV_BIAS` | Multiply the matrix by the vector and add the bias Tile. | CUBE |  | 17/None | `TGEMV_BIAS` | destination0=destination, source0=matrix, source1=vector, source2=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_BIAS.md) |
| <a id="tgemv_mx"></a>`TGEMV_MX` | Multiply the matrix by the vector using row and column scale Tiles. | CUBE |  | 20/None | `TGEMV_MX` | destination0=destination, source0=matrix, source1=row-scale, source2=vector, source3=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX.md) |
| <a id="tgemv_mx_acc"></a>`TGEMV_MX_ACC` | Multiply the scaled matrix and vector and accumulate into the supplied Tile. | CUBE |  | 22/None | `TGEMV_MX_ACC` | destination0=destination, source0=accumulator, source1=matrix, source2=row-scale, source3=vector, source4=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_ACC.md) |
| <a id="tgemv_mx_bias"></a>`TGEMV_MX_BIAS` | Multiply the scaled matrix and vector and add the bias Tile. | CUBE |  | 21/None | `TGEMV_MX_BIAS` | destination0=destination, source0=matrix, source1=row-scale, source2=vector, source3=column-scale, source4=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_BIAS.md) |

## <a id="layout-and-rearrangement-section"></a>layout-and-rearrangement

Tile 布局重排与初始化,含转置、拼接、插入、抽取、im2col、填充、移动。

### <a id="layout-and-rearrangement-initialization"></a>initialization

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tfillpad"></a>`TFILLPAD` | Copy the source and fill destination padding elements with the bound scalar. | SFU | 0x065 |  | `TFILLPAD` | destination0=destination, source0=source, scalar0=padding | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/initialization/TFILLPAD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/initialization/TFILLPAD.md) |

### <a id="layout-and-rearrangement-layout"></a>layout

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tconcat"></a>`TCONCAT` | Concatenate two source Tiles along the selected axis. | SFU | 0x060 |  | `TCONCAT` | destination0=destination, source0=source-left, source1=source-right, axis=axis | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TCONCAT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TCONCAT.md) |
| <a id="textract"></a>`TEXTRACT` | Extract a rectangular source region at the encoded row and column offsets. | SFU | 0x062 |  | `TEXTRACT` | destination0=destination, source0=source, natural0=row-offset, natural1=column-offset | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TEXTRACT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TEXTRACT.md) |
| <a id="timg2col"></a>`TIMG2COL` | Transform an image Tile into kernel-column layout using kernel, stride, padding, and fill operands. | SFU | 0x064 |  | `TIMG2COL` | destination0=destination, source0=source, positive0=kernel-rows, positive1=kernel-columns, positive2=stride-rows, positive3=stride-columns, natural0=pad-rows, natural1=pad-columns, scalar0=padding | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TIMG2COL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TIMG2COL.md) |
| <a id="tinsert"></a>`TINSERT` | Insert the source Tile into the destination region at the encoded row and column offsets. | SFU | 0x063 |  | `TINSERT` | destination0=destination, source0=source, natural0=row-offset, natural1=column-offset | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TINSERT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TINSERT.md) |
| <a id="tmov"></a>`TMOV` | Copy the source Tile payload and definedness into the destination. | TLSU |  |  | `TMOV` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TMOV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TMOV.md) |
| <a id="ttrans"></a>`TTRANS` | Transpose the source Tile into the destination. | SFU | 0x06E |  | `TTRANS` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TTRANS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TTRANS.md) |

## <a id="irregular-and-complex-section"></a>irregular-and-complex

不规则/复杂操作,含量化、直方图、三角化、排序/归并排序、scatter/gather、partition。

### <a id="irregular-and-complex-format-conversion"></a>format-conversion

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tdequant"></a>`TDEQUANT` | Dequantize source elements using scale, zero point, rounding, and saturation controls. | SFU | 0x06B |  | `TDEQUANT` | destination0=destination, source0=source, scalar0=scale, scalar1=zero-point, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/format-conversion/TDEQUANT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/format-conversion/TDEQUANT.md) |
| <a id="tquant"></a>`TQUANT` | Quantize source elements using scale, zero point, rounding, and saturation controls. | SFU | 0x06A |  | `TQUANT` | destination0=destination, source0=source, scalar0=scale, scalar1=zero-point, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/format-conversion/TQUANT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/format-conversion/TQUANT.md) |

### <a id="irregular-and-complex-initialization"></a>initialization

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tci"></a>`TCI` | Initialize destination elements as an ascending or descending counter sequence. | SFU | 0x066 |  | `TCI` | destination0=destination, scalar0=start, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/TCI.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/TCI.md) |
| <a id="thistogram"></a>`THISTOGRAM` | Accumulate a histogram from source values and selected-byte indices. | SFU | 0x068 |  | `THISTOGRAM` | destination0=destination, source0=source, source1=indices, selected_byte=selected-byte | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/THISTOGRAM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/THISTOGRAM.md) |
| <a id="ttri"></a>`TTRI` | Initialize the selected upper or lower triangular region relative to the diagonal. | SFU | 0x067 |  | `TTRI` | destination0=destination, flag0=upper, diagonal=diagonal | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/TTRI.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/TTRI.md) |

### <a id="irregular-and-complex-layout"></a>layout

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tgather"></a>`TGATHER` | Gather source elements by Tile indices into the destination. | SFU | 0x06F |  | `TGATHER` | destination0=destination, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/layout/TGATHER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/layout/TGATHER.md) |
| <a id="tscatter"></a>`TSCATTER` | Scatter source elements by Tile indices into the destination. | SFU | 0x070 |  | `TSCATTER` | destination0=destination, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/layout/TSCATTER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/layout/TSCATTER.md) |

### <a id="irregular-and-complex-sorting"></a>sorting

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tmrgsort"></a>`TMRGSORT` | Merge two sorted source Tiles in the selected ascending or descending order. | SFU | 0x06D |  | `TMRGSORT` | destination0=destination, source0=source-left, source1=source-right, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/sorting/TMRGSORT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/sorting/TMRGSORT.md) |
| <a id="tsort"></a>`TSORT` | Sort source groups, returning ordered values and original U32 indices. | SFU | 0x06C |  | `TSORT` | destination0=destination, destination1=original-indices-u32, source0=source, sort_width=sort-width, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/sorting/TSORT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/sorting/TSORT.md) |

### <a id="irregular-and-complex-union"></a>union

| 助记符 | 摘要 | 引擎 | 选择子 | Func/Mode | 处理器 | 操作数 | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tpartadd"></a>`TPARTADD` | Combine corresponding source partitions by addition. | SFU | 0x071 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTADD.md) |
| <a id="tpartmax"></a>`TPARTMAX` | Combine corresponding source partitions by maximum selection. | SFU | 0x073 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMAX.md) |
| <a id="tpartmin"></a>`TPARTMIN` | Combine corresponding source partitions by minimum selection. | SFU | 0x074 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMIN.md) |
| <a id="tpartmul"></a>`TPARTMUL` | Combine corresponding source partitions by multiplication. | SFU | 0x072 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMUL.md) |

---

## 附录 A: bundle 组合模板示例

每条 tile 指令在程序中必须以下面的 bundle 形式出现(以 `TADD` 为例):

```asm
BSTART.VEC TADD, DataType
B.DATR (optional)        # 数据属性
B.DIM LB0                # 第一维
B.DIM (LB1/LB2 for 2D)   # 第二/三维
B.IOT                    # 输入输出 tile 绑定
BSTOP
```

`BSTART.TEPL <MNEMONIC>` 是兼容写法;0.58.0 起 `BSTART.VEC`/`BSTART.SFU` 是规范拆分。

## 附录 B: 助记符索引

每条助记符带可点击跳转链接,直达主表中的指令行(内文锚点)。

| 助记符 | 类 | 引擎 | 跳转 |
| --- | --- | :---: | :---: |
| `GMOV` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#gmov) |
| `MGATHER` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#mgather) |
| `MGATHER_CAS` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#mgather_cas) |
| `MGATHER_MASK` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#mgather_mask) |
| `MSCATTER` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#mscatter) |
| `MSCATTER_MASK` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#mscatter_mask) |
| `TABS` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tabs) |
| `TADD` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tadd) |
| `TADDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tadds) |
| `TAND` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tand) |
| `TANDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tands) |
| `TCI` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tci) |
| `TCMP` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tcmp) |
| `TCMPS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tcmps) |
| `TCOLARGMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolargmax) |
| `TCOLARGMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolargmin) |
| `TCOLEXPAND` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpand) |
| `TCOLEXPANDADD` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandadd) |
| `TCOLEXPANDDIV` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpanddiv) |
| `TCOLEXPANDEXPDIF` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandexpdif) |
| `TCOLEXPANDMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandmax) |
| `TCOLEXPANDMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandmin) |
| `TCOLEXPANDMUL` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandmul) |
| `TCOLEXPANDSUB` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolexpandsub) |
| `TCOLMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolmax) |
| `TCOLMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolmin) |
| `TCOLPROD` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolprod) |
| `TCOLSUM` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#tcolsum) |
| `TCONCAT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#tconcat) |
| `TCVT` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tcvt) |
| `TDEQUANT` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tdequant) |
| `TDIV` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tdiv) |
| `TDIVS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tdivs) |
| `TEXP` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [跳转 →](#texp) |
| `TEXPANDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#texpands) |
| `TEXTRACT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#textract) |
| `TFILLPAD` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#tfillpad) |
| `TFMA` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tfma) |
| `TGATHER` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tgather) |
| `TGEMV` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv) |
| `TGEMV_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv_acc) |
| `TGEMV_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv_bias) |
| `TGEMV_MX` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv_mx) |
| `TGEMV_MX_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv_mx_acc) |
| `TGEMV_MX_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tgemv_mx_bias) |
| `THISTOGRAM` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#thistogram) |
| `TIMG2COL` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#timg2col) |
| `TINSERT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#tinsert) |
| `TLOAD` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#tload) |
| `TLOG` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [跳转 →](#tlog) |
| `TMATMUL` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul) |
| `TMATMUL_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul_acc) |
| `TMATMUL_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul_bias) |
| `TMATMUL_MX` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul_mx) |
| `TMATMUL_MX_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul_mx_acc) |
| `TMATMUL_MX_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [跳转 →](#tmatmul_mx_bias) |
| `TMAX` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tmax) |
| `TMAXS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tmaxs) |
| `TMIN` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tmin) |
| `TMINS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tmins) |
| `TMOV` | [layout-and-rearrangement](#layout-and-rearrangement) | TLSU | [跳转 →](#tmov) |
| `TMRGSORT` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tmrgsort) |
| `TMUL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tmul) |
| `TMULS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tmuls) |
| `TNEG` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tneg) |
| `TNOT` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tnot) |
| `TOR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tor) |
| `TORS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tors) |
| `TPARTADD` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tpartadd) |
| `TPARTMAX` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tpartmax) |
| `TPARTMIN` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tpartmin) |
| `TPARTMUL` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tpartmul) |
| `TPREFETCH` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#tprefetch) |
| `TQUANT` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tquant) |
| `TRECIP` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [跳转 →](#trecip) |
| `TRELU` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#trelu) |
| `TREM` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#trem) |
| `TREMS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#trems) |
| `TROWARGMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowargmax) |
| `TROWARGMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowargmin) |
| `TROWEXPAND` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpand) |
| `TROWEXPANDADD` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandadd) |
| `TROWEXPANDDIV` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpanddiv) |
| `TROWEXPANDEXPDIF` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandexpdif) |
| `TROWEXPANDMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandmax) |
| `TROWEXPANDMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandmin) |
| `TROWEXPANDMUL` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandmul) |
| `TROWEXPANDSUB` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowexpandsub) |
| `TROWMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowmax) |
| `TROWMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowmin) |
| `TROWPROD` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowprod) |
| `TROWSUM` | [reduce-and-expand](#reduce-and-expand) | SFU | [跳转 →](#trowsum) |
| `TRSQRT` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [跳转 →](#trsqrt) |
| `TSCATTER` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tscatter) |
| `TSEL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tsel) |
| `TSELS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tsels) |
| `TSHL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tshl) |
| `TSHLS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tshls) |
| `TSHR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tshr) |
| `TSHRS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tshrs) |
| `TSORT` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#tsort) |
| `TSQRT` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [跳转 →](#tsqrt) |
| `TSTORE` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [跳转 →](#tstore) |
| `TSUB` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#tsub) |
| `TSUBS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#tsubs) |
| `TTRANS` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [跳转 →](#ttrans) |
| `TTRI` | [irregular-and-complex](#irregular-and-complex) | SFU | [跳转 →](#ttri) |
| `TXOR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [跳转 →](#txor) |
| `TXORS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [跳转 →](#txors) |

## 附录 C: 编目剔除与保留名(来自 `asl/tile/model/dispatch/top-level.asl`)

- **已删除名**(曾在历史编目中,0.58.0 不再保留):

  - `ACCCVT`
  - `TADDC`
  - `TADDSC`
  - `TALLOC`
  - `TAXPY`
  - `TDEINTERLEAVE`
  - `TFMOD`
  - `TFMODS`
  - `TFREE`
  - `TGATHERB`
  - `TINTERLEAVE`
  - `TLRELU`
  - `TPARTARGMAX`
  - `TPARTARGMIN`
  - `TPOP`
  - `TPRELU`
  - `TPUSH`
  - `TRESHAPE`
  - `TSUBC`
  - `TSUBSC`
  - `TRANDOM`
  - `TSORT32`

- **拒绝名**(架构层拒绝,不在编目中):

  - `TEXRACT`
  - `TFILL/TEXPANDS`
  - `TPOW`
  - `TPOWS`
