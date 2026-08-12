# PTO ISA Tile Instruction Set Reference

> Generated from the [`pto-spec`](https://github.com/PTO-ISA/pto-spec) repository (`docs/tile/` and `asl/tile/` formal specification). Covers all 109 tile operations of PTO ISA 0.58.0. Each instruction row carries an in-document anchor; the index in Appendix B links to it.

**Quick navigation**: [Appendix B Mnemonic Index](#appendix-b-mnemonic-index) | [Overview](#overview)

## Overview

- **Total tile operations**: 109
- **Seven semantic classes**:

- <a id="elementwise-tile-tile"></a>[`elementwise-tile-tile`](#elementwise-tile-tile): 25 ops — Elementwise Tile-Tile operations (arithmetic / logical / transcendental / format-conversion). Two source tiles in, one destination tile out.
- <a id="tile-scalar-and-immediate"></a>[`tile-scalar-and-immediate`](#tile-scalar-and-immediate): 15 ops — Elementwise Tile-Scalar/Immediate operations. One source tile plus a scalar/immediate operand; the trailing `S` distinguishes these from the Tile-Tile form (e.g. TADD vs TADDS).
- <a id="reduce-and-expand"></a>[`reduce-and-expand`](#reduce-and-expand): 28 ops — Reductions (row/column reduction) and expansions (row/column expansion). Reductions collapse a tile to a 1D result; expansions broadcast a 1D source to a 2D tile.
- <a id="memory-and-data-movement"></a>[`memory-and-data-movement`](#memory-and-data-movement): 9 ops — Data movement between tiles and main memory: regular TLOAD/TSTORE/TPREFETCH, irregular MGATHER/MSCATTER, and PE-to-PE GMOV.
- <a id="matrix-and-matrix-vector"></a>[`matrix-and-matrix-vector`](#matrix-and-matrix-vector): 12 ops — Matrix-Matrix and Matrix-Vector multiply family (GEMM/GEMV) with MX (mixed precision), ACC (accumulate), and BIAS (bias-fused) variants.
- <a id="layout-and-rearrangement"></a>[`layout-and-rearrangement`](#layout-and-rearrangement): 7 ops — Tile layout rearrangement and initialization: transpose, concat, insert, extract, im2col, fill, move.
- <a id="irregular-and-complex"></a>[`irregular-and-complex`](#irregular-and-complex): 13 ops — Irregular/complex operations: quantization, histogram, triangularization, sort/merge-sort, scatter/gather, partition.

- **Four execution engines**:

  - `VEC` (Vector engine): 35 ops
  - `SFU` (Scalar Functional Unit): 52 ops
  - `CUBE` (Cube engine): 12 ops
  - `TLSU` (Tile Load/Store Unit): 10 ops

- **Encoding carrier**: TEPL binary carrier (selected via `BSTART.VEC` / `BSTART.SFU` / `C.BSTART`); CUBE uses the Local C/D cube encoding.
- **Tile registers**: 64 flat T/U/M/N tiles, 128-byte CELL, B.IOT allocation 128 B – 8 KiB.
- **Bundle model**: every tile instruction is wrapped in `BSTART.<engine> <MNEMONIC>, DataType` / `B.DIM` / `B.IOT` / `BSTOP`.

## Per-instruction lookup fields

| Field | Meaning |
| --- | --- |
| Mnemonic | Operation name (e.g. `TADD`), with an in-document anchor |
| Summary | From the `PTO-INSTRUCTION` JSON `summary` field |
| Engine | VEC / SFU / CUBE / TLSU |
| Selector | TEPL `selector` (3-digit hex) or CUBE `function`/`mode` |
| Handler | ASL `semantic_handler` |
| Operands | Roles such as `destination0`/`source0`/`source1` |
| ASL / Doc | Absolute URLs to the pto-spec normative source and mirror page |

---

## <a id="elementwise-tile-tile-section"></a>elementwise-tile-tile

Elementwise Tile-Tile operations (arithmetic / logical / transcendental / format-conversion). Two source tiles in, one destination tile out.

### <a id="elementwise-tile-tile-arithmetic"></a>arithmetic

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tadd"></a>`TADD` | Apply elementwise addition to the two source Tiles. | VEC | 0x000 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TADD.md) |
| <a id="tfma"></a>`TFMA` | Compute a fused elementwise left-times-right plus addend result. | VEC | 0x01C |  | `TFMA` | destination0=destination, source0=multiplicand-left, source1=multiplicand-right, source2=addend | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TFMA.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TFMA.md) |
| <a id="tmax"></a>`TMAX` | Apply elementwise maximum selection to the two source Tiles. | VEC | 0x00B |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMAX.md) |
| <a id="tmin"></a>`TMIN` | Apply elementwise minimum selection to the two source Tiles. | VEC | 0x00C |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMIN.md) |
| <a id="tmul"></a>`TMUL` | Apply elementwise multiplication to the two source Tiles. | VEC | 0x002 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TMUL.md) |
| <a id="tsub"></a>`TSUB` | Apply elementwise subtraction to the two source Tiles. | VEC | 0x001 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/arithmetic/TSUB.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/arithmetic/TSUB.md) |

### <a id="elementwise-tile-tile-format-conversion"></a>format-conversion

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tcvt"></a>`TCVT` | Convert source elements to the destination data type under rounding and saturation controls. | VEC | 0x01B |  | `TCVT` | destination0=destination, source0=source, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/format-conversion/TCVT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/format-conversion/TCVT.md) |

### <a id="elementwise-tile-tile-logical"></a>logical

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
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

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tdiv"></a>`TDIV` | Apply elementwise division to the two source Tiles. | VEC | 0x003 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TDIV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TDIV.md) |
| <a id="texp"></a>`TEXP` | Apply elementwise exponential to the source Tile. | SFU | 0x012 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TEXP.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TEXP.md) |
| <a id="tlog"></a>`TLOG` | Apply elementwise logarithm to the source Tile. | SFU | 0x013 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TLOG.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TLOG.md) |
| <a id="trecip"></a>`TRECIP` | Apply elementwise reciprocal to the source Tile. | SFU | 0x014 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TRECIP.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TRECIP.md) |
| <a id="trem"></a>`TREM` | Apply elementwise remainder to the two source Tiles. | VEC | 0x004 |  | `ExecuteTileBinary` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TREM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TREM.md) |
| <a id="trsqrt"></a>`TRSQRT` | Apply elementwise reciprocal square root to the source Tile. | SFU | 0x016 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TRSQRT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TRSQRT.md) |
| <a id="tsqrt"></a>`TSQRT` | Apply elementwise square root to the source Tile. | SFU | 0x015 |  | `ExecuteTileUnary` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/elementwise-tile-tile/transcendental/TSQRT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/elementwise-tile-tile/transcendental/TSQRT.md) |

## <a id="tile-scalar-and-immediate-section"></a>tile-scalar-and-immediate

Elementwise Tile-Scalar/Immediate operations. One source tile plus a scalar/immediate operand; the trailing `S` distinguishes these from the Tile-Tile form (e.g. TADD vs TADDS).

### <a id="tile-scalar-and-immediate-arithmetic"></a>arithmetic

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tadds"></a>`TADDS` | Apply elementwise addition between the source Tile and bound scalar. | VEC | 0x020 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TADDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TADDS.md) |
| <a id="tdivs"></a>`TDIVS` | Apply elementwise division between the source Tile and bound scalar. | VEC | 0x023 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TDIVS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TDIVS.md) |
| <a id="tmaxs"></a>`TMAXS` | Apply elementwise maximum selection between the source Tile and bound scalar. | VEC | 0x02B |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMAXS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMAXS.md) |
| <a id="tmins"></a>`TMINS` | Apply elementwise minimum selection between the source Tile and bound scalar. | VEC | 0x02C |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMINS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMINS.md) |
| <a id="tmuls"></a>`TMULS` | Apply elementwise multiplication between the source Tile and bound scalar. | VEC | 0x022 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TMULS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TMULS.md) |
| <a id="trems"></a>`TREMS` | Apply elementwise remainder between the source Tile and bound scalar. | VEC | 0x024 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TREMS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TREMS.md) |
| <a id="tsubs"></a>`TSUBS` | Apply elementwise subtraction between the source Tile and bound scalar. | VEC | 0x021 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/arithmetic/TSUBS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/arithmetic/TSUBS.md) |

### <a id="tile-scalar-and-immediate-initialization"></a>initialization

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="texpands"></a>`TEXPANDS` | Fill the destination Tile by expanding the bound scalar value. | VEC | 0x03B |  | `ExecuteTileFillScalar` | destination0=destination, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/initialization/TEXPANDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/initialization/TEXPANDS.md) |

### <a id="tile-scalar-and-immediate-logical"></a>logical

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tands"></a>`TANDS` | Apply elementwise bitwise AND between the source Tile and bound scalar. | VEC | 0x026 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TANDS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TANDS.md) |
| <a id="tcmps"></a>`TCMPS` | Apply elementwise comparison between the source Tile and bound scalar. | VEC | 0x02D |  | `ExecuteTileCompareScalar` | destination0=destination, source0=source, scalar0=scalar, comparison=comparison | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TCMPS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TCMPS.md) |
| <a id="tors"></a>`TORS` | Apply elementwise bitwise OR between the source Tile and bound scalar. | VEC | 0x027 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TORS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TORS.md) |
| <a id="tsels"></a>`TSELS` | Select each destination element from the Tile source or scalar alternative under the mask Tile. | VEC | 0x03A |  | `ExecuteTileSelectScalar` | destination0=destination, source0=mask, source1=source-true, scalar0=scalar-false | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSELS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSELS.md) |
| <a id="tshls"></a>`TSHLS` | Apply elementwise left shift between the source Tile and bound scalar. | VEC | 0x029 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSHLS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSHLS.md) |
| <a id="tshrs"></a>`TSHRS` | Apply elementwise right shift between the source Tile and bound scalar. | VEC | 0x02A |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TSHRS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TSHRS.md) |
| <a id="txors"></a>`TXORS` | Apply elementwise bitwise XOR between the source Tile and bound scalar. | VEC | 0x028 |  | `ExecuteTileScalar` | destination0=destination, source0=source, scalar0=scalar | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/tile-scalar-and-immediate/logical/TXORS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/tile-scalar-and-immediate/logical/TXORS.md) |

## <a id="reduce-and-expand-section"></a>reduce-and-expand

Reductions (row/column reduction) and expansions (row/column expansion). Reductions collapse a tile to a 1D result; expansions broadcast a 1D source to a 2D tile.

### <a id="reduce-and-expand-column-expansion"></a>column-expansion

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
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

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tcolargmax"></a>`TCOLARGMAX` | Reduce each source col to its maximum index. | SFU | 0x05C |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLARGMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLARGMAX.md) |
| <a id="tcolargmin"></a>`TCOLARGMIN` | Reduce each source col to its minimum index. | SFU | 0x05D |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLARGMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLARGMIN.md) |
| <a id="tcolmax"></a>`TCOLMAX` | Reduce each source col to its maximum. | SFU | 0x051 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLMAX.md) |
| <a id="tcolmin"></a>`TCOLMIN` | Reduce each source col to its minimum. | SFU | 0x052 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLMIN.md) |
| <a id="tcolprod"></a>`TCOLPROD` | Reduce each source col to its product. | SFU | 0x053 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLPROD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLPROD.md) |
| <a id="tcolsum"></a>`TCOLSUM` | Reduce each source col to its sum. | SFU | 0x050 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/column-reduction/TCOLSUM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/column-reduction/TCOLSUM.md) |

### <a id="reduce-and-expand-row-expansion"></a>row-expansion

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
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

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="trowargmax"></a>`TROWARGMAX` | Reduce each source row to its maximum index. | SFU | 0x04C |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWARGMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWARGMAX.md) |
| <a id="trowargmin"></a>`TROWARGMIN` | Reduce each source row to its minimum index. | SFU | 0x04D |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWARGMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWARGMIN.md) |
| <a id="trowmax"></a>`TROWMAX` | Reduce each source row to its maximum. | SFU | 0x041 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWMAX.md) |
| <a id="trowmin"></a>`TROWMIN` | Reduce each source row to its minimum. | SFU | 0x042 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWMIN.md) |
| <a id="trowprod"></a>`TROWPROD` | Reduce each source row to its product. | SFU | 0x043 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWPROD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWPROD.md) |
| <a id="trowsum"></a>`TROWSUM` | Reduce each source row to its sum. | SFU | 0x040 |  | `ExecuteTileReduction` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/reduce-and-expand/row-reduction/TROWSUM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/reduce-and-expand/row-reduction/TROWSUM.md) |

## <a id="memory-and-data-movement-section"></a>memory-and-data-movement

Data movement between tiles and main memory: regular TLOAD/TSTORE/TPREFETCH, irregular MGATHER/MSCATTER, and PE-to-PE GMOV.

### <a id="memory-and-data-movement-irregular"></a>irregular

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="mgather"></a>`MGATHER` | Gather GM elements at Tile-provided indices into the destination. | TLSU |  |  | `MGATHER` | destination0=destination, address=base-address, source0=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER.md) |
| <a id="mgather_cas"></a>`MGATHER_CAS` | Atomically compare and conditionally replace GM elements at Tile-provided indices. | TLSU |  |  | `MGATHER_CAS` | destination0=destination, address=base-address, source0=indices, source1=expected, source2=replacement | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER_CAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER_CAS.md) |
| <a id="mgather_mask"></a>`MGATHER_MASK` | Gather masked GM elements at Tile-provided indices into the destination. | TLSU |  |  | `MGATHER_MASK` | destination0=destination, address=base-address, source0=indices, source1=mask | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MGATHER_MASK.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MGATHER_MASK.md) |
| <a id="mscatter"></a>`MSCATTER` | Scatter source Tile elements to GM addresses selected by Tile indices. | TLSU |  |  | `MSCATTER` | address=base-address, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MSCATTER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MSCATTER.md) |
| <a id="mscatter_mask"></a>`MSCATTER_MASK` | Scatter masked source elements to GM addresses selected by Tile indices. | TLSU |  |  | `MSCATTER_MASK` | address=base-address, source0=source, source1=indices, source2=mask | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/irregular/MSCATTER_MASK.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/irregular/MSCATTER_MASK.md) |

### <a id="memory-and-data-movement-pe-movement"></a>pe-movement

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="gmov"></a>`GMOV` | Copy the resolved peer-PE Tile fragment selected by the bound peer TID. | TLSU |  |  | `GMOV` | destination0=destination, source0=resolved-peer-source, scalar0=peer-tid | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/pe-movement/GMOV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/pe-movement/GMOV.md) |

### <a id="memory-and-data-movement-regular"></a>regular

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tload"></a>`TLOAD` | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. | TLSU |  |  | `TLOAD` | destination0=destination, address=base-address, scalar0=row-stride-elements | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TLOAD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TLOAD.md) |
| <a id="tprefetch"></a>`TPREFETCH` | Prefetch the requested GM byte range without producing a Tile destination. | TLSU |  |  | `TPREFETCH` | address=base-address, byte_count=byte-count | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TPREFETCH.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TPREFETCH.md) |
| <a id="tstore"></a>`TSTORE` | Store the valid Tile rectangle to GM using the encoded base and logical row stride. | TLSU |  |  | `TSTORE` | address=base-address, scalar0=row-stride-elements, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/memory-and-data-movement/regular/TSTORE.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/memory-and-data-movement/regular/TSTORE.md) |

## <a id="matrix-and-matrix-vector-section"></a>matrix-and-matrix-vector

Matrix-Matrix and Matrix-Vector multiply family (GEMM/GEMV) with MX (mixed precision), ACC (accumulate), and BIAS (bias-fused) variants.

### <a id="matrix-and-matrix-vector-matrix-matrix"></a>matrix-matrix

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tmatmul"></a>`TMATMUL` | Multiply the left and right matrices into the destination. | CUBE |  | 0/None | `TMATMUL` | destination0=destination, source0=left, source1=right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.md) |
| <a id="tmatmul_acc"></a>`TMATMUL_ACC` | Multiply matrices and accumulate into the supplied accumulator Tile. | CUBE |  | 2/None | `TMATMUL_ACC` | destination0=destination, source0=accumulator, source1=left, source2=right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_ACC.md) |
| <a id="tmatmul_bias"></a>`TMATMUL_BIAS` | Multiply matrices and add the bias Tile into the destination. | CUBE |  | 1/None | `TMATMUL_BIAS` | destination0=destination, source0=left, source1=right, source2=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_BIAS.md) |
| <a id="tmatmul_mx"></a>`TMATMUL_MX` | Multiply matrices using row and column scale Tiles. | CUBE |  | 4/None | `TMATMUL_MX` | destination0=destination, source0=left, source1=row-scale, source2=right, source3=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX.md) |
| <a id="tmatmul_mx_acc"></a>`TMATMUL_MX_ACC` | Multiply scaled matrices and accumulate into the supplied accumulator Tile. | CUBE |  | 6/None | `TMATMUL_MX_ACC` | destination0=destination, source0=accumulator, source1=left, source2=row-scale, source3=right, source4=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_ACC.md) |
| <a id="tmatmul_mx_bias"></a>`TMATMUL_MX_BIAS` | Multiply scaled matrices and add the bias Tile. | CUBE |  | 5/None | `TMATMUL_MX_BIAS` | destination0=destination, source0=left, source1=row-scale, source2=right, source3=column-scale, source4=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL_MX_BIAS.md) |

### <a id="matrix-and-matrix-vector-matrix-vector"></a>matrix-vector

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tgemv"></a>`TGEMV` | Multiply the matrix by the vector into the destination. | CUBE |  | 16/None | `TGEMV` | destination0=destination, source0=matrix, source1=vector | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV.md) |
| <a id="tgemv_acc"></a>`TGEMV_ACC` | Multiply the matrix by the vector and accumulate into the supplied Tile. | CUBE |  | 18/None | `TGEMV_ACC` | destination0=destination, source0=accumulator, source1=matrix, source2=vector | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_ACC.md) |
| <a id="tgemv_bias"></a>`TGEMV_BIAS` | Multiply the matrix by the vector and add the bias Tile. | CUBE |  | 17/None | `TGEMV_BIAS` | destination0=destination, source0=matrix, source1=vector, source2=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_BIAS.md) |
| <a id="tgemv_mx"></a>`TGEMV_MX` | Multiply the matrix by the vector using row and column scale Tiles. | CUBE |  | 20/None | `TGEMV_MX` | destination0=destination, source0=matrix, source1=row-scale, source2=vector, source3=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX.md) |
| <a id="tgemv_mx_acc"></a>`TGEMV_MX_ACC` | Multiply the scaled matrix and vector and accumulate into the supplied Tile. | CUBE |  | 22/None | `TGEMV_MX_ACC` | destination0=destination, source0=accumulator, source1=matrix, source2=row-scale, source3=vector, source4=column-scale | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_ACC.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_ACC.md) |
| <a id="tgemv_mx_bias"></a>`TGEMV_MX_BIAS` | Multiply the scaled matrix and vector and add the bias Tile. | CUBE |  | 21/None | `TGEMV_MX_BIAS` | destination0=destination, source0=matrix, source1=row-scale, source2=vector, source3=column-scale, source4=bias | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_BIAS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/matrix-and-matrix-vector/matrix-vector/TGEMV_MX_BIAS.md) |

## <a id="layout-and-rearrangement-section"></a>layout-and-rearrangement

Tile layout rearrangement and initialization: transpose, concat, insert, extract, im2col, fill, move.

### <a id="layout-and-rearrangement-initialization"></a>initialization

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tfillpad"></a>`TFILLPAD` | Copy the source and fill destination padding elements with the bound scalar. | SFU | 0x065 |  | `TFILLPAD` | destination0=destination, source0=source, scalar0=padding | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/initialization/TFILLPAD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/initialization/TFILLPAD.md) |

### <a id="layout-and-rearrangement-layout"></a>layout

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tconcat"></a>`TCONCAT` | Concatenate two source Tiles along the selected axis. | SFU | 0x060 |  | `TCONCAT` | destination0=destination, source0=source-left, source1=source-right, axis=axis | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TCONCAT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TCONCAT.md) |
| <a id="textract"></a>`TEXTRACT` | Extract a rectangular source region at the encoded row and column offsets. | SFU | 0x062 |  | `TEXTRACT` | destination0=destination, source0=source, natural0=row-offset, natural1=column-offset | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TEXTRACT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TEXTRACT.md) |
| <a id="timg2col"></a>`TIMG2COL` | Transform an image Tile into kernel-column layout using kernel, stride, padding, and fill operands. | SFU | 0x064 |  | `TIMG2COL` | destination0=destination, source0=source, positive0=kernel-rows, positive1=kernel-columns, positive2=stride-rows, positive3=stride-columns, natural0=pad-rows, natural1=pad-columns, scalar0=padding | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TIMG2COL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TIMG2COL.md) |
| <a id="tinsert"></a>`TINSERT` | Insert the source Tile into the destination region at the encoded row and column offsets. | SFU | 0x063 |  | `TINSERT` | destination0=destination, source0=source, natural0=row-offset, natural1=column-offset | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TINSERT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TINSERT.md) |
| <a id="tmov"></a>`TMOV` | Copy the source Tile payload and definedness into the destination. | TLSU |  |  | `TMOV` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TMOV.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TMOV.md) |
| <a id="ttrans"></a>`TTRANS` | Transpose the source Tile into the destination. | SFU | 0x06E |  | `TTRANS` | destination0=destination, source0=source | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/layout-and-rearrangement/layout/TTRANS.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/layout-and-rearrangement/layout/TTRANS.md) |

## <a id="irregular-and-complex-section"></a>irregular-and-complex

Irregular/complex operations: quantization, histogram, triangularization, sort/merge-sort, scatter/gather, partition.

### <a id="irregular-and-complex-format-conversion"></a>format-conversion

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tdequant"></a>`TDEQUANT` | Dequantize source elements using scale, zero point, rounding, and saturation controls. | SFU | 0x06B |  | `TDEQUANT` | destination0=destination, source0=source, scalar0=scale, scalar1=zero-point, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/format-conversion/TDEQUANT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/format-conversion/TDEQUANT.md) |
| <a id="tquant"></a>`TQUANT` | Quantize source elements using scale, zero point, rounding, and saturation controls. | SFU | 0x06A |  | `TQUANT` | destination0=destination, source0=source, scalar0=scale, scalar1=zero-point, numeric_control=rounding-and-saturation | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/format-conversion/TQUANT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/format-conversion/TQUANT.md) |

### <a id="irregular-and-complex-initialization"></a>initialization

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tci"></a>`TCI` | Initialize destination elements as an ascending or descending counter sequence. | SFU | 0x066 |  | `TCI` | destination0=destination, scalar0=start, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/TCI.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/TCI.md) |
| <a id="thistogram"></a>`THISTOGRAM` | Accumulate a histogram from source values and selected-byte indices. | SFU | 0x068 |  | `THISTOGRAM` | destination0=destination, source0=source, source1=indices, selected_byte=selected-byte | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/THISTOGRAM.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/THISTOGRAM.md) |
| <a id="ttri"></a>`TTRI` | Initialize the selected upper or lower triangular region relative to the diagonal. | SFU | 0x067 |  | `TTRI` | destination0=destination, flag0=upper, diagonal=diagonal | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/initialization/TTRI.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/initialization/TTRI.md) |

### <a id="irregular-and-complex-layout"></a>layout

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tgather"></a>`TGATHER` | Gather source elements by Tile indices into the destination. | SFU | 0x06F |  | `TGATHER` | destination0=destination, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/layout/TGATHER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/layout/TGATHER.md) |
| <a id="tscatter"></a>`TSCATTER` | Scatter source elements by Tile indices into the destination. | SFU | 0x070 |  | `TSCATTER` | destination0=destination, source0=source, source1=indices | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/layout/TSCATTER.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/layout/TSCATTER.md) |

### <a id="irregular-and-complex-sorting"></a>sorting

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tmrgsort"></a>`TMRGSORT` | Merge two sorted source Tiles in the selected ascending or descending order. | SFU | 0x06D |  | `TMRGSORT` | destination0=destination, source0=source-left, source1=source-right, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/sorting/TMRGSORT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/sorting/TMRGSORT.md) |
| <a id="tsort"></a>`TSORT` | Sort source groups, returning ordered values and original U32 indices. | SFU | 0x06C |  | `TSORT` | destination0=destination, destination1=original-indices-u32, source0=source, sort_width=sort-width, flag0=descending | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/sorting/TSORT.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/sorting/TSORT.md) |

### <a id="irregular-and-complex-union"></a>union

| Mnemonic | Summary | Engine | Selector | Func/Mode | Handler | Operands | ASL / Doc |
| --- | --- | :---: | :---: | :---: | --- | --- | --- |
| <a id="tpartadd"></a>`TPARTADD` | Combine corresponding source partitions by addition. | SFU | 0x071 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTADD.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTADD.md) |
| <a id="tpartmax"></a>`TPARTMAX` | Combine corresponding source partitions by maximum selection. | SFU | 0x073 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMAX.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMAX.md) |
| <a id="tpartmin"></a>`TPARTMIN` | Combine corresponding source partitions by minimum selection. | SFU | 0x074 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMIN.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMIN.md) |
| <a id="tpartmul"></a>`TPARTMUL` | Combine corresponding source partitions by multiplication. | SFU | 0x072 |  | `ExecuteTilePartial` | destination0=destination, source0=source-left, source1=source-right | [ASL](https://github.com/PTO-ISA/pto-spec/blob/main/asl/tile/irregular-and-complex/union/TPARTMUL.asl) / [Doc](https://github.com/PTO-ISA/pto-spec/blob/main/docs/tile/irregular-and-complex/union/TPARTMUL.md) |

---

## Appendix A: Bundle composition template

Every tile instruction must appear inside the following bundle shape (using `TADD` as an example):

```asm
BSTART.VEC TADD, DataType
B.DATR (optional)        # data attributes
B.DIM LB0                # first dimension
B.DIM (LB1/LB2 for 2D)   # second/third dimension
B.IOT                    # input/output tile binding
BSTOP
```

`BSTART.TEPL <MNEMONIC>` is an accepted compatibility spelling; from 0.58.0 the canonical split is `BSTART.VEC` / `BSTART.SFU`.

## Appendix B: Mnemonic index

Every mnemonic below is a clickable link that jumps to the instruction row in the main tables (in-document anchor).

| Mnemonic | Class | Engine | Jump |
| --- | --- | :---: | :---: |
| `GMOV` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#gmov) |
| `MGATHER` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#mgather) |
| `MGATHER_CAS` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#mgather_cas) |
| `MGATHER_MASK` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#mgather_mask) |
| `MSCATTER` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#mscatter) |
| `MSCATTER_MASK` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#mscatter_mask) |
| `TABS` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tabs) |
| `TADD` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tadd) |
| `TADDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tadds) |
| `TAND` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tand) |
| `TANDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tands) |
| `TCI` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tci) |
| `TCMP` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tcmp) |
| `TCMPS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tcmps) |
| `TCOLARGMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolargmax) |
| `TCOLARGMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolargmin) |
| `TCOLEXPAND` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpand) |
| `TCOLEXPANDADD` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandadd) |
| `TCOLEXPANDDIV` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpanddiv) |
| `TCOLEXPANDEXPDIF` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandexpdif) |
| `TCOLEXPANDMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandmax) |
| `TCOLEXPANDMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandmin) |
| `TCOLEXPANDMUL` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandmul) |
| `TCOLEXPANDSUB` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolexpandsub) |
| `TCOLMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolmax) |
| `TCOLMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolmin) |
| `TCOLPROD` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolprod) |
| `TCOLSUM` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#tcolsum) |
| `TCONCAT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#tconcat) |
| `TCVT` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tcvt) |
| `TDEQUANT` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tdequant) |
| `TDIV` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tdiv) |
| `TDIVS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tdivs) |
| `TEXP` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [jump →](#texp) |
| `TEXPANDS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#texpands) |
| `TEXTRACT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#textract) |
| `TFILLPAD` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#tfillpad) |
| `TFMA` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tfma) |
| `TGATHER` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tgather) |
| `TGEMV` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv) |
| `TGEMV_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv_acc) |
| `TGEMV_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv_bias) |
| `TGEMV_MX` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv_mx) |
| `TGEMV_MX_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv_mx_acc) |
| `TGEMV_MX_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tgemv_mx_bias) |
| `THISTOGRAM` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#thistogram) |
| `TIMG2COL` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#timg2col) |
| `TINSERT` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#tinsert) |
| `TLOAD` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#tload) |
| `TLOG` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [jump →](#tlog) |
| `TMATMUL` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul) |
| `TMATMUL_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul_acc) |
| `TMATMUL_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul_bias) |
| `TMATMUL_MX` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul_mx) |
| `TMATMUL_MX_ACC` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul_mx_acc) |
| `TMATMUL_MX_BIAS` | [matrix-and-matrix-vector](#matrix-and-matrix-vector) | CUBE | [jump →](#tmatmul_mx_bias) |
| `TMAX` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tmax) |
| `TMAXS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tmaxs) |
| `TMIN` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tmin) |
| `TMINS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tmins) |
| `TMOV` | [layout-and-rearrangement](#layout-and-rearrangement) | TLSU | [jump →](#tmov) |
| `TMRGSORT` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tmrgsort) |
| `TMUL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tmul) |
| `TMULS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tmuls) |
| `TNEG` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tneg) |
| `TNOT` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tnot) |
| `TOR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tor) |
| `TORS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tors) |
| `TPARTADD` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tpartadd) |
| `TPARTMAX` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tpartmax) |
| `TPARTMIN` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tpartmin) |
| `TPARTMUL` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tpartmul) |
| `TPREFETCH` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#tprefetch) |
| `TQUANT` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tquant) |
| `TRECIP` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [jump →](#trecip) |
| `TRELU` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#trelu) |
| `TREM` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#trem) |
| `TREMS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#trems) |
| `TROWARGMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowargmax) |
| `TROWARGMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowargmin) |
| `TROWEXPAND` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpand) |
| `TROWEXPANDADD` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandadd) |
| `TROWEXPANDDIV` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpanddiv) |
| `TROWEXPANDEXPDIF` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandexpdif) |
| `TROWEXPANDMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandmax) |
| `TROWEXPANDMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandmin) |
| `TROWEXPANDMUL` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandmul) |
| `TROWEXPANDSUB` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowexpandsub) |
| `TROWMAX` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowmax) |
| `TROWMIN` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowmin) |
| `TROWPROD` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowprod) |
| `TROWSUM` | [reduce-and-expand](#reduce-and-expand) | SFU | [jump →](#trowsum) |
| `TRSQRT` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [jump →](#trsqrt) |
| `TSCATTER` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tscatter) |
| `TSEL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tsel) |
| `TSELS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tsels) |
| `TSHL` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tshl) |
| `TSHLS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tshls) |
| `TSHR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tshr) |
| `TSHRS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tshrs) |
| `TSORT` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#tsort) |
| `TSQRT` | [elementwise-tile-tile](#elementwise-tile-tile) | SFU | [jump →](#tsqrt) |
| `TSTORE` | [memory-and-data-movement](#memory-and-data-movement) | TLSU | [jump →](#tstore) |
| `TSUB` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#tsub) |
| `TSUBS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#tsubs) |
| `TTRANS` | [layout-and-rearrangement](#layout-and-rearrangement) | SFU | [jump →](#ttrans) |
| `TTRI` | [irregular-and-complex](#irregular-and-complex) | SFU | [jump →](#ttri) |
| `TXOR` | [elementwise-tile-tile](#elementwise-tile-tile) | VEC | [jump →](#txor) |
| `TXORS` | [tile-scalar-and-immediate](#tile-scalar-and-immediate) | VEC | [jump →](#txors) |

## Appendix C: Catalog deletions and rejections (from `asl/tile/model/dispatch/top-level.asl`)

- **Deleted names** (once present in historical catalogs, no longer retained in 0.58.0):

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

- **Rejected names** (architecturally rejected, not in the catalog):

  - `TEXRACT`
  - `TFILL/TEXPANDS`
  - `TPOW`
  - `TPOWS`
