# GMMA 多线程编程模型：硬件、内存与程序员可见抽象

本文档根据 `matmul_gmma_pesudo.cpp` 和 `matmul_gmma_pesudo_report.md` 总结 GMMA ReuseB 的多线程编程模型。重点不是矩阵乘算法本身，而是该模型如何抽象硬件、线程、内存层级、tile 搬运方式，以及哪些对象对程序员可见。

## 1. 模型总览
 
该模型可以概括为：

```text
4 个线程分别映射到 4 个 PE。

每个 PE 只看见自己的 PE-local tile：
  A_pe / C_pe

右矩阵 B 不按 PE 切分，而是进入 shared / staging B：
  B_shared

GMMA 作为 collective 指令，把 4 个 PE-local A tile
和一个 shared B tile 组合成一次逻辑 GEMM。
```

从程序员视角看，kernel 像是 SPMD 程序：每个线程写的是同一份代码，但通过 `thread_id` 访问不同的 PE-local 数据分片。

## 2. 硬件抽象

该 multi-thread model 暴露了三类硬件资源抽象：

<table>
  <colgroup>
    <col style="width: 100px;">
    <col>
    <col>
    <col>
  </colgroup>
  <thead>
    <tr>
      <th style="width: 100px; min-width: 100px;">层级</th>
      <th>硬件抽象</th>
      <th>多PE并行编程</th>
      <th>多PE group编程</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" style="width: 100px; min-width: 100px;">执行模型</td>
      <td>考虑维度</td>
      <td colspan="2">
        <ul>
          <li>如何暴露线程</li>
          <li>vector / CUBE 对外呈现形式</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>PE / thread</td>
      <td colspan="2"> 4个图灵完备的superscalar, 由thread_id区分PE</td>
    </tr>
    <tr>
      <td> cube</td>
      <td> <code>tmatmul</code>调用mma指令</code>。</td>
      <td><code>tgmatmul(tA, tB, tC)</code>调用gmma指令，其中 <code>tA/tC</code> 是当前 PE cell中的分片，逻辑拼成完整 <code>A_big/C_big</code>。</td>
    </tr>
    <tr>
      <td>vector</td>
      <td> <code>tileop</code></td>
      <td><code>tileop / group_tile_op</code></td>
    </tr>
    <tr>
      <td rowspan="6" style="width: 100px; min-width: 100px;">内存模型</td>
      <td>考虑维度</td>
      <td colspan="2">
        <ul>
          <li>cell / staging B 是否对外可见</li>
          <li>staging B 是 reg 还是 buffer；如果是 buffer，需要程序员管理内存</li>
          <li>layout 转换是否显式需要：cube 需要 <code>Nz/Zn</code>，vector 需要 <code>ND/DN</code></li>
          <li> 寄存器对外呈现(缺乏资料)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td rowspan="2">PE-local cell</td>
      <td colspan="2">线程控制当前 PE 的 <code>tileA</code>加载、<code>tileC</code>写回；其它 PE 的 tile 不可见。</td>
    </tr>
    <tr>
      <td></td>
      <td>逻辑构成完整大 <code>tileA [tM,tK]</code>、<code>tileC [tM,tN]</code>。</td>
    </tr>
    <tr>
      <td>staging B</td>
      <td> staging B 按 PE 切分, 各线程维护自己在staging B中的tile。</td>
      <td> staging B 不按 PE 切分；<code>tileB</code> 通过<code>gmma.ld</code>加载到shared staging B，被多个 PE 和多个 M block 复用。</td>
    </tr>
    <tr>
      <td>global memory</td>
      <td><code>global_iterator</code> 直接索引完整 tile，例如 <code>gAIter(i,k)</code>、<code>gCIter(i,j)</code>。</td>
      <td>  根据<code>thread_id</code>手动切分M轴tile，如 <code>gAIter(i*thread_num+thread_id,k)</code>。</td>
    </tr>
    <tr>
      <td>layout 转换</td>
      <td colspan="2">待定，目前假定cell reg中数据均为ND格式，<code>ND2Nz/ND2Zn</code> 等 layout 由 GMMA/cube 处理，layout软件不感知。</td>
    </tr>
    <tr>
      <td rowspan="3" style="width: 100px; min-width: 100px;">模型对比</td>
      <td style="vertical-align: top;">code example<br><span style="color: #d1242f; font-size: 11px;"></span></td>
      <td style="vertical-align: top;">
        <div style="box-sizing: border-box; width: 100%; padding: 8px 10px; overflow-x: auto; white-space: pre; tab-size: 4; border: 1px solid #d8dee4; border-radius: 5px; background: #f6f8fa; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 10px; font-weight: 400; line-height: 1.35; letter-spacing: 0;">template &lt;typename dtype, int gM, int gN, int gK,
          int tM, int tN, int tK>
void matmul_gmma_ReuseB(<span style="color: #d1242f; font-weight: 600;">float (*c_ptr)[gM * gN]</span>,
                        <span style="color: #d1242f; font-weight: 600;">dtype (*a_ptr)[gM * gK]</span>,
                        <span style="color: #d1242f; font-weight: 600;">dtype (*b_ptr)[gK * gN]</span>) {
    const uint32_t thread_id = get_thread_id();
    // global memory shape
    using gmA = global_tensor&lt;dtype, RowMajor&lt;gM, gK>>;
    using gmB = global_tensor&lt;dtype, RowMajor&lt;gK, gN>>;
    using gmC = global_tensor&lt;float, RowMajor&lt;gM, gN>>;
    // tile shape
    using tileA = TileLeft&lt;dtype, <span style="color: #d1242f; font-weight: 600;">tM</span>, tK>;
    // tile B存在staging B
    using tileB = TileRight&lt;dtype, tK, tN>;
    using tileC = Tile&lt;Location::Vec, float, <span style="color: #d1242f; font-weight: 600;">tM</span>, tN,
                       BLayout::RowMajor>;
    using itA = global_iterator&lt;gmA, tileA>;
    using itB = global_iterator&lt;gmB, tileB>;
    using itC = global_iterator&lt;gmC, tileC>;
    // 每个PE读取各自的矩阵
    itA gAIter(<span style="color: #d1242f; font-weight: 600;">a_ptr[thread_id]</span>);
    itB gBIter(<span style="color: #d1242f; font-weight: 600;">b_ptr[thread_id]</span>);
    itC gCIter(<span style="color: #d1242f; font-weight: 600;">c_ptr[thread_id]</span>);
    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    // 各个PE并行处理各自的矩阵计算
    for (int j = 0; j &lt; Nb; ++j) {
        tileB tB;
        <span style="color: #d1242f; font-weight: 600;">TLOAD</span>(tB, gBIter(0, j));
        for (int i = 0; i &lt; Mb; ++i) {
            tileA tA;
            tileC tC;
            TLOAD(tA, gAIter(<span style="color: #d1242f; font-weight: 600;">i</span>, 0));
            // 各PE并行执行mma指令
            <span style="color: #d1242f; font-weight: 600;">tmatmul</span>(tA, tB, tC);
            TSTORE(gCIter(<span style="color: #d1242f; font-weight: 600;">i</span>, j), tC);
        }
    }
}
int main() {
    constexpr uint32_t thread_num = 4;
    <span style="color: #d1242f; font-weight: 600;">dtype srcA[thread_num][M * K];</span>
    <span style="color: #d1242f; font-weight: 600;">dtype srcB[thread_num][K * N];</span>
    <span style="color: #d1242f; font-weight: 600;">float dstC[thread_num][M * N];</span>
    matmul_gmma_ReuseB&lt;dtype, M, N, K, tileM, tileN, tileK>
        &lt;&lt;&lt;thread_num>>>(dstC, srcA, srcB);
}</div>
      </td>
      <td style="vertical-align: top;">
        <div style="box-sizing: border-box; width: 100%; padding: 8px 10px; overflow-x: auto; white-space: pre; tab-size: 4; border: 1px solid #d8dee4; border-radius: 5px; background: #f6f8fa; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 10px; font-weight: 400; line-height: 1.35; letter-spacing: 0;">template &lt;typename dtype, int gM, int gN, int gK,
          int tM, int tN, int tK>
void matmul_gmma_ReuseB(<span style="color: #d1242f; font-weight: 600;">float *c_ptr</span>,
                        <span style="color: #d1242f; font-weight: 600;">dtype *a_ptr</span>,
                        <span style="color: #d1242f; font-weight: 600;">dtype *b_ptr</span>,
                        <span style="color: #d1242f; font-weight: 600;">const uint32_t thread_num = 4</span>) {
    const uint32_t thread_id = get_thread_id();
    // global memory shape
    using gmA = global_tensor&lt;dtype, RowMajor&lt;gM, gK>>;
    using gmB = global_tensor&lt;dtype, RowMajor&lt;gK, gN>>;
    using gmC = global_tensor&lt;float, RowMajor&lt;gM, gN>>;
    // tile shape, 按PE切分
    using tileA = TileLeft&lt;dtype, <span style="color: #d1242f; font-weight: 600;">tM / thread_num</span>, tK>;
    // tile B存在staging B, B矩阵跨PE共享
    using tileB = TileRight&lt;dtype, tK, tN>;
    using tileC = Tile&lt;Location::Vec, float,
                       <span style="color: #d1242f; font-weight: 600;">tM / thread_num</span>, tN,
                       BLayout::RowMajor>;
    using itA = global_iterator&lt;gmA, tileA>;
    using itB = global_iterator&lt;gmB, tileB>;
    using itC = global_iterator&lt;gmC, tileC>;
    itA gAIter(<span style="color: #d1242f; font-weight: 600;">a_ptr</span>);
    itB gBIter(<span style="color: #d1242f; font-weight: 600;">b_ptr</span>);
    itC gCIter(<span style="color: #d1242f; font-weight: 600;">c_ptr</span>);
    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    for (int j = 0; j &lt; Nb; ++j) {
        tileB tB;
        // 对应gmma.ld指令
        <span style="color: #d1242f; font-weight: 600;">TgLOAD</span>(tB, gBIter(0, j));
        for (int i = 0; i &lt; Mb; ++i) {
            <span style="color: #d1242f; font-weight: 600;">const int row = i * thread_num + thread_id;</span>
            tileA tA;
            tileC tC;
            TLOAD(tA, gAIter(<span style="color: #d1242f; font-weight: 600;">row</span>, 0));
            // 4 PE合力执行gmma指令
            <span style="color: #d1242f; font-weight: 600;">tgmatmul</span>(tA, tB, tC);
            TSTORE(gCIter(<span style="color: #d1242f; font-weight: 600;">row</span>, j), tC);
        }
    }
}
int main() {
    constexpr uint32_t thread_num = 4;
    <span style="color: #d1242f; font-weight: 600;">dtype srcA[M * K];</span>
    <span style="color: #d1242f; font-weight: 600;">dtype srcB[K * N];</span>
    <span style="color: #d1242f; font-weight: 600;">float dstC[M * N];</span>
    matmul_gmma_ReuseB&lt;dtype, M, N, K, tileM, tileN, tileK>
        &lt;&lt;&lt;thread_num>>>(dstC, srcA, srcB<span style="color: #d1242f; font-weight: 600;">, thread_num</span>);
}</div>
      </td>
    </tr>
    <!-- <tr>
      <td>优劣势分析</td>
      <td>
        <ul>
          <li>优点：编程直观简单，纯api编程，不感知gmma和mma差异。</li>
          <li>缺点：无法表达PE间多个小tile的矩阵计算并行,完全由硬件/编译器控制,负担较重。</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>优点：控制力度更灵活。可以做更细粒度的流水编排</li>
          <li>缺点：尾块处理多线程编程可能比较复杂。</li>
        </ul>
      </td>
    </tr> -->
    <tr>
      <td>算子实现</td>
      <td>
        <ul>
            可快速迁移复用现有一层架构的代码
        </ul>
      </td>
      <td>
        <ul>
          <li>✅ matmul</li>
          <li>✅ fa</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>遗留问题</td>
      <td> - </td>
      <td> gmma限定4 PE锁步，理论上thread也只能是4，thread无法脱离PE做抽象</td>
    </tr>
    <!-- <tr>
      <td>TODO</td>
      <td colspan="3">
        <ul>
          <li>生态兼容问题
            <ul>
              <li> NVIDIA 代码迁移是否方便</li>
              <li>warp specialization 是否需要，如何映射</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr> -->
  </tbody>
</table>

其中最关键的抽象边界是：

```text
PE-local tile 是当前 PE 独占的。
shared B tile 是 GMMA collective 可共享的。
GMMA engine 负责把多个 PE-local tile 逻辑组织成一个 big tile。
```

## 3. 内存层级

该模型中的内存层级可以按可见性分为三层。

### 3.1 Global Memory

Global memory 对所有 PE 可见，用 `global_tensor` 描述布局：

```cpp
using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
using gmC = global_tensor<float, RowMajor<gM, gN>>;
```

Global memory 中的数据形状：

```text
A: [gM, gK], RowMajor
B: [gK, gN], RowMajor
C: [gM, gN], RowMajor
```

程序员可见内容：

- tensor layout
- tile iterator
- 从 global memory 取哪个 tile
- 写回 global memory 的哪个 tile

### 3.2 PE-local Cell

PE-local cell 是每个 PE 独占的 tile 存储。程序员用普通 tile 变量表示当前 PE 自己的 tile：

```cpp
tileA tA;       // 当前 PE 的 lhs tile
tileC tC;       // 当前 PE 的 output tile
```

这类 tile 不应该在单个 PE 程序中写成 `tA[4]`、`tC[4]` 来表示所有 PE 的 tile，因为其它 PE 的 cell 对当前 PE 不可见。

PE-local tile 的形状：

```text
tA   : [tM / thread_num, tK]
tC   : [tM / thread_num, tN]
```

### 3.3 Staging B / Shared RHS

B tile 不按 PE 切分，而是进入 shared / staging B：

```cpp
tileB tB[kReuseK];
TLOAD(tB[k], gB);
```

其形状为：

```text
tB[k]: [tK, tN]
```

它的语义是：

- 对 GMMA collective 可见
- 作为 rhs / `Sb`
- 在 ReuseB 模型中跨多个 M block 复用
- 不属于某个单独 PE 的私有 cell

## 4. 线程抽象与交互方式

### 4.1 线程映射

模型中启动 4 个线程：

```cpp
uint32_t thread_num = 4;
matmul_gmma_ReuseB<<<thread_num>>>(..., thread_num);
```

每个线程通过 `thread_id` 选择自己的 M 维分片：

```cpp
auto gA = gAIter(i * thread_num + thread_id, k);
auto gC = gCIter(i * thread_num + thread_id, j);
```

对应关系：

| `thread_id` | PE | A/C tile row slice |
|---:|---:|---|
| 0 | PE0 | 第 0 个 `[tM/4, *]` |
| 1 | PE1 | 第 1 个 `[tM/4, *]` |
| 2 | PE2 | 第 2 个 `[tM/4, *]` |
| 3 | PE3 | 第 3 个 `[tM/4, *]` |

### 4.2 线程之间如何交互

线程之间不通过普通 C++ 变量交换 PE-local tile。

错误理解：

```cpp
tileA tA[4];      // 一个线程看见所有 PE 的 A tile
tileC tC[4];      // 一个线程看见所有 PE 的 C tile
```

正确理解：

```cpp
tileA tA;       // 当前 PE 私有
tileC tC;       // 当前 PE 私有
```

多个 PE 的 `tA` 只在 GMMA collective 指令内部被硬件逻辑组合：

```text
PE0.tA + PE1.tA + PE2.tA + PE3.tA
    --GMMA collective-->
logical A_big [tM, tK]
```

也就是说，线程间交互不是显式 load/store 或共享数组，而是通过 GMMA collective engine 的隐式同步和硬件组织完成。

## 5. Tile 加载方式

### 5.1 A tile 加载

A 是 lhs，按 M 维切分到 PE：

```cpp
using tileA = TileLeft<dtype, tM / thread_num, tK>;

auto gA = gAIter(i * thread_num + thread_id, k);
TLOAD(tA, gA);
```

每个 PE 加载自己的 A slice：

```text
PE0: A[i*tM + 0*tM/4 : i*tM + 1*tM/4, k*tK : (k+1)*tK]
PE1: A[i*tM + 1*tM/4 : i*tM + 2*tM/4, k*tK : (k+1)*tK]
PE2: A[i*tM + 2*tM/4 : i*tM + 3*tM/4, k*tK : (k+1)*tK]
PE3: A[i*tM + 3*tM/4 : i*tM + 4*tM/4, k*tK : (k+1)*tK]
```

程序员可见：

- 当前 PE 的 `thread_id`
- 当前 PE 的 `tileA tA`
- `TLOAD(tA, gA)`

程序员不可见：

- 其它 PE 的 `tA`
- 其它 PE cell 内部地址

### 5.2 B tile 加载

B 是 rhs，不按 PE 切分：

```cpp
using tileB = TileRight<dtype, tK, tN>;

auto gB = gBIter(k, j);
TLOAD(tB[k], gB);
```

B tile 进入 staging B：

```text
B[k*tK : (k+1)*tK, j*tN : (j+1)*tN]
    -> tB[k] [tK, tN]
```

在注释模型中，这类 load 对应 `GMMA.ld` / staging B load。它的关键语义是：B tile 不属于某一个 PE，而是作为 GMMA rhs 被 collective 使用。

### 5.3 C tile store

C 是 output，按 M 维由各 PE 分片写回：

```cpp
using tileC = Tile<Location::Vec, float,
                   tM / thread_num, tN,
                   BLayout::RowMajor>;

auto gC = gCIter(i * thread_num + thread_id, j);
TSTORE(gC, tC);
```

每个 PE 只写自己的 C slice。最终 4 个 PE 的 store 共同形成逻辑 `C_big [tM,tN]`。

## 6. GMMA 用法

### 6.1 程序中看到的调用

在单个线程/PE 的程序中，GMMA 调用形式是：

```cpp
gmma(tA, tB[k], tC);
```

其中：

| 参数 | 当前 PE 视角 | GMMA collective 视角 |
|---|---|---|
| `tA` | 当前 PE 的 `[tM/4,tK]` lhs tile | 4 个 PE 的 `tA` 合成 `A_big [tM,tK]` |
| `tB[k]` | shared rhs tile | `B_shared [tK,tN]` |
| `tC` | 当前 PE 的 `[tM/4,tN]` output tile | 4 个 PE 的 `tC` 合成 `C_big [tM,tN]` |

### 6.2 GMMA 的硬件语义

GMMA 执行时的逻辑计算是：

```text
A_big [tM, tK] * B_shared [tK, tN] -> C_big [tM, tN]
```

但物理分布是：

```text
A_big = concat(PE0.tA, PE1.tA, PE2.tA, PE3.tA) along M
C_big = concat(PE0.tC, PE1.tC, PE2.tC, PE3.tC) along M
B_shared = shared staging B tile
```

程序员需要提供：

- 当前 PE 的 lhs tile
- shared rhs tile
- 当前 PE 的 output tile

硬件/GMMA collective 负责：

- 收集/解释 4 个 PE 的 lhs tile
- 读取 shared staging B
- 组织 ND 到内部矩阵布局的转换
- 将输出分发回各 PE 的 `tC`

## 7. ReuseB 与内存流

ReuseB 的目的是减少 B tile 的重复加载。循环顺序是：

```text
for j in output-N blocks:
    load B[k,j] tiles into staging B
    for i in output-M blocks:
        load A[i,k] slice for current PE
        gmma(A_pe, B_shared, C_pe)
        store C_pe
```

这意味着：

```text
B_shared 的生命周期覆盖多个 i。
A_pe 和 C_pe 的生命周期只属于当前 i 和当前 PE。
```

内存流如下：

```text
Global B -> staging B -> GMMA rhs

Global A -> PE-local cell -> GMMA lhs

GMMA output -> PE-local C -> Global C
```

## 8. 程序员可见与不可见对象

| 类别 | 对象 | 程序员是否可见 | 说明 |
|---|---|---|---|
| 全局内存 | `global_tensor`, `global_iterator` | 可见 | 用于描述 global layout 和 tile 访问 |
| 当前 PE id | `thread_id` | 可见 | 决定当前 PE 的 A/C row slice |
| PE 私有 tile | `tA`, `tC` | 可见 | 只能表示当前 PE 的 tile |
| 其它 PE 私有 tile | `PE1.tA` 等 | 不可直接可见 | 不能用普通数组表达为当前 PE 可访问对象 |
| Staging B | `tB[k]` | 可见为 shared rhs | 作为 GMMA rhs，被 collective 使用 |
| GMMA 合并后的 big tile | `A_big`, `C_big` | 逻辑可见，物理不可见 | 程序员理解其语义，但不直接声明完整 tile |
| 内部布局转换 | ND2Nz / ND2Zn 等 | 不直接可见 | 由 GMMA/cube 内部处理 |

## 9. 编程规则总结

1. PE 私有 tile 用单个变量表示，例如 `tileA tA`，不要用数组表示所有 PE 的私有 tile。
2. `thread_id` 是当前线程/PE 选择 A/C 分片的唯一显式依据。
3. A 按 M 维切分，进入 PE-local cell。
4. B 不按 PE 切分，进入 staging B，作为 shared rhs。
5. C 按 M 维分布在各 PE 中。
6. GMMA 是 collective 指令，负责把 4 个 PE 的 lhs tile 逻辑合并为 `A_big`。
7. 程序员写的是局部 PE 程序；GMMA 的 big tile 是逻辑概念，不是单个线程持有的对象。
8. ReuseB 的核心优化点是 B tile 在 M loop 外加载，在多个 M tile 间复用。

## 10. 总结

这个多线程 GMMA 编程模型的重点是硬件可见性的划分：

```text
程序员显式管理：
  global tensor layout
  thread_id 到 PE-local tile 的映射
  PE-local A/C tile
  shared staging B tile
  gmma 调用顺序

硬件/GMMA 隐式完成：
  多 PE lhs tile 的 collective 组合
  shared rhs tile 的 collective 读取
  内部矩阵布局转换
  输出结果到各 PE `tC` 的分发
```

因此，这不是“一个线程拥有完整 tile”的编程模型，而是“每个 PE 拥有局部分片，GMMA 在硬件层把分片组合成逻辑大矩阵”的编程模型。
