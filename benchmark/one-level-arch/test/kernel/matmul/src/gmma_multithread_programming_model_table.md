# GMMA 多线程编程模型：硬件抽象表

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

