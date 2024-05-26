# DeepSeek-V2高性能推理优化笔记：MLA算子优化

## 前言

最近，幻方发布的DeepSeek-V2模型得到了学术界和产业界的广泛关注。作为一款236B参数的MoE大模型，DeepSeek-V2通过独特的DeepSeekMoE架构设计，每token仅需激活21B的参数，且通过新提出的MLA机制替换传统的MHA和MQA注意力机制，实现了推理过程中的KV Cache大小的大幅降低。因此，DeepSeek-V2能够以较低的推理成本，取得了与GPT-4相当的模型性能。

MLA机制是DeepSeek-V2中的一个核心创新点。作为计算机系统方向的研究人员，我们自然不敢从AI/ML的角度对MLA的算法设计妄加评论。然而，从System的角度来看，MLA无疑是一个非常优秀的设计。近年来，大模型推理成本居高不下的一大原因就是GPU的算力利用率低下。由于Tensor Core等专门电路的的出现，现代GPU的计算能力已经远强于其内存带宽，GPU每读入的一个字节的数据，往往要参与到数百次计算中，才能保证GPU取得较好的计算资源利用率（即MFU）。然而，大模型推理的任务负载通常难以提供如此高的计算强度，即GPU读入的参数未参与足够多次的计算便要被丢弃而读入下一个参数，这导致显存带宽成为了整个推理过程的性能瓶颈。这其中的一大障碍就是KV Cache的空间占用问题：GPU的显存空间往往非常有限，较大的KV Cache会导致同时处理的request数量变少，也即batch size较小；以vLLM为代表的一众工作就是从这个角度入手，优化KV Cache的显存利用率，从而提高推理过程的硬件资源利用率。另一方面，针对传统的MHA或GQA算子，在计算注意力的过程中，所有KV Cache中的数据读取后都仅参与一次或几次计算，导致该算子的MFU极低，并且由于每个request有自己的KV Cache，这一问题无法通过提高batch size的方式解决。而MLA算子，从其计算特征来看，同时解决了这两方面的问题：一方面，通过低秩压缩大幅降低了KV Cache的大小，另一方面，MLA解压缩后的多头注意力机制能够提供较高的计算强度，有助于充分利用GPU的算力资源。很明显，MLA算子是针对现代GPU硬件特点“量体裁衣”定制的一个注意力机制，通过对存储和计算的再平衡，能够充分发挥现代GPU的各项优势。

DeepSeek-V2开源的代码中未对MLA算子进行过多的优化。我们尝试复现了一些MLA算子在推理阶段可能涉及的优化点，并对其进行了评测和分析。

## MLA算子的计算过程

给定输入向量$h_t \in \mathbb{R}^{B \times L \times 5120}$，其中$B$为batch size，$L$为sequence length，MLA的计算过程如下。

### Q向量

在DeepSeek-V2中，Q向量也采用了低秩压缩的方式。首先，将输入向量投影到一个1536维的低维空间：
$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{B \times L \times 1536} $$
然后，将其投影到$\mathbb{R}^{H \times 128}$的多头向量空间上（其中$H=128$是heads数），得到了Q向量的第一部分：
$$ q_t^C = W^{UQ} c_t^Q \in \mathbb{R}^{B \times L \times H \times 128} $$
再将其投影到$\mathbb{R}^{H \times 64}$上并使用RoPE嵌入位置信息，得到Q向量的第二部分：
$$ q_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times H \times 64} $$
将两部分拼接的到最终的Q向量：
$$ q_t = [q_t^C, q_t^R] \in \mathbb{R}^{B \times L \times H \times 192} $$

### KV向量

计算KV向量时，首先需要将输入向量投影为512维的联合压缩表示：
$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{B \times L \times 512} $$

与Q向量的计算过程类似，K向量的第一部分是将$c_t^{KV}$通过投影解压缩到$\mathbb{R}^{H \times 128}$的多头向量空间：
$$ k_t^C = W^{UK} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$
K的第二部分是将输入向量投影到64维向量空间并施加RoPE嵌入位置信息：
$$ k_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times 64} $$
与Q不同的是，完整的K是将K的第二部分广播到每个head后与第一部分拼接得到：
$$ k_t = \begin{bmatrix}
    k_{t,1}^C & k_t^R \\ 
    k_{t,2}^C & k_t^R \\
    \vdots & \vdots \\
    \end{bmatrix} \in \mathbb{R}^{B \times L \times H \times 192} $$
也就是说，每个head的RoPE部分是完全相同的。

V向量的计算较为简单，直接将$c_t^{KV}$解压缩到$\mathbb{R}^{H \times 128}$即可：
$$ v_t = W^{UV} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$

### Attention计算

Attention的计算过程和传统的MHA并无差异。首先计算attention score：
$$ a = (q_t^\top k_t + \mathrm{Mask}) / \sqrt{192} = ({q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R + \mathrm{Mask}) / \sqrt{128 + 64}\in \mathbb{R}^{B \times L \times H \times L} $$
对a的最后一维做softmax，并计算对V的加权和，得到Attention输出：
$$ o = \mathrm{softmax}(a) v_t \in \mathbb{R}^{B \times L \times H \times 128} $$
经过另一个矩阵的投影，得到MLA的最终输出：
$$ u = W^o o \in \mathbb{R}^{B \times L \times 5120} $$


## 开源代码MLA算子分析

``` python
def forward(...):
    bsz, q_len, _ = hidden_states.size()
    
    # 计算Q：先降维再升维，好处是相比直接使用大小为 [5120, 24576] 的矩阵
    # [5120, 1536] * [1536, 24576] 这样的低秩分解在存储空间和计算量上都大幅度降低
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # 切分 rope 和非 rope 部分
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    
    # 计算KV
    # 一个优化的 MLA KVCache 实现只需要缓存这个 compressed_kv 就行，不过后面实际上展开了
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # 此处compressed_kv 对应公式中的 c_t^{KV}
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # 将 MLA 展开成标准 MHA 的形式
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    # 因为 kv_b_proj 打包了 W^{UK} 和 W^{UV} 把他们分离出来
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    ...
    # 给需要 rope 的部分加 rope
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    
    # 更新和拼接历史 KVCache，可以看到这里存储的是展开后的 MHA KVCache
    # 其中 q_head_dim 等于 qk_nope_head_dim + qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # 后续就是标准的 MHA 代码，不再赘述
    ...
```

## MLA实现优化

### Caching Compressed KV

在原始的transformer计算过程中，每次都需要计算完整的KV向量，这部分的计算往往会带来较大的开销。实际上，每次模型迭代过程中，这些KV向量的值都是一样的。因此，我们可以采用“空间换时间”的策略，将先前迭代过程中KV向量的值缓存下来，这样在后续的迭代过程中，就不需要重复计算KV向量了，从而大大减小了模型推理过程中的计算量。

然而，在以MHA为代表的传统Attention算子中，这种空间换时间的策略往往会矫枉过正。由于KV cache占用的空间很大，并且KV cache中的数据在每次迭代过程中仅参与一次计算，在使用KV cache后，虽然计算量减小了，但是显存占用量以及显存带宽需求却急剧上升，成为了制约大模型推理效率的新瓶颈。MLA的设计通过多头共用压缩后的KV表示，一方面大幅减少了KV cache的占用，另一方面，由于Compressed KV在每个head中都参与了计算，DeepSeek-V2的128个heads能够提供足够的计算强度，因此Attention部分的MFU也得到了大幅提高。

在开源的版本中， MLA算子了完整的KV Cache，丧失了MLA的上述种种好处。我们尝试改为缓存压缩后的KV Cache，并与缓存完整的KV Cache进行对比。此外我们还实现了一个不缓存任何KV Cache的版本作为Baseline。各种实现的KV Cache占用和计算量如下：

| 实现版本 | 每token每层Cache大小 | 每token每层计算量 |
| :---: | :---: | :---: |
| Baseline | 0 | 19.77 MFLOP |
| CacheDecompressed | 81.92 kB | 0.04 MFLOP |
| CacheCompressed | 1.152 kB | 16.82 MFLOP |

我们分别在A100-PCIe-40G（Compute80架构）和GeForce RTX 4080（Compute89架构）上对上述实现进行性能测试。对于单个request，各种实现的性能表现如下图所示：

![](data/caching-B1.png)

此时，CacheDecompressed的性能最好，CacheCompressed次之，Baseline最差。这正好与三种实现的计算量相对应。

当Batch Size=32时，各实现的性能如下图所示：

![](data/caching-B32.png)

测试结果基本和单个查询时的相同。

### Projection Absorption

上述分析和实验结果表明，相比缓存完整的KV Cache，缓存压缩后的KV Cache会带来较大的性能下降。另外一个重要的问题是，当前的CacheDecompressed实现实际上并不能缓解KV Cache过大的问题，这是由于在计算MLA的时候，仍然需要存储解压后的完整的KV Cache，这很可能引起OOM崩溃。

所幸DeepSeek-V2的论文中提出，可以将KV的解压缩矩阵吸收到Q-projection和Out-projection中，从而可以在不解压缩KV Cache的情况下直接计算最终的Attention结果。

## 后续优化


