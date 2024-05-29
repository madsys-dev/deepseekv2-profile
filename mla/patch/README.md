This patched DeepseekV2Model contains the following modifications to DeepseekV2Attention for reducing VRAM consumption and improve efficiency:

1. Instead of caching the decompressed key/value states, we cache only the low-rank key-value joint compression as well 
as the decoupled RoPE part of the keys. For the sake of reusing the cache utility of transformers library, we treat 
k_pe as key_states and compressed_kv as value_states.
2. We implement the absorption technique described in the DeepseekV2 paper, by changing the multiplication order when 
computing query and output vectors. This not only saves memory consumption of intermediate tensors but also reduces 
the number of floating-point operations.
3. We compute the RoPE part and non-RoPE part of the attention score separately and then sum them up. The original 
implementation concatenates the two parts of the query/key vectors, which has proven to be quite inefficient when 
caching compressed key/value states due to unnecessary data broadcast and memory round-trips.

By applying the above changes, the MLA module can achieve up to 20.4x speedup for single request and 3.63x for 32 
batched requests on an NVIDIA A100-PCIE-40GB GPU during the decoding phase, as well as 26.2x and 3.52x speedup on 
NVIDIA GeForce RTX 4080 for single and batched requests, respectively.

More detailed description of the modification can be found in 
https://zhuanlan.zhihu.com/p/700214123?utm_psn=1779287628619632640 (in Chinese).
